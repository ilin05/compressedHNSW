#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <omp.h>

#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>

namespace {

class StopWatch {
 public:
  StopWatch() : start_(std::chrono::steady_clock::now()) {}

  double elapsed_us() const {
    const auto end = std::chrono::steady_clock::now();
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count());
  }

 private:
  std::chrono::steady_clock::time_point start_;
};

float* load_fvecs(const std::string& filename, size_t& num_vectors, size_t& dim) {
  std::ifstream input(filename, std::ios::binary);
  if (!input) {
    return nullptr;
  }

  int32_t file_dim = 0;
  input.read(reinterpret_cast<char*>(&file_dim), sizeof(file_dim));
  if (!input || file_dim <= 0) {
    return nullptr;
  }
  dim = static_cast<size_t>(file_dim);

  input.seekg(0, std::ios::end);
  const size_t file_size = static_cast<size_t>(input.tellg());
  const size_t record_size = sizeof(int32_t) + dim * sizeof(float);
  if (file_size % record_size != 0) {
    return nullptr;
  }
  num_vectors = file_size / record_size;

  auto* data = new float[num_vectors * dim];
  input.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num_vectors; ++i) {
    input.read(reinterpret_cast<char*>(&file_dim), sizeof(file_dim));
    if (!input || static_cast<size_t>(file_dim) != dim) {
      delete[] data;
      return nullptr;
    }
    input.read(reinterpret_cast<char*>(data + i * dim), dim * sizeof(float));
    if (!input) {
      delete[] data;
      return nullptr;
    }
  }
  return data;
}

unsigned int* load_ivecs(const std::string& filename, size_t& num_vectors, size_t& dim) {
  std::ifstream input(filename, std::ios::binary);
  if (!input) {
    return nullptr;
  }

  int32_t file_dim = 0;
  input.read(reinterpret_cast<char*>(&file_dim), sizeof(file_dim));
  if (!input || file_dim <= 0) {
    return nullptr;
  }
  dim = static_cast<size_t>(file_dim);

  input.seekg(0, std::ios::end);
  const size_t file_size = static_cast<size_t>(input.tellg());
  const size_t record_size = sizeof(int32_t) + dim * sizeof(unsigned int);
  if (file_size % record_size != 0) {
    return nullptr;
  }
  num_vectors = file_size / record_size;

  auto* data = new unsigned int[num_vectors * dim];
  input.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num_vectors; ++i) {
    input.read(reinterpret_cast<char*>(&file_dim), sizeof(file_dim));
    if (!input || static_cast<size_t>(file_dim) != dim) {
      delete[] data;
      return nullptr;
    }
    input.read(reinterpret_cast<char*>(data + i * dim), dim * sizeof(unsigned int));
    if (!input) {
      delete[] data;
      return nullptr;
    }
  }
  return data;
}

size_t file_size_bytes(const std::string& filename) {
  std::ifstream input(filename, std::ios::binary | std::ios::ate);
  if (!input) {
    throw std::runtime_error("Cannot open index: " + filename);
  }
  return static_cast<size_t>(input.tellg());
}

double recall_at_k(size_t query_count, size_t gt_width, const unsigned int* ground_truth,
                   const std::vector<faiss::idx_t>& results, size_t k) {
  if (k == 0 || k > gt_width) {
    throw std::runtime_error("Requested k is outside the ground-truth width");
  }

  size_t hits = 0;
  for (size_t qi = 0; qi < query_count; ++qi) {
    std::unordered_set<faiss::idx_t> expected;
    for (size_t rank = 0; rank < k; ++rank) {
      expected.insert(static_cast<faiss::idx_t>(ground_truth[qi * gt_width + rank]));
    }
    for (size_t rank = 0; rank < k; ++rank) {
      if (expected.count(results[qi * k + rank]) != 0) {
        ++hits;
      }
    }
  }
  return static_cast<double>(hits) / static_cast<double>(query_count * k);
}

template <typename T>
double average(const std::vector<T>& values) {
  return std::accumulate(values.begin(), values.end(), 0.0) /
         static_cast<double>(values.size());
}

void print_usage(const char* program) {
  std::cerr
      << "Usage: " << program
      << " [--base_dir DIR] [--index_dir DIR] [--output CSV] [--dataset NAME]"
      << " [--pq_dims_per_subq P1 P2 ...] [--ef E1 E2 ...] [--k K1 K2 ...]"
      << " [--num_rounds N]\n"
      << "\nFor HNSWPQ_p, p is the number of original dimensions per PQ subquantizer.\n"
      << "On SIFT-128, p={1,2,4,8} corresponds to m={128,64,32,16} with 8-bit codes.\n";
}

}  // namespace

int main(int argc, char** argv) {
  omp_set_num_threads(1);

  std::string base_dir = "../datasets/hdf5files/";
  std::string index_dir = ".";
  std::string output_path = "hnswpq_tradeoff_sift.csv";
  std::string dataset = "sift-128-euclidean";
  std::vector<int> dims_per_subquantizer = {1, 2, 4, 8};
  std::vector<int> efs = {10, 20, 40, 80, 120, 200, 400, 600, 800, 1000};
  std::vector<int> ks = {1, 10};
  int num_rounds = 3;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto consume_values = [&](std::vector<int>& destination) {
      destination.clear();
      while (i + 1 < argc && argv[i + 1][0] != '-') {
        destination.push_back(std::stoi(argv[++i]));
      }
    };

    if (arg == "--base_dir" && i + 1 < argc) {
      base_dir = argv[++i];
    } else if (arg == "--index_dir" && i + 1 < argc) {
      index_dir = argv[++i];
    } else if (arg == "--output" && i + 1 < argc) {
      output_path = argv[++i];
    } else if (arg == "--dataset" && i + 1 < argc) {
      dataset = argv[++i];
    } else if (arg == "--pq_dims_per_subq") {
      consume_values(dims_per_subquantizer);
    } else if (arg == "--ef") {
      consume_values(efs);
    } else if (arg == "--k") {
      consume_values(ks);
    } else if (arg == "--num_rounds" && i + 1 < argc) {
      num_rounds = std::stoi(argv[++i]);
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown or incomplete argument: " << arg << "\n";
      print_usage(argv[0]);
      return 1;
    }
  }

  if (base_dir.back() != '/' && base_dir.back() != '\\') {
    base_dir += '/';
  }
  if (index_dir.back() != '/' && index_dir.back() != '\\') {
    index_dir += '/';
  }
  if (num_rounds <= 0 || dims_per_subquantizer.empty() || efs.empty() || ks.empty()) {
    std::cerr << "All sweeps must be non-empty and --num_rounds must be positive.\n";
    return 1;
  }

  const std::string query_path = base_dir + dataset + "_test.fvecs";
  const std::string gt_path = base_dir + dataset + "_neighbors.ivecs";
  size_t query_count = 0;
  size_t dim = 0;
  std::unique_ptr<float[]> queries(load_fvecs(query_path, query_count, dim));
  if (!queries) {
    std::cerr << "Cannot load queries from " << query_path << "\n";
    return 1;
  }

  size_t gt_count = 0;
  size_t gt_width = 0;
  std::unique_ptr<unsigned int[]> ground_truth(load_ivecs(gt_path, gt_count, gt_width));
  if (!ground_truth || query_count != gt_count) {
    std::cerr << "Cannot load ground truth or query/ground-truth counts differ.\n";
    return 1;
  }

  std::ofstream csv(output_path);
  if (!csv) {
    std::cerr << "Cannot open output CSV " << output_path << "\n";
    return 1;
  }
  csv << "Dataset,IndexType,Param,K,Recall,QPS,Latency(us),IndexSizeKB,VectorsPerKB,VQ,"
         "PQSubquantizers,PQSubvectorDim,PQCodeBytes,PayloadCompressionRatio\n";
  csv << std::fixed << std::setprecision(6);

  std::cout << "[HNSW+PQ trade-off] dataset=" << dataset << " dim=" << dim
            << " rounds=" << num_rounds << " threads=1\n";

  for (const int dims_per_subq : dims_per_subquantizer) {
    if (dims_per_subq <= 0 || dim % static_cast<size_t>(dims_per_subq) != 0) {
      std::cerr << "Skip HNSWPQ_" << dims_per_subq << ": dim=" << dim
                << " is not divisible by p=" << dims_per_subq << "\n";
      continue;
    }

    const size_t pq_m = dim / static_cast<size_t>(dims_per_subq);
    const size_t pq_code_bytes = pq_m;  // m 8-bit PQ codes.
    const double payload_compression = static_cast<double>(dim * sizeof(float)) /
                                       static_cast<double>(pq_code_bytes);
    const std::string index_name = "HNSWPQ_" + std::to_string(dims_per_subq);
    const std::string index_path =
        index_dir + dataset + "_train.fvecs_" + index_name + ".bin";

    std::unique_ptr<faiss::Index> index;
    try {
      index.reset(faiss::read_index(index_path.c_str()));
    } catch (const std::exception& error) {
      std::cerr << "Skip " << index_name << ": " << error.what() << "\n";
      continue;
    }
    if (!index) {
      std::cerr << "Skip " << index_name << ": failed to load " << index_path << "\n";
      continue;
    }

    auto* hnsw_index = dynamic_cast<faiss::IndexHNSW*>(index.get());
    if (hnsw_index == nullptr) {
      std::cerr << "Skip " << index_name << ": loaded index is not faiss::IndexHNSW\n";
      continue;
    }

    const double index_kb = static_cast<double>(file_size_bytes(index_path)) / 1024.0;
    const double vectors_per_kb = static_cast<double>(index->ntotal) / index_kb;
    std::cout << "Loaded " << index_name << " (m=" << pq_m
              << ", code=" << pq_code_bytes << " B/vector, payload CR="
              << payload_compression << "x)\n";

    for (const int k : ks) {
      if (k <= 0 || static_cast<size_t>(k) > gt_width) {
        std::cerr << "Skip invalid k=" << k << " for ground-truth width=" << gt_width << "\n";
        continue;
      }

      for (const int ef : efs) {
        if (ef < k) {
          continue;
        }
        hnsw_index->hnsw.efSearch = ef;

        std::vector<double> recalls;
        std::vector<double> qps_values;
        std::vector<double> latency_values;
        recalls.reserve(num_rounds);
        qps_values.reserve(num_rounds);
        latency_values.reserve(num_rounds);

        for (int round = 0; round < num_rounds; ++round) {
          std::vector<faiss::idx_t> ids(query_count * static_cast<size_t>(k));
          std::vector<float> distances(query_count * static_cast<size_t>(k));
          StopWatch timer;
          index->search(static_cast<faiss::idx_t>(query_count), queries.get(), k,
                        distances.data(), ids.data());
          const double elapsed_us = timer.elapsed_us();
          recalls.push_back(recall_at_k(query_count, gt_width, ground_truth.get(), ids, k));
          qps_values.push_back(static_cast<double>(query_count) * 1e6 / elapsed_us);
          latency_values.push_back(elapsed_us / static_cast<double>(query_count));
        }

        const double mean_recall = average(recalls);
        const double mean_qps = average(qps_values);
        const double mean_latency = average(latency_values);
        const double vq = vectors_per_kb * mean_qps;
        const std::string param = "ef=" + std::to_string(ef) + ";m=" +
                                  std::to_string(pq_m) + ";bits=8";

        csv << dataset << ",HNSW+PQ," << param << ',' << k << ',' << mean_recall << ','
            << mean_qps << ',' << mean_latency << ',' << index_kb << ',' << vectors_per_kb
            << ',' << vq << ',' << pq_m << ',' << dims_per_subq << ',' << pq_code_bytes
            << ',' << payload_compression << '\n';
        csv.flush();

        std::cout << index_name << " k=" << k << " ef=" << ef
                  << " recall=" << mean_recall << " QPS=" << mean_qps
                  << " latency=" << mean_latency << " us VQ=" << vq << '\n';
      }
    }
  }

  return 0;
}
