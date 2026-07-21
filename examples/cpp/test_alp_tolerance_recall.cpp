#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sstream>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include <omp.h>

#include "../../hnswlib/hnswalg.h"
#include "../../hnswlib/hnswalg_simplified_alp_PQ.h"

using namespace hnswlib;

namespace {

struct Config {
    std::string base_dir = "../datasets/hdf5files/";
    std::string dataset = "deep-image-96-angular";
    std::string output_csv = "alp_tolerance_deep_results.csv";
    std::string raw_index_path;
    size_t M = 16;
    size_t ef_construction = 200;
    int build_threads = 32;
    int search_threads = 1;
    int num_rounds = 3;
    std::vector<size_t> efs = {10, 20, 40, 80, 120, 200, 400, 600, 800, 1000};
    std::vector<size_t> ks = {1, 10};
    std::vector<float> tolerances = {1e-4f, 1e-10f};
};

struct Evaluation {
    double recall = 0.0;
    double qps = 0.0;
    double latency_us = 0.0;
};

std::vector<float> load_fvecs(const std::string& path, size_t& count, size_t& dim) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot open fvecs file: " + path);
    }

    int32_t file_dim = 0;
    input.read(reinterpret_cast<char*>(&file_dim), sizeof(file_dim));
    if (!input || file_dim <= 0) {
        throw std::runtime_error("Invalid fvecs header: " + path);
    }
    dim = static_cast<size_t>(file_dim);

    input.seekg(0, std::ios::end);
    const size_t bytes = static_cast<size_t>(input.tellg());
    const size_t record_bytes = sizeof(int32_t) + dim * sizeof(float);
    if (bytes % record_bytes != 0) {
        throw std::runtime_error("Unexpected fvecs size: " + path);
    }
    count = bytes / record_bytes;
    input.seekg(0, std::ios::beg);

    std::vector<float> data(count * dim);
    for (size_t i = 0; i < count; ++i) {
        int32_t record_dim = 0;
        input.read(reinterpret_cast<char*>(&record_dim), sizeof(record_dim));
        input.read(reinterpret_cast<char*>(data.data() + i * dim),
                   static_cast<std::streamsize>(dim * sizeof(float)));
        if (!input || record_dim != file_dim) {
            throw std::runtime_error("Malformed fvecs record in: " + path);
        }
    }
    return data;
}

std::vector<uint32_t> load_ivecs(const std::string& path, size_t& count, size_t& dim) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot open ivecs file: " + path);
    }

    int32_t file_dim = 0;
    input.read(reinterpret_cast<char*>(&file_dim), sizeof(file_dim));
    if (!input || file_dim <= 0) {
        throw std::runtime_error("Invalid ivecs header: " + path);
    }
    dim = static_cast<size_t>(file_dim);

    input.seekg(0, std::ios::end);
    const size_t bytes = static_cast<size_t>(input.tellg());
    const size_t record_bytes = sizeof(int32_t) + dim * sizeof(uint32_t);
    if (bytes % record_bytes != 0) {
        throw std::runtime_error("Unexpected ivecs size: " + path);
    }
    count = bytes / record_bytes;
    input.seekg(0, std::ios::beg);

    std::vector<uint32_t> data(count * dim);
    for (size_t i = 0; i < count; ++i) {
        int32_t record_dim = 0;
        input.read(reinterpret_cast<char*>(&record_dim), sizeof(record_dim));
        input.read(reinterpret_cast<char*>(data.data() + i * dim),
                   static_cast<std::streamsize>(dim * sizeof(uint32_t)));
        if (!input || record_dim != file_dim) {
            throw std::runtime_error("Malformed ivecs record in: " + path);
        }
    }
    return data;
}

template <typename T>
std::unique_ptr<HierarchicalNSW<T>> build_raw_index(
    SpaceInterface<T>* space,
    const std::vector<T>& base,
    size_t count,
    size_t dim,
    size_t M,
    size_t ef_construction,
    int build_threads) {
    auto index = std::make_unique<HierarchicalNSW<T>>(space, count, M, ef_construction, 100);
    omp_set_num_threads(build_threads);
#pragma omp parallel for schedule(dynamic)
    for (long long i = 0; i < static_cast<long long>(count); ++i) {
        index->addPoint(base.data() + static_cast<size_t>(i) * dim,
                        static_cast<labeltype>(i));
    }
    return index;
}

void copy_raw_index_to_alp(
    const HierarchicalNSW<float>& source,
    HierarchicalNSWALPSIMPLIFIEDPQ<float>& destination) {
    if (source.max_elements_ != destination.max_elements_ ||
        source.M_ != destination.M_ ||
        source.size_data_per_element_ != destination.size_data_per_element_ ||
        source.size_links_per_element_ != destination.size_links_per_element_) {
        throw std::runtime_error("Incompatible raw/ALP HNSW layouts");
    }

    const size_t count = source.cur_element_count;
    destination.cur_element_count = count;
    destination.num_deleted_ = source.num_deleted_.load();
    destination.ef_ = source.ef_;
    destination.maxlevel_ = source.maxlevel_;
    destination.enterpoint_node_ = source.enterpoint_node_;
    destination.element_levels_ = source.element_levels_;
    destination.label_lookup_ = source.label_lookup_;
    destination.deleted_elements = source.deleted_elements;
    destination.is_compacted_ = false;
    destination.pq_data_.clear();
    destination.pq_centroids_.clear();
    destination.uncompressed_mask_.clear();
    std::fill(destination.level0_element_start_positions_.begin(),
              destination.level0_element_start_positions_.end(), 0);
    std::memcpy(destination.data_level0_memory_.data(), source.data_level0_memory_,
                count * source.size_data_per_element_);
    std::memset(destination.linkLists_, 0, sizeof(char*) * destination.max_elements_);

    for (tableint id = 0; id < count; ++id) {
        if (source.element_levels_[id] > 0) {
            const size_t bytes = static_cast<size_t>(source.element_levels_[id]) *
                                 source.size_links_per_element_;
            destination.linkLists_[id] = static_cast<char*>(std::malloc(bytes));
            if (destination.linkLists_[id] == nullptr) {
                throw std::bad_alloc();
            }
            std::memcpy(destination.linkLists_[id], source.linkLists_[id], bytes);
        }
    }
}

double recall_at_k(const std::vector<labeltype>& results,
                   const std::vector<uint32_t>& ground_truth,
                   size_t query_count,
                   size_t ground_truth_k,
                   size_t k) {
    if (k > ground_truth_k) {
        throw std::runtime_error("Requested k exceeds the ground-truth width");
    }

    size_t correct = 0;
    for (size_t query_id = 0; query_id < query_count; ++query_id) {
        std::unordered_set<labeltype> expected;
        for (size_t rank = 0; rank < k; ++rank) {
            expected.insert(static_cast<labeltype>(ground_truth[query_id * ground_truth_k + rank]));
        }
        for (size_t rank = 0; rank < k; ++rank) {
            if (expected.count(results[query_id * k + rank]) != 0) {
                ++correct;
            }
        }
    }
    return static_cast<double>(correct) / static_cast<double>(query_count * k);
}

template <typename Index>
Evaluation evaluate(Index& index,
                    const std::vector<float>& queries,
                    size_t query_count,
                    size_t dim,
                    const std::vector<uint32_t>& ground_truth,
                    size_t ground_truth_k,
                    size_t ef,
                    size_t k,
                    int num_rounds,
                    int search_threads) {
    index.setEf(ef);
    omp_set_num_threads(search_threads);

    std::vector<double> recalls;
    std::vector<double> qps_values;
    std::vector<double> latency_values;
    recalls.reserve(num_rounds);
    qps_values.reserve(num_rounds);
    latency_values.reserve(num_rounds);

    for (int round = 0; round < num_rounds; ++round) {
        std::vector<labeltype> results(query_count * k);
        const auto start = std::chrono::steady_clock::now();
        for (size_t query_id = 0; query_id < query_count; ++query_id) {
            auto queue = index.searchKnn(queries.data() + query_id * dim, k);
            for (size_t rank = k; rank > 0; --rank) {
                if (queue.empty()) {
                    results[query_id * k + rank - 1] = static_cast<labeltype>(-1);
                } else {
                    results[query_id * k + rank - 1] = queue.top().second;
                    queue.pop();
                }
            }
        }
        const auto end = std::chrono::steady_clock::now();
        const double elapsed_us = static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

        recalls.push_back(recall_at_k(results, ground_truth, query_count, ground_truth_k, k));
        qps_values.push_back(static_cast<double>(query_count) * 1e6 / elapsed_us);
        latency_values.push_back(elapsed_us / static_cast<double>(query_count));
    }

    return {
        std::accumulate(recalls.begin(), recalls.end(), 0.0) / recalls.size(),
        std::accumulate(qps_values.begin(), qps_values.end(), 0.0) / qps_values.size(),
        std::accumulate(latency_values.begin(), latency_values.end(), 0.0) / latency_values.size()
    };
}

template <typename Index>
void write_rows(std::ofstream& csv,
                const Config& config,
                const std::string& index_type,
                const std::string& tolerance,
                Index& index,
                size_t index_bytes,
                const std::vector<float>& queries,
                size_t query_count,
                size_t dim,
                const std::vector<uint32_t>& ground_truth,
                size_t ground_truth_k) {
    const double index_kb = static_cast<double>(index_bytes) / 1024.0;
    const double vectors_per_kb = static_cast<double>(index.getCurrentElementCount()) / index_kb;

    for (size_t k : config.ks) {
        for (size_t ef : config.efs) {
            const Evaluation metrics = evaluate(index, queries, query_count, dim, ground_truth,
                                                ground_truth_k, ef, k, config.num_rounds,
                                                config.search_threads);
            const double vq_capacity = vectors_per_kb * metrics.qps;
            csv << config.dataset << ',' << index_type << ',' << tolerance << ',' << ef << ',' << k << ','
                << std::fixed << std::setprecision(6) << metrics.recall << ','
                << std::setprecision(2) << metrics.qps << ',' << metrics.latency_us << ','
                << index_kb << ',' << vectors_per_kb << ',' << vq_capacity << ','
                << config.num_rounds << ',' << config.search_threads << '\n';
            csv.flush();
            std::cout << index_type << " tolerance=" << tolerance << " ef=" << ef << " k=" << k
                      << " recall=" << std::setprecision(6) << metrics.recall
                      << " qps=" << std::setprecision(2) << metrics.qps
                      << " latency_us=" << metrics.latency_us
                      << " vq=" << vq_capacity << std::endl;
        }
    }
}

template <typename Parser>
void parse_values(int argc, char** argv, int& i,
                  std::vector<typename std::invoke_result<Parser, const std::string&>::type>& output,
                  Parser parser) {
    output.clear();
    while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
        output.push_back(parser(argv[++i]));
    }
    if (output.empty()) {
        throw std::runtime_error("Expected at least one value after " + std::string(argv[i]));
    }
}

Config parse_config(int argc, char** argv) {
    Config config;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--base_dir" && i + 1 < argc) {
            config.base_dir = argv[++i];
        } else if (arg == "--dataset" && i + 1 < argc) {
            config.dataset = argv[++i];
        } else if (arg == "--output_csv" && i + 1 < argc) {
            config.output_csv = argv[++i];
        } else if (arg == "--raw_index" && i + 1 < argc) {
            config.raw_index_path = argv[++i];
        } else if (arg == "--M" && i + 1 < argc) {
            config.M = static_cast<size_t>(std::stoull(argv[++i]));
        } else if (arg == "--ef_construction" && i + 1 < argc) {
            config.ef_construction = static_cast<size_t>(std::stoull(argv[++i]));
        } else if (arg == "--build_threads" && i + 1 < argc) {
            config.build_threads = std::stoi(argv[++i]);
        } else if (arg == "--search_threads" && i + 1 < argc) {
            config.search_threads = std::stoi(argv[++i]);
        } else if (arg == "--num_rounds" && i + 1 < argc) {
            config.num_rounds = std::stoi(argv[++i]);
        } else if (arg == "--ef") {
            parse_values(argc, argv, i, config.efs,
                         [](const std::string& value) { return static_cast<size_t>(std::stoull(value)); });
        } else if (arg == "--k") {
            parse_values(argc, argv, i, config.ks,
                         [](const std::string& value) { return static_cast<size_t>(std::stoull(value)); });
        } else if (arg == "--tolerance") {
            parse_values(argc, argv, i, config.tolerances,
                         [](const std::string& value) { return std::stof(value); });
        } else {
            throw std::runtime_error("Unknown or incomplete argument: " + arg);
        }
    }

    if (config.base_dir.empty()) {
        throw std::runtime_error("--base_dir must not be empty");
    }
    if (config.base_dir.back() != '/' && config.base_dir.back() != '\\') {
        config.base_dir.push_back('/');
    }
    if (config.build_threads <= 0 || config.search_threads <= 0 || config.num_rounds <= 0) {
        throw std::runtime_error("thread counts and --num_rounds must be positive");
    }
    return config;
}

std::string tolerance_tag(float tolerance) {
    std::ostringstream output;
    output << std::scientific << std::setprecision(0) << tolerance;
    return output.str();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Config config = parse_config(argc, argv);
        size_t base_count = 0, base_dim = 0;
        size_t query_count = 0, query_dim = 0;
        size_t ground_truth_count = 0, ground_truth_k = 0;
        const std::vector<float> base = load_fvecs(
            config.base_dir + config.dataset + "_train.fvecs", base_count, base_dim);
        const std::vector<float> queries = load_fvecs(
            config.base_dir + config.dataset + "_test.fvecs", query_count, query_dim);
        const std::vector<uint32_t> ground_truth = load_ivecs(
            config.base_dir + config.dataset + "_neighbors.ivecs", ground_truth_count, ground_truth_k);
        if (base_dim != query_dim || query_count != ground_truth_count) {
            throw std::runtime_error("Dataset/query/ground-truth metadata mismatch");
        }

        L2Space space(base_dim);
        std::unique_ptr<HierarchicalNSW<float>> raw;
        if (config.raw_index_path.empty()) {
            std::cout << "Building shared raw HNSW graph: dataset=" << config.dataset
                      << ", vectors=" << base_count << ", dim=" << base_dim
                      << ", M=" << config.M << ", efConstruction=" << config.ef_construction
                      << ", build_threads=" << config.build_threads << std::endl;
            raw = build_raw_index<float>(&space, base, base_count, base_dim, config.M,
                                         config.ef_construction, config.build_threads);
        } else {
            std::cout << "Loading shared raw HNSW graph: " << config.raw_index_path << std::endl;
            raw = std::make_unique<HierarchicalNSW<float>>(&space, config.raw_index_path, false);
            if (raw->getCurrentElementCount() != base_count) {
                throw std::runtime_error("Loaded raw index count does not match the base data");
            }
        }

        std::ofstream csv(config.output_csv);
        if (!csv) {
            throw std::runtime_error("Cannot open output CSV: " + config.output_csv);
        }
        csv << "Dataset,IndexType,Tolerance,EF,K,Recall,QPS,Latency(us),IndexSizeKB,"
               "VectorsPerKB,VQCapacity,NumRounds,SearchThreads\n";
        csv.flush();

        write_rows(csv, config, "HNSW", "N/A", *raw, raw->indexFileSize(), queries, query_count,
                   base_dim, ground_truth, ground_truth_k);

        for (float tolerance : config.tolerances) {
            std::cout << "Building HNSW+ALP from the shared raw graph, tolerance="
                      << tolerance_tag(tolerance) << std::endl;
            HierarchicalNSWALPSIMPLIFIEDPQ<float> compressed(
                &space, raw->max_elements_, config.M, config.ef_construction, 100, false, tolerance);
            copy_raw_index_to_alp(*raw, compressed);
            compressed.setUseTLS(false);
            compressed.compress_dataset(0);
            write_rows(csv, config, "HNSW+ALP", tolerance_tag(tolerance), compressed,
                       compressed.getCompressedIndexSize(), queries, query_count, base_dim,
                       ground_truth, ground_truth_k);
        }

        std::cout << "Results saved to: " << config.output_csv << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << std::endl;
        return 1;
    }
}
