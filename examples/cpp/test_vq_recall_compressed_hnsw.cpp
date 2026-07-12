#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#include <omp.h>

#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/compressed_hnsw_framework.h"

using namespace std;
using namespace hnswlib;

class StopW {
    std::chrono::steady_clock::time_point time_begin;
public:
    StopW() : time_begin(std::chrono::steady_clock::now()) {}
    double getElapsedTimeMicro() const {
        return static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - time_begin).count());
    }
};

static double* load_fvecs_as_double(const std::string& filename, size_t& num_vectors, size_t& dim) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        std::cerr << "Cannot open " << filename << std::endl;
        return nullptr;
    }

    int32_t d = 0;
    input.read(reinterpret_cast<char*>(&d), 4);
    if (d <= 0) return nullptr;
    dim = static_cast<size_t>(d);

    input.seekg(0, std::ios::end);
    size_t file_size = static_cast<size_t>(input.tellg());
    num_vectors = file_size / (4 + dim * 4);

    double* data = new double[num_vectors * dim];
    std::vector<float> tmp(dim);
    input.seekg(0, std::ios::beg);

    for (size_t i = 0; i < num_vectors; ++i) {
        input.read(reinterpret_cast<char*>(&d), 4);
        input.read(reinterpret_cast<char*>(tmp.data()), dim * 4);
        for (size_t j = 0; j < dim; ++j) {
            data[i * dim + j] = static_cast<double>(tmp[j]);
        }
    }

    return data;
}

static unsigned int* load_ivecs(const std::string& filename, size_t& num_vectors, size_t& dim) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        std::cerr << "Cannot open " << filename << std::endl;
        return nullptr;
    }

    int32_t d = 0;
    input.read(reinterpret_cast<char*>(&d), 4);
    if (d <= 0) return nullptr;
    dim = static_cast<size_t>(d);

    input.seekg(0, std::ios::end);
    size_t file_size = static_cast<size_t>(input.tellg());
    num_vectors = file_size / (4 + dim * 4);

    unsigned int* data = new unsigned int[num_vectors * dim];
    input.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num_vectors; ++i) {
        input.read(reinterpret_cast<char*>(&d), 4);
        input.read(reinterpret_cast<char*>(data + i * dim), dim * 4);
    }

    return data;
}

static size_t count_fvecs(const std::string& filename) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) return 0;

    int32_t d = 0;
    input.read(reinterpret_cast<char*>(&d), 4);
    if (d <= 0) return 0;

    input.seekg(0, std::ios::end);
    size_t file_size = static_cast<size_t>(input.tellg());
    return file_size / (4 + static_cast<size_t>(d) * 4);
}

static std::string chain_tag(int chain_max_length) {
    if (chain_max_length < 0) return "chunlim";
    return "ch" + std::to_string(chain_max_length);
}

static double compute_recall_from_gt(
    size_t qn,
    size_t gt_k,
    const unsigned int* gt_rows,
    const std::vector<labeltype>& results,
    size_t k) {
    if (k > gt_k) {
        k = gt_k;
    }

    size_t correct = 0;
    for (size_t qi = 0; qi < qn; ++qi) {
        std::unordered_set<labeltype> gt_set;
        gt_set.reserve(k * 2 + 1);
        for (size_t j = 0; j < k; ++j) {
            gt_set.insert(static_cast<labeltype>(gt_rows[qi * gt_k + j]));
        }
        for (size_t j = 0; j < k; ++j) {
            if (gt_set.find(results[qi * k + j]) != gt_set.end()) {
                ++correct;
            }
        }
    }

    return static_cast<double>(correct) / static_cast<double>(qn * k);
}

template <typename CodecPolicy>
void run_one_index(const std::string& dataset,
                   const std::string& base_dir,
                   const std::string& algo,
                   int chain_max_length,
                   const std::vector<double>& tls_ratios,
                   const std::vector<int>& efs,
                   const std::vector<int>& ks,
                   int num_rounds,
                   std::ofstream& csv) {
    const std::string query_file = base_dir + dataset + "_test.fvecs";
    const std::string gt_file = base_dir + dataset + "_neighbors.ivecs";
    const std::string train_file = base_dir + dataset + "_train.fvecs";
    const std::string index_file = dataset + "_" + algo + "_" + chain_tag(chain_max_length) + "_pq.bin";

    size_t qn = 0, qdim = 0;
    double* queries = load_fvecs_as_double(query_file, qn, qdim);
    if (!queries) return;

    size_t gt_n = 0, gt_dim = 0;
    unsigned int* gt_rows = load_ivecs(gt_file, gt_n, gt_dim);
    if (!gt_rows) {
        delete[] queries;
        return;
    }
    if (qn != gt_n) {
        std::cerr << "Query/GT size mismatch for " << dataset << std::endl;
        delete[] queries;
        delete[] gt_rows;
        return;
    }

    size_t train_n = count_fvecs(train_file);
    size_t cache_size = train_n > 0 ? std::max<size_t>(1, train_n / 100) : 0;

    std::cout << "\n--- Dataset=" << dataset
              << " codec=" << algo
              << " index=" << index_file
              << " cache_size=" << cache_size << " ---" << std::endl;

    L2SpaceDouble l2space(qdim);
    HierarchicalNSWCABFRAMEWORK<double, CodecPolicy>* index = nullptr;
    try {
        index = new HierarchicalNSWCABFRAMEWORK<double, CodecPolicy>(&l2space, index_file, true, cache_size);
    } catch (const std::exception& e) {
        std::cerr << "Load compressed index failed: " << e.what() << std::endl;
        delete[] queries;
        delete[] gt_rows;
        return;
    }

    const size_t nvecs = index->getCurrentElementCount();
    const size_t index_size_bytes = index->getCompressedIndexSize();
    const double index_kb = static_cast<double>(index_size_bytes) / 1024.0;
    const double vectors_per_kb = index_kb > 0.0 ? static_cast<double>(nvecs) / index_kb : 0.0;

    for (double tls_ratio : tls_ratios) {
        const bool use_tls = tls_ratio > 0.0;
        index->setUseTls(use_tls);
        index->setTlsRatio(tls_ratio);

        for (int k_raw : ks) {
            const size_t k = std::min(static_cast<size_t>(k_raw), gt_dim);
            for (int ef : efs) {
                index->setEf(static_cast<size_t>(ef));

                std::vector<double> recalls;
                std::vector<double> qps_values;
                std::vector<double> latencies;
                recalls.reserve(num_rounds);
                qps_values.reserve(num_rounds);
                latencies.reserve(num_rounds);

                for (int round = 0; round < num_rounds; ++round) {
                    std::vector<labeltype> results(qn * k, static_cast<labeltype>(-1));
                    StopW timer;

                    for (size_t qi = 0; qi < qn; ++qi) {
                        auto pq = index->searchKnn(queries + qi * qdim, k);
                        for (int j = static_cast<int>(k) - 1; j >= 0; --j) {
                            if (pq.empty()) break;
                            results[qi * k + static_cast<size_t>(j)] = pq.top().second;
                            pq.pop();
                        }
                    }

                    double elapsed_us = timer.getElapsedTimeMicro();
                    double qps = elapsed_us > 0.0 ? static_cast<double>(qn) * 1e6 / elapsed_us : 0.0;
                    double latency = qn > 0 ? elapsed_us / static_cast<double>(qn) : 0.0;
                    double recall = compute_recall_from_gt(qn, gt_dim, gt_rows, results, k);

                    recalls.push_back(recall);
                    qps_values.push_back(qps);
                    latencies.push_back(latency);
                }

                double avg_recall = std::accumulate(recalls.begin(), recalls.end(), 0.0) / recalls.size();
                double avg_qps = std::accumulate(qps_values.begin(), qps_values.end(), 0.0) / qps_values.size();
                double avg_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
                double vq = vectors_per_kb * avg_qps;

                std::string index_type = "HNSW+" + algo;
                std::string param = "ef=" + std::to_string(ef) +
                                    ";tls=" + std::to_string(tls_ratio) +
                                    ";chain=" + std::to_string(chain_max_length) +
                                    ";cache=0.01";

                csv << dataset << ','
                    << index_type << ','
                    << param << ','
                    << k << ','
                    << std::fixed << std::setprecision(6) << avg_recall << ','
                    << avg_qps << ','
                    << avg_latency << ','
                    << index_kb << ','
                    << vectors_per_kb << ','
                    << vq << '\n';

                std::cout << index_type
                          << " " << param
                          << " k=" << k
                          << " recall=" << std::fixed << std::setprecision(4) << avg_recall
                          << " qps=" << std::setprecision(2) << avg_qps
                          << " latency=" << avg_latency << "us"
                          << " vq=" << vq << std::endl;

                if (avg_recall > 0.999) {
                    break;
                }
            }
        }
    }

    delete index;
    delete[] queries;
    delete[] gt_rows;
}

static void dispatch_run(const std::string& dataset,
                         const std::string& base_dir,
                         const std::string& algo,
                         int chain_max_length,
                         const std::vector<double>& tls_ratios,
                         const std::vector<int>& efs,
                         const std::vector<int>& ks,
                         int num_rounds,
                         std::ofstream& csv) {
    if (algo == "DeXOR") return run_one_index<codecs::DeXORCodecPolicy>(dataset, base_dir, algo, chain_max_length, tls_ratios, efs, ks, num_rounds, csv);
    if (algo == "Gorilla") return run_one_index<codecs::GorillaCodecPolicy>(dataset, base_dir, algo, chain_max_length, tls_ratios, efs, ks, num_rounds, csv);
    if (algo == "Camel") return run_one_index<codecs::CamelCodecPolicy>(dataset, base_dir, algo, chain_max_length, tls_ratios, efs, ks, num_rounds, csv);
    if (algo == "Elf") return run_one_index<codecs::ElfCodecPolicy>(dataset, base_dir, algo, chain_max_length, tls_ratios, efs, ks, num_rounds, csv);
    throw std::runtime_error("Unknown codec: " + algo);
}

int main(int argc, char** argv) {
    omp_set_num_threads(1);

    std::string base_dir = "../datasets/hdf5files/";
    std::string out_csv = "compressed_hnsw_vq_recall_results.csv";
    int chain_max_length = 2;
    int num_rounds = 1;

    std::vector<std::string> datasets = {
        "sift-128-euclidean",
        "deep-image-96-angular",
        "mnist-784-euclidean",
        "fashion-mnist-784-euclidean",
        "gist-960-euclidean"
    };
    std::vector<std::string> algorithms = {"DeXOR", "Gorilla", "Camel", "Elf"};
    std::vector<double> tls_ratios = {0.0, 0.1, 0.2, 0.3, 0.5};
    std::vector<int> efs = {10, 20, 30, 40, 60, 80, 100, 150, 200, 250, 300, 350, 400, 500};
    std::vector<int> ks = {1, 10};

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--base_dir" && i + 1 < argc) {
            base_dir = argv[++i];
        } else if (arg == "--output_csv" && i + 1 < argc) {
            out_csv = argv[++i];
        } else if (arg == "--chain_max" && i + 1 < argc) {
            chain_max_length = std::stoi(argv[++i]);
        } else if (arg == "--num_rounds" && i + 1 < argc) {
            num_rounds = std::max(1, std::stoi(argv[++i]));
        } else if (arg == "--dataset" && i + 1 < argc) {
            datasets.clear();
            while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                datasets.push_back(argv[++i]);
            }
        } else if (arg == "--algorithm" && i + 1 < argc) {
            algorithms.clear();
            while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                algorithms.push_back(argv[++i]);
            }
        } else if (arg == "--tls_ratio" && i + 1 < argc) {
            tls_ratios.clear();
            while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                tls_ratios.push_back(std::stod(argv[++i]));
            }
        } else if (arg == "--ef" && i + 1 < argc) {
            efs.clear();
            while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                efs.push_back(std::stoi(argv[++i]));
            }
        } else if (arg == "--k" && i + 1 < argc) {
            ks.clear();
            while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                ks.push_back(std::stoi(argv[++i]));
            }
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << std::endl;
            return -1;
        }
    }

    if (!base_dir.empty() && base_dir.back() != '/' && base_dir.back() != '\\') {
        base_dir += '/';
    }

    std::ofstream csv(out_csv);
    if (!csv.is_open()) {
        std::cerr << "Failed to open csv: " << out_csv << std::endl;
        return -1;
    }
    csv << "Dataset,IndexType,Param,K,Recall,QPS,Latency(us),IndexSizeKB,VectorsPerKB,VQ\n";

    for (const auto& ds : datasets) {
        for (const auto& algo : algorithms) {
            dispatch_run(ds, base_dir, algo, chain_max_length, tls_ratios, efs, ks, num_rounds, csv);
        }
    }

    csv.close();
    std::cout << "\nSaved CSV: " << out_csv << std::endl;
    return 0;
}
