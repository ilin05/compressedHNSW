#include <chrono>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <queue>
#include <string>
#include <unordered_set>
#include <vector>
#include <omp.h>

#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/compressed_hnsw_framework.h"

using namespace std;
using namespace hnswlib;

class StopW {
    std::chrono::steady_clock::time_point begin_;
public:
    StopW() : begin_(std::chrono::steady_clock::now()) {}
    double elapsed_us() const {
        return static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - begin_).count());
    }
};

double* load_fvecs_as_double(const std::string& filename, size_t& num_vectors, size_t& dim) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        std::cerr << "Cannot open " << filename << std::endl;
        return nullptr;
    }

    int32_t d = 0;
    input.read((char*)&d, 4);
    dim = static_cast<size_t>(d);

    input.seekg(0, std::ios::end);
    size_t file_size = static_cast<size_t>(input.tellg());
    num_vectors = file_size / (4 + dim * 4);

    double* data = new double[num_vectors * dim];
    float* tmp = new float[dim];
    input.seekg(0, std::ios::beg);

    for (size_t i = 0; i < num_vectors; ++i) {
        input.read((char*)&d, 4);
        input.read((char*)tmp, dim * 4);
        for (size_t j = 0; j < dim; ++j) {
            data[i * dim + j] = static_cast<double>(tmp[j]);
        }
    }

    delete[] tmp;
    return data;
}

unsigned int* load_ivecs(const std::string& filename, size_t& num_vectors, size_t& dim) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        std::cerr << "Cannot open " << filename << std::endl;
        return nullptr;
    }

    int32_t d = 0;
    input.read((char*)&d, 4);
    dim = static_cast<size_t>(d);

    input.seekg(0, std::ios::end);
    size_t file_size = static_cast<size_t>(input.tellg());
    num_vectors = file_size / (4 + dim * 4);

    unsigned int* data = new unsigned int[num_vectors * dim];
    input.seekg(0, std::ios::beg);

    for (size_t i = 0; i < num_vectors; ++i) {
        input.read((char*)&d, 4);
        input.read((char*)(data + i * dim), dim * 4);
    }

    return data;
}

size_t count_fvecs(const std::string& filename) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) return 0;

    int32_t d = 0;
    input.read((char*)&d, 4);
    if (d <= 0) return 0;

    input.seekg(0, std::ios::end);
    size_t file_size = static_cast<size_t>(input.tellg());
    return file_size / (4 + static_cast<size_t>(d) * 4);
}

static std::string chain_tag(int chain_max_length) {
    if (chain_max_length < 0) return "chunlim";
    return "ch" + std::to_string(chain_max_length);
}

struct SearchRoundMetrics {
    size_t query_count = 0;
    double recall = 0.0;
    double mean_latency_us = 0.0;
    double p99_latency_us = 0.0;
    double qps = 0.0;
    long decoding_calls = 0;
    long get_original_data_calls = 0;
    long backtrack_hops = 0;
    double avg_decode_per_query = 0.0;
    double avg_backtrack_hops_per_call = 0.0;
};

template <typename CodecPolicy>
SearchRoundMetrics run_search_round(
    HierarchicalNSWCABFRAMEWORK<double, CodecPolicy>* index,
    const double* queries,
    size_t qsize,
    size_t qdim,
    const unsigned int* gt,
    size_t gt_dim,
    size_t topk,
    bool enable_profiling) {
    SearchRoundMetrics metrics;
    metrics.query_count = qsize;

    index->resetProfilingMetrics();
    index->setProfilingMetrics(enable_profiling);

    size_t correct = 0;
    std::vector<double> latencies_us;
    latencies_us.reserve(qsize);

    StopW total_timer;
    for (long long i = 0; i < static_cast<long long>(qsize); ++i) {
        StopW query_timer;
        auto result = index->searchKnn(queries + i * qdim, topk);
        latencies_us.push_back(query_timer.elapsed_us());

        std::unordered_set<labeltype> gt_set;
        for (size_t j = 0; j < topk; ++j) {
            gt_set.insert(gt[i * gt_dim + j]);
        }

        while (!result.empty()) {
            if (gt_set.find(result.top().second) != gt_set.end()) {
                ++correct;
            }
            result.pop();
        }
    }

    double total_us = total_timer.elapsed_us();
    metrics.recall = static_cast<double>(correct) / static_cast<double>(qsize * topk);
    metrics.mean_latency_us = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0)
                              / static_cast<double>(latencies_us.size());
    std::sort(latencies_us.begin(), latencies_us.end());
    size_t p99_idx = std::min(
        latencies_us.size() - 1,
        static_cast<size_t>(std::ceil(latencies_us.size() * 0.99)) - 1);
    metrics.p99_latency_us = latencies_us[p99_idx];
    metrics.qps = total_us > 0.0 ? static_cast<double>(qsize) / (total_us / 1e6) : 0.0;

    metrics.decoding_calls = index->getDecodingCallCount();
    metrics.get_original_data_calls = index->getOriginalDataCallCount();
    metrics.backtrack_hops = index->getOriginalDataBacktrackHops();
    metrics.avg_decode_per_query = qsize > 0
                                       ? static_cast<double>(metrics.decoding_calls) / static_cast<double>(qsize)
                                       : 0.0;
    metrics.avg_backtrack_hops_per_call = metrics.get_original_data_calls > 0
                                              ? static_cast<double>(metrics.backtrack_hops) /
                                                    static_cast<double>(metrics.get_original_data_calls)
                                              : 0.0;

    index->setProfilingMetrics(false);
    return metrics;
}

template <typename CodecPolicy>
void search_one_config(const std::string& dataset,
                       const std::string& base_dir,
                       const std::string& algo,
                       int chain_max_length,
                       bool use_cache,
                       bool use_tls,
                       double tls_ratio,
                       size_t k,
                       const std::vector<size_t>& efs,
                       bool enable_profiling,
                       int num_rounds,
                       std::ofstream& csv) {
    const std::string query_file = base_dir + dataset + "_test.fvecs";
    const std::string gt_file = base_dir + dataset + "_neighbors.ivecs";
    const std::string train_file = base_dir + dataset + "_train.fvecs";

    size_t qsize = 0, qdim = 0;
    double* queries = load_fvecs_as_double(query_file, qsize, qdim);
    if (!queries) return;

    size_t gt_num = 0, gt_dim = 0;
    unsigned int* gt = load_ivecs(gt_file, gt_num, gt_dim);
    if (!gt) {
        delete[] queries;
        return;
    }

    if (qsize != gt_num) {
        std::cerr << "Query size and GT size mismatch on dataset: " << dataset << std::endl;
        delete[] queries;
        delete[] gt;
        return;
    }

    size_t train_num = count_fvecs(train_file);
    size_t cache_size = use_cache ? std::max<size_t>(1, train_num / 100) : 0;

    const std::string index_file = dataset + "_" + algo + "_" + chain_tag(chain_max_length) + "_pq.bin";

    L2SpaceDouble l2space(qdim);
    auto* index = new HierarchicalNSWCABFRAMEWORK<double, CodecPolicy>(
        &l2space, index_file, true, cache_size);
    index->setUseTls(use_tls);
    index->setTlsRatio(tls_ratio);

    size_t topk = std::min(k, gt_dim);

    std::cout << "\n=== Search dataset=" << dataset
              << " algo=" << algo
              << " chain_max=" << chain_max_length
              << " cache=" << (use_cache ? 1 : 0)
              << " tls=" << (use_tls ? 1 : 0)
              << " tls_ratio=" << tls_ratio
              << " profiling=" << (enable_profiling ? 1 : 0)
              << " rounds=" << num_rounds << " ===" << std::endl;

    for (size_t ef : efs) {
        index->setEf(ef);

        SearchRoundMetrics avg;
        for (int round = 0; round < num_rounds; ++round) {
            SearchRoundMetrics metrics = run_search_round(
                index, queries, qsize, qdim, gt, gt_dim, topk, enable_profiling);
            avg.recall += metrics.recall;
            avg.mean_latency_us += metrics.mean_latency_us;
            avg.p99_latency_us += metrics.p99_latency_us;
            avg.qps += metrics.qps;
            avg.decoding_calls += metrics.decoding_calls;
            avg.get_original_data_calls += metrics.get_original_data_calls;
            avg.backtrack_hops += metrics.backtrack_hops;

            std::cout << "  round " << (round + 1)
                      << " ef=" << ef
                      << " recall=" << std::fixed << std::setprecision(4) << metrics.recall
                      << " qps=" << std::setprecision(2) << metrics.qps
                      << " mean=" << metrics.mean_latency_us << " us/query"
                      << " p99=" << metrics.p99_latency_us << " us"
                      << " decoding_calls=" << metrics.decoding_calls
                      << std::endl;
        }

        const double denom = static_cast<double>(num_rounds);
        avg.recall /= denom;
        avg.mean_latency_us /= denom;
        avg.p99_latency_us /= denom;
        avg.qps /= denom;
        avg.decoding_calls /= num_rounds;
        avg.get_original_data_calls /= num_rounds;
        avg.backtrack_hops /= num_rounds;
        avg.avg_decode_per_query = qsize > 0
                                       ? static_cast<double>(avg.decoding_calls) / static_cast<double>(qsize)
                                       : 0.0;
        avg.avg_backtrack_hops_per_call = avg.get_original_data_calls > 0
                                              ? static_cast<double>(avg.backtrack_hops) /
                                                    static_cast<double>(avg.get_original_data_calls)
                                              : 0.0;

        std::cout << "ef=" << ef
                  << " recall=" << std::fixed << std::setprecision(4) << avg.recall
                  << " qps=" << std::setprecision(2) << avg.qps
                  << " mean=" << avg.mean_latency_us << " us/query"
                  << " p99=" << avg.p99_latency_us << " us"
                  << " decoding_calls=" << avg.decoding_calls
                  << " getOriginalData_calls=" << avg.get_original_data_calls
                  << " avg_decode_per_query=" << avg.avg_decode_per_query
                  << " avg_backtrack_hops_per_call=" << avg.avg_backtrack_hops_per_call
                  << std::endl;

        csv << dataset << ','
            << algo << ','
            << chain_max_length << ','
            << (use_cache ? 1 : 0) << ','
            << (use_tls ? 1 : 0) << ','
            << std::fixed << std::setprecision(4) << tls_ratio << ','
            << topk << ','
            << ef << ','
            << num_rounds << ','
            << (enable_profiling ? 1 : 0) << ','
            << std::setprecision(6) << avg.recall << ','
            << std::setprecision(6) << avg.mean_latency_us << ','
            << std::setprecision(2) << avg.p99_latency_us << ','
            << std::setprecision(2) << avg.qps << ','
            << avg.decoding_calls << ','
            << avg.get_original_data_calls << ','
            << std::setprecision(6) << avg.avg_decode_per_query << ','
            << std::setprecision(6) << avg.avg_backtrack_hops_per_call << '\n';

        if (avg.recall >= 0.99) {
            break;
        }
    }

    delete index;
    delete[] queries;
    delete[] gt;
}

void dispatch_search(const std::string& dataset,
                     const std::string& base_dir,
                     const std::string& algo,
                     int chain_max_length,
                     bool use_cache,
                     bool use_tls,
                     double tls_ratio,
                     size_t k,
                     const std::vector<size_t>& efs,
                     bool enable_profiling,
                     int num_rounds,
                     std::ofstream& csv) {
    if (algo == "DeXOR") return search_one_config<codecs::DeXORCodecPolicy>(dataset, base_dir, algo, chain_max_length, use_cache, use_tls, tls_ratio, k, efs, enable_profiling, num_rounds, csv);
    if (algo == "Gorilla") return search_one_config<codecs::GorillaCodecPolicy>(dataset, base_dir, algo, chain_max_length, use_cache, use_tls, tls_ratio, k, efs, enable_profiling, num_rounds, csv);
    if (algo == "Elf") return search_one_config<codecs::ElfCodecPolicy>(dataset, base_dir, algo, chain_max_length, use_cache, use_tls, tls_ratio, k, efs, enable_profiling, num_rounds, csv);
    if (algo == "Camel") return search_one_config<codecs::CamelCodecPolicy>(dataset, base_dir, algo, chain_max_length, use_cache, use_tls, tls_ratio, k, efs, enable_profiling, num_rounds, csv);
    if (algo == "DeXORPlus") return search_one_config<codecs::DeXORPlusCodecPolicy>(dataset, base_dir, algo, chain_max_length, use_cache, use_tls, tls_ratio, k, efs, enable_profiling, num_rounds, csv);
    throw std::runtime_error("Unknown algorithm: " + algo);
}

int main(int argc, char** argv) {
    std::string base_dir = "../datasets/hdf5files/";
    std::string out_csv = "compressed_hnsw_ablation_search_results.csv";
    int threads = 32;
    int num_rounds = 1;
    bool enable_profiling = false;
    size_t k = 1;

    std::vector<std::string> datasets = {
        "fashion-mnist-784-euclidean",
        "gist-960-euclidean",
        "mnist-784-euclidean",
        "sift-128-euclidean"
    };

    std::vector<std::string> algorithms = {
        "DeXOR",
        "Gorilla",
        "Elf",
        "Camel",
        "DeXORPlus"
    };

    std::vector<int> chain_max_list = {2, 5, -1};
    std::vector<int> cache_flags = {0, 1};
    std::vector<int> tls_flags = {0, 1};
    std::vector<double> tls_ratios = {0.2};
    std::vector<size_t> efs = {10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 300, 400};

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--base_dir" && i + 1 < argc) {
            base_dir = argv[++i];
        } else if (arg == "--output_csv" && i + 1 < argc) {
            out_csv = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            threads = std::stoi(argv[++i]);
        } else if (arg == "--num_rounds" && i + 1 < argc) {
            num_rounds = std::max(1, std::stoi(argv[++i]));
        } else if (arg == "--enable_profiling" && i + 1 < argc) {
            enable_profiling = std::stoi(argv[++i]) != 0;
        } else if (arg == "--k" && i + 1 < argc) {
            k = static_cast<size_t>(std::stoul(argv[++i]));
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
        } else if (arg == "--chain_max" && i + 1 < argc) {
            chain_max_list.clear();
            while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                chain_max_list.push_back(std::stoi(argv[++i]));
            }
        } else if (arg == "--use_cache" && i + 1 < argc) {
            cache_flags.clear();
            while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                cache_flags.push_back(std::stoi(argv[++i]));
            }
        } else if (arg == "--use_tls" && i + 1 < argc) {
            tls_flags.clear();
            while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                tls_flags.push_back(std::stoi(argv[++i]));
            }
        } else if (arg == "--tls_ratio" && i + 1 < argc) {
            tls_ratios.clear();
            while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                tls_ratios.push_back(std::stod(argv[++i]));
            }
        } else if (arg == "--ef" && i + 1 < argc) {
            efs.clear();
            while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                efs.push_back(static_cast<size_t>(std::stoul(argv[++i])));
            }
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << std::endl;
            return -1;
        }
    }

    if (!base_dir.empty() && base_dir.back() != '/' && base_dir.back() != '\\') {
        base_dir += '/';
    }

    omp_set_num_threads(threads);

    std::ofstream csv(out_csv);
    if (!csv.is_open()) {
        std::cerr << "Failed to open csv: " << out_csv << std::endl;
        return -1;
    }

    csv << "Dataset,Algorithm,ChainMaxLength,UseCacheTop1PctByLevel,UseTwoLevelSearch,TlsRatio,K,ef,"
        << "Num_Rounds,Enable_Profiling,Recall,TimePerQuery(us),P99_Latency(us),QPS,"
        << "Decoding_Calls,GetOriginalData_Calls,Avg_Decode_Per_Query,Avg_Backtrack_Hops_Per_Call\n";

    for (const auto& ds : datasets) {
        for (const auto& algo : algorithms) {
            for (int chain_max : chain_max_list) {
                for (int cache_flag : cache_flags) {
                    for (int tls_flag : tls_flags) {
                        if (tls_flag == 0) {
                            dispatch_search(ds, base_dir, algo, chain_max, cache_flag != 0, false, 0.0, k, efs, enable_profiling, num_rounds, csv);
                        } else {
                            for (double ratio : tls_ratios) {
                                dispatch_search(ds, base_dir, algo, chain_max, cache_flag != 0, true, ratio, k, efs, enable_profiling, num_rounds, csv);
                            }
                        }
                    }
                }
            }
        }
    }

    csv.close();
    std::cout << "\nSaved CSV: " << out_csv << std::endl;
    return 0;
}
