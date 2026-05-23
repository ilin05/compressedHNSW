#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <queue>
#include <sstream>
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
    StopW() {
        time_begin = std::chrono::steady_clock::now();
    }

    float getElapsedTimeMicro() {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count();
    }
};

double* load_fvecs_as_double(const std::string& filename, size_t& num_vectors, size_t& dim) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        std::cerr << "Cannot open " << filename << std::endl;
        return nullptr;
    }

    int32_t d;
    input.read(reinterpret_cast<char*>(&d), 4);
    dim = d;

    input.seekg(0, std::ios::end);
    size_t file_size = input.tellg();
    num_vectors = file_size / (4 + dim * 4);

    double* data = new double[num_vectors * dim];
    float* tmp = new float[dim];
    input.seekg(0, std::ios::beg);

    for (size_t i = 0; i < num_vectors; ++i) {
        input.read(reinterpret_cast<char*>(&d), 4);
        input.read(reinterpret_cast<char*>(tmp), dim * 4);
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

    int32_t d;
    input.read(reinterpret_cast<char*>(&d), 4);
    dim = d;

    input.seekg(0, std::ios::end);
    size_t file_size = input.tellg();
    num_vectors = file_size / (4 + dim * 4);

    unsigned int* data = new unsigned int[num_vectors * dim];
    input.seekg(0, std::ios::beg);

    for (size_t i = 0; i < num_vectors; ++i) {
        input.read(reinterpret_cast<char*>(&d), 4);
        input.read(reinterpret_cast<char*>(data + i * dim), dim * 4);
    }

    return data;
}

bool file_exists(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    return input.good();
}

string strip_train_suffix(const string& dataset_file) {
    size_t pos = dataset_file.find("_train.fvecs");
    if (pos != string::npos) return dataset_file.substr(0, pos);
    pos = dataset_file.find(".fvecs");
    if (pos != string::npos) return dataset_file.substr(0, pos);
    return dataset_file;
}

CompressionRootPolicy parse_root_policy(const string& name) {
    if (name == "level") return CompressionRootPolicy::Level;
    if (name == "random") return CompressionRootPolicy::Random;
    if (name == "level0degree" || name == "level0_degree") return CompressionRootPolicy::Level0Degree;
    throw std::runtime_error("Unknown root policy: " + name);
}

string root_policy_name(CompressionRootPolicy policy) {
    switch (policy) {
        case CompressionRootPolicy::Level:
            return "level";
        case CompressionRootPolicy::Random:
            return "random";
        case CompressionRootPolicy::Level0Degree:
            return "level0degree";
        default:
            return "unknown";
    }
}

string root_policy_index_path(const string& prefix, CompressionRootPolicy policy, int chain_max_length) {
    return prefix + "_DeXOR_chain" + std::to_string(chain_max_length) + "_root_" + root_policy_name(policy) + ".bin";
}

string format_ratio_for_filename(double ratio) {
    std::ostringstream oss;
    oss << fixed << setprecision(4) << ratio;
    string s = oss.str();
    for (char& ch : s) {
        if (ch == '.') ch = 'p';
    }
    while (!s.empty() && s.back() == '0') s.pop_back();
    if (!s.empty() && s.back() == 'p') s.pop_back();
    return s;
}

string root_policy_index_path(
    const string& prefix,
    CompressionRootPolicy policy,
    int chain_max_length,
    double root_ratio) {
    return prefix + "_DeXOR_chain" + std::to_string(chain_max_length)
           + "_root_" + root_policy_name(policy)
           + "_ratio_" + format_ratio_for_filename(root_ratio) + ".bin";
}

struct ExperimentResult {
    string experiment;
    string dataset_name;
    string root_policy;
    double root_ratio = 0.01;
    double compression_time_sec = 0.0;
    double cache_ratio = 0.0;
    size_t cache_size = 0;
    int chain_max_length = -1;
    double recall = 0.0;
    double qps = 0.0;
    double qps_std_dev = 0.0;
    double mean_latency_us = 0.0;
    double p99_latency_us = 0.0;
    long decoding_calls = 0;
    long distance_computations = 0;
    long get_original_data_calls = 0;
    long backtrack_hops = 0;
    double avg_decode_per_query = 0.0;
    double avg_backtrack_hops_per_call = 0.0;
    double cache_hit_rate = 0.0;
    double avg_encoding_chain_length = 0.0;
    size_t index_size = 0;
    size_t compressed_index_size = 0;
    double compression_ratio = 0.0;
    int num_rounds = 0;
};

struct SearchMetrics {
    size_t query_count = 0;
    double recall = 0.0;
    double qps = 0.0;
    double mean_latency_us = 0.0;
    double p99_latency_us = 0.0;
    long decoding_calls = 0;
    long distance_computations = 0;
    long get_original_data_calls = 0;
    long backtrack_hops = 0;
    double cache_hit_rate = 0.0;
};

SearchMetrics perform_search(
    const string& query_file,
    const string& gt_file,
    HierarchicalNSWCABFRAMEWORK<double, codecs::DeXORCodecPolicy>* index,
    int search_threads) {
    SearchMetrics metrics;
    omp_set_num_threads(search_threads);

    size_t qsize = 0, qdim = 0;
    double* queries = load_fvecs_as_double(query_file, qsize, qdim);
    if (!queries) return metrics;

    size_t gt_num = 0, gt_dim = 0;
    unsigned int* groundtruth = load_ivecs(gt_file, gt_num, gt_dim);
    if (!groundtruth) {
        delete[] queries;
        return metrics;
    }

    if (qsize != gt_num) {
        cerr << "Query size and ground-truth size mismatch." << endl;
        delete[] queries;
        delete[] groundtruth;
        return metrics;
    }

    index->resetProfilingMetrics();
    index->setProfilingMetrics(true);

    const size_t k = 1;
    size_t correct = 0;
    vector<double> latencies_us;
    latencies_us.reserve(qsize);

    StopW total_timer;
    for (size_t i = 0; i < qsize; ++i) {
        StopW query_timer;
        auto result = index->searchKnn(queries + qdim * i, k);
        latencies_us.push_back(query_timer.getElapsedTimeMicro());

        unordered_set<labeltype> gt;
        for (size_t j = 0; j < k && j < gt_dim; ++j) {
            gt.insert(groundtruth[i * gt_dim + j]);
        }

        while (!result.empty()) {
            if (gt.find(result.top().second) != gt.end()) {
                ++correct;
            }
            result.pop();
        }
    }

    double total_us = total_timer.getElapsedTimeMicro();
    metrics.query_count = qsize;
    metrics.recall = static_cast<double>(correct) / static_cast<double>(qsize * k);
    metrics.qps = static_cast<double>(qsize) / (total_us / 1e6);
    metrics.mean_latency_us = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0) / latencies_us.size();
    std::sort(latencies_us.begin(), latencies_us.end());
    size_t p99_idx = std::min(latencies_us.size() - 1, static_cast<size_t>(std::ceil(latencies_us.size() * 0.99)) - 1);
    metrics.p99_latency_us = latencies_us[p99_idx];

    metrics.decoding_calls = index->getDecodingCallCount();
    metrics.distance_computations = index->metric_distance_computations;
    metrics.get_original_data_calls = index->getOriginalDataCallCount();
    metrics.backtrack_hops = index->getOriginalDataBacktrackHops();
    long cache_hits = index->getCacheHitTimes();
    metrics.cache_hit_rate = metrics.get_original_data_calls > 0
                                 ? static_cast<double>(cache_hits) / metrics.get_original_data_calls
                                 : 0.0;

    delete[] queries;
    delete[] groundtruth;
    return metrics;
}

double average_encoding_chain_length(
    HierarchicalNSWCABFRAMEWORK<double, codecs::DeXORCodecPolicy>* index,
    size_t num_vectors) {
    if (num_vectors == 0) return 0.0;
    long long total = 0;
    for (size_t i = 0; i < num_vectors; ++i) {
        total += index->getEncodingChainLength(static_cast<tableint>(i));
    }
    return static_cast<double>(total) / num_vectors;
}

void build_index_if_missing(
    const string& index_path,
    const string& dataset_path,
    size_t num_vectors,
    size_t dim,
    CompressionRootPolicy root_policy,
    double root_ratio,
    int chain_max_length,
    size_t M,
    size_t ef_construction,
    int build_threads,
    double* compression_time_sec = nullptr) {
    if (file_exists(index_path)) {
        cout << "Index exists, skip build: " << index_path << endl;
        if (compression_time_sec) {
            *compression_time_sec = 0.0;
        }
        return;
    }

    cout << "Building index: " << index_path << endl;
    size_t loaded_vectors = 0, loaded_dim = 0;
    double* data = load_fvecs_as_double(dataset_path, loaded_vectors, loaded_dim);
    if (!data) {
        throw std::runtime_error("Failed to load dataset for build: " + dataset_path);
    }
    if (loaded_vectors != num_vectors || loaded_dim != dim) {
        delete[] data;
        throw std::runtime_error("Loaded dataset shape mismatch.");
    }

    L2SpaceDouble space(dim);
    HierarchicalNSWCABFRAMEWORK<double, codecs::DeXORCodecPolicy> index(
        &space, num_vectors, M, ef_construction, true, 0, 100, false);

    omp_set_num_threads(build_threads);
    #pragma omp parallel for schedule(dynamic)
    for (long long i = 0; i < static_cast<long long>(num_vectors); ++i) {
        index.addPoint(data + i * dim, i);
    }

    StopW timer;
    omp_set_num_threads(build_threads);
    index.compress_dataset(chain_max_length, root_ratio, root_policy);
    double elapsed_sec = timer.getElapsedTimeMicro() / 1e6;
    if (compression_time_sec) {
        *compression_time_sec = elapsed_sec;
    }
    cout << "  compression time: " << elapsed_sec << " sec" << endl;
    index.saveIndex(index_path);
    cout << "Saved index: " << index_path << endl;

    delete[] data;
}

ExperimentResult evaluate_loaded_index(
    const string& experiment,
    const string& dataset_name,
    const string& query_file,
    const string& gt_file,
    const string& index_path,
    CompressionRootPolicy root_policy,
    double root_ratio,
    double compression_time_sec,
    double cache_ratio,
    size_t cache_size,
    size_t num_vectors,
    size_t dim,
    int chain_max_length,
    int ef,
    bool use_tls,
    double tls_ratio,
    int search_threads,
    int num_rounds) {
    L2SpaceDouble space(dim);
    HierarchicalNSWCABFRAMEWORK<double, codecs::DeXORCodecPolicy> index(
        &space, index_path, true, cache_size);
    index.setEf(ef);
    index.setUseTls(use_tls);
    index.setTlsRatio(tls_ratio);

    vector<SearchMetrics> rounds;
    rounds.reserve(num_rounds);
    for (int round = 0; round < num_rounds; ++round) {
        SearchMetrics metrics = perform_search(query_file, gt_file, &index, search_threads);
        rounds.push_back(metrics);
        cout << "  round " << (round + 1)
             << ": qps=" << fixed << setprecision(2) << metrics.qps
             << ", recall=" << setprecision(4) << metrics.recall
             << ", mean_us=" << setprecision(2) << metrics.mean_latency_us
             << ", p99_us=" << metrics.p99_latency_us
             << ", decoding_calls=" << metrics.decoding_calls
             << endl;
    }

    ExperimentResult result;
    result.experiment = experiment;
    result.dataset_name = dataset_name;
    result.root_policy = root_policy_name(root_policy);
    result.root_ratio = root_ratio;
    result.compression_time_sec = compression_time_sec;
    result.cache_ratio = cache_ratio;
    result.cache_size = cache_size;
    result.chain_max_length = chain_max_length;
    result.index_size = index.getIndexSize();
    result.compressed_index_size = index.getCompressedIndexSize();
    result.compression_ratio = result.index_size > 0
                                   ? static_cast<double>(result.compressed_index_size) / result.index_size
                                   : 0.0;
    result.avg_encoding_chain_length = average_encoding_chain_length(&index, num_vectors);
    result.num_rounds = num_rounds;

    double qps_sum = 0.0;
    double qps_sq_sum = 0.0;
    size_t query_count = rounds.empty() ? 0 : rounds[0].query_count;
    for (const auto& round : rounds) {
        result.recall += round.recall;
        result.mean_latency_us += round.mean_latency_us;
        result.p99_latency_us += round.p99_latency_us;
        result.decoding_calls += round.decoding_calls;
        result.distance_computations += round.distance_computations;
        result.get_original_data_calls += round.get_original_data_calls;
        result.backtrack_hops += round.backtrack_hops;
        result.cache_hit_rate += round.cache_hit_rate;
        qps_sum += round.qps;
        qps_sq_sum += round.qps * round.qps;
    }

    double denom = static_cast<double>(num_rounds);
    result.qps = qps_sum / denom;
    result.recall /= denom;
    result.mean_latency_us /= denom;
    result.p99_latency_us /= denom;
    result.decoding_calls /= num_rounds;
    result.distance_computations /= num_rounds;
    result.get_original_data_calls /= num_rounds;
    result.backtrack_hops /= num_rounds;
    result.cache_hit_rate /= denom;
    result.avg_decode_per_query = query_count > 0
                                      ? static_cast<double>(result.decoding_calls) / static_cast<double>(query_count)
                                      : 0.0;
    result.avg_backtrack_hops_per_call = result.get_original_data_calls > 0
                                             ? static_cast<double>(result.backtrack_hops) / result.get_original_data_calls
                                             : 0.0;

    double variance = qps_sq_sum / denom - result.qps * result.qps;
    result.qps_std_dev = variance > 0 ? std::sqrt(variance) : 0.0;
    return result;
}

void write_results_to_csv(const string& path, const vector<ExperimentResult>& results) {
    ofstream file(path);
    if (!file) {
        cerr << "Failed to open CSV: " << path << endl;
        return;
    }

    file << "Experiment,Dataset,RootPolicy,Root_Ratio,Compression_Time(s),Cache_Ratio(%),Cache_Size,ChainMaxLength,"
         << "Recall@1,QPS(avg),QPS_StdDev,Mean_Latency(us),P99_Latency(us),"
         << "Decoding_Calls,Distance_Computations,GetOriginalData_Calls,Backtrack_Hops,"
         << "Avg_Decode_Per_Query,Avg_Backtrack_Hops_Per_Call,Cache_Hit_Rate,"
         << "Avg_Encoding_Chain_Length,Index_Size,Compressed_Index_Size,Compression_Ratio,Num_Rounds\n";

    for (const auto& r : results) {
        file << r.experiment << ","
             << r.dataset_name << ","
             << r.root_policy << ","
             << fixed << setprecision(4) << r.root_ratio << ","
             << fixed << setprecision(6) << r.compression_time_sec << ","
             << fixed << setprecision(2) << r.cache_ratio << ","
             << r.cache_size << ","
             << r.chain_max_length << ","
             << fixed << setprecision(6) << r.recall << ","
             << fixed << setprecision(2) << r.qps << ","
             << fixed << setprecision(2) << r.qps_std_dev << ","
             << fixed << setprecision(2) << r.mean_latency_us << ","
             << fixed << setprecision(2) << r.p99_latency_us << ","
             << r.decoding_calls << ","
             << r.distance_computations << ","
             << r.get_original_data_calls << ","
             << r.backtrack_hops << ","
             << fixed << setprecision(6) << r.avg_decode_per_query << ","
             << fixed << setprecision(6) << r.avg_backtrack_hops_per_call << ","
             << fixed << setprecision(6) << r.cache_hit_rate << ","
             << fixed << setprecision(6) << r.avg_encoding_chain_length << ","
             << r.index_size << ","
             << r.compressed_index_size << ","
             << fixed << setprecision(6) << r.compression_ratio << ","
             << r.num_rounds << "\n";
    }

    cout << "Results written to " << path << endl;
}

int main(int argc, char** argv) {
    string base_dir = "../datasets/hdf5files/";
    vector<string> datasets = {"sift-128-euclidean_train.fvecs"};
    vector<float> cache_ratios = {0.0, 0.5, 1.0, 2.0, 5.0, 10.0};
    vector<double> root_ratios = {0.001, 0.002, 0.005, 0.01, 0.02, 0.05};
    vector<CompressionRootPolicy> root_policies = {
        CompressionRootPolicy::Random,
        CompressionRootPolicy::Level,
        CompressionRootPolicy::Level0Degree,
    };
    int num_rounds = 1;
    int chain_max_length = -1;
    size_t M = 16;
    size_t ef_construction = 200;
    int ef = 200;
    int build_threads = 32;
    int search_threads = 1;
    bool run_root_policy_ablation = true;
    bool run_cache_ratio_sensitivity = true;
    bool run_root_ratio_sensitivity = true;
    bool use_tls = false;
    double tls_ratio = 0.2;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--dataset" && i + 1 < argc) {
            datasets.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                datasets.push_back(argv[++i]);
            }
        } else if (arg == "--base-dir" && i + 1 < argc) {
            base_dir = argv[++i];
            if (!base_dir.empty() && base_dir.back() != '/' && base_dir.back() != '\\') {
                base_dir += "/";
            }
        } else if (arg == "--cache-ratios" && i + 1 < argc) {
            cache_ratios.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                cache_ratios.push_back(std::stof(argv[++i]));
            }
        } else if (arg == "--root-ratios" && i + 1 < argc) {
            root_ratios.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                root_ratios.push_back(std::stod(argv[++i]));
            }
        } else if (arg == "--root-policies" && i + 1 < argc) {
            root_policies.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                root_policies.push_back(parse_root_policy(argv[++i]));
            }
        } else if (arg == "--num-rounds" && i + 1 < argc) {
            num_rounds = std::stoi(argv[++i]);
        } else if (arg == "--chain-max-length" && i + 1 < argc) {
            chain_max_length = std::stoi(argv[++i]);
        } else if (arg == "--M" && i + 1 < argc) {
            M = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (arg == "--ef-construction" && i + 1 < argc) {
            ef_construction = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (arg == "--ef" && i + 1 < argc) {
            ef = std::stoi(argv[++i]);
        } else if (arg == "--build-threads" && i + 1 < argc) {
            build_threads = std::stoi(argv[++i]);
        } else if (arg == "--search-threads" && i + 1 < argc) {
            search_threads = std::stoi(argv[++i]);
        } else if (arg == "--root-policy-only") {
            run_cache_ratio_sensitivity = false;
            run_root_ratio_sensitivity = false;
        } else if (arg == "--cache-ratio-only") {
            run_root_policy_ablation = false;
            run_root_ratio_sensitivity = false;
        } else if (arg == "--root-ratio-only") {
            run_root_policy_ablation = false;
            run_cache_ratio_sensitivity = false;
        } else if (arg == "--no-root-ratio") {
            run_root_ratio_sensitivity = false;
        } else if (arg == "--use-tls" && i + 1 < argc) {
            use_tls = std::stoi(argv[++i]) != 0;
        } else if (arg == "--tls-ratio" && i + 1 < argc) {
            tls_ratio = std::stod(argv[++i]);
        }
    }

    vector<ExperimentResult> root_policy_results;
    vector<ExperimentResult> cache_ratio_results;
    vector<ExperimentResult> root_ratio_results;

    for (const auto& ds : datasets) {
        string prefix = strip_train_suffix(ds);
        string dataset_path = base_dir + ds;
        string query_file = base_dir + prefix + "_test.fvecs";
        string gt_file = base_dir + prefix + "_neighbors.ivecs";

        size_t num_vectors = 0, dim = 0;
        double* temp_data = load_fvecs_as_double(dataset_path, num_vectors, dim);
        if (!temp_data) {
            cerr << "Failed to load dataset metadata: " << dataset_path << endl;
            continue;
        }
        delete[] temp_data;

        cout << "\nDataset: " << prefix
             << ", vectors=" << num_vectors
             << ", dim=" << dim
             << ", chain_max_length=" << chain_max_length
             << ", ef=" << ef
             << ", build_threads=" << build_threads
             << ", search_threads=" << search_threads << endl;

        if (run_root_policy_ablation) {
            cout << "\nRoot selection policy ablation" << endl;
            for (CompressionRootPolicy policy : root_policies) {
                double build_time_sec = 0.0;
                string index_path = root_policy_index_path(prefix, policy, chain_max_length);
                build_index_if_missing(
                    index_path,
                    dataset_path,
                    num_vectors,
                    dim,
                    policy,
                    0.01,
                    chain_max_length,
                    M,
                    ef_construction,
                    build_threads,
                    &build_time_sec);

                cout << "Evaluating root policy " << root_policy_name(policy)
                     << " from index " << index_path << endl;
                root_policy_results.push_back(evaluate_loaded_index(
                    "root_policy_ablation",
                    prefix,
                    query_file,
                    gt_file,
                    index_path,
                    policy,
                    0.01,
                    build_time_sec,
                    0.0,
                    0,
                    num_vectors,
                    dim,
                    chain_max_length,
                    ef,
                    use_tls,
                    tls_ratio,
                    search_threads,
                    num_rounds));
            }
        }

        if (run_cache_ratio_sensitivity) {
            cout << "\nRoot cache ratio sensitivity" << endl;
            CompressionRootPolicy cache_policy = CompressionRootPolicy::Level;
            string index_path = root_policy_index_path(prefix, cache_policy, chain_max_length);
            build_index_if_missing(
                index_path,
                dataset_path,
                num_vectors,
                dim,
                cache_policy,
                0.01,
                chain_max_length,
                M,
                ef_construction,
                build_threads);

            for (float ratio : cache_ratios) {
                size_t cache_size = static_cast<size_t>(num_vectors * ratio / 100.0);
                cout << "Evaluating cache ratio " << fixed << setprecision(2)
                     << ratio << "%, cache_size=" << cache_size << endl;
                cache_ratio_results.push_back(evaluate_loaded_index(
                    "root_cache_ratio_sensitivity",
                    prefix,
                    query_file,
                    gt_file,
                    index_path,
                    cache_policy,
                    0.01,
                    0.0,
                    ratio,
                    cache_size,
                    num_vectors,
                    dim,
                    chain_max_length,
                    ef,
                    use_tls,
                    tls_ratio,
                    search_threads,
                    num_rounds));
            }
        }

        if (run_root_ratio_sensitivity) {
            cout << "\nRoot ratio parameter sensitivity" << endl;
            CompressionRootPolicy policy = CompressionRootPolicy::Level;
            for (double root_ratio : root_ratios) {
                double build_time_sec = 0.0;
                string index_path = root_policy_index_path(prefix, policy, chain_max_length, root_ratio);
                build_index_if_missing(
                    index_path,
                    dataset_path,
                    num_vectors,
                    dim,
                    policy,
                    root_ratio,
                    chain_max_length,
                    M,
                    ef_construction,
                    build_threads,
                    &build_time_sec);

                cout << "Evaluating root ratio " << fixed << setprecision(4)
                     << root_ratio << " from index " << index_path << endl;
                root_ratio_results.push_back(evaluate_loaded_index(
                    "root_ratio_sensitivity",
                    prefix,
                    query_file,
                    gt_file,
                    index_path,
                    policy,
                    root_ratio,
                    build_time_sec,
                    0.0,
                    0,
                    num_vectors,
                    dim,
                    chain_max_length,
                    ef,
                    use_tls,
                    tls_ratio,
                    search_threads,
                    num_rounds));
            }
        }
    }

    if (!root_policy_results.empty()) {
        write_results_to_csv("root_selection_policy_ablation_dexor_results_chain_" + std::to_string(chain_max_length) + ".csv", root_policy_results);
    }
    if (!cache_ratio_results.empty()) {
        write_results_to_csv("root_cache_ratio_sensitivity_dexor_results_chain_" + std::to_string(chain_max_length) + ".csv", cache_ratio_results);
    }
    if (!root_ratio_results.empty()) {
        write_results_to_csv("root_ratio_sensitivity_dexor_results_chain_" + std::to_string(chain_max_length) + ".csv", root_ratio_results);
    }

    cout << "\nDone." << endl;
    return 0;
}
