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
#include "../../hnswlib/hnswalg_simplified_alp_PQ.h"

using namespace hnswlib;

class StopW {
    std::chrono::steady_clock::time_point begin_;
public:
    StopW() : begin_(std::chrono::steady_clock::now()) {}
    double elapsed_us() const {
        return static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - begin_).count());
    }
};

struct FvecsData {
    std::vector<float> values;
    size_t count = 0;
    size_t dim = 0;
};

struct IvecsData {
    std::vector<unsigned int> values;
    size_t count = 0;
    size_t dim = 0;
};

struct RunResult {
    std::string dataset;
    std::string method;
    std::string index_path;
    int chain_max = 0;
    size_t cache_size = 0;
    double cache_ratio = 0.0;
    int use_tls = 0;
    double tls_ratio = 0.0;
    size_t ef = 0;
    size_t k = 0;
    int num_rounds = 0;
    double recall = 0.0;
    double mean_latency_us = 0.0;
    double p99_latency_us = 0.0;
    double qps = 0.0;
    long decoding_calls = 0;
    long get_original_data_calls = 0;
    long distance_computations = 0;
    double avg_decode_per_query = 0.0;
};

static FvecsData load_fvecs(const std::string& filename) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot open " + filename);
    }

    int32_t d = 0;
    input.read(reinterpret_cast<char*>(&d), 4);
    if (d <= 0) {
        throw std::runtime_error("Invalid fvecs dimension in " + filename);
    }

    FvecsData data;
    data.dim = static_cast<size_t>(d);
    input.seekg(0, std::ios::end);
    const size_t file_size = static_cast<size_t>(input.tellg());
    data.count = file_size / (4 + data.dim * 4);
    data.values.resize(data.count * data.dim);

    input.seekg(0, std::ios::beg);
    for (size_t i = 0; i < data.count; ++i) {
        input.read(reinterpret_cast<char*>(&d), 4);
        input.read(reinterpret_cast<char*>(data.values.data() + i * data.dim), data.dim * 4);
    }
    return data;
}

static IvecsData load_ivecs(const std::string& filename) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot open " + filename);
    }

    int32_t d = 0;
    input.read(reinterpret_cast<char*>(&d), 4);
    if (d <= 0) {
        throw std::runtime_error("Invalid ivecs dimension in " + filename);
    }

    IvecsData data;
    data.dim = static_cast<size_t>(d);
    input.seekg(0, std::ios::end);
    const size_t file_size = static_cast<size_t>(input.tellg());
    data.count = file_size / (4 + data.dim * 4);
    data.values.resize(data.count * data.dim);

    input.seekg(0, std::ios::beg);
    for (size_t i = 0; i < data.count; ++i) {
        input.read(reinterpret_cast<char*>(&d), 4);
        input.read(reinterpret_cast<char*>(data.values.data() + i * data.dim), data.dim * 4);
    }
    return data;
}

static size_t count_fvecs(const std::string& filename) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot open " + filename);
    }
    int32_t d = 0;
    input.read(reinterpret_cast<char*>(&d), 4);
    if (d <= 0) {
        throw std::runtime_error("Invalid fvecs dimension in " + filename);
    }
    input.seekg(0, std::ios::end);
    const size_t file_size = static_cast<size_t>(input.tellg());
    return file_size / (4 + static_cast<size_t>(d) * 4);
}

static std::vector<double> to_double_queries(const FvecsData& queries) {
    std::vector<double> out(queries.values.size());
    std::transform(queries.values.begin(), queries.values.end(), out.begin(),
                   [](float v) { return static_cast<double>(v); });
    return out;
}

template <typename DistT>
static size_t count_correct(std::priority_queue<std::pair<DistT, labeltype>>& result,
                            const unsigned int* gt,
                            size_t gt_dim,
                            size_t k) {
    std::unordered_set<labeltype> gt_set;
    gt_set.reserve(k);
    for (size_t j = 0; j < k; ++j) {
        gt_set.insert(gt[j]);
    }

    size_t correct = 0;
    while (!result.empty()) {
        if (gt_set.find(result.top().second) != gt_set.end()) {
            ++correct;
        }
        result.pop();
    }
    return correct;
}

static double percentile(std::vector<double> values, double pct) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t idx = std::min(values.size() - 1,
                                static_cast<size_t>(std::ceil(values.size() * pct)) - 1);
    return values[idx];
}

template <typename IndexT, typename QueryT>
static RunResult run_search(IndexT* index,
                            const QueryT* queries,
                            const IvecsData& gt,
                            size_t qcount,
                            size_t qdim,
                            size_t k,
                            int num_rounds,
                            const std::string& dataset,
                            const std::string& method,
                            const std::string& index_path,
                            int chain_max,
                            size_t cache_size,
                            double cache_ratio,
                            int use_tls,
                            double tls_ratio) {
    RunResult avg;
    avg.dataset = dataset;
    avg.method = method;
    avg.index_path = index_path;
    avg.chain_max = chain_max;
    avg.cache_size = cache_size;
    avg.cache_ratio = cache_ratio;
    avg.use_tls = use_tls;
    avg.tls_ratio = tls_ratio;
    avg.k = k;
    avg.num_rounds = num_rounds;

    for (int round = 0; round < num_rounds; ++round) {
        size_t correct = 0;
        std::vector<double> latencies_us;
        latencies_us.reserve(qcount);

        StopW total_timer;
        for (size_t qi = 0; qi < qcount; ++qi) {
            StopW query_timer;
            auto result = index->searchKnn(queries + qi * qdim, k);
            latencies_us.push_back(query_timer.elapsed_us());
            correct += count_correct(result, gt.values.data() + qi * gt.dim, gt.dim, k);
        }
        const double total_us = total_timer.elapsed_us();

        avg.recall += static_cast<double>(correct) / static_cast<double>(qcount * k);
        avg.mean_latency_us += std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0) /
                               static_cast<double>(latencies_us.size());
        avg.p99_latency_us += percentile(latencies_us, 0.99);
        avg.qps += total_us > 0.0 ? static_cast<double>(qcount) / (total_us / 1e6) : 0.0;
    }

    const double denom = static_cast<double>(num_rounds);
    avg.recall /= denom;
    avg.mean_latency_us /= denom;
    avg.p99_latency_us /= denom;
    avg.qps /= denom;
    return avg;
}

static RunResult run_original_hnsw(const std::string& dataset,
                                   const FvecsData& queries,
                                   const IvecsData& gt,
                                   size_t ef,
                                   size_t k,
                                   int num_rounds) {
    const std::string index_path = dataset + "_train.fvecs_hnsw_float.bin";
    L2Space l2space(static_cast<int>(queries.dim));
    HierarchicalNSW<float> index(&l2space, index_path, false);
    index.setEf(ef);

    RunResult result = run_search(&index, queries.values.data(), gt, queries.count, queries.dim,
                                  k, num_rounds, dataset, "HNSW", index_path, -1, 0, 0.0, 0, 0.0);
    result.ef = ef;
    return result;
}

static RunResult run_hnswalp_tls(const std::string& dataset,
                                 const FvecsData& queries,
                                 const IvecsData& gt,
                                 size_t ef,
                                 size_t k,
                                 int num_rounds,
                                 double tls_ratio) {
    const std::string index_path = dataset + "_train.fvecs_hnswalp_simplified_pq.bin";
    L2Space l2space(static_cast<int>(queries.dim));
    HierarchicalNSWALPSIMPLIFIEDPQ<float> index(&l2space, index_path, false);
    index.setEf(ef);
    index.setUseTLS(true);
    index.setTLSRatio(static_cast<float>(tls_ratio));
    index.setProfilingMetrics(true);

    index.decoding_call_count = 0;
    index.get_original_data_call_count = 0;
    index.metric_distance_computations = 0;

    RunResult result = run_search(&index, queries.values.data(), gt, queries.count, queries.dim,
                                  k, num_rounds, dataset, "HNSWALP_TLS", index_path, -1, 0, 0.0, 1, tls_ratio);
    result.ef = ef;
    result.decoding_calls = index.getDecodingCallCount();
    result.get_original_data_calls = index.getGetOriginalDataCallCount();
    result.distance_computations = index.metric_distance_computations;
    result.avg_decode_per_query = queries.count > 0
                                      ? static_cast<double>(result.decoding_calls) /
                                            static_cast<double>(queries.count * num_rounds)
                                      : 0.0;
    index.setProfilingMetrics(false);
    return result;
}

static std::string chain_tag(int chain_max) {
    return chain_max < 0 ? "chunlim" : "ch" + std::to_string(chain_max);
}

static RunResult run_dexor(const std::string& dataset,
                           const FvecsData& queries,
                           const IvecsData& gt,
                           size_t ef,
                           size_t k,
                           int num_rounds,
                           int chain_max,
                           size_t cache_size,
                           double cache_ratio,
                           bool use_tls,
                           double tls_ratio) {
    const std::vector<double> double_queries = to_double_queries(queries);
    const std::string index_path = dataset + "_DeXOR_" + chain_tag(chain_max) + "_pq.bin";
    L2SpaceDouble l2space(queries.dim);
    HierarchicalNSWCABFRAMEWORK<double, codecs::DeXORCodecPolicy> index(&l2space, index_path, true, cache_size);
    index.setEf(ef);
    index.setUseTls(use_tls);
    index.setTlsRatio(tls_ratio);
    index.setProfilingMetrics(true);
    index.resetProfilingMetrics();

    RunResult result = run_search(&index, double_queries.data(), gt, queries.count, queries.dim,
                                  k, num_rounds, dataset,
                                  use_tls ? "DeXOR_Optimized" : "DeXOR_NoOpt",
                                  index_path, chain_max, cache_size, cache_ratio,
                                  use_tls ? 1 : 0, use_tls ? tls_ratio : 0.0);
    result.ef = ef;
    result.decoding_calls = index.getDecodingCallCount();
    result.get_original_data_calls = index.getOriginalDataCallCount();
    result.distance_computations = index.metric_distance_computations;
    result.avg_decode_per_query = queries.count > 0
                                      ? static_cast<double>(result.decoding_calls) /
                                            static_cast<double>(queries.count * num_rounds)
                                      : 0.0;
    index.setProfilingMetrics(false);
    return result;
}

static void write_csv_header(std::ofstream& csv) {
    csv << "Dataset,Method,IndexPath,ChainMaxLength,UseTwoLevelSearch,TlsRatio,ef,K,Num_Rounds,"
        << "CacheRatio,CacheSize,Recall,TimePerQuery(us),P99_Latency(us),QPS,Decoding_Calls,GetOriginalData_Calls,"
        << "Distance_Computations,Avg_Decode_Per_Query\n";
}

static void write_result(std::ofstream& csv, const RunResult& r) {
    csv << r.dataset << ','
        << r.method << ','
        << r.index_path << ','
        << r.chain_max << ','
        << r.use_tls << ','
        << std::fixed << std::setprecision(4) << r.tls_ratio << ','
        << r.ef << ','
        << r.k << ','
        << r.num_rounds << ','
        << std::setprecision(4) << r.cache_ratio << ','
        << r.cache_size << ','
        << std::setprecision(6) << r.recall << ','
        << std::setprecision(2) << r.mean_latency_us << ','
        << std::setprecision(2) << r.p99_latency_us << ','
        << std::setprecision(2) << r.qps << ','
        << r.decoding_calls << ','
        << r.get_original_data_calls << ','
        << r.distance_computations << ','
        << std::setprecision(6) << r.avg_decode_per_query << '\n';
}

int main(int argc, char** argv) {
    std::string base_dir = "../datasets/hdf5files/";
    std::string dataset = "sift-128-euclidean";
    std::string mode = "all";
    std::string output_csv = "perf_breakdown_hnsw_summary.csv";
    size_t ef = 80;
    size_t k = 1;
    int num_rounds = 5;
    int append_csv = 0;
    int write_csv = 1;
    double tls_ratio = 0.2;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--base_dir" && i + 1 < argc) {
            base_dir = argv[++i];
        } else if (arg == "--dataset" && i + 1 < argc) {
            dataset = argv[++i];
        } else if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        } else if (arg == "--output_csv" && i + 1 < argc) {
            output_csv = argv[++i];
        } else if (arg == "--ef" && i + 1 < argc) {
            ef = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (arg == "--k" && i + 1 < argc) {
            k = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (arg == "--num_rounds" && i + 1 < argc) {
            num_rounds = std::max(1, std::stoi(argv[++i]));
        } else if (arg == "--tls_ratio" && i + 1 < argc) {
            tls_ratio = std::stod(argv[++i]);
        } else if (arg == "--append" && i + 1 < argc) {
            append_csv = std::stoi(argv[++i]);
        } else if (arg == "--write_csv" && i + 1 < argc) {
            write_csv = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << std::endl;
            return 1;
        }
    }

    if (!base_dir.empty() && base_dir.back() != '/' && base_dir.back() != '\\') {
        base_dir += '/';
    }

    omp_set_num_threads(1);

    const FvecsData queries = load_fvecs(base_dir + dataset + "_test.fvecs");
    const IvecsData gt = load_ivecs(base_dir + dataset + "_neighbors.ivecs");
    const size_t train_count = count_fvecs(base_dir + dataset + "_train.fvecs");
    if (queries.count != gt.count) {
        throw std::runtime_error("Query and ground-truth counts do not match");
    }
    k = std::min(k, gt.dim);

    std::vector<RunResult> results;
    auto should_run = [&](const std::string& target) {
        return mode == "all" || mode == target;
    };

    if (should_run("dexor_noopt")) {
        results.push_back(run_dexor(dataset, queries, gt, ef, k, num_rounds, -1, 0, 0.0, false, 0.0));
    }
    if (should_run("dexor_opt")) {
        const double cache_ratio = 0.01;
        const size_t cache_size = std::max<size_t>(1, static_cast<size_t>(train_count * cache_ratio));
        results.push_back(run_dexor(dataset, queries, gt, ef, k, num_rounds, 2, cache_size, cache_ratio, true, tls_ratio));
    }
    if (should_run("hnswalp_tls")) {
        results.push_back(run_hnswalp_tls(dataset, queries, gt, ef, k, num_rounds, tls_ratio));
    }
    if (should_run("hnsw")) {
        results.push_back(run_original_hnsw(dataset, queries, gt, ef, k, num_rounds));
    }

    if (results.empty()) {
        std::cerr << "No matching mode: " << mode
                  << " (expected all, dexor_noopt, dexor_opt, hnswalp_tls, or hnsw)" << std::endl;
        return 1;
    }

    if (write_csv != 0) {
        const bool existing = std::ifstream(output_csv).good();
        std::ofstream csv(output_csv, append_csv ? std::ios::app : std::ios::out);
        if (!csv) {
            throw std::runtime_error("Cannot open output csv: " + output_csv);
        }
        if (!append_csv || !existing) {
            write_csv_header(csv);
        }
        for (const auto& result : results) {
            write_result(csv, result);
        }
    }

    for (const auto& result : results) {
        std::cout << result.method
                  << " dataset=" << result.dataset
                  << " ef=" << result.ef
                  << " rounds=" << result.num_rounds
                  << " recall=" << std::fixed << std::setprecision(6) << result.recall
                  << " mean_us=" << std::setprecision(2) << result.mean_latency_us
                  << " p99_us=" << result.p99_latency_us
                  << " qps=" << result.qps
                  << " decoding_calls=" << result.decoding_calls
                  << " avg_decode_per_query=" << std::setprecision(4) << result.avg_decode_per_query
                  << std::endl;
    }

    return 0;
}
