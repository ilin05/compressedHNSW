#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/hnswalg_simplified_alp_PQ.h"

using namespace hnswlib;

namespace {

template <typename T>
std::vector<T> load_vecs(const std::string& filename, size_t& count, size_t& dim) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot open: " + filename);
    }

    int32_t file_dim = 0;
    input.read(reinterpret_cast<char*>(&file_dim), sizeof(file_dim));
    if (!input || file_dim <= 0) {
        throw std::runtime_error("Invalid vector-file header: " + filename);
    }
    dim = static_cast<size_t>(file_dim);

    input.seekg(0, std::ios::end);
    const size_t bytes = static_cast<size_t>(input.tellg());
    const size_t record_bytes = sizeof(int32_t) + dim * sizeof(T);
    if (bytes % record_bytes != 0) {
        throw std::runtime_error("Unexpected vector-file size: " + filename);
    }
    count = bytes / record_bytes;
    input.seekg(0, std::ios::beg);

    std::vector<T> values(count * dim);
    for (size_t i = 0; i < count; ++i) {
        int32_t record_dim = 0;
        input.read(reinterpret_cast<char*>(&record_dim), sizeof(record_dim));
        input.read(reinterpret_cast<char*>(values.data() + i * dim),
                   static_cast<std::streamsize>(dim * sizeof(T)));
        if (!input || record_dim != file_dim) {
            throw std::runtime_error("Malformed vector record in: " + filename);
        }
    }
    return values;
}

struct SearchRound {
    double recall_at_1 = 0.0;
    double recall_at_10 = 0.0;
    double mean_latency_ms = 0.0;
    double p99_latency_ms = 0.0;
    double decode_calls_per_query = 0.0;
};

template <typename Queue>
std::vector<labeltype> ordered_labels(Queue queue) {
    std::vector<labeltype> labels(queue.size());
    for (size_t i = queue.size(); i > 0; --i) {
        labels[i - 1] = queue.top().second;
        queue.pop();
    }
    return labels;
}

SearchRound run_search_round(HierarchicalNSWALPSIMPLIFIEDPQ<float>& index,
                             const std::vector<float>& queries,
                             const std::vector<unsigned int>& gt,
                             size_t query_count,
                             size_t query_dim,
                             size_t gt_dim,
                             size_t k,
                             bool collect_decode_calls) {
    if (collect_decode_calls) {
        index.decoding_call_count.store(0, std::memory_order_relaxed);
        index.decoding_time.store(0, std::memory_order_relaxed);
        index.get_original_data_call_count.store(0, std::memory_order_relaxed);
    }
    index.setProfilingMetrics(collect_decode_calls);

    size_t recall_at_1_hits = 0;
    size_t recall_at_10_hits = 0;
    std::vector<double> per_query_latency_ms;
    per_query_latency_ms.reserve(query_count);

    for (size_t qi = 0; qi < query_count; ++qi) {
        const auto start = std::chrono::steady_clock::now();
        const auto labels = ordered_labels(index.searchKnn(queries.data() + qi * query_dim, k));
        const auto end = std::chrono::steady_clock::now();
        per_query_latency_ms.push_back(
            std::chrono::duration<double, std::milli>(end - start).count());

        if (!labels.empty() && gt_dim > 0 && labels.front() == gt[qi * gt_dim]) {
            ++recall_at_1_hits;
        }

        std::unordered_set<labeltype> truth;
        const size_t truth_k = std::min<size_t>(10, gt_dim);
        truth.reserve(truth_k * 2);
        for (size_t rank = 0; rank < truth_k; ++rank) {
            truth.insert(static_cast<labeltype>(gt[qi * gt_dim + rank]));
        }
        for (labeltype label : labels) {
            if (truth.find(label) != truth.end()) {
                ++recall_at_10_hits;
            }
        }
    }

    std::sort(per_query_latency_ms.begin(), per_query_latency_ms.end());
    SearchRound result;
    result.recall_at_1 = static_cast<double>(recall_at_1_hits) / static_cast<double>(query_count);
    result.recall_at_10 = static_cast<double>(recall_at_10_hits) /
                          static_cast<double>(query_count * std::min<size_t>(10, gt_dim));
    result.mean_latency_ms = std::accumulate(per_query_latency_ms.begin(), per_query_latency_ms.end(), 0.0) /
                             static_cast<double>(query_count);
    const size_t p99_index = static_cast<size_t>(std::ceil(0.99 * query_count)) - 1;
    result.p99_latency_ms = per_query_latency_ms[p99_index];
    result.decode_calls_per_query = collect_decode_calls
        ? static_cast<double>(index.decoding_call_count.load(std::memory_order_relaxed)) /
              static_cast<double>(query_count)
        : 0.0;
    index.setProfilingMetrics(false);
    return result;
}

struct Summary {
    double median = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
};

Summary summarize(std::vector<double> values) {
    if (values.empty()) return {};
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    const double median = values.size() % 2 == 0
        ? (values[middle - 1] + values[middle]) * 0.5
        : values[middle];
    return {median, values.front(), values.back()};
}

std::string ensure_trailing_separator(std::string path) {
    if (!path.empty() && path.back() != '/' && path.back() != '\\') {
        path.push_back('/');
    }
    return path;
}

std::string default_index_path(const std::string& dataset, const std::string& index_dir) {
    return ensure_trailing_separator(index_dir) + dataset + "_train.fvecs_hnswalp_simplified_pq.bin";
}

struct Config {
    std::string base_dir = "../datasets/hdf5files/";
    std::string index_dir;
    std::string output_path = "tls_accuracy_stability.csv";
    std::vector<std::string> datasets = {
        "sift-128-euclidean", "gist-960-euclidean", "fashion-mnist-784-euclidean"};
    std::vector<float> alphas = {0.1f, 0.2f, 0.3f, 0.5f, 1.0f};
    size_t ef = 100;
    size_t k = 10;
    size_t num_rounds = 3;
    size_t warmup_rounds = 1;
    size_t max_queries = 0;
};

template <typename T, typename Parser>
void parse_list(int argc, char** argv, int& i, std::vector<T>& output, Parser parser) {
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
        } else if (arg == "--index_dir" && i + 1 < argc) {
            config.index_dir = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            config.output_path = argv[++i];
        } else if (arg == "--dataset") {
            parse_list(argc, argv, i, config.datasets, [](const std::string& value) { return value; });
        } else if (arg == "--alpha") {
            parse_list(argc, argv, i, config.alphas, [](const std::string& value) { return std::stof(value); });
        } else if (arg == "--ef" && i + 1 < argc) {
            config.ef = static_cast<size_t>(std::stoull(argv[++i]));
        } else if (arg == "--k" && i + 1 < argc) {
            config.k = static_cast<size_t>(std::stoull(argv[++i]));
        } else if (arg == "--num_rounds" && i + 1 < argc) {
            config.num_rounds = static_cast<size_t>(std::stoull(argv[++i]));
        } else if (arg == "--warmup_rounds" && i + 1 < argc) {
            config.warmup_rounds = static_cast<size_t>(std::stoull(argv[++i]));
        } else if (arg == "--max_queries" && i + 1 < argc) {
            config.max_queries = static_cast<size_t>(std::stoull(argv[++i]));
        } else {
            throw std::runtime_error("Unknown or incomplete argument: " + arg);
        }
    }
    config.base_dir = ensure_trailing_separator(config.base_dir);
    for (float alpha : config.alphas) {
        if (alpha <= 0.0f || alpha > 1.0f) {
            throw std::runtime_error("Each alpha must be in (0, 1]");
        }
    }
    if (config.k < 10) {
        throw std::runtime_error("--k must be at least 10 to report Recall@10");
    }
    if (config.num_rounds == 0) {
        throw std::runtime_error("--num_rounds must be positive");
    }
    return config;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Config config = parse_config(argc, argv);
        std::ofstream csv(config.output_path);
        if (!csv) {
            throw std::runtime_error("Cannot open output CSV: " + config.output_path);
        }
        csv << "Dataset,Mode,Alpha,EF,K,Queries,NumRounds,DecodedFraction,CandidatePoolPerQuery,"
               "ShortlistedCandidatesPerQuery,Top1Retention,Top5Retention,ExactAdmissibleRetention,"
               "Recall@1_Median,Recall@1_Min,Recall@1_Max,Recall@10_Median,Recall@10_Min,Recall@10_Max,"
               "MeanLatencyMs_Median,MeanLatencyMs_Min,MeanLatencyMs_Max,"
               "P99LatencyMs_Median,P99LatencyMs_Min,P99LatencyMs_Max,"
               "DecodeCallsPerQuery_Median,DecodeCallsPerQuery_Min,DecodeCallsPerQuery_Max\n";

        for (const std::string& dataset : config.datasets) {
            size_t query_total = 0, query_dim = 0, gt_total = 0, gt_dim = 0;
            const auto queries = load_vecs<float>(config.base_dir + dataset + "_test.fvecs", query_total, query_dim);
            const auto gt = load_vecs<unsigned int>(config.base_dir + dataset + "_neighbors.ivecs", gt_total, gt_dim);
            if (query_total != gt_total) {
                throw std::runtime_error("Query/ground-truth count mismatch for " + dataset);
            }
            const size_t query_count = config.max_queries == 0
                ? query_total
                : std::min(config.max_queries, query_total);

            const std::string index_path = default_index_path(dataset, config.index_dir);
            std::cout << "[TLS Accuracy] dataset=" << dataset << ", index=" << index_path
                      << ", queries=" << query_count << ", ef=" << config.ef << std::endl;

            L2Space space(query_dim);
            HierarchicalNSWALPSIMPLIFIEDPQ<float> index(&space, index_path, false);
            index.setEf(config.ef);
            index.setUseTLS(true);

            for (float alpha : config.alphas) {
                index.setTLSRatio(alpha);

                // Diagnostic pass: it deliberately performs extra exact checks and is never timed.
                index.resetTLSRetentionStats();
                index.setTLSRetentionDiagnostic(true);
                (void)run_search_round(index, queries, gt, query_count, query_dim, gt_dim, config.k, false);
                index.setTLSRetentionDiagnostic(false);
                const auto retention = index.getTLSRetentionStats();

                for (size_t warmup = 0; warmup < config.warmup_rounds; ++warmup) {
                    (void)run_search_round(index, queries, gt, query_count, query_dim, gt_dim, config.k, false);
                }

                std::vector<double> recall1_values, recall10_values, mean_latency_values, p99_latency_values, decode_values;
                recall1_values.reserve(config.num_rounds);
                recall10_values.reserve(config.num_rounds);
                mean_latency_values.reserve(config.num_rounds);
                p99_latency_values.reserve(config.num_rounds);
                decode_values.reserve(config.num_rounds);

                for (size_t round = 0; round < config.num_rounds; ++round) {
                    const SearchRound metrics = run_search_round(
                        index, queries, gt, query_count, query_dim, gt_dim, config.k, true);
                    recall1_values.push_back(metrics.recall_at_1);
                    recall10_values.push_back(metrics.recall_at_10);
                    mean_latency_values.push_back(metrics.mean_latency_ms);
                    p99_latency_values.push_back(metrics.p99_latency_ms);
                    decode_values.push_back(metrics.decode_calls_per_query);
                }

                const Summary recall1 = summarize(recall1_values);
                const Summary recall10 = summarize(recall10_values);
                const Summary mean_latency = summarize(mean_latency_values);
                const Summary p99_latency = summarize(p99_latency_values);
                const Summary decode_calls = summarize(decode_values);
                const std::string mode = alpha == 1.0f ? "Exact-LVC (alpha=1.0)" : "Filtered-LVC";
                const double pool_per_query = retention.expansion_pools == 0 ? 0.0
                    : static_cast<double>(retention.pool_candidates) / static_cast<double>(query_count);
                const double shortlisted_per_query = retention.expansion_pools == 0 ? 0.0
                    : static_cast<double>(retention.shortlisted_candidates) / static_cast<double>(query_count);

                csv << dataset << ',' << mode << ',' << std::fixed << std::setprecision(2) << alpha << ','
                    << config.ef << ',' << config.k << ',' << query_count << ',' << config.num_rounds << ','
                    << std::setprecision(6) << retention.decoded_fraction() << ','
                    << pool_per_query << ',' << shortlisted_per_query << ','
                    << retention.top1_retention() << ',' << retention.top5_retention() << ','
                    << retention.exact_admissible_retention() << ','
                    << recall1.median << ',' << recall1.minimum << ',' << recall1.maximum << ','
                    << recall10.median << ',' << recall10.minimum << ',' << recall10.maximum << ','
                    << mean_latency.median << ',' << mean_latency.minimum << ',' << mean_latency.maximum << ','
                    << p99_latency.median << ',' << p99_latency.minimum << ',' << p99_latency.maximum << ','
                    << decode_calls.median << ',' << decode_calls.minimum << ',' << decode_calls.maximum << '\n';
                csv.flush();

                std::cout << "  alpha=" << std::fixed << std::setprecision(2) << alpha
                          << " mode=" << mode
                          << " top1-retention=" << std::setprecision(4) << retention.top1_retention()
                          << " recall@1=" << recall1.median
                          << " recall@10=" << recall10.median
                          << " mean/p99=" << mean_latency.median << '/' << p99_latency.median << " ms"
                          << " decode/query=" << decode_calls.median << std::endl;
            }
        }
        std::cout << "Saved TLS accuracy/stability results to " << config.output_path << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "TLS retention diagnostic failed: " << error.what() << std::endl;
        return 1;
    }
}
