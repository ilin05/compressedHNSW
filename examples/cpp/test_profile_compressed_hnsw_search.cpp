#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
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

struct TimedL2Params {
    size_t dim;
    std::atomic<long>* distance_calls;
    std::atomic<long>* distance_time_us;
    bool* enable_metrics;
};

class TimedL2SpaceDouble final : public SpaceInterface<double> {
private:
    TimedL2Params params_;

    static double timed_l2_distance(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        const auto* p = reinterpret_cast<const TimedL2Params*>(qty_ptr);
        const auto* a = reinterpret_cast<const double*>(pVect1v);
        const auto* b = reinterpret_cast<const double*>(pVect2v);

        if (*(p->enable_metrics)) {
            auto t0 = std::chrono::high_resolution_clock::now();
            double dist = 0.0;
            for (size_t i = 0; i < p->dim; ++i) {
                const double d = a[i] - b[i];
                dist += d * d;
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            (*(p->distance_calls))++;
            *(p->distance_time_us) += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            return dist;
        }

        double dist = 0.0;
        for (size_t i = 0; i < p->dim; ++i) {
            const double d = a[i] - b[i];
            dist += d * d;
        }
        return dist;
    }

public:
    TimedL2SpaceDouble(size_t dim,
                       std::atomic<long>* distance_calls,
                       std::atomic<long>* distance_time_us,
                       bool* enable_metrics)
        : params_{dim, distance_calls, distance_time_us, enable_metrics} {}

    size_t get_data_size() override {
        return params_.dim * sizeof(double);
    }

    DISTFUNC<double> get_dist_func() override {
        return timed_l2_distance;
    }

    void* get_dist_func_param() override {
        return &params_;
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

static std::string chain_tag(int chain_max_length) {
    if (chain_max_length < 0) return "chunlim";
    return "ch" + std::to_string(chain_max_length);
}

template <typename CodecPolicy>
int run_profile(const std::string& dataset,
                const std::string& base_dir,
                const std::string& algo,
                int chain_max_length,
                bool use_cache,
                bool use_tls,
                double tls_ratio,
                size_t k,
                size_t ef,
                const std::string& out_csv) {
    const std::string query_file = base_dir + dataset + "_test.fvecs";
    const std::string gt_file = base_dir + dataset + "_neighbors.ivecs";

    size_t qsize = 0, qdim = 0;
    double* queries = load_fvecs_as_double(query_file, qsize, qdim);
    if (!queries) return -1;

    size_t gt_num = 0, gt_dim = 0;
    unsigned int* gt = load_ivecs(gt_file, gt_num, gt_dim);
    bool has_gt = (gt != nullptr && gt_num == qsize);

    size_t cache_size = 0;
    if (use_cache) {
        const std::string train_file = base_dir + dataset + "_train.fvecs";
        std::ifstream train_in(train_file, std::ios::binary);
        if (train_in) {
            int32_t d = 0;
            train_in.read((char*)&d, 4);
            if (d > 0) {
                train_in.seekg(0, std::ios::end);
                size_t file_size = static_cast<size_t>(train_in.tellg());
                size_t train_num = file_size / (4 + static_cast<size_t>(d) * 4);
                cache_size = std::max<size_t>(1, train_num / 100);
            }
        }
    }

    const std::string index_file = dataset + "_" + algo + "_" + chain_tag(chain_max_length) + "_pq.bin";
    
    // Create distance computation metrics for this test run
    std::atomic<long> distance_calls{0};
    std::atomic<long> distance_time_us{0};
    bool enable_metrics = true;

    TimedL2SpaceDouble l2space(qdim, &distance_calls, &distance_time_us, &enable_metrics);
    auto* index = new HierarchicalNSWCABFRAMEWORK<double, CodecPolicy>(
        &l2space, index_file, true, cache_size);

    index->setUseTls(use_tls);
    index->setTlsRatio(tls_ratio);
    index->setProfilingMetrics(true);
    index->setEf(ef);

    const size_t topk = has_gt ? std::min(k, gt_dim) : k;

    index->resetProfilingMetrics();
    StopW timer;

    size_t correct = 0;
    for (size_t i = 0; i < qsize; ++i) {
        auto result = index->searchKnn(queries + i * qdim, topk);

        if (has_gt) {
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

    const double total_us = timer.elapsed_us();
    
    // Disable metrics collection after search
    enable_metrics = false;
    
    // Get distance computation metrics from TimedL2SpaceDouble
    const double dist_us = static_cast<double>(distance_time_us.load());
    const long dist_calls = distance_calls.load();
    
    const double dec_us = static_cast<double>(index->getTotalTimeDecoding());
    const double get_raw_us = static_cast<double>(index->getTotalTimeGetOriginalData());
    const double dist_share = (total_us > 0.0) ? (dist_us / total_us * 100.0) : 0.0;
    const double dec_share = (total_us > 0.0) ? (dec_us / total_us * 100.0) : 0.0;

    const long dec_calls = index->getDecodingCallCount();
    const long get_raw_calls = index->getOriginalDataCallCount();
    const long backtrack_hops = index->getOriginalDataBacktrackHops();

    const double avg_total_us_per_query = total_us / static_cast<double>(qsize);
    const double avg_dist_us_per_query = dist_us / static_cast<double>(qsize);
    const double avg_dec_us_per_query = dec_us / static_cast<double>(qsize);
    const double avg_get_raw_us_per_query = get_raw_us / static_cast<double>(qsize);
    const double avg_dec_calls_per_query = static_cast<double>(dec_calls) / static_cast<double>(qsize);
    const double avg_get_raw_calls_per_query = static_cast<double>(get_raw_calls) / static_cast<double>(qsize);
    const double avg_backtrack_per_get = (get_raw_calls > 0)
        ? static_cast<double>(backtrack_hops) / static_cast<double>(get_raw_calls)
        : 0.0;
    const double avg_backtrack_per_query = static_cast<double>(backtrack_hops) / static_cast<double>(qsize);

    const double recall = (has_gt && topk > 0)
        ? static_cast<double>(correct) / static_cast<double>(qsize * topk)
        : -1.0;

    std::ofstream csv(out_csv);
    if (!csv.is_open()) {
        std::cerr << "Failed to open csv: " << out_csv << std::endl;
        delete[] queries;
        if (gt) delete[] gt;
        delete index;
        return -1;
    }

    csv << "Dataset,Algorithm,ChainMaxLength,UseCache,UseTLS,TlsRatio,K,ef,Queries,Recall,"
        << "TotalTimePerQuery(us),DistanceTimePerQuery(us),DecodingTimePerQuery(us),GetOriginalDataTimePerQuery(us),"
        << "DistanceTimeShare(%),DecodingTimeShare(%),DistanceCalls,DecodingCalls,GetOriginalDataCalls,"
        << "AvgDecodingCallsPerQuery,AvgGetOriginalDataCallsPerQuery,BacktrackHops,AvgBacktrackPerGet,AvgBacktrackPerQuery\n";

    csv << dataset << ','
        << algo << ','
        << chain_max_length << ','
        << (use_cache ? 1 : 0) << ','
        << (use_tls ? 1 : 0) << ','
        << std::fixed << std::setprecision(4) << tls_ratio << ','
        << topk << ','
        << ef << ','
        << qsize << ','
        << std::setprecision(6) << recall << ','
        << std::setprecision(6) << avg_total_us_per_query << ','
        << avg_dist_us_per_query << ','
        << avg_dec_us_per_query << ','
        << avg_get_raw_us_per_query << ','
        << dist_share << ','
        << dec_share << ','
        << dist_calls << ','
        << dec_calls << ','
        << get_raw_calls << ','
        << avg_dec_calls_per_query << ','
        << avg_get_raw_calls_per_query << ','
        << backtrack_hops << ','
        << avg_backtrack_per_get << ','
        << avg_backtrack_per_query << '\n';

    csv.close();

    std::cout << "\n[Profile Summary]" << std::endl;
    std::cout << "dataset=" << dataset << ", algo=" << algo
              << ", chain_max=" << chain_max_length
              << ", use_cache=" << (use_cache ? 1 : 0)
              << ", use_tls=" << (use_tls ? 1 : 0)
              << ", ef=" << ef
              << ", k=" << topk << std::endl;
    if (has_gt) {
        std::cout << "recall@" << topk << "=" << std::fixed << std::setprecision(6) << recall << std::endl;
    }
    std::cout << "avg_total_us/query=" << std::fixed << std::setprecision(4) << avg_total_us_per_query << std::endl;
    std::cout << "avg_distance_us/query=" << avg_dist_us_per_query
              << " (" << dist_share << "%)" << std::endl;
    std::cout << "avg_decoding_us/query=" << avg_dec_us_per_query
              << " (" << dec_share << "%)" << std::endl;
    std::cout << "avg_getOriginalData_us/query=" << avg_get_raw_us_per_query << std::endl;
    std::cout << "avg_decoding_calls/query=" << avg_dec_calls_per_query << std::endl;
    std::cout << "avg_getOriginalData_calls/query=" << avg_get_raw_calls_per_query << std::endl;
    std::cout << "avg_backtrack_per_getOriginalData=" << avg_backtrack_per_get << std::endl;
    std::cout << "avg_backtrack_per_query=" << avg_backtrack_per_query << std::endl;
    std::cout << "saved_csv=" << out_csv << std::endl;

    delete[] queries;
    if (gt) delete[] gt;
    delete index;
    return 0;
}

int main(int argc, char** argv) {
    std::string base_dir = "../datasets/hdf5files/";
    std::string dataset = "sift-128-euclidean";
    std::string algorithm = "DeXOR";
    std::string out_csv = "compressed_hnsw_profile_results.csv";

    int chain_max = -1;
    int threads = 32;
    int use_cache = 0;
    int use_tls = 0;
    double tls_ratio = 0.0;
    size_t k = 10;
    size_t ef = 100;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--base_dir" && i + 1 < argc) {
            base_dir = argv[++i];
        } else if (arg == "--dataset" && i + 1 < argc) {
            dataset = argv[++i];
        } else if (arg == "--algorithm" && i + 1 < argc) {
            algorithm = argv[++i];
        } else if (arg == "--output_csv" && i + 1 < argc) {
            out_csv = argv[++i];
        } else if (arg == "--chain_max" && i + 1 < argc) {
            chain_max = std::stoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            threads = std::stoi(argv[++i]);
        } else if (arg == "--k" && i + 1 < argc) {
            k = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (arg == "--ef" && i + 1 < argc) {
            ef = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (arg == "--use_cache" && i + 1 < argc) {
            use_cache = std::stoi(argv[++i]);
        } else if (arg == "--use_tls" && i + 1 < argc) {
            use_tls = std::stoi(argv[++i]);
        } else if (arg == "--tls_ratio" && i + 1 < argc) {
            tls_ratio = std::stod(argv[++i]);
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << std::endl;
            return -1;
        }
    }

    if (!base_dir.empty() && base_dir.back() != '/' && base_dir.back() != '\\') {
        base_dir += '/';
    }

    omp_set_num_threads(threads);

    if (algorithm == "DeXOR") {
        return run_profile<codecs::DeXORCodecPolicy>(dataset, base_dir, algorithm, chain_max, use_cache != 0, use_tls != 0, tls_ratio, k, ef, out_csv);
    }
    if (algorithm == "Gorilla") {
        return run_profile<codecs::GorillaCodecPolicy>(dataset, base_dir, algorithm, chain_max, use_cache != 0, use_tls != 0, tls_ratio, k, ef, out_csv);
    }
    if (algorithm == "Elf") {
        return run_profile<codecs::ElfCodecPolicy>(dataset, base_dir, algorithm, chain_max, use_cache != 0, use_tls != 0, tls_ratio, k, ef, out_csv);
    }
    if (algorithm == "Camel") {
        return run_profile<codecs::CamelCodecPolicy>(dataset, base_dir, algorithm, chain_max, use_cache != 0, use_tls != 0, tls_ratio, k, ef, out_csv);
    }
    if (algorithm == "DeXORPlus") {
        return run_profile<codecs::DeXORPlusCodecPolicy>(dataset, base_dir, algorithm, chain_max, use_cache != 0, use_tls != 0, tls_ratio, k, ef, out_csv);
    }

    std::cerr << "Unknown algorithm: " << algorithm << std::endl;
    return -1;
}
