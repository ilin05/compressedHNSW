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

int main(int argc, char** argv) {
    std::string base_dir = "../datasets/hdf5files/";
    std::string dataset = "sift-128-euclidean";
    std::string out_csv = "hnsw_profile_results.csv";
    int threads = 32;
    size_t k = 10;
    size_t ef = 100;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--base_dir" && i + 1 < argc) {
            base_dir = argv[++i];
        } else if (arg == "--dataset" && i + 1 < argc) {
            dataset = argv[++i];
        } else if (arg == "--output_csv" && i + 1 < argc) {
            out_csv = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            threads = std::stoi(argv[++i]);
        } else if (arg == "--k" && i + 1 < argc) {
            k = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (arg == "--ef" && i + 1 < argc) {
            ef = static_cast<size_t>(std::stoul(argv[++i]));
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << std::endl;
            return -1;
        }
    }

    if (!base_dir.empty() && base_dir.back() != '/' && base_dir.back() != '\\') {
        base_dir += '/';
    }

    omp_set_num_threads(threads);

    const std::string query_file = base_dir + dataset + "_test.fvecs";
    const std::string gt_file = base_dir + dataset + "_neighbors.ivecs";
    const std::string index_file = dataset + "_train.fvecs_hnsw.bin";

    size_t qsize = 0, qdim = 0;
    double* queries = load_fvecs_as_double(query_file, qsize, qdim);
    if (!queries) return -1;

    size_t gt_num = 0, gt_dim = 0;
    unsigned int* gt = load_ivecs(gt_file, gt_num, gt_dim);
    const bool has_gt = (gt != nullptr && gt_num == qsize);

    std::atomic<long> distance_calls{0};
    std::atomic<long> distance_time_us{0};
    bool enable_metrics = true;

    TimedL2SpaceDouble l2space(qdim, &distance_calls, &distance_time_us, &enable_metrics);
    HierarchicalNSW<double>* index = nullptr;
    try {
        index = new HierarchicalNSW<double>(&l2space, index_file, false);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load index: " << e.what() << std::endl;
        delete[] queries;
        if (gt) delete[] gt;
        return -1;
    }

    index->setEf(ef);

    const size_t topk = has_gt ? std::min(k, gt_dim) : k;
    size_t correct = 0;

    StopW timer;
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
    }
    const double total_us = timer.elapsed_us();

    enable_metrics = false;
    const long dist_calls = distance_calls.load();
    const long dist_us = distance_time_us.load();

    const double avg_total_us_per_query = total_us / static_cast<double>(qsize);
    const double avg_dist_us_per_query = static_cast<double>(dist_us) / static_cast<double>(qsize);
    const double dist_share = (total_us > 0.0) ? (static_cast<double>(dist_us) / total_us * 100.0) : 0.0;
    const double avg_other_us_per_query = avg_total_us_per_query - avg_dist_us_per_query;

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

    csv << "Dataset,Algorithm,K,ef,Queries,Recall,"
        << "TotalTimePerQuery(us),DistanceTimePerQuery(us),OtherTimePerQuery(us),"
        << "DistanceTimeShare(%),DistanceCalls,AvgDistanceCallsPerQuery\n";

    csv << dataset << ','
        << "HNSW" << ','
        << topk << ','
        << ef << ','
        << qsize << ','
        << std::fixed << std::setprecision(6) << recall << ','
        << avg_total_us_per_query << ','
        << avg_dist_us_per_query << ','
        << avg_other_us_per_query << ','
        << dist_share << ','
        << dist_calls << ','
        << (static_cast<double>(dist_calls) / static_cast<double>(qsize))
        << '\n';

    csv.close();

    std::cout << "\n[Profile Summary]" << std::endl;
    std::cout << "dataset=" << dataset
              << ", ef=" << ef
              << ", k=" << topk
              << ", queries=" << qsize << std::endl;
    if (has_gt) {
        std::cout << "recall@" << topk << "=" << std::fixed << std::setprecision(6) << recall << std::endl;
    }
    std::cout << "avg_total_us/query=" << std::fixed << std::setprecision(4) << avg_total_us_per_query << std::endl;
    std::cout << "avg_distance_us/query=" << avg_dist_us_per_query
              << " (" << dist_share << "%)" << std::endl;
    std::cout << "avg_other_us/query=" << avg_other_us_per_query << std::endl;
    std::cout << "avg_distance_calls/query="
              << (static_cast<double>(dist_calls) / static_cast<double>(qsize)) << std::endl;
    std::cout << "saved_csv=" << out_csv << std::endl;

    delete[] queries;
    if (gt) delete[] gt;
    delete index;
    return 0;
}