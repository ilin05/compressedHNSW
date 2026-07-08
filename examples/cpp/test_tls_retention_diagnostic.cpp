#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/hnswalg_simplified_alp_PQ.h"

using namespace hnswlib;

class StopW {
    std::chrono::steady_clock::time_point time_begin_;

 public:
    StopW() : time_begin_(std::chrono::steady_clock::now()) {}

    double getElapsedTimeMicro() const {
        const auto time_end = std::chrono::steady_clock::now();
        return static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin_).count());
    }
};

static float* load_fvecs_as_float(const std::string& filename, size_t& num_vectors, size_t& dim) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        std::cerr << "Cannot open " << filename << std::endl;
        return nullptr;
    }

    int32_t d;
    input.read(reinterpret_cast<char*>(&d), 4);
    dim = static_cast<size_t>(d);

    input.seekg(0, std::ios::end);
    const size_t file_size = static_cast<size_t>(input.tellg());
    num_vectors = file_size / (4 + dim * 4);

    float* data = new float[num_vectors * dim];
    input.seekg(0, std::ios::beg);

    for (size_t i = 0; i < num_vectors; ++i) {
        input.read(reinterpret_cast<char*>(&d), 4);
        input.read(reinterpret_cast<char*>(data + i * dim), dim * 4);
    }
    return data;
}

static unsigned int* load_ivecs(const std::string& filename, size_t& num_vectors, size_t& dim) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        std::cerr << "Cannot open " << filename << std::endl;
        return nullptr;
    }

    int32_t d;
    input.read(reinterpret_cast<char*>(&d), 4);
    dim = static_cast<size_t>(d);

    input.seekg(0, std::ios::end);
    const size_t file_size = static_cast<size_t>(input.tellg());
    num_vectors = file_size / (4 + dim * 4);

    unsigned int* data = new unsigned int[num_vectors * dim];
    input.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num_vectors; ++i) {
        input.read(reinterpret_cast<char*>(&d), 4);
        input.read(reinterpret_cast<char*>(data + i * dim), dim * 4);
    }
    return data;
}

static double run_recall_latency(
    HierarchicalNSWALPSIMPLIFIEDPQ<float>& index,
    const float* queries,
    const unsigned int* gt,
    size_t qsize,
    size_t qdim,
    size_t gt_dim,
    size_t k,
    double& latency_ms) {

    size_t correct = 0;
    const StopW stopw;

    for (size_t qi = 0; qi < qsize; ++qi) {
        auto result = index.searchKnn(queries + qi * qdim, k);

        std::unordered_set<labeltype> truth;
        for (size_t j = 0; j < k && j < gt_dim; ++j) {
            truth.insert(static_cast<labeltype>(gt[qi * gt_dim + j]));
        }

        while (!result.empty()) {
            if (truth.find(result.top().second) != truth.end()) {
                ++correct;
            }
            result.pop();
        }
    }

    latency_ms = stopw.getElapsedTimeMicro() / 1000.0 / static_cast<double>(qsize);
    return static_cast<double>(correct) / static_cast<double>(qsize * k);
}

int main(int argc, char** argv) {
    std::string base_dir = "../datasets/hdf5files/";
    std::string dataset = "sift-128-euclidean";
    std::string index_path = dataset + "_train.fvecs_hnswalp_simplified_pq.bin";
    std::string output_path = "tls_first_stage_retention_sift.csv";
    size_t ef = 100;
    size_t k = 1;
    std::vector<float> alphas = {0.05f, 0.1f, 0.2f, 0.3f, 0.5f, 1.0f};

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--base_dir" && i + 1 < argc) {
            base_dir = argv[++i];
            if (!base_dir.empty() && base_dir.back() != '/' && base_dir.back() != '\\') {
                base_dir += "/";
            }
        } else if (arg == "--dataset" && i + 1 < argc) {
            dataset = argv[++i];
            index_path = dataset + "_train.fvecs_hnswalp_simplified_pq.bin";
        } else if (arg == "--index_path" && i + 1 < argc) {
            index_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--ef" && i + 1 < argc) {
            ef = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (arg == "--alpha" && i + 1 < argc) {
            alphas.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                alphas.push_back(std::stof(argv[++i]));
            }
        }
    }

    const std::string query_file = base_dir + dataset + "_test.fvecs";
    const std::string gt_file = base_dir + dataset + "_neighbors.ivecs";

    size_t qsize = 0;
    size_t qdim = 0;
    float* queries = load_fvecs_as_float(query_file, qsize, qdim);
    if (!queries) return 1;

    size_t gt_num = 0;
    size_t gt_dim = 0;
    unsigned int* gt = load_ivecs(gt_file, gt_num, gt_dim);
    if (!gt) {
        delete[] queries;
        return 1;
    }

    if (qsize != gt_num) {
        std::cerr << "Query size and ground truth size mismatch: " << qsize << " vs " << gt_num << std::endl;
        delete[] queries;
        delete[] gt;
        return 1;
    }

    L2Space space(qdim);
    HierarchicalNSWALPSIMPLIFIEDPQ<float> index(&space, index_path, false);
    index.setEf(ef);
    index.setUseTLS(true);
    index.setProfilingMetrics(false);

    std::ofstream csv(output_path);
    csv << "Dataset,Alpha,ef,Decoded_Fraction,Top1_Retention,Top5_Retention,"
           "Exact_Admissible_Retention,Recall@1,Latency(ms)\n";

    std::cout << "TLS first-stage retention diagnostic: dataset=" << dataset
              << ", ef=" << ef << ", queries=" << qsize << std::endl;

    for (float alpha : alphas) {
        index.setTLSRatio(alpha);

        index.resetTLSRetentionStats();
        index.setTLSRetentionDiagnostic(true);
        double diagnostic_latency_ms = 0.0;
        (void)run_recall_latency(index, queries, gt, qsize, qdim, gt_dim, k, diagnostic_latency_ms);
        index.setTLSRetentionDiagnostic(false);
        const auto stats = index.getTLSRetentionStats();

        double latency_ms = 0.0;
        const double recall = run_recall_latency(index, queries, gt, qsize, qdim, gt_dim, k, latency_ms);

        csv << dataset << ","
            << std::fixed << std::setprecision(2) << alpha << ","
            << ef << ","
            << std::setprecision(6) << stats.decoded_fraction() << ","
            << stats.top1_retention() << ","
            << stats.top5_retention() << ","
            << stats.exact_admissible_retention() << ","
            << recall << ","
            << latency_ms << "\n";

        std::cout << "alpha=" << std::fixed << std::setprecision(2) << alpha
                  << " decoded=" << std::setprecision(4) << stats.decoded_fraction()
                  << " top1=" << stats.top1_retention()
                  << " top5=" << stats.top5_retention()
                  << " admissible=" << stats.exact_admissible_retention()
                  << " recall@1=" << recall
                  << " latency=" << latency_ms << " ms" << std::endl;
    }

    csv.close();
    delete[] queries;
    delete[] gt;
    return 0;
}
