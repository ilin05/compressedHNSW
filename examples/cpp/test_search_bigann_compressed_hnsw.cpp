#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <string>
#include <unordered_set>
#include <vector>

#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/compressed_hnsw_framework.h"
#include "../../examples/data_processor/data_loader.h"
#include "bigann_test_utils.h"

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

template <typename CodecPolicy>
void search_one_algo(size_t subset_m,
                     const std::string& algo,
                     const std::vector<size_t>& efs,
                     double* queries,
                     int qdim,
                     size_t qsize,
                     size_t k,
                     const std::vector<std::vector<unsigned int>>& gt,
                     std::ofstream& csv,
                     bool use_tls,
                     double tls_ratio) {
    const size_t cache_size = (subset_m * 1000000ULL) / 100;
    const std::string index_file = "bigann_" + std::to_string(subset_m) + "M_" + algo + "_pq.bin";

    L2SpaceDouble l2space(static_cast<size_t>(qdim));
    HierarchicalNSWCABFRAMEWORK<double, CodecPolicy> index(&l2space, index_file, true, cache_size);
    index.setUseTls(use_tls);
    index.setTlsRatio(tls_ratio);

    for (size_t ef : efs) {
        index.setEf(ef);
        size_t correct = 0;
        StopW timer;

        for (size_t i = 0; i < qsize; ++i) {
            auto result = index.searchKnn(queries + i * qdim, k);
            std::unordered_set<labeltype> gt_set(gt[i].begin(), gt[i].end());
            while (!result.empty()) {
                if (gt_set.find(result.top().second) != gt_set.end()) ++correct;
                result.pop();
            }
        }

        const double recall = static_cast<double>(correct) / static_cast<double>(qsize * k);
        const double time_ms = timer.elapsed_us() / 1000.0 / static_cast<double>(qsize);

        std::cout << "  " << algo << " ef=" << ef
                  << " recall=" << std::fixed << std::setprecision(4) << recall
                  << " time=" << std::setprecision(3) << time_ms << " ms" << std::endl;

        csv << bigann_test_utils::subset_label(subset_m) << ',' << subset_m << ',' << algo << ','
            << k << ',' << ef << ',' << std::fixed << std::setprecision(6)
            << recall << ',' << time_ms << "\n";

        if (recall >= 0.99) break;
    }
}

int main(int argc, char** argv) {
    std::string query_file = "../bigann/bigann_query.bvecs";
    std::string gt_dir = "../bigann/gnd";
    std::vector<size_t> subsets = bigann_test_utils::parse_subsets_from_cli(argc, argv, {1, 10, 100});
    std::vector<std::string> algorithms = {"DeXOR", "Gorilla", "Elf", "Camel", "DeXORPlus"};
    bool use_tls = false;
    double tls_ratio = 0.2;
    size_t k = 10;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--query_file" && i + 1 < argc) query_file = argv[++i];
        else if (arg == "--gt_dir" && i + 1 < argc) gt_dir = argv[++i];
        else if (arg == "--k" && i + 1 < argc) k = static_cast<size_t>(std::stoul(argv[++i]));
        else if (arg == "--use_tls" && i + 1 < argc) use_tls = std::stoi(argv[++i]) != 0;
        else if (arg == "--tls_ratio" && i + 1 < argc) tls_ratio = std::stod(argv[++i]);
        else if (arg == "--algorithm" && i + 1 < argc) {
            algorithms.clear();
            while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                algorithms.push_back(argv[++i]);
            }
        }
    }

    int qdim = 0;
    const size_t qsize = 10000;
    double* queries = data_loader::loadBvecsChunk(query_file, 0, qsize, qdim);
    if (!queries) return -1;

    std::vector<size_t> efs = {10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 300, 400, 500};
    std::ofstream csv("bigann_compressed_hnsw_search_results.csv");
    csv << "Dataset,SubsetMillion,Algorithm,K,ef,Recall,TimePerQuery(ms)\n";

    for (size_t m : subsets) {
        const std::string gt_file = gt_dir + "/idx_" + std::to_string(m) + "M.ivecs";
        std::vector<std::vector<unsigned int>> gt;
        if (!bigann_test_utils::load_gt_ivecs_topk(gt_file, qsize, k, gt)) continue;

        std::cout << "\n=== Searching compressed_hnsw for " << bigann_test_utils::subset_label(m) << " ===" << std::endl;

        for (const auto& algo : algorithms) {
            if (algo == "DeXOR") search_one_algo<codecs::DeXORCodecPolicy>(m, algo, efs, queries, qdim, qsize, k, gt, csv, use_tls, tls_ratio);
            else if (algo == "Gorilla") search_one_algo<codecs::GorillaCodecPolicy>(m, algo, efs, queries, qdim, qsize, k, gt, csv, use_tls, tls_ratio);
            else if (algo == "Elf") search_one_algo<codecs::ElfCodecPolicy>(m, algo, efs, queries, qdim, qsize, k, gt, csv, use_tls, tls_ratio);
            else if (algo == "Camel") search_one_algo<codecs::CamelCodecPolicy>(m, algo, efs, queries, qdim, qsize, k, gt, csv, use_tls, tls_ratio);
            else if (algo == "DeXORPlus") search_one_algo<codecs::DeXORPlusCodecPolicy>(m, algo, efs, queries, qdim, qsize, k, gt, csv, use_tls, tls_ratio);
        }
    }

    delete[] queries;
    std::cout << "\nSaved CSV: bigann_compressed_hnsw_search_results.csv" << std::endl;
    return 0;
}
