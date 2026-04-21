#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <string>
#include <unordered_set>
#include <vector>

#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/hnswalg_simplified_alp_PQ.h"
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

int main(int argc, char** argv) {
    std::string query_file = "../bigann/bigann_query.bvecs";
    std::string gt_dir = "../bigann/gnd";
    std::vector<size_t> subsets = bigann_test_utils::parse_subsets_from_cli(argc, argv, {1, 10, 100});
    size_t k = 10;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--query_file" && i + 1 < argc) query_file = argv[++i];
        else if (arg == "--gt_dir" && i + 1 < argc) gt_dir = argv[++i];
        else if (arg == "--k" && i + 1 < argc) k = static_cast<size_t>(std::stoul(argv[++i]));
    }

    int qdim = 0;
    const size_t qsize = 10000;
    double* queries = data_loader::loadBvecsChunk(query_file, 0, qsize, qdim);
    if (!queries) return -1;

    L2SpaceDouble l2space(static_cast<size_t>(qdim));
    std::vector<size_t> efs = {10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 300, 400, 500};

    std::ofstream csv("bigann_hnswalp_pq_search_results_recall@10.csv");
    csv << "Dataset,SubsetMillion,K,ef,Recall,TimePerQuery(ms)\n";

    for (size_t m : subsets) {
        const std::string index_file = "bigann_" + std::to_string(m) + "M_hnswalp_pq.bin";
        const std::string gt_file = gt_dir + "/idx_" + std::to_string(m) + "M.ivecs";

        std::vector<std::vector<unsigned int>> gt;
        if (!bigann_test_utils::load_gt_ivecs_topk(gt_file, qsize, k, gt)) continue;

        std::cout << "\n=== Searching HNSWALP_PQ for " << bigann_test_utils::subset_label(m) << " ===" << std::endl;
        HierarchicalNSWALPSIMPLIFIEDPQ<double> index(&l2space, index_file, false);

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
            std::cout << "  ef=" << ef << " recall=" << std::fixed << std::setprecision(4) << recall
                      << " time=" << std::setprecision(3) << time_ms << " ms" << std::endl;

            csv << bigann_test_utils::subset_label(m) << ',' << m << ',' << k << ',' << ef << ','
                << std::fixed << std::setprecision(6) << recall << ',' << time_ms << "\n";
            if (recall >= 0.99) break;
        }
    }

    delete[] queries;
    std::cout << "\nSaved CSV: bigann_hnswalp_pq_search_results_recall@10.csv" << std::endl;
    return 0;
}
