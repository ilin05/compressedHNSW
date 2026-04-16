#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_set>
#include <iomanip>

#include "../../hnswlib/hnswlib.h"

using namespace std;
using namespace hnswlib;

// Load fvecs as double to match L2SpaceDouble.
double* load_fvecs_as_double(const std::string& filename, size_t& num_vectors, size_t& dim) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        std::cerr << "Cannot open " << filename << std::endl;
        return nullptr;
    }

    int32_t d;
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

    int32_t d;
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
    std::vector<std::string> datasets = {
        "fashion-mnist-784-euclidean",
        "gist-960-euclidean",
        "mnist-784-euclidean",
        "sift-128-euclidean"
    };

    size_t ef = 100;
    size_t k = 1;

    // Optional args:
    //   --base_dir <dir>
    //   --dataset <name1> [name2 ...]
    //   --ef <value>
    //   --k <value>
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--base_dir" && i + 1 < argc) {
            base_dir = argv[++i];
            if (base_dir.back() != '/' && base_dir.back() != '\\') {
                base_dir += "/";
            }
        } else if (arg == "--dataset" && i + 1 < argc) {
            datasets.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                datasets.push_back(argv[++i]);
            }
        } else if (arg == "--ef" && i + 1 < argc) {
            ef = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (arg == "--k" && i + 1 < argc) {
            k = static_cast<size_t>(std::stoul(argv[++i]));
        }
    }

    for (const auto& dataset_name : datasets) {
        std::cout << "\n==============================================\n";
        std::cout << "Profiling dataset: " << dataset_name << std::endl;

        const std::string query_file = base_dir + dataset_name + "_test.fvecs";
        const std::string gt_file = base_dir + dataset_name + "_neighbors.ivecs";
        const std::string index_path = dataset_name + "_train.fvecs_hnsw.bin";

        size_t qsize = 0, qdim = 0;
        double* massQ = load_fvecs_as_double(query_file, qsize, qdim);
        if (!massQ) {
            continue;
        }

        size_t gt_num = 0, gt_dim = 0;
        unsigned int* massQA = load_ivecs(gt_file, gt_num, gt_dim);
        if (!massQA) {
            delete[] massQ;
            continue;
        }

        if (qsize != gt_num) {
            std::cerr << "Query size and GT size mismatch for " << dataset_name << std::endl;
            delete[] massQ;
            delete[] massQA;
            continue;
        }

        if (k > gt_dim) {
            k = gt_dim;
        }

        L2SpaceDouble l2space(qdim);
        HierarchicalNSW<double>* appr_alg = nullptr;

        try {
            appr_alg = new HierarchicalNSW<double>(&l2space, index_path, false);
        } catch (const std::exception& e) {
            std::cerr << "Failed to load index " << index_path << ": " << e.what() << std::endl;
            delete[] massQ;
            delete[] massQA;
            continue;
        }

        appr_alg->setEf(ef);
        appr_alg->resetAccessCounts();

        size_t correct = 0;
        const size_t total = qsize * k;

        for (long i = 0; i < static_cast<long>(qsize); ++i) {
            std::priority_queue<std::pair<double, labeltype>> result = appr_alg->searchKnn(massQ + qdim * i, k);

            std::unordered_set<labeltype> gt;
            for (size_t j = 0; j < k; ++j) {
                gt.insert(massQA[i * gt_dim + j]);
            }

            while (!result.empty()) {
                if (gt.find(result.top().second) != gt.end()) {
                    correct++;
                }
                result.pop();
            }
        }

        const double recall = (total == 0) ? 0.0 : (1.0 * correct / total);
        std::cout << "Recall@" << k << " at ef=" << ef << ": " << std::fixed << std::setprecision(4) << recall << std::endl;

        const std::vector<unsigned long long> access_counts = appr_alg->getAccessCounts();
        const std::vector<int> levels = appr_alg->getNodeLevels();
        const size_t node_count = appr_alg->getCurrentElementCount();

        const std::string node_csv = "hnsw_node_level_access_degree_" + dataset_name + ".csv";
        std::ofstream out(node_csv);
        out << "internal_id,level,access_count,degree0\n";

        for (size_t id = 0; id < node_count; ++id) {
            const unsigned long long cnt = (id < access_counts.size()) ? access_counts[id] : 0ULL;
            const int level = (id < levels.size()) ? levels[id] : 0;
            const int degree0 = appr_alg->getLevel0LinkListSize(static_cast<tableint>(id));
            out << id << "," << level << "," << cnt << "," << degree0 << "\n";
        }
        out.close();

        const std::string summary_csv = "hnsw_profile_summary_" + dataset_name + ".csv";
        std::ofstream summary(summary_csv);
        summary << "dataset,queries,dim,k,ef,recall,node_count\n";
        summary << dataset_name << "," << qsize << "," << qdim << "," << k << "," << ef << "," << recall << "," << node_count << "\n";
        summary.close();

        std::cout << "Wrote node stats: " << node_csv << std::endl;
        std::cout << "Wrote summary  : " << summary_csv << std::endl;

        delete appr_alg;
        delete[] massQ;
        delete[] massQA;
    }

    return 0;
}
