#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>
#include <omp.h>

#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>

#include "../../examples/data_processor/data_loader.h"
#include "bigann_test_utils.h"

using namespace std;

class StopW {
    std::chrono::steady_clock::time_point time_begin;
 public:
    StopW() { time_begin = std::chrono::steady_clock::now(); }
    double getElapsedTimeMicro() const {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
            time_end - time_begin)
            .count());
    }
};

static vector<float> to_float_buffer(const double* src, size_t n) {
    vector<float> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(src[i]);
    return out;
}

static bool resolve_hnswpq_params(
        size_t dim,
        int p,
        size_t n_train,
        int& M_pq,
        int& pq_nbits) {
    if (p <= 0) {
        return false;
    }
    int base_M_pq = static_cast<int>(dim) / p;
    if (base_M_pq <= 0) {
        return false;
    }

    if (n_train >= 65536) {
        M_pq = base_M_pq;
        pq_nbits = 16;
    } else {
        M_pq = base_M_pq * 2;
        pq_nbits = 8;
    }

    if (M_pq <= 0 || (dim % static_cast<size_t>(M_pq) != 0)) {
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    omp_set_num_threads(1);

    string query_file = "../bigann/bigann_query.bvecs";
    string gt_dir = "../bigann/gnd";
    vector<size_t> subsets = bigann_test_utils::parse_subsets_from_cli(argc, argv, {1, 10, 100});
    vector<string> algos = {"HNSWPQ", "HNSWSQ"};
    vector<int> pq_ms = {8};
    vector<int> sq_nbits = {8};

    size_t qsize = 10000;
    size_t k = 1;
    vector<size_t> efs = {10, 20, 40, 80, 120, 200, 400, 600, 800, 1000};
    size_t train_size = 1000000;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--query_file" && i + 1 < argc) {
            query_file = argv[++i];
        } else if (arg == "--gt_dir" && i + 1 < argc) {
            gt_dir = argv[++i];
        } else if (arg == "--qsize" && i + 1 < argc) {
            qsize = static_cast<size_t>(stoull(argv[++i]));
        } else if (arg == "--k" && i + 1 < argc) {
            k = static_cast<size_t>(stoull(argv[++i]));
        } else if (arg == "--train_size" && i + 1 < argc) {
            train_size = static_cast<size_t>(stoull(argv[++i]));
        } else if (arg == "--algorithm" && i + 1 < argc) {
            algos.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                string a = argv[++i];
                if (a == "HNSWPQ" || a == "HNSWSQ") algos.push_back(a);
            }
        } else if (arg == "--pq_m" && i + 1 < argc) {
            pq_ms.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                pq_ms.push_back(stoi(argv[++i]));
            }
        } else if (arg == "--sq_nbits" && i + 1 < argc) {
            sq_nbits.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                int bit = stoi(argv[++i]);
                if (bit == 4 || bit == 8 || bit == 16) sq_nbits.push_back(bit);
            }
        }
    }

    int qdim = 0;
    double* qraw = data_loader::loadBvecsChunk(query_file, 0, qsize, qdim);
    if (!qraw) {
        cerr << "Failed to load query file: " << query_file << endl;
        return -1;
    }
    vector<float> queries = to_float_buffer(qraw, qsize * static_cast<size_t>(qdim));
    delete[] qraw;

    ofstream csv("bigann_faiss_hnsw_pq_sq_search_results.csv");
    csv << "Dataset,Algorithm,Param,K,ef,Recall,TimePerQuery(ms)\n";

    for (size_t subset_m : subsets) {
        const size_t vec_count = subset_m * 1000000ULL;
        const size_t train_count = std::min(train_size, vec_count);
        string gt_file = gt_dir + "/idx_" + to_string(subset_m) + "M.ivecs";
        vector<vector<unsigned int>> gt;
        if (!bigann_test_utils::load_gt_ivecs_topk(gt_file, qsize, k, gt)) {
            cerr << "Skip " << bigann_test_utils::subset_label(subset_m) << " due to GT load failure." << endl;
            continue;
        }

        for (const auto& algo : algos) {
            const vector<int>& params = (algo == "HNSWPQ") ? pq_ms : sq_nbits;
            for (int p : params) {
                string current_algo_name = algo + "_" + to_string(p);
                if (algo == "HNSWPQ") {
                    int M_pq = 0;
                    int pq_nbits = 0;
                    if (!resolve_hnswpq_params(static_cast<size_t>(qdim), p, train_count, M_pq, pq_nbits)) {
                        cerr << "Skip invalid HNSWPQ setting in search: p=" << p
                             << ", dim=" << qdim << ", n_train=" << train_count << endl;
                        continue;
                    }
                    current_algo_name += "_M" + to_string(M_pq) + "x" + to_string(pq_nbits);
                }

                string index_path = bigann_test_utils::subset_label(subset_m) + "_" + current_algo_name + ".bin";
                faiss::Index* index = nullptr;
                try {
                    index = faiss::read_index(index_path.c_str());
                } catch (const exception& e) {
                    cerr << "Cannot load index " << index_path << ": " << e.what() << endl;
                    continue;
                }

                 cout << "\n=== Searching " << current_algo_name << " on "
                     << bigann_test_utils::subset_label(subset_m) << " ===" << endl;

                vector<faiss::idx_t> I(qsize * k);
                vector<float> D(qsize * k);

                for (size_t ef : efs) {
                    if (auto* hnsw = dynamic_cast<faiss::IndexHNSW*>(index)) {
                        hnsw->hnsw.efSearch = static_cast<int>(ef);
                    }

                    StopW timer;
                    index->search(qsize, queries.data(), k, D.data(), I.data());
                    const double time_ms = timer.getElapsedTimeMicro() / 1000.0 / static_cast<double>(qsize);

                    size_t correct = 0;
                    for (size_t i = 0; i < qsize; ++i) {
                        unordered_set<unsigned int> gset(gt[i].begin(), gt[i].end());
                        for (size_t j = 0; j < k; ++j) {
                            faiss::idx_t id = I[i * k + j];
                            if (id >= 0 && gset.find(static_cast<unsigned int>(id)) != gset.end()) {
                                ++correct;
                            }
                        }
                    }
                    const double recall = static_cast<double>(correct) / static_cast<double>(qsize * k);

                    cout << "ef=" << ef << " recall=" << fixed << setprecision(4) << recall
                         << " time=" << setprecision(3) << time_ms << " ms" << endl;

                    csv << bigann_test_utils::subset_label(subset_m) << ',' << current_algo_name << ',' << p << ',' << k << ',' << ef
                        << ',' << fixed << setprecision(6) << recall << ',' << time_ms << "\n";

                    if (recall >= 0.99) break;
                }

                delete index;
            }
        }
    }

    cout << "\nSaved CSV: bigann_faiss_hnsw_pq_sq_search_results.csv" << endl;
    return 0;
}
