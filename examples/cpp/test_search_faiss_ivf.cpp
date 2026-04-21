#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>
#include <omp.h>

#include <faiss/IndexIVF.h>
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
        return static_cast<double>(
                std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin)
                        .count());
    }
};

static float* load_fvecs(const string& filename, size_t& num_vectors, size_t& dim) {
    ifstream input(filename, ios::binary);
    if (!input) {
        throw runtime_error("Cannot open file " + filename);
    }
    int32_t d = 0;
    input.read(reinterpret_cast<char*>(&d), 4);
    dim = static_cast<size_t>(d);
    input.seekg(0, ios::end);
    size_t file_size = static_cast<size_t>(input.tellg());
    num_vectors = file_size / (4 + dim * 4);
    float* data = new float[num_vectors * dim];
    input.seekg(0, ios::beg);
    for (size_t i = 0; i < num_vectors; ++i) {
        input.read(reinterpret_cast<char*>(&d), 4);
        input.read(reinterpret_cast<char*>(data + i * dim), dim * 4);
    }
    return data;
}

static unsigned int* load_ivecs(const string& filename, size_t& num_vectors, size_t& dim) {
    ifstream input(filename, ios::binary);
    if (!input) {
        return nullptr;
    }
    int32_t d = 0;
    input.read(reinterpret_cast<char*>(&d), 4);
    dim = static_cast<size_t>(d);
    input.seekg(0, ios::end);
    size_t file_size = static_cast<size_t>(input.tellg());
    num_vectors = file_size / (4 + dim * 4);
    unsigned int* data = new unsigned int[num_vectors * dim];
    input.seekg(0, ios::beg);
    for (size_t i = 0; i < num_vectors; ++i) {
        input.read(reinterpret_cast<char*>(&d), 4);
        input.read(reinterpret_cast<char*>(data + i * dim), dim * 4);
    }
    return data;
}

static vector<float> to_float_buffer(const double* src, size_t n) {
    vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = static_cast<float>(src[i]);
    }
    return out;
}

static bool is_bigann_dataset_token(const string& token, size_t& subset_million) {
    return bigann_test_utils::parse_subset_token(token, subset_million);
}

int main(int argc, char** argv) {
    omp_set_num_threads(1);

    string normal_base_dir = "../datasets/hdf5files/";
    string bigann_query_file = "../bigann/bigann_query.bvecs";
    string bigann_gt_dir = "../bigann/gnd";

    vector<string> datasets = {
            "fashion-mnist-784-euclidean",
            "gist-960-euclidean",
            "mnist-784-euclidean",
            "sift-128-euclidean"};
    vector<string> algos = {"IVFFlat", "IVFPQ"};
    vector<int> nlists = {1024, 4096};
    vector<int> pq_ms = {8};
    vector<int> nprobes = {1, 2, 4, 8, 16, 32, 64};

    size_t qsize = 10000;
    size_t k = 10;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--dataset" && i + 1 < argc) {
            datasets.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                datasets.push_back(argv[++i]);
            }
        } else if (arg == "--algorithm" && i + 1 < argc) {
            algos.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                const string a = argv[++i];
                if (a == "IVFFlat" || a == "IVFPQ") {
                    algos.push_back(a);
                }
            }
        } else if (arg == "--nlist" && i + 1 < argc) {
            nlists.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                nlists.push_back(stoi(argv[++i]));
            }
        } else if (arg == "--pq_m" && i + 1 < argc) {
            pq_ms.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                pq_ms.push_back(stoi(argv[++i]));
            }
        } else if (arg == "--nprobe" && i + 1 < argc) {
            nprobes.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                nprobes.push_back(stoi(argv[++i]));
            }
        } else if (arg == "--normal_base_dir" && i + 1 < argc) {
            normal_base_dir = argv[++i];
        } else if (arg == "--bigann_query_file" && i + 1 < argc) {
            bigann_query_file = argv[++i];
        } else if (arg == "--bigann_gt_dir" && i + 1 < argc) {
            bigann_gt_dir = argv[++i];
        } else if (arg == "--qsize" && i + 1 < argc) {
            qsize = static_cast<size_t>(stoull(argv[++i]));
        } else if (arg == "--k" && i + 1 < argc) {
            k = static_cast<size_t>(stoull(argv[++i]));
        }
    }

    if (datasets.empty()) {
        cerr << "No datasets specified." << endl;
        return -1;
    }
    if (algos.empty()) {
        cerr << "No valid algorithms specified. Use --algorithm IVFFlat IVFPQ" << endl;
        return -1;
    }

    vector<float> bigann_queries;
    int qdim_bigann = 0;
    try {
        double* qraw = data_loader::loadBvecsChunk(bigann_query_file, 0, qsize, qdim_bigann);
        bigann_queries = to_float_buffer(qraw, qsize * static_cast<size_t>(qdim_bigann));
        delete[] qraw;
    } catch (const exception& e) {
        cerr << "Warning: cannot preload bigann queries: " << e.what() << endl;
    }

    ofstream csv("faiss_ivf_search_results_recall@10.csv");
    csv << "Dataset,Type,Algorithm,NList,Param,K,NProbe,Recall,TimePerQuery(ms)\n";

    for (const auto& ds : datasets) {
        size_t subset_m = 0;
        const bool is_bigann = is_bigann_dataset_token(ds, subset_m);

        float* queries = nullptr;
        size_t qn = 0;
        size_t qdim = 0;
        unsigned int* gt_dense = nullptr;
        size_t gt_dim_normal = 0;
        vector<vector<unsigned int>> gt_rows;

        if (is_bigann) {
            if (bigann_queries.empty()) {
                cerr << "Skip " << ds << " because bigann queries are unavailable." << endl;
                continue;
            }
            queries = bigann_queries.data();
            qn = qsize;
            qdim = static_cast<size_t>(qdim_bigann);
            const string gt_file = bigann_gt_dir + "/idx_" + to_string(subset_m) + "M.ivecs";
            if (!bigann_test_utils::load_gt_ivecs_topk(gt_file, qn, k, gt_rows)) {
                cerr << "Skip " << ds << " due to GT load failure." << endl;
                continue;
            }
        } else {
            const string query_file = normal_base_dir + ds + "_test.fvecs";
            const string gt_file = normal_base_dir + ds + "_neighbors.ivecs";
            try {
                queries = load_fvecs(query_file, qn, qdim);
                size_t gt_n = 0;
                size_t gt_d = 0;
                gt_dense = load_ivecs(gt_file, gt_n, gt_d);
                gt_dim_normal = gt_d;
                if (!gt_dense) {
                    cerr << "Skip " << ds << " due to GT file load failure." << endl;
                    delete[] queries;
                    continue;
                }
            } catch (const exception& e) {
                cerr << "Skip " << ds << " due to load error: " << e.what() << endl;
                if (queries) delete[] queries;
                continue;
            }
        }

        for (const auto& algo : algos) {
            for (int nlist : nlists) {
                vector<int> params = {0};
                if (algo == "IVFPQ") {
                    params = pq_ms;
                }

                for (int p : params) {
                    string algo_name;
                    if (algo == "IVFFlat") {
                        algo_name = "IVFFlat_nlist" + to_string(nlist);
                    } else {
                        algo_name = "IVFPQ_" + to_string(p) + "_nlist" + to_string(nlist);
                    }

                    const string index_path = ds + "_" + algo_name + ".bin";
                    faiss::Index* index = nullptr;
                    try {
                        index = faiss::read_index(index_path.c_str());
                    } catch (const exception& e) {
                        cerr << "Cannot load index " << index_path << ": " << e.what() << endl;
                        continue;
                    }

                    vector<faiss::idx_t> I(qn * k);
                    vector<float> D(qn * k);

                    for (int nprobe : nprobes) {
                        if (auto* ivf = dynamic_cast<faiss::IndexIVF*>(index)) {
                            ivf->nprobe = static_cast<size_t>(nprobe);
                        }

                        StopW timer;
                        index->search(static_cast<faiss::idx_t>(qn), queries, static_cast<faiss::idx_t>(k), D.data(), I.data());
                        const double time_ms = timer.getElapsedTimeMicro() / 1000.0 / static_cast<double>(qn);

                        size_t correct = 0;
                        for (size_t i = 0; i < qn; ++i) {
                            unordered_set<unsigned int> gset;
                            if (is_bigann) {
                                gset = unordered_set<unsigned int>(gt_rows[i].begin(), gt_rows[i].end());
                            } else {
                                const size_t topk = min(k, gt_dim_normal);
                                gset = unordered_set<unsigned int>(gt_dense + i * gt_dim_normal,
                                                                   gt_dense + i * gt_dim_normal + topk);
                            }

                            for (size_t j = 0; j < k; ++j) {
                                faiss::idx_t id = I[i * k + j];
                                if (id >= 0 && gset.find(static_cast<unsigned int>(id)) != gset.end()) {
                                    ++correct;
                                }
                            }
                        }
                        const double recall = static_cast<double>(correct) / static_cast<double>(qn * k);

                        cout << "Dataset=" << ds << " algo=" << algo_name << " nprobe=" << nprobe
                             << " recall=" << fixed << setprecision(4) << recall << " time="
                             << setprecision(3) << time_ms << " ms" << endl;

                        csv << ds << ',' << (is_bigann ? "bigann" : "normal") << ',' << algo << ',' << nlist
                            << ',' << p << ',' << k << ',' << nprobe << ',' << fixed << setprecision(6)
                            << recall << ',' << time_ms << "\n";

                        if (recall >= 0.99) {
                            break;
                        }
                    }

                    delete index;
                }
            }
        }

        if (!is_bigann) {
            delete[] queries;
            if (gt_dense) {
                delete[] gt_dense;
            }
        }
    }

    cout << "\nSaved CSV: faiss_ivf_search_results_recall@10.csv" << endl;
    return 0;
}
