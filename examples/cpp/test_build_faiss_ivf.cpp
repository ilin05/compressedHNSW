#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <omp.h>

#include <faiss/IndexIVF.h>
#include <faiss/index_factory.h>
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
    void reset() { time_begin = std::chrono::steady_clock::now(); }
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
    omp_set_num_threads(32);

    string normal_base_dir = "../datasets/hdf5files/";
    string bigann_base_file = "../bigann/bigann_base.bvecs";

    vector<string> datasets = {
            "fashion-mnist-784-euclidean",
            "gist-960-euclidean",
            "mnist-784-euclidean",
            "sift-128-euclidean",
            "deep-image-96-angular"
        };
    vector<string> algos = {"IVFFlat", "IVFPQ"};
    vector<int> nlists = {1024, 4096};
    vector<int> pq_ms = {8};

    size_t train_size = 100000;
    size_t add_chunk = 500000;

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
        } else if (arg == "--normal_base_dir" && i + 1 < argc) {
            normal_base_dir = argv[++i];
        } else if (arg == "--bigann_base_file" && i + 1 < argc) {
            bigann_base_file = argv[++i];
        } else if (arg == "--train_size" && i + 1 < argc) {
            train_size = static_cast<size_t>(stoull(argv[++i]));
        } else if (arg == "--add_chunk" && i + 1 < argc) {
            add_chunk = static_cast<size_t>(stoull(argv[++i]));
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

    ofstream csv("faiss_ivf_build_results.csv");
    csv << "Dataset,Type,Algorithm,NList,Param,Vectors,Dim,TrainSize,TrainTime(s),AddTime(s),TotalTime(s),IndexFile\n";

    for (const auto& ds : datasets) {
        size_t subset_m = 0;
        const bool is_bigann = is_bigann_dataset_token(ds, subset_m);

        size_t n = 0;
        size_t dim = 0;
        float* data = nullptr;

        if (is_bigann) {
            const size_t vec_count = subset_m * 1000000ULL;
            int dim_detect = 0;
            double* probe = nullptr;
            try {
                probe = data_loader::loadBvecsChunk(bigann_base_file, 0, 1, dim_detect);
            } catch (const exception& e) {
                cerr << "Skip " << ds << " due to error: " << e.what() << endl;
                continue;
            }
            delete[] probe;
            dim = static_cast<size_t>(dim_detect);
            n = vec_count;
        } else {
            const string train_file = normal_base_dir + ds + "_train.fvecs";
            try {
                data = load_fvecs(train_file, n, dim);
            } catch (const exception& e) {
                cerr << "Skip " << ds << " due to error: " << e.what() << endl;
                continue;
            }
        }

        for (const auto& algo : algos) {
            for (int nlist : nlists) {
                if (nlist <= 0) {
                    cerr << "Skip invalid nlist=" << nlist << endl;
                    continue;
                }

                vector<int> params = {0};
                if (algo == "IVFPQ") {
                    params = pq_ms;
                }

                for (int p : params) {
                    string algo_name;
                    string factory;
                    if (algo == "IVFFlat") {
                        algo_name = "IVFFlat_nlist" + to_string(nlist);
                        factory = "IVF" + to_string(nlist) + ",Flat";
                    } else {
                        int M_pq = static_cast<int>(dim) / p;
                        if (p <= 0 || M_pq <= 0) {
                            cerr << "Skip invalid IVFPQ setting: pq_m=" << p << ", dim=" << dim << endl;
                            continue;
                        }
                        algo_name = "IVFPQ_" + to_string(p) + "_nlist" + to_string(nlist);
                        factory = "IVF" + to_string(nlist) + ",PQ" + to_string(M_pq);
                    }

                    cout << "\n=== Building " << algo_name << " for " << ds << " with " << factory
                         << " ===" << endl;

                    faiss::Index* index = nullptr;
                    try {
                        index = faiss::index_factory(static_cast<int>(dim), factory.c_str(), faiss::METRIC_L2);
                    } catch (const exception& e) {
                        cerr << "index_factory failed on " << ds << ": " << e.what() << endl;
                        continue;
                    }

                    StopW timer;
                    size_t train_count = 0;
                    if (is_bigann) {
                        train_count = std::min(train_size, n);
                        int tdim = 0;
                        double* train_raw = nullptr;
                        try {
                            train_raw = data_loader::loadBvecsChunk(bigann_base_file, 0, train_count, tdim);
                        } catch (const exception& e) {
                            cerr << "Failed to load train chunk for " << ds << ": " << e.what() << endl;
                            delete index;
                            continue;
                        }
                        vector<float> train_float = to_float_buffer(train_raw, train_count * dim);
                        delete[] train_raw;
                        index->train(static_cast<faiss::idx_t>(train_count), train_float.data());
                    } else {
                        train_count = n;
                        index->train(static_cast<faiss::idx_t>(n), data);
                    }
                    const double train_s = timer.getElapsedTimeMicro() / 1e6;

                    timer.reset();
                    if (is_bigann) {
                        size_t added = 0;
                        while (added < n) {
                            const size_t take = std::min(add_chunk, n - added);
                            int cdim = 0;
                            double* raw = nullptr;
                            try {
                                raw = data_loader::loadBvecsChunk(bigann_base_file, added, take, cdim);
                            } catch (const exception& e) {
                                cerr << "Add chunk load failed for " << ds << " at " << added << ": "
                                     << e.what() << endl;
                                break;
                            }
                            vector<float> fbuf = to_float_buffer(raw, take * dim);
                            delete[] raw;
                            index->add(static_cast<faiss::idx_t>(take), fbuf.data());
                            added += take;
                        }
                    } else {
                        index->add(static_cast<faiss::idx_t>(n), data);
                    }
                    const double add_s = timer.getElapsedTimeMicro() / 1e6;

                    const string index_path = ds + "_" + algo_name + ".bin";
                    faiss::write_index(index, index_path.c_str());

                    csv << ds << ',' << (is_bigann ? "bigann" : "normal") << ',' << algo << ',' << nlist
                        << ',' << p << ',' << n << ',' << dim << ',' << train_count << ',' << fixed
                        << setprecision(6) << train_s << ',' << add_s << ',' << (train_s + add_s) << ','
                        << index_path << "\n";

                    cout << "train=" << train_s << "s add=" << add_s << "s index=" << index_path << endl;
                    delete index;
                }
            }
        }

        if (!is_bigann && data) {
            delete[] data;
        }
    }

    cout << "\nSaved CSV: faiss_ivf_build_results.csv" << endl;
    return 0;
}
