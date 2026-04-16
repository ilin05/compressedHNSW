#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <omp.h>

#include <faiss/IndexHNSW.h>
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
        return static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
            time_end - time_begin)
            .count());
    }
    void reset() { time_begin = std::chrono::steady_clock::now(); }
};

static vector<float> to_float_buffer(const double* src, size_t n) {
    vector<float> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(src[i]);
    return out;
}

static size_t file_size_bytes(const string& path) {
    ifstream in(path, ios::binary | ios::ate);
    if (!in) return 0;
    return static_cast<size_t>(in.tellg());
}

int main(int argc, char** argv) {
    omp_set_num_threads(32);

    string base_file = "../bigann/bigann_base.bvecs";
    vector<size_t> subsets = bigann_test_utils::parse_subsets_from_cli(argc, argv, {1, 10, 100});
    vector<string> algos = {"HNSWPQ", "HNSWSQ"};
    vector<int> pq_ms = {8};
    vector<int> sq_nbits = {8};

    int M_hnsw = 16;
    int ef_construction = 200;
    size_t train_size = 1000000;
    size_t add_chunk = 500000;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--base_file" && i + 1 < argc) {
            base_file = argv[++i];
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
        } else if (arg == "--M" && i + 1 < argc) {
            M_hnsw = stoi(argv[++i]);
        } else if (arg == "--ef_construction" && i + 1 < argc) {
            ef_construction = stoi(argv[++i]);
        } else if (arg == "--train_size" && i + 1 < argc) {
            train_size = static_cast<size_t>(stoull(argv[++i]));
        } else if (arg == "--add_chunk" && i + 1 < argc) {
            add_chunk = static_cast<size_t>(stoull(argv[++i]));
        }
    }

    if (algos.empty()) {
        cerr << "No valid algorithms. Use --algorithm HNSWPQ HNSWSQ" << endl;
        return -1;
    }

    ofstream csv("bigann_faiss_hnsw_pq_sq_build_results.csv");
    csv << "Dataset,Algorithm,Param,Vectors,Dim,TrainSize,TrainTime(s),AddTime(s),TotalTime(s),IndexFile,IndexSize(Bytes)\n";

    for (size_t subset_m : subsets) {
        const size_t vec_count = subset_m * 1000000ULL;

        int dim_detect = 0;
        double* dim_probe = data_loader::loadBvecsChunk(base_file, 0, 1, dim_detect);
        if (!dim_probe) {
            cerr << "Failed to read base file: " << base_file << endl;
            continue;
        }
        delete[] dim_probe;
        const size_t dim = static_cast<size_t>(dim_detect);

        const size_t train_count = std::min(train_size, vec_count);
        double* train_raw = data_loader::loadBvecsChunk(base_file, 0, train_count, dim_detect);
        if (!train_raw) {
            cerr << "Failed to load training sample for " << bigann_test_utils::subset_label(subset_m) << endl;
            continue;
        }
        vector<float> train_float = to_float_buffer(train_raw, train_count * dim);
        delete[] train_raw;

        for (const auto& algo : algos) {
            const vector<int>& params = (algo == "HNSWPQ") ? pq_ms : sq_nbits;
            for (int p : params) {
                string factory;
                if (algo == "HNSWPQ") {
                    int M_pq = static_cast<int>(dim) / p;
                    if (M_pq <= 0) {
                        cerr << "Skip invalid PQ param p=" << p << " for dim=" << dim << endl;
                        continue;
                    }
                    factory = "HNSW" + to_string(M_hnsw) + ",PQ" + to_string(M_pq) + "x16";
                } else {
                    string sq_suffix = (p == 16) ? "SQfp16" : "SQ" + to_string(p);
                    factory = "HNSW" + to_string(M_hnsw) + "," + sq_suffix;
                }

                cout << "\n=== Building " << algo << "(" << p << ") for "
                     << bigann_test_utils::subset_label(subset_m) << " with " << factory << " ===" << endl;

                faiss::Index* index = nullptr;
                try {
                    index = faiss::index_factory(static_cast<int>(dim), factory.c_str(), faiss::METRIC_L2);
                } catch (const exception& e) {
                    cerr << "index_factory failed: " << e.what() << endl;
                    continue;
                }

                if (auto* hnsw = dynamic_cast<faiss::IndexHNSW*>(index)) {
                    hnsw->hnsw.efConstruction = ef_construction;
                }

                StopW timer;
                index->train(train_count, train_float.data());
                const double train_s = timer.getElapsedTimeMicro() / 1e6;

                timer.reset();
                size_t added = 0;
                while (added < vec_count) {
                    const size_t take = std::min(add_chunk, vec_count - added);
                    int chunk_dim = 0;
                    double* raw = data_loader::loadBvecsChunk(base_file, added, take, chunk_dim);
                    if (!raw) {
                        cerr << "loadBvecsChunk failed at offset " << added << endl;
                        break;
                    }
                    vector<float> fbuf = to_float_buffer(raw, take * dim);
                    delete[] raw;
                    index->add(take, fbuf.data());
                    added += take;
                }
                const double add_s = timer.getElapsedTimeMicro() / 1e6;

                string index_path = bigann_test_utils::subset_label(subset_m) + "_" + algo + "_" + to_string(p) + ".bin";
                faiss::write_index(index, index_path.c_str());
                const size_t idx_bytes = file_size_bytes(index_path);

                csv << bigann_test_utils::subset_label(subset_m) << ',' << algo << ',' << p << ','
                    << vec_count << ',' << dim << ',' << train_count << ','
                    << fixed << setprecision(6) << train_s << ',' << add_s << ',' << (train_s + add_s) << ','
                    << index_path << ',' << idx_bytes << "\n";

                cout << "train=" << train_s << "s add=" << add_s << "s index=" << index_path << endl;
                delete index;
            }
        }
    }

    cout << "\nSaved CSV: bigann_faiss_hnsw_pq_sq_build_results.csv" << endl;
    return 0;
}
