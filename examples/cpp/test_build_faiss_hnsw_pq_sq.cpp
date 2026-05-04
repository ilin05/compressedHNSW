#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <omp.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/IndexPQ.h>
#include <faiss/index_factory.h>
#include <faiss/index_io.h>

using namespace std;

class StopW {
    std::chrono::steady_clock::time_point time_begin;
 public:
    StopW() {
        time_begin = std::chrono::steady_clock::now();
    }
    float getElapsedTimeMicro() {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
    }
    void reset() {
        time_begin = std::chrono::steady_clock::now();
    }
};

float* load_fvecs(const std::string& filename, size_t& num_vectors, size_t& dim) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot open file " + filename);
    }
    int32_t d;
    input.read((char*)&d, 4);
    dim = d;
    input.seekg(0, std::ios::end);
    size_t file_size = input.tellg();
    num_vectors = file_size / (4 + dim * 4);
    float* data = new float[num_vectors * dim];
    input.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num_vectors; ++i) {
        input.read((char*)&d, 4);
        input.read((char*)(data + i * dim), dim * 4);
    }
    return data;
}

int main(int argc, char** argv) {
    omp_set_num_threads(32);
    std::string base_dir = "../datasets/hdf5files/";
    vector<string> datasets = {
        "fashion-mnist-784-euclidean_train.fvecs",
        "gist-960-euclidean_train.fvecs",
        "mnist-784-euclidean_train.fvecs",
        "sift-128-euclidean_train.fvecs"
    };

    vector<string> algos;
    vector<int> pq_ms = {8};
    vector<int> sq_nbits = {8};

    for(int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--dataset" && i + 1 < argc) {
            datasets.clear();
            while(i + 1 < argc && argv[i + 1][0] != '-') {
                datasets.push_back(argv[++i]);
            }
        } else if (arg == "--algorithm" && i + 1 < argc) {
            algos.clear();
            while(i + 1 < argc && argv[i + 1][0] != '-') {
                string a = argv[++i];
                if (a == "HNSWPQ" || a == "HNSWSQ") algos.push_back(a);
            }
        } else if (arg == "--pq_m" && i + 1 < argc) {
            pq_ms.clear();
            while(i + 1 < argc && argv[i + 1][0] != '-') {
                pq_ms.push_back(std::stoi(argv[++i]));
            }
        } else if (arg == "--sq_nbits" && i + 1 < argc) {
            sq_nbits.clear();
            while(i + 1 < argc && argv[i + 1][0] != '-') {
                int bit = std::stoi(argv[++i]);
                if (bit == 4 || bit == 8 || bit == 16) sq_nbits.push_back(bit);
            }
        }
    }

    if (algos.empty()) {
        cerr << "No valid algorithms specified! Please specify --algorithm HNSWPQ or HNSWSQ." << endl;
        return -1;
    }

    // 严谨对齐 hnswlib 构建参数
    int M_hnsw = 16;
    int efConstruction = 200;

    for (const auto& ds : datasets) {
        std::string filepath = base_dir + ds;
        size_t n, dim;
        float* data = nullptr;
        try {
            data = load_fvecs(filepath, n, dim);
        } catch (...) { continue; }

        for (const auto& algo : algos) {
            std::vector<int> ns = (algo == "HNSWPQ") ? pq_ms : sq_nbits;
            for (int n_val : ns) {
                cout << "Building " << ds << " with " << algo << " (n=" << n_val << ")" << endl;
                faiss::Index* index = nullptr;
                std::string current_algo_name = algo + "_" + std::to_string(n_val);

                if (algo == "HNSWSQ") {
                    std::string sq_suffix = "SQ" + std::to_string(n_val);
                    if (n_val == 16) sq_suffix = "SQfp16"; // FAISS natively uses SQfp16 instead of SQ16
                    std::string factory_string = "HNSW" + std::to_string(M_hnsw) + "," + sq_suffix;
                    index = faiss::index_factory(dim, factory_string.c_str(), faiss::METRIC_L2);
                } else if (algo == "HNSWPQ") {
                    int M_pq = static_cast<int>(dim) / n_val;
                    if (M_pq <= 0) {
                        cerr << "Skip invalid HNSWPQ setting: p=" << n_val << ", dim=" << dim << endl;
                        continue;
                    }
                    std::string factory_string = "HNSW" + std::to_string(M_hnsw) + ",PQ" +
                            std::to_string(M_pq);
                    index = faiss::index_factory(dim, factory_string.c_str(), faiss::METRIC_L2);
                }

                // 对齐 efConstruction
                faiss::IndexHNSW* hnsw_idx = dynamic_cast<faiss::IndexHNSW*>(index);
                if(hnsw_idx != nullptr){
                    hnsw_idx->hnsw.efConstruction = efConstruction;
                } else {
                    cerr << "Failed to cast to IndexHNSW to set efConstruction!" << endl;
                }

                cout << "Building " << algo << " with HNSW M=" << M_hnsw << " efConstruction=" << efConstruction << " (n=" << n_val << ")" << endl;

                StopW sw;
                index->train(n, data);
                double train_time = sw.getElapsedTimeMicro() / 1e6;

                sw.reset();
                index->add(n, data);
                double build_time = sw.getElapsedTimeMicro() / 1e6;

                 cout << "Train + Build Time for " << current_algo_name << " on " << ds << ": "
                     << (train_time + build_time) << "s\n";

                 std::string index_path = ds + "_" + current_algo_name + ".bin";
                faiss::write_index(index, index_path.c_str());

                delete index;
            }
        }
        delete[] data;
    }
    return 0;
}
