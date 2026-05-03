#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <omp.h>

#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>

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

struct TestResult {
    std::string dataset_name;
    double build_time;
    size_t index_size_bytes;
};

static void write_results_to_csv(const string& csv_file_path, const vector<TestResult>& results) {
    ofstream file(csv_file_path);
    file << "Dataset,Build time (s),Index size (bytes)\n";
    for (const auto& res : results) {
        file << res.dataset_name << ',' << fixed << setprecision(6) << res.build_time << ','
             << res.index_size_bytes << '\n';
    }
}

static TestResult test_build_index(const string& dataset_name, const string& base_dir) {
    TestResult res{dataset_name, 0.0, 0};

    const string filepath = base_dir + dataset_name;
    cout << "\n==============================================" << endl;
    cout << "Testing Faiss HNSW construction: " << dataset_name << endl;

    size_t num_vectors = 0, dim = 0;
    float* data = nullptr;
    try {
        data = load_fvecs(filepath, num_vectors, dim);
    } catch (const exception& e) {
        cerr << "Skip " << dataset_name << " due to error: " << e.what() << endl;
        return res;
    }

    cout << "Loaded " << num_vectors << " vectors of dimension " << dim << endl;

    constexpr int M = 16;
    constexpr int efConstruction = 200;

    faiss::IndexHNSWFlat index(static_cast<int>(dim), M, faiss::METRIC_L2);
    index.hnsw.efConstruction = efConstruction;

    StopW timer;
    index.add(static_cast<faiss::idx_t>(num_vectors), data);
    res.build_time = timer.getElapsedTimeMicro() / 1e6;
    // res.index_size_bytes = static_cast<size_t>(index.sa_code_size()) * num_vectors;
    res.index_size_bytes = 0; // Faiss does not provide a direct way to get the index size in bytes, so we set it to 0 for now.

    const string index_path = dataset_name + "_faiss_hnsw_M16_efConstruction200.bin";
    cout << "Saving index to " << index_path << "..." << endl;
    faiss::write_index(&index, index_path.c_str());

    cout << "Build time: " << res.build_time << " seconds" << endl;
    cout << "Index saved." << endl;

    delete[] data;
    return res;
}

int main(int argc, char** argv) {
    omp_set_num_threads(32);

    string base_dir = "../datasets/hdf5files/";
    vector<string> datasets = {
            "fashion-mnist-784-euclidean_train.fvecs",
            "gist-960-euclidean_train.fvecs",
            "mnist-784-euclidean_train.fvecs",
            "sift-128-euclidean_train.fvecs"};

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--dataset" && i + 1 < argc) {
            datasets.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                datasets.push_back(argv[++i]);
            }
        } else if (arg == "--base_dir" && i + 1 < argc) {
            base_dir = argv[++i];
            if (base_dir.back() != '/' && base_dir.back() != '\\') {
                base_dir += "/";
            }
        }
    }

    vector<TestResult> all_results;
    for (const auto& ds : datasets) {
        TestResult res = test_build_index(ds, base_dir);
        if (res.build_time > 0) {
            all_results.push_back(res);
        }
    }

    write_results_to_csv("faiss_hnsw_build_results.csv", all_results);
    cout << "\nAll build tests (Faiss HNSW) completed." << endl;
    return 0;
}