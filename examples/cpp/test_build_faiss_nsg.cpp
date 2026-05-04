#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <omp.h>
#include <filesystem>

#include <faiss/IndexNSG.h>
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
            "sift-128-euclidean"};

    int R = 32; // NSG parameter (degree)

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--dataset" && i + 1 < argc) {
            datasets.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                datasets.push_back(argv[++i]);
            }
        } else if (arg == "--R" && i + 1 < argc) {
            R = stoi(argv[++i]);
        } else if (arg == "--normal_base_dir" && i + 1 < argc) {
            normal_base_dir = argv[++i];
            if (normal_base_dir.back() != '/' && normal_base_dir.back() != '\\') normal_base_dir += "/";
        }
    }

    ofstream csv("faiss_nsg_build_results.csv");
    csv << "Dataset,Vectors,Dim,BuildTime(s),IndexFile\n";

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
            cerr << "Bigann dataset: only metadata used; building from external source not implemented." << endl;
            continue; // avoid incomplete bigann path for now
        } else {
            const string train_file = normal_base_dir + ds + "_train.fvecs";
            try {
                data = load_fvecs(train_file, n, dim);
            } catch (const exception& e) {
                cerr << "Skip " << ds << " due to error: " << e.what() << endl;
                continue;
            }
        }

        cout << "\n=== Building NSG for " << ds << " (n=" << n << ", dim=" << dim << ") R=" << R << " ===" << endl;

        StopW timer;
        try {
            // Use NSG flat index type
            faiss::IndexNSGFlat index(dim, R, faiss::METRIC_L2);

            // Add vectors (this will build the index internals)
            index.add(static_cast<faiss::idx_t>(n), data);

            double build_s = timer.getElapsedTimeMicro() / 1e6;

            string index_path = ds + "_NSG_R" + to_string(R) + ".bin";
            faiss::write_index(&index, index_path.c_str());

            cout << "Built NSG: " << index_path << " (build time=" << build_s << "s)" << endl;
            csv << ds << ',' << n << ',' << dim << ',' << fixed << setprecision(6) << build_s << ',' << index_path << "\n";
        } catch (const exception& e) {
            cerr << "NSG build failed for " << ds << ": " << e.what() << endl;
        }

        if (data) delete[] data;
    }

    cout << "\nSaved CSV: faiss_nsg_build_results.csv" << endl;
    return 0;
}
