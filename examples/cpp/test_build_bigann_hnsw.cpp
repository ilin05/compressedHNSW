#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <omp.h>

#include "../../hnswlib/hnswlib.h"
#include "../../examples/data_processor/data_loader.h"
#include "bigann_test_utils.h"

using namespace hnswlib;

class StopW {
    std::chrono::steady_clock::time_point begin_;
public:
    StopW() : begin_(std::chrono::steady_clock::now()) {}
    void reset() { begin_ = std::chrono::steady_clock::now(); }
    double elapsed_us() const {
        return static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - begin_).count());
    }
};

struct BuildResult {
    std::string dataset;
    size_t subset_million;
    size_t vectors;
    size_t dim;
    double build_seconds;
    size_t index_size_bytes;
    std::string index_file;
};

int main(int argc, char** argv) {
    omp_set_num_threads(32);

    std::string base_file = "../bigann/bigann_base.bvecs";
    std::vector<size_t> subsets = bigann_test_utils::parse_subsets_from_cli(argc, argv, {1, 10, 100});

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--base_file" && i + 1 < argc) {
            base_file = argv[++i];
        }
    }

    std::vector<BuildResult> results;

    for (size_t m : subsets) {
        const size_t vec_count = m * 1000000ULL;
        const int M = 16;
        const int ef_construction = 200;

        std::cout << "\n=== Building HNSW for " << bigann_test_utils::subset_label(m) << " ===" << std::endl;

        int dim = 0;
        L2SpaceDouble* l2space = nullptr;
        HierarchicalNSW<double>* index = nullptr;

        try {
            l2space = new L2SpaceDouble(128);
            index = new HierarchicalNSW<double>(l2space, vec_count, M, ef_construction);

            size_t loaded = 0;
            const size_t chunk = 500000;
            StopW timer;

            while (loaded < vec_count) {
                const size_t take = std::min(chunk, vec_count - loaded);
                double* data = data_loader::loadBvecsChunk(base_file, loaded, take, dim);
                if (!data) {
                    throw std::runtime_error("loadBvecsChunk returned null");
                }

                #pragma omp parallel for
                for (long long i = 0; i < static_cast<long long>(take); ++i) {
                    index->addPoint(data + i * dim, loaded + static_cast<size_t>(i));
                }

                delete[] data;
                loaded += take;
                if (loaded % 10000000ULL == 0 || loaded == vec_count) {
                    std::cout << "  inserted " << loaded / 1000000ULL << "M / "
                              << vec_count / 1000000ULL << "M" << std::endl;
                }
            }

            const double build_s = timer.elapsed_us() * 1e-6;
            const std::string index_file = "bigann_" + std::to_string(m) + "M_hnsw.bin";
            index->saveIndex(index_file);
            const size_t index_size = index->indexFileSize();

            results.push_back({bigann_test_utils::subset_label(m), m, vec_count,
                               static_cast<size_t>(dim), build_s, index_size, index_file});

            std::cout << "  build time: " << build_s << " s" << std::endl;
            std::cout << "  index file: " << index_file << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "Failed on subset " << m << "M: " << e.what() << std::endl;
        }

        delete index;
        delete l2space;
    }

    std::ofstream csv("bigann_hnsw_build_results.csv");
    csv << "Dataset,SubsetMillion,Vectors,Dim,BuildTime(s),IndexSize(Bytes),IndexFile\n";
    for (const auto& r : results) {
        csv << r.dataset << ',' << r.subset_million << ',' << r.vectors << ',' << r.dim << ','
            << std::fixed << std::setprecision(6) << r.build_seconds << ','
            << r.index_size_bytes << ',' << r.index_file << "\n";
    }
    std::cout << "\nSaved CSV: bigann_hnsw_build_results.csv" << std::endl;
    return 0;
}
