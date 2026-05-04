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

int main(int argc, char** argv) {
    omp_set_num_threads(32);

    std::string base_file = "../bigann/bigann_base.bvecs";
    std::vector<size_t> subsets = bigann_test_utils::parse_subsets_from_cli(argc, argv, {1, 10, 100});

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--base_file" && i + 1 < argc) base_file = argv[++i];
    }

    std::ofstream csv("bigann_hnswdexor_pq_build_results.csv");
    csv << "Dataset,SubsetMillion,Vectors,Dim,BuildTime(s),CompressTime(s),DataCompressionRatio,IndexCompressionRatio,IndexFile\n";

    for (size_t m : subsets) {
        const size_t vec_count = m * 1000000ULL;
        const int M = 16;
        const int ef_construction = 200;
        const size_t cache_max_size = vec_count / 100;

        std::cout << "\n=== Building HNSWDEXOR_PQ for " << bigann_test_utils::subset_label(m) << " ===" << std::endl;

        int dim = 0;
        L2SpaceDouble l2space(128);
        HierarchicalNSWCABLEANNPQ<double> index(&l2space, vec_count, "DeXOR", M, ef_construction, true, cache_max_size);

        size_t loaded = 0;
        const size_t chunk = 500000;
        StopW timer;

        while (loaded < vec_count) {
            const size_t take = std::min(chunk, vec_count - loaded);
            double* data = data_loader::loadBvecsChunk(base_file, loaded, take, dim);
            #pragma omp parallel for
            for (long long i = 0; i < static_cast<long long>(take); ++i) {
                index.addPoint(data + i * dim, loaded + static_cast<size_t>(i));
            }
            delete[] data;
            loaded += take;
        }

        const double build_s = timer.elapsed_us() * 1e-6;
        timer.reset();
        index.compress_dataset();
        const double comp_s = timer.elapsed_us() * 1e-6;

        const size_t original_data = vec_count * static_cast<size_t>(dim) * sizeof(double);
        const size_t compressed_data = index.getCompressedDataSize();
        const size_t original_index = index.getIndexSize();
        const size_t compressed_index = index.getCompressedIndexSize();

        const double data_ratio = static_cast<double>(original_data) / static_cast<double>(compressed_data);
        const double index_ratio = static_cast<double>(original_index) / static_cast<double>(compressed_index);

        const std::string index_file = "bigann_" + std::to_string(m) + "M_hnswdexor_pq.bin";
        index.saveIndex(index_file);

        csv << bigann_test_utils::subset_label(m) << ',' << m << ',' << vec_count << ',' << dim << ','
            << std::fixed << std::setprecision(6) << build_s << ',' << comp_s << ','
            << data_ratio << ',' << index_ratio << ',' << index_file << "\n";
    }

    std::cout << "\nSaved CSV: bigann_hnswdexor_pq_build_results.csv" << std::endl;
    return 0;
}
