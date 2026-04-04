#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <omp.h>
#include <iomanip>

#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/compressed_hnsw_framework.h"

using namespace std;
using namespace hnswlib;

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

double* load_fvecs_as_double(const std::string& filename, size_t& num_vectors, size_t& dim) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        std::cerr << "Cannot open " << filename << std::endl;
        return nullptr;
    }
    
    int32_t d;
    input.read((char*)&d, 4);
    dim = d;
    
    input.seekg(0, std::ios::end);
    size_t file_size = input.tellg();
    num_vectors = file_size / (4 + dim * 4);
    
    double* data = new double[num_vectors * dim];
    float* tmp = new float[dim];
    input.seekg(0, std::ios::beg);
    
    for (size_t i = 0; i < num_vectors; ++i) {
        input.read((char*)&d, 4);
        input.read((char*)tmp, dim * 4);
        for(size_t j = 0; j < dim; ++j) {
            data[i * dim + j] = static_cast<double>(tmp[j]);
        }
    }
    
    delete[] tmp;
    return data;
}

struct TestResult {
    std::string dataset_name;
    std::string algorithm_name;
    double build_time;
    double compress_time;
    double data_compression_ratio;
    double index_compression_ratio;
};

void write_results_to_csv(const std::string& csv_file_path, const std::vector<TestResult>& results) {
    std::ofstream file(csv_file_path);
    if (file.is_open()) {
        file << "Dataset,Algorithm,BuildTime(s),CompressTime(s),DataCompressionRatio,IndexCompressionRatio\n";
        for (const auto& res : results) {
            file << res.dataset_name << ","
                 << res.algorithm_name << ","
                 << std::fixed << std::setprecision(4) << res.build_time << ","
                 << std::fixed << std::setprecision(4) << res.compress_time << ","
                 << std::fixed << std::setprecision(4) << res.data_compression_ratio << ","
                 << std::fixed << std::setprecision(4) << res.index_compression_ratio << "\n";
        }
        file.close();
    } else {
        std::cerr << "Unable to open file " << csv_file_path << " for writing." << std::endl;
    }
}


template<typename dist_t, class Codec>
TestResult run_build_test_for_codec(
    SpaceInterface<dist_t>* l2space,
    size_t num_vectors,
    const std::string& algo_name,
    int M,
    int efConstruction,
    size_t cache_max_size,
    const std::string& dataset_name,
    double* data,
    size_t dim) 
{
    TestResult res;
    res.dataset_name = dataset_name;
    res.algo_name = algo_name;
    res.build_time = 0.0;
    res.compress_time = 0.0;
    res.data_compression_ratio = 0.0;
    res.index_compression_ratio = 0.0;

    cout << "Allocating memory for index..." << endl;
    HierarchicalNSWCABFRAMEWORK<dist_t, Codec>* appr_alg = 
        new HierarchicalNSWCABFRAMEWORK<dist_t, Codec>(l2space, num_vectors, algo_name, M, efConstruction, true, cache_max_size);

    StopW stopw;
    
    cout << "Inserting elements into HNSW... " << endl;
    // 多线程并发插入
    #pragma omp parallel for
    for (long i = 0; i < (long)num_vectors; ++i) {
        appr_alg->addPoint(data + i * dim, i);
    }
    
    res.build_time = 1e-6 * stopw.getElapsedTimeMicro();
    cout << "Graph Construction Time: " << res.build_time << " seconds" << endl;
    
    stopw.reset();
    cout << "Compressing dataset using " << algo_name << "..." << endl;
    
    appr_alg->compress_dataset();
    
    res.compress_time = 1e-6 * stopw.getElapsedTimeMicro();
    cout << "Compression Time: " << res.compress_time << " seconds" << endl;
    
    size_t original_data_size = num_vectors * dim * sizeof(double);
    size_t compressed_data_size = appr_alg->getCompressedDataSize();
    cout << "Original data size: " << original_data_size << " bytes." << endl;
    cout << "Total compressed data size: " << compressed_data_size << " bytes." << endl;
    res.data_compression_ratio = static_cast<double>(original_data_size) / static_cast<double>(compressed_data_size);
    cout << "Data compression ratio: " << res.data_compression_ratio << endl;

    size_t original_index_size = appr_alg->getIndexSize();
    size_t compressed_index_size = appr_alg->getCompressedIndexSize();
    cout << "Original index size: " << original_index_size << " bytes." << endl;
    cout << "Total compressed index size: " << compressed_index_size << " bytes." << endl;
    res.index_compression_ratio = static_cast<double>(original_index_size) / static_cast<double>(compressed_index_size);
    cout << "Index compression ratio: " << res.index_compression_ratio << endl;
    
    std::string index_path = dataset_name + "_compressed_hnsw_framework.bin";
    appr_alg->saveIndex(index_path);

    delete appr_alg;
    return res;
}


TestResult test_build_index(const std::string& dataset_name, const std::string& base_dir, const std::string& algo_name) {
    TestResult res;
    res.dataset_name = dataset_name;
    res.algorithm_name = algo_name;
    res.build_time = 0.0;
    res.compress_time = 0.0;
    res.data_compression_ratio = 0.0;
    res.index_compression_ratio = 0.0;

    std::string filepath = base_dir + dataset_name;
    cout << "\n==============================================" << endl;
    cout << "Testing dataset construction: " << dataset_name << " using " << algo_name << endl;
    
    size_t num_vectors = 0, dim = 0;
    double* data = load_fvecs_as_double(filepath, num_vectors, dim);
    if (!data) return res;
    
    cout << "Loaded " << num_vectors << " vectors of dimension " << dim << " (converted to double)" << endl;
    
    size_t cache_max_size = num_vectors / 100;

    L2SpaceDouble l2space(dim);
    int M = 16;
    int efConstruction = 200;
    
    if (algo_name == "DeXOR") {
        res = run_build_test_for_codec<double, codecs::DeXORCodec>(&l2space, num_vectors, algo_name, M, efConstruction, cache_max_size, dataset_name, data, dim);
    } else if (algo_name == "Gorilla") {
        res = run_build_test_for_codec<double, codecs::GorillaCodec>(&l2space, num_vectors, algo_name, M, efConstruction, cache_max_size, dataset_name, data, dim);
    } else if (algo_name == "Elf") {
        res = run_build_test_for_codec<double, codecs::ElfCodec>(&l2space, num_vectors, algo_name, M, efConstruction, cache_max_size, dataset_name, data, dim);
    } else if (algo_name == "Camel") {
        res = run_build_test_for_codec<double, codecs::CamelCodec>(&l2space, num_vectors, algo_name, M, efConstruction, cache_max_size, dataset_name, data, dim);
    } else {
        res = run_build_test_for_codec<double, codecs::DeXORCodec>(&l2space, num_vectors, algo_name, M, efConstruction, cache_max_size, dataset_name, data, dim);
    }

    return res;
}

int main(int argc, char** argv) {
    omp_set_num_threads(32);

    std::string base_dir = "../datasets/hdf5files/";
    
    if(argc > 1) {
        base_dir = argv[1];
        if(base_dir.back() != '/' && base_dir.back() != '\\') {
            base_dir += '/';
        }
    }

    vector<string> datasets = {
        // "fashion-mnist-784-euclidean_train.fvecs",
        // "gist-960-euclidean_train.fvecs",
        // "mnist-784-euclidean_train.fvecs",
        "sift-128-euclidean_train.fvecs"
    };

    vector<string> algorithms = {
        "DeXOR", "Gorilla", "Elf", "Camel"
    };
    
    std::vector<TestResult> all_results;

    for (const auto& ds : datasets) {
        for (const auto& algo : algorithms) {
            all_results.push_back(test_build_index(ds, base_dir, algo));
        }
    }
    
    write_results_to_csv("compressed_hnsw_build_results.csv", all_results);
    
    cout << "\nAll build tests completed." << endl;
    return 0;
}