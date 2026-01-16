#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/hnswalg_ALP_leann.h"
#include "../data_processor/data_loader.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

// Global parameters
const size_t M = 32;
const size_t ef_construction = 200;
const std::string hnsw_path = "bigann_hnsw.bin";
const std::string hnswalp_path = "bigann_hnswalp.bin";

void build_hnsw(double* data, int rows, int dim) {
    size_t max_elements = rows;
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "Building Standard HNSW Index..." << std::endl;
    hnswlib::L2SpaceDouble space(dim);
    hnswlib::HierarchicalNSW<double> alg_hnsw(&space, max_elements, M, ef_construction);

    auto start = std::chrono::high_resolution_clock::now();
    
    // #pragma omp parallel for
    for (long i = 0; i < static_cast<long>(max_elements); ++i) {
         alg_hnsw.addPoint(data + i * dim, i);
         if (i % 100000 == 0 && i > 0) {
            //  #pragma omp critical 
             {
                std::cout << "HNSW Added " << i << " points" << std::endl;
             }
         }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "HNSW build time: " << std::chrono::duration<double>(end - start).count() << " s" << std::endl;
    
    std::cout << "Saving HNSW index to " << hnsw_path << "..." << std::endl;
    alg_hnsw.saveIndex(hnsw_path);
}

void search_hnsw(double* data, int rows, int dim) {
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "Searching Standard HNSW Index..." << std::endl;
    hnswlib::L2SpaceDouble space(dim);
    
    std::cout << "Loading HNSW index from " << hnsw_path << "..." << std::endl;
    hnswlib::HierarchicalNSW<double> alg_hnsw(&space, hnsw_path, false, false, rows);
    std::cout << "Index loaded." << std::endl;

    float correct = 0;
    // Limit query count for large datasets
    int step = std::max(1, rows / 50000); 
    int query_count = 0;
    
    std::cout << "Querying..." << std::endl;
    auto query_start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < rows; i += step) {
        std::priority_queue<std::pair<double, hnswlib::labeltype>> result = alg_hnsw.searchKnn(data + i * dim, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i) correct++;
        query_count++;
        if (query_count % 10000 == 0) std::cout << "Queried " << query_count << " vectors..." << std::endl;
    }
    
    auto query_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> query_duration = query_end - query_start;
    float avg_query_time = query_duration.count() / query_count; 
    float recall = correct / query_count;
    
    std::cout << "HNSW Recall@1: " << recall << "\n";
    std::cout << "HNSW Average Query Time: " << avg_query_time << " ms" << std::endl;
}

void build_hnswalp(double* data, int rows, int dim) {
    size_t max_elements = rows;
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "Building HNSWALP_LEANN Index..." << std::endl;
    hnswlib::L2SpaceDouble space(dim);
    hnswlib::HierarchicalNSWALPLEANN<double> alg_alp(&space, max_elements, M, ef_construction);

    auto start = std::chrono::high_resolution_clock::now();
    
    // #pragma omp parallel for
    for (long i = 0; i < static_cast<long>(max_elements); ++i) {
        alg_alp.addPoint(data + i * dim, i);
        if (i % 100000 == 0 && i > 0) {
            //  #pragma omp critical
             {
                std::cout << "HNSWALP Added " << i << " points" << std::endl;
             }
         }
    }
    
    std::cout << "Compressing dataset..." << std::endl;
    auto compress_start = std::chrono::high_resolution_clock::now();
    alg_alp.compress_dataset();
    auto compress_end = std::chrono::high_resolution_clock::now();
    std::cout << "Compression time: " << std::chrono::duration<double>(compress_end - compress_start).count() << " s" << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Total HNSWALP build time (incl. compression): " << std::chrono::duration<double>(end - start).count() << " s" << std::endl;

    std::cout << "Saving HNSWALP index to " << hnswalp_path << "..." << std::endl;
    alg_alp.saveIndex(hnswalp_path);
    
    size_t compressed_size = alg_alp.getCompressedDataSize();
    size_t original_size = (size_t)max_elements * dim * sizeof(double);
    std::cout << "Original Raw Data Size: " << original_size / (1024*1024) << " MB" << std::endl;
    std::cout << "Compressed Data Size: " << compressed_size / (1024*1024) << " MB" << std::endl;
    std::cout << "Compression Ratio: " << (double)original_size / compressed_size << std::endl;
}

void search_hnswalp(double* data, int rows, int dim) {
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "Searching HNSWALP_LEANN Index..." << std::endl;
    hnswlib::L2SpaceDouble space(dim);
    
    std::cout << "Loading HNSWALP index from " << hnswalp_path << "..." << std::endl;
    hnswlib::HierarchicalNSWALPLEANN<double> alg_alp(&space, hnswalp_path);
    std::cout << "Index loaded." << std::endl;

    float correct = 0;
    int step = std::max(1, rows / 50000); 
    int query_count = 0;
    
    std::cout << "Querying..." << std::endl;
    auto query_start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < rows; i += step) {
        std::priority_queue<std::pair<double, hnswlib::labeltype>> result = alg_alp.searchKnn(data + i * dim, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i) correct++;
        query_count++;
        if (query_count % 10000 == 0) std::cout << "Queried " << query_count << " vectors..." << std::endl;
    }
    
    auto query_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> query_duration = query_end - query_start;
    float avg_query_time = query_duration.count() / query_count; 
    float recall = correct / query_count;
    
    std::cout << "HNSWALP Recall@1: " << recall << "\n";
    std::cout << "HNSWALP Average Query Time: " << avg_query_time << " ms" << std::endl;
    
    float decoding_time_per_query = static_cast<float>(alg_alp.getTotalTimeDecoding()) / query_count / 1e3; // ms
    float decoding_call_count_per_query = static_cast<float>(alg_alp.getDecodingCallCount()) / query_count;
    std::cout << "HNSWALP Avg Decoding Time: " << decoding_time_per_query << " ms" << std::endl;
    std::cout << "HNSWALP Avg Decoding Calls: " << decoding_call_count_per_query << std::endl;
}

int main() {
    std::string data_path = "../datasets/bigann_learn.bvecs"; 

    int rows, dim;
    double* data = nullptr;
    try {
        data = data_loader::loadBvecs(data_path, rows, dim);
    } catch (const std::exception& e) {
        std::cerr << "Error loading data: " << e.what() << std::endl;
        return 1;
    }

    if (!data) return 1;
    std::cout << "Data loaded. Rows: " << rows << ", Dim: " << dim << std::endl;

    // Uncomment sections to run specific stages
    build_hnsw(data, rows, dim);
    search_hnsw(data, rows, dim);
    
    build_hnswalp(data, rows, dim);
    search_hnswalp(data, rows, dim);

    delete[] data;
    return 0;
}
