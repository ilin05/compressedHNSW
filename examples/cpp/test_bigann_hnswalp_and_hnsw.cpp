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
const size_t ef_construction = 300;
const std::string hnsw_path = "bigann_hnsw.bin";
const std::string hnswalp_path = "bigann_hnswalp.bin";

size_t TOTAL_LOAD_COUNT = 10000000; // Default 10M
size_t CHUNK_SIZE = 5000000;        // Default 5M

void build_hnsw(const std::string& data_path, size_t total_vectors, int dim) {
    size_t max_elements = total_vectors;
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "Building Standard HNSW Index (Incremental)..." << std::endl;
    hnswlib::L2SpaceDouble space(dim);
    hnswlib::HierarchicalNSW<double> alg_hnsw(&space, max_elements, M, ef_construction);

    auto start = std::chrono::high_resolution_clock::now();
    
    size_t loaded_count = 0;
    while (loaded_count < total_vectors) {
        size_t this_batch = std::min(CHUNK_SIZE, total_vectors - loaded_count);
        std::cout << "Loading chunk: offset " << loaded_count << ", size " << this_batch << std::endl;
        
        int loaded_dim;
        double* chunk_data = data_loader::loadBvecsChunk(data_path, loaded_count, this_batch, loaded_dim);
        
        if (loaded_dim != dim) {
            std::cerr << "Error: Dimension mismatch in chunk!" << std::endl;
            delete[] chunk_data;
            break;
        }

        std::cout << "Adding chunk to HNSW..." << std::endl;
        // #pragma omp parallel for // addPoint is thread-safe internally usually, but parallel loop depends on implementation.
        // HNSW addPoint is thread-safe only if we manage visited lists carefully, usually parallel batch add is better if supported or just separate additions.
        // Standard HNSW addPoint is thread safe.
        for (long i = 0; i < static_cast<long>(this_batch); ++i) {
             alg_hnsw.addPoint(chunk_data + i * dim, loaded_count + i);
        }
        
        delete[] chunk_data; // Release memory for this chunk
        loaded_count += this_batch;
        std::cout << "HNSW Total Added " << loaded_count << " points" << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "HNSW build time: " << std::chrono::duration<double>(end - start).count() << " s" << std::endl;
    
    std::cout << "Saving HNSW index to " << hnsw_path << "..." << std::endl;
    alg_hnsw.saveIndex(hnsw_path);
}

void search_hnsw(const std::string& data_path, int rows, int dim) {
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "Searching Standard HNSW Index..." << std::endl;
    hnswlib::L2SpaceDouble space(dim);
    
    std::cout << "Loading HNSW index from " << hnsw_path << "..." << std::endl;
    hnswlib::HierarchicalNSW<double> alg_hnsw(&space, hnsw_path, false, false, rows);
    std::cout << "Index loaded." << std::endl;

    // Load a small subset for query (e.g. first 10k or first chunk)
    // For fair comparison, we use the first 10,000 vectors as queries if rows allows
    int query_limit = std::min(rows, 10000);
    int d_dummy;
    double* query_data = data_loader::loadBvecsChunk(data_path, 0, query_limit, d_dummy);
    
    float correct = 0;
    int query_count = 0;
    
    std::cout << "Querying " << query_limit << " vectors..." << std::endl;
    auto query_start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < query_limit; i++) {
        std::priority_queue<std::pair<double, hnswlib::labeltype>> result = alg_hnsw.searchKnn(query_data + i * dim, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i) correct++;
        query_count++;
    }
    
    auto query_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> query_duration = query_end - query_start;
    float avg_query_time = query_duration.count() / query_count; 
    float recall = correct / query_count;
    
    std::cout << "HNSW Recall@1 (First " << query_limit << "): " << recall << "\n";
    std::cout << "HNSW Average Query Time: " << avg_query_time << " ms" << std::endl;
    
    delete[] query_data;
}

void build_hnswalp(const std::string& data_path, size_t total_vectors, int dim) {
    size_t max_elements = total_vectors;
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "Building HNSWALP_LEANN Index (Incremental)..." << std::endl;
    hnswlib::L2SpaceDouble space(dim);
    hnswlib::HierarchicalNSWALPSIMPLIFIED<double> alg_alp(&space, max_elements, M, ef_construction);

    auto start = std::chrono::high_resolution_clock::now();
    
    size_t loaded_count = 0;
    while (loaded_count < total_vectors) {
        size_t this_batch = std::min(CHUNK_SIZE, total_vectors - loaded_count);
        std::cout << "Loading chunk: offset " << loaded_count << ", size " << this_batch << std::endl;
        
        int loaded_dim;
        double* chunk_data = data_loader::loadBvecsChunk(data_path, loaded_count, this_batch, loaded_dim);

        std::cout << "Adding chunk to HNSWALP..." << std::endl;
        for (long i = 0; i < static_cast<long>(this_batch); ++i) {
            alg_alp.addPoint(chunk_data + i * dim, loaded_count + i);
        }
        
        delete[] chunk_data; // Release memory
        loaded_count += this_batch;
        std::cout << "HNSWALP Added " << loaded_count << " points" << std::endl;
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

void search_hnswalp(const std::string& data_path, int rows, int dim) {
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "Searching HNSWALP_LEANN Index..." << std::endl;
    hnswlib::L2SpaceDouble space(dim);
    
    std::cout << "Loading HNSWALP index from " << hnswalp_path << "..." << std::endl;
    hnswlib::HierarchicalNSWALPSIMPLIFIED<double> alg_alp(&space, hnswalp_path);
    std::cout << "Index loaded." << std::endl;

    int query_limit = std::min(rows, 10000);
    int d_dummy;
    double* query_data = data_loader::loadBvecsChunk(data_path, 0, query_limit, d_dummy);

    float correct = 0;
    int query_count = 0;
    
    std::cout << "Querying " << query_limit << " vectors..." << std::endl;
    auto query_start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < query_limit; i++) {
        std::priority_queue<std::pair<double, hnswlib::labeltype>> result = alg_alp.searchKnn(query_data + i * dim, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i) correct++;
        query_count++;
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
    
    delete[] query_data;
}

int main(int argc, char** argv) {
    std::string data_path = "../datasets/bigann_learn.bvecs"; 

    if (argc > 1) TOTAL_LOAD_COUNT = std::stoull(argv[1]);
    if (argc > 2) CHUNK_SIZE = std::stoull(argv[2]);

    std::cout << "Settings:" << std::endl;
    std::cout << "Total Vectors: " << TOTAL_LOAD_COUNT << std::endl;
    std::cout << "Chunk Size: " << CHUNK_SIZE << std::endl;

    // Detect Dim from first chunk or header
    int dim;
    {
       std::ifstream in(data_path, std::ios::binary);
       if(!in) {
           std::cerr << "Cannot open " << data_path << std::endl;
           return 1;
       }
       in.read((char*)&dim, 4);
    }
    std::cout << "Detected Dimension: " << dim << std::endl;

    // Uncomment sections to run specific stages
    build_hnsw(data_path, TOTAL_LOAD_COUNT, dim);
    search_hnsw(data_path, TOTAL_LOAD_COUNT, dim);
    
    build_hnswalp(data_path, TOTAL_LOAD_COUNT, dim);
    search_hnswalp(data_path, TOTAL_LOAD_COUNT, dim);

    return 0;
}
