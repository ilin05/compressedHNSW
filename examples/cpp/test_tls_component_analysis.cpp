#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <unordered_set>
#include <omp.h>
#include <iomanip>

#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/hnswalg_simplified_alp_PQ.h"

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

// Load fvecs file
float* load_fvecs_as_float(const std::string& filename, size_t& num_vectors, size_t& dim) {
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
    
    float* data = new float[num_vectors * dim];
    input.seekg(0, std::ios::beg);
    
    for (size_t i = 0; i < num_vectors; ++i) {
        input.read((char*)&d, 4);
        input.read((char*)(data + i * dim), dim * 4);
    }
    
    return data;
}

// Load ivecs file (Ground Truth)
unsigned int* load_ivecs(const std::string& filename, size_t& num_vectors, size_t& dim) {
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
    
    unsigned int* data = new unsigned int[num_vectors * dim];
    input.seekg(0, std::ios::beg);
    
    for (size_t i = 0; i < num_vectors; ++i) {
        input.read((char*)&d, 4);
        input.read((char*)(data + i * dim), dim * 4);
    }
    
    return data;
}

struct TLSResult {
    std::string dataset_name;
    bool use_tls;
    double tls_ratio;
    size_t k;
    size_t ef;
    double recall;
    double latency_ms_per_query;
    long decoding_calls;
    long distance_computations;
    double avg_decoding_per_query;
};

void write_results_to_csv(const std::string& csv_file_path, const std::vector<TLSResult>& results) {
    std::ofstream file(csv_file_path);
    if (file.is_open()) {
        file << "Dataset,UseTLS,TLSRatio,K,ef,Recall,Latency(ms/query),"
             << "DecodingCalls,DistanceComputations,AvgDecodingPerQuery\n";
        for (const auto& res : results) {
            file << res.dataset_name << ","
                 << (res.use_tls ? 1 : 0) << ","
                 << std::fixed << std::setprecision(2) << res.tls_ratio << ","
                 << res.k << ","
                 << res.ef << ","
                 << std::setprecision(6) << res.recall << ","
                 << std::setprecision(4) << res.latency_ms_per_query << ","
                 << res.decoding_calls << ","
                 << res.distance_computations << ","
                 << std::setprecision(4) << res.avg_decoding_per_query << "\n";
        }
        file.close();
        std::cout << "Results written to " << csv_file_path << std::endl;
    } else {
        std::cerr << "Failed to open CSV file for writing." << std::endl;
    }
}

TLSResult test_tls_configuration(
    const std::string& dataset_name,
    const std::string& base_dir,
    size_t k,
    size_t ef,
    bool use_tls,
    double tls_ratio) {
    
    TLSResult result;
    result.dataset_name = dataset_name;
    result.use_tls = use_tls;
    result.tls_ratio = tls_ratio;
    result.k = k;
    result.ef = ef;
    result.recall = 0.0;
    result.latency_ms_per_query = 0.0;
    result.decoding_calls = 0;
    result.distance_computations = 0;
    result.avg_decoding_per_query = 0.0;
    
    // Construct file paths
    std::string query_file = base_dir + dataset_name + "_test.fvecs";
    std::string gt_file = base_dir + dataset_name + "_neighbors.ivecs";
    std::string index_path = dataset_name + "_train.fvecs_hnswalp_simplified_pq.bin";
    
    // Load queries
    size_t qsize = 0, qdim = 0;
    float* massQ = load_fvecs_as_float(query_file, qsize, qdim);
    if (!massQ) return result;
    
    // Load ground truth
    size_t gt_num = 0, gt_dim = 0;
    unsigned int* massQA = load_ivecs(gt_file, gt_num, gt_dim);
    if (!massQA) {
        delete[] massQ;
        return result;
    }
    
    if (qsize != gt_num) {
        std::cerr << "Query size and Ground Truth size mismatch!" << std::endl;
        delete[] massQ;
        delete[] massQA;
        return result;
    }
    
    // Load index
    L2Space l2space(qdim);
    HierarchicalNSWALPSIMPLIFIEDPQ<float>* appr_alg = nullptr;
    
    try {
        appr_alg = new HierarchicalNSWALPSIMPLIFIEDPQ<float>(&l2space, index_path, false);
    } catch (std::exception& e) {
        std::cerr << "Failed to load index: " << e.what() << std::endl;
        delete[] massQ;
        delete[] massQA;
        return result;
    }
    
    // Configure search parameters
    appr_alg->setEf(ef);
    appr_alg->setProfilingMetrics(true);  // Enable metrics collection
    
    if (use_tls) {
        appr_alg->setUseTLS(true);
        appr_alg->setTLSRatio(tls_ratio);
    } else {
        appr_alg->setUseTLS(false);
    }
    
    // Adjust k
    size_t actual_k = k;
    if (actual_k > gt_dim) actual_k = gt_dim;
    
    // Reset metrics before search
    appr_alg->decoding_call_count = 0;
    appr_alg->metric_distance_computations = 0;
    
    // Perform search and collect metrics
    size_t correct = 0;
    size_t total = qsize * actual_k;
    
    StopW stopw;
    
    for (long i = 0; i < (long)qsize; i++) {
        std::priority_queue<std::pair<float, labeltype>> search_result =
            appr_alg->searchKnn(massQ + qdim * i, actual_k);
        
        std::unordered_set<labeltype> gt_set;
        for (size_t j = 0; j < actual_k; j++) {
            gt_set.insert(massQA[i * gt_dim + j]);
        }
        
        while (!search_result.empty()) {
            if (gt_set.find(search_result.top().second) != gt_set.end()) {
                correct++;
            }
            search_result.pop();
        }
    }
    
    double total_time_us = stopw.getElapsedTimeMicro();
    
    // Collect results
    result.recall = (total > 0) ? (1.0 * correct / total) : 0.0;
    result.latency_ms_per_query = total_time_us / 1000.0 / qsize;
    result.decoding_calls = appr_alg->getDecodingCallCount();
    result.distance_computations = appr_alg->metric_distance_computations;
    result.avg_decoding_per_query = (qsize > 0) ? (1.0 * result.decoding_calls / qsize) : 0.0;
    
    delete[] massQ;
    delete[] massQA;
    delete appr_alg;
    
    return result;
}

int main(int argc, char** argv) {
    omp_set_num_threads(1);

    std::string base_dir = "../datasets/hdf5files/";
    std::vector<std::string> datasets = {"fashion-mnist-784-euclidean", "gist-960-euclidean", "mnist-784-euclidean", "sift-128-euclidean"};
    std::vector<double> tls_ratios = {0.1, 0.2, 0.3};
    std::vector<size_t> efs = {10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 300};  // Multiple ef values for sweep
    
    size_t k = 1;      // Recall@1
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--dataset" && i + 1 < argc) {
            datasets.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                datasets.push_back(argv[++i]);
            }
        } else if (arg == "--tls-ratios" && i + 1 < argc) {
            tls_ratios.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                tls_ratios.push_back(std::stod(argv[++i]));
            }
        } else if (arg == "--efs" && i + 1 < argc) {
            efs.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                efs.push_back(std::stoull(argv[++i]));
            }
        } else if (arg == "--k" && i + 1 < argc) {
            k = std::stoull(argv[++i]);
        }
    }
    
    std::vector<TLSResult> all_results;
    
    // Test configurations: sweep over multiple ef values
    for (const auto& dataset : datasets) {
        std::cout << "\n=====================================================" << std::endl;
        std::cout << "Testing dataset: " << dataset << std::endl;
        std::cout << "=====================================================" << std::endl;
        
        for (size_t ef : efs) {
            std::cout << "\nTesting ef=" << ef << std::endl;
            
            // Test without TLS
            std::cout << "  WITHOUT TLS: ";
            TLSResult result_no_tls = test_tls_configuration(dataset, base_dir, k, ef, false, 0.0);
            all_results.push_back(result_no_tls);
            
            std::cout << "Recall@" << k << "=" << std::fixed << std::setprecision(4) << result_no_tls.recall
                      << " Latency=" << std::setprecision(2) << result_no_tls.latency_ms_per_query << "ms"
                      << " AvgDecode=" << std::setprecision(2) << result_no_tls.avg_decoding_per_query << std::endl;
            
            // Test with TLS at different ratios
            for (double ratio : tls_ratios) {
                std::cout << "  TLS ratio=" << std::fixed << std::setprecision(2) << ratio << ": ";
                TLSResult result_tls = test_tls_configuration(dataset, base_dir, k, ef, true, ratio);
                all_results.push_back(result_tls);
                
                std::cout << "Recall@" << k << "=" << std::setprecision(4) << result_tls.recall
                          << " Latency=" << std::setprecision(2) << result_tls.latency_ms_per_query << "ms"
                          << " AvgDecode=" << std::setprecision(2) << result_tls.avg_decoding_per_query << std::endl;
            }
        }
    }
    
    std::string csv_file_path = "hnswalp_tls_component_analysis_results.csv";
    write_results_to_csv(csv_file_path, all_results);
    
    std::cout << "\nAll component analysis tests completed." << std::endl;
    return 0;
}
