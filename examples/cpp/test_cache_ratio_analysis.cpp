#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <unordered_set>
#include <omp.h>
#include <iomanip>
#include <cmath>

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

// 读取 fvecs 文件 (原数据为 float)，直接存储为 float 数组
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

// 读取 ivecs 文件，作为 Ground Truth
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

struct CacheAnalysisResult {
    std::string dataset_name;
    float cache_ratio;                    // Cache ratio (0.5%, 1%, 5%, 10%)
    size_t cache_size;                    // Number of nodes to cache
    double compress_time;                 // Compression time (seconds)
    double qps;                           // Queries per second (averaged across rounds)
    double qps_std_dev;                   // Standard deviation of QPS
    double recall;                        // Recall@1 (averaged across rounds)
    long decoding_calls;                  // Number of decoding calls during search (averaged)
    long total_distance_computations;     // Total distance computations (averaged)
    long get_original_data_calls;         // Number of getOriginalData calls (averaged)
    double avg_decode_per_query;          // Average decoding per query
    double cache_efficiency_ratio;        // QPS improvement vs cache investment
    double cache_hit_rate;               // Cache hit rate
    int num_rounds;                       // Number of rounds tested
};

void build_base_index(const std::string& dataset_name, const std::string& base_dir, 
                      const std::string& index_output) {
    std::string filepath = base_dir + dataset_name;
    cout << "\n" << string(50, '=') << endl;
    cout << "Building base index for: " << dataset_name << endl;
    cout << string(50, '=') << endl;
    
    size_t num_vectors = 0, dim = 0;
    float* data = load_fvecs_as_float(filepath, num_vectors, dim);
    if (!data) return;
    
    cout << "Loaded " << num_vectors << " vectors of dimension " << dim << " (float)" << endl;
    
    L2Space l2space(dim);
    int M = 16;
    int efConstruction = 200;
    
    cout << "Allocating and building HNSW index..." << endl;
    HierarchicalNSWALPSIMPLIFIEDPQ<float>* appr_alg = 
        new HierarchicalNSWALPSIMPLIFIEDPQ<float>(&l2space, num_vectors, M, efConstruction);
    
    StopW stopw;
    #pragma omp parallel for
    for (long i = 0; i < (long)num_vectors; ++i) {
        appr_alg->addPoint(data + i * dim, i);
    }
    double build_time = 1e-6 * stopw.getElapsedTimeMicro();
    cout << "Graph construction time: " << build_time << " seconds" << endl;
    
    // Save base index (uncompressed)
    cout << "Saving base index to " << index_output << "..." << endl;
    appr_alg->saveIndex(index_output);
    cout << "Base index saved successfully." << endl;
    
    delete[] data;
    delete appr_alg;
}

CacheAnalysisResult test_with_cache_ratio(const std::string& dataset_name,
                                          const std::string& base_index_path,
                                          const std::string& query_file,
                                          const std::string& gt_file,
                                          float cache_ratio,
                                          size_t num_vectors,
                                          size_t query_dim) {
    CacheAnalysisResult result;
    result.dataset_name = dataset_name;
    result.cache_ratio = cache_ratio;
    result.cache_size = (size_t)(num_vectors * cache_ratio / 100.0);
    result.compress_time = 0.0;
    result.qps = 0.0;
    result.recall = 0.0;
    result.decoding_calls = 0;
    result.total_distance_computations = 0;
    result.avg_decode_per_query = 0.0;
    result.cache_efficiency_ratio = 0.0;
    
    cout << "\n--- Testing with cache ratio: " << fixed << setprecision(2) << cache_ratio 
         << "% (cache_size=" << result.cache_size << ") ---" << endl;
    
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
        cerr << "Query size and Ground Truth size mismatch!" << endl;
        delete[] massQ;
        delete[] massQA;
        return result;
    }
    
    // Load base index
    L2Space l2space(qdim);
    HierarchicalNSWALPSIMPLIFIEDPQ<float>* appr_alg = nullptr;
    
    try {
        cout << "Loading base index from " << base_index_path << "..." << endl;
        appr_alg = new HierarchicalNSWALPSIMPLIFIEDPQ<float>(&l2space, base_index_path, false);
        cout << "Index loaded successfully." << endl;
    } catch (std::exception& e) {
        cerr << "Failed to load index: " << e.what() << endl;
        delete[] massQ;
        delete[] massQA;
        return result;
    }
    
    // Enable TLS (Two-Level Search) if cache is being used
    if (result.cache_size > 0) {
        appr_alg->setUseTLS(true);
        // TLS ratio: only check top 20% of candidates in exact evaluation
        appr_alg->setTLSRatio(0.2);
    }
    
    // Compress dataset with specified cache size
    StopW stopw;
    cout << "Compressing dataset with cache_size=" << result.cache_size << "..." << endl;
    appr_alg->compress_dataset(result.cache_size);
    result.compress_time = 1e-6 * stopw.getElapsedTimeMicro();
    cout << "Compression completed in " << result.compress_time << " seconds" << endl;
    
    // Set search parameters
    appr_alg->setEf(200);
    appr_alg->setProfilingMetrics(true);
    
    size_t k = 1;  // Recall@1
    if (k > gt_dim) k = gt_dim;
    
    // Test search and collect metrics
    size_t correct = 0;
    size_t total = qsize * k;
    
    stopw.reset();
    
    for (long i = 0; i < (long)qsize; i++) {
        std::priority_queue<std::pair<float, labeltype>> search_result = 
            appr_alg->searchKnn(massQ + qdim * i, k);
        
        unordered_set<labeltype> g;
        for (size_t j = 0; j < k; j++) {
            g.insert(massQA[i * gt_dim + j]);
        }
        
        while (search_result.size()) {
            if (g.find(search_result.top().second) != g.end()) {
                correct++;
            }
            search_result.pop();
        }
    }
    
    double search_time_us = stopw.getElapsedTimeMicro();
    double qps = (double)qsize / (search_time_us / 1e6);
    result.qps = qps;
    result.recall = 1.0 * correct / total;
    
    // Collect statistics
    result.decoding_calls = appr_alg->getDecodingCallCount();
    result.total_distance_computations = appr_alg->metric_distance_computations;
    result.get_original_data_calls = appr_alg->getGetOriginalDataCallCount();
    cout << "Search Results:" << endl;
    cout << "  Recall@" << k << ": " << fixed << setprecision(4) << result.recall << endl;
    cout << "  QPS: " << fixed << setprecision(2) << result.qps << endl;
    cout << "  Decoding calls: " << result.decoding_calls << endl;
    cout << "  Total distance computations: " << result.total_distance_computations << endl;
    cout << "  Get original data calls: " << result.get_original_data_calls << endl;

    // Calculate additional metrics
    result.avg_decode_per_query = (double)result.decoding_calls / qsize;
    result.cache_efficiency_ratio = result.qps / (result.cache_ratio + 0.01);  // Avoid division by zero
    // 缓存命中率
    result.cache_hit_rate = (double)(result.get_original_data_calls - result.decoding_calls) / result.get_original_data_calls;
    
    cout << "  Avg decode per query: " << fixed << setprecision(4) << result.avg_decode_per_query << endl;
    cout << "  Cache efficiency ratio: " << fixed << setprecision(4) << result.cache_efficiency_ratio << endl;
    cout << "  Cache hit rate: " << fixed << setprecision(4) << result.cache_hit_rate << endl;
    
    delete[] massQ;
    delete[] massQA;
    delete appr_alg;
    
    return result;
}

void write_results_to_csv(const std::string& csv_path, 
                          const std::vector<CacheAnalysisResult>& results) {
    std::ofstream file(csv_path);
    if (file.is_open()) {
        file << "Dataset,Cache_Ratio(%),Cache_Size,Compress_Time(s),"
             << "QPS(avg),QPS_StdDev,Recall@1,Decoding_Calls,Distance_Computations,"
             << "Avg_Decode_Per_Query,Cache_Efficiency_Ratio,Cache_Hit_Rate,Num_Rounds\n";
        
        for (const auto& res : results) {
            file << res.dataset_name << ","
                 << fixed << setprecision(2) << res.cache_ratio << ","
                 << res.cache_size << ","
                 << fixed << setprecision(4) << res.compress_time << ","
                 << fixed << setprecision(2) << res.qps << ","
                 << fixed << setprecision(2) << res.qps_std_dev << ","
                 << fixed << setprecision(4) << res.recall << ","
                 << res.decoding_calls << ","
                 << res.total_distance_computations << ","
                 << fixed << setprecision(6) << res.avg_decode_per_query << ","
                 << fixed << setprecision(6) << res.cache_efficiency_ratio << ","
                 << fixed << setprecision(6) << res.cache_hit_rate << ","
                 << res.num_rounds << "\n";
        }
        file.close();
        cout << "\nResults written to " << csv_path << endl;
    } else {
        cerr << "Failed to open CSV file for writing." << endl;
    }
}

int main(int argc, char** argv) {
    omp_set_num_threads(32);
    
    std::string base_dir = "../datasets/hdf5files/";
    
    vector<string> datasets = {
        "fashion-mnist-784-euclidean_train.fvecs",
        "gist-960-euclidean_train.fvecs",
        "mnist-784-euclidean_train.fvecs",
        "sift-128-euclidean_train.fvecs",
        "deep-image-96-angular_train.fvecs"
    };
    
    // Cache ratios to test: 0.0 (baseline), 0.5%, 1%, 5%, 10%
    vector<float> cache_ratios = {0.0, 0.5, 1.0, 5.0, 10.0};
    
    // Number of rounds to run for stability
    int num_rounds = 1;
    
    // Parse command line
    for(int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--dataset" && i + 1 < argc) {
            datasets.clear();
            while(i + 1 < argc && argv[i + 1][0] != '-') {
                datasets.push_back(argv[++i]);
            }
        } else if (arg == "--cache-ratios" && i + 1 < argc) {
            cache_ratios.clear();
            while(i + 1 < argc && argv[i + 1][0] != '-') {
                cache_ratios.push_back(std::stof(argv[++i]));
            }
        } else if (arg == "--num-rounds" && i + 1 < argc) {
            num_rounds = std::stoi(argv[++i]);
        }
    }
    
    std::vector<CacheAnalysisResult> all_results;
    
    for (const auto& ds : datasets) {
        // Extract dataset name prefix
        size_t pos = ds.find("_train.fvecs");
        string prefix = ds.substr(0, pos);
        
        // Build base index
        string base_index_path = prefix + "_base_uncompressed.bin";
        build_base_index(ds, base_dir, base_index_path);
        
        // Load dataset info for cache ratio calculation
        size_t num_vectors = 0, dim = 0;
        float* temp_data = load_fvecs_as_float(base_dir + ds, num_vectors, dim);
        if (!temp_data) continue;
        delete[] temp_data;
        
        // Test with different cache ratios
        string query_file = base_dir + prefix + "_test.fvecs";
        string gt_file = base_dir + prefix + "_neighbors.ivecs";
        
        cout << "\n" << string(60, '*') << endl;
        cout << "Cache Ratio Analysis for: " << prefix << endl;
        cout << "Number of rounds per configuration: " << num_rounds << endl;
        cout << string(60, '*') << endl;
        
        for (float ratio : cache_ratios) {
            cout << "\n--- Testing cache_ratio=" << fixed << setprecision(2) << ratio << "% (" 
                 << num_rounds << " rounds) ---" << endl;
            
            // Run multiple rounds and collect results
            vector<CacheAnalysisResult> round_results;
            vector<double> qps_values;
            
            for (int round = 0; round < num_rounds; ++round) {
                CacheAnalysisResult res = test_with_cache_ratio(
                    prefix, base_index_path, query_file, gt_file,
                    ratio, num_vectors, dim
                );
                if (res.qps > 0) {
                    round_results.push_back(res);
                    qps_values.push_back(res.qps);
                    cout << "  Round " << (round + 1) << ": QPS=" << fixed << setprecision(2) 
                         << res.qps << ", Recall=" << setprecision(4) << res.recall << endl;
                }
            }
            
            // Aggregate results from multiple rounds
            if (!round_results.empty()) {
                CacheAnalysisResult aggregated = round_results[0];
                aggregated.num_rounds = num_rounds;
                
                // Calculate average and std dev for QPS
                double sum_qps = 0.0, sum_qps_sq = 0.0;
                double sum_recall = 0.0;
                long sum_decoding = 0;
                long sum_distance = 0;
                
                for (const auto& res : round_results) {
                    sum_qps += res.qps;
                    sum_qps_sq += res.qps * res.qps;
                    sum_recall += res.recall;
                    sum_decoding += res.decoding_calls;
                    sum_distance += res.total_distance_computations;
                }
                
                aggregated.qps = sum_qps / num_rounds;
                aggregated.recall = sum_recall / num_rounds;
                aggregated.decoding_calls = sum_decoding / num_rounds;
                aggregated.total_distance_computations = sum_distance / num_rounds;
                
                // Calculate standard deviation
                double variance = (sum_qps_sq / num_rounds) - (aggregated.qps * aggregated.qps);
                aggregated.qps_std_dev = std::sqrt(variance);
                
                // Recalculate derived metrics
                aggregated.avg_decode_per_query = (double)aggregated.decoding_calls / num_vectors;
                aggregated.cache_efficiency_ratio = aggregated.qps / (ratio + 0.01);
                
                cout << "  Average QPS: " << fixed << setprecision(2) << aggregated.qps 
                     << " ± " << setprecision(2) << aggregated.qps_std_dev << endl;
                
                all_results.push_back(aggregated);
            }
        }
    }
    
    // Output results to CSV
    write_results_to_csv("cache_ratio_analysis_results.csv", all_results);
    
    cout << "\n" << string(60, '=') << endl;
    cout << "Cache ratio analysis completed!" << endl;
    cout << "Results saved to cache_ratio_analysis_results.csv" << endl;
    cout << "Ran " << num_rounds << " round(s) per configuration for stability" << endl;
    cout << string(60, '=') << endl;
    
    return 0;
}
