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

// Load fvecs file as double array
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

struct CacheAnalysisResult {
    std::string dataset_name;
    float cache_ratio;
    size_t cache_size;
    double qps;
    double qps_std_dev;
    double recall;
    long decoding_calls;
    long distance_computations;
    long getOriginalData_calls;
    long backtrack_hops;
    double avg_decode_per_query;
    double cache_efficiency_ratio;
    double cache_hit_rate;
    int num_rounds;
};

// Perform search on an already-loaded and cached index
CacheAnalysisResult perform_search_on_compressed_index(
    const std::string& dataset_name,
    const std::string& query_file,
    const std::string& gt_file,
    HierarchicalNSWCABFRAMEWORK<double, codecs::DeXORCodecPolicy>* appr_alg,
    float cache_ratio,
    size_t num_vectors,
    size_t query_dim) {
    
    CacheAnalysisResult result;
    result.dataset_name = dataset_name;
    result.cache_ratio = cache_ratio;
    result.cache_size = (size_t)(num_vectors * cache_ratio / 100.0);
    result.qps = 0.0;
    result.recall = 0.0;
    result.decoding_calls = 0;
    result.distance_computations = 0;
    result.getOriginalData_calls = 0;
    result.backtrack_hops = 0;
    result.avg_decode_per_query = 0.0;
    result.cache_efficiency_ratio = 0.0;
    result.cache_hit_rate = 0.0;
    // Load queries
    size_t qsize = 0, qdim = 0;
    double* massQ = load_fvecs_as_double(query_file, qsize, qdim);
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
    
    // Enable profiling metrics collection
    appr_alg->setProfilingMetrics(true);
    
    size_t k = 1;  // Recall@1
    if (k > gt_dim) k = gt_dim;
    
    // Test search and collect metrics
    size_t correct = 0;
    size_t total = qsize * k;
    
    StopW stopw;
    
    for (long i = 0; i < (long)qsize; i++) {
        std::priority_queue<std::pair<double, labeltype>> search_result = 
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
    
    // Collect statistics from profiling metrics
    result.decoding_calls = appr_alg->decoding_count;
    result.distance_computations = appr_alg->metric_distance_computations;
    result.getOriginalData_calls = appr_alg->getOriginalData_calls;
    result.backtrack_hops = appr_alg->getOriginalData_backtrack_hops;
    long cache_hits = appr_alg->getCacheHitTimes();
    result.cache_hit_rate = (result.getOriginalData_calls > 0) ? (double)cache_hits / result.getOriginalData_calls : 0.0;
    
    // Calculate additional metrics
    result.avg_decode_per_query = (double)result.decoding_calls / qsize;
    result.cache_efficiency_ratio = result.qps / (result.cache_ratio + 0.01);
    
    delete[] massQ;
    delete[] massQA;
    
    return result;
}

void write_results_to_csv(const std::string& csv_path, 
                          const std::vector<CacheAnalysisResult>& results) {
    std::ofstream file(csv_path);
    if (file.is_open()) {
        file << "Dataset,Cache_Ratio(%),Cache_Size,"
             << "QPS(avg),QPS_StdDev,Recall@1,Decoding_Calls,Distance_Computations,"
             << "GetOriginalData_Calls,Backtrack_Hops,Avg_Decode_Per_Query,"
             << "Cache_Efficiency_Ratio,Cache_Hit_Rate,Num_Rounds\n";
        
        for (const auto& res : results) {
            file << res.dataset_name << ","
                 << fixed << setprecision(2) << res.cache_ratio << ","
                 << res.cache_size << ","
                 << fixed << setprecision(2) << res.qps << ","
                 << fixed << setprecision(2) << res.qps_std_dev << ","
                 << fixed << setprecision(4) << res.recall << ","
                 << res.decoding_calls << ","
                 << res.distance_computations << ","
                 << res.getOriginalData_calls << ","
                 << res.backtrack_hops << ","
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
    
    // Default: SIFT dataset
    vector<string> datasets = {
        "sift-128-euclidean_train.fvecs"
    };
    
    // Cache ratios to test: 0.0 (baseline), 0.5%, 1%, 2%, 3%, 5%, 10%, 20%
    vector<float> cache_ratios = {0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0};
    
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
        
        // Index file path (for DeXOR algorithm)
        string index_path = prefix + "_DeXOR_chunlim_pq.bin";
        
        // Load dataset info for cache ratio calculation
        size_t num_vectors = 0, dim = 0;
        double* temp_data = load_fvecs_as_double(base_dir + ds, num_vectors, dim);
        if (!temp_data) {
            cerr << "Failed to load dataset " << ds << endl;
            continue;
        }
        delete[] temp_data;
        
        cout << "Dataset: " << prefix << ", Vectors: " << num_vectors << ", Dimension: " << dim << endl;
        
        // Test with different cache ratios
        string query_file = base_dir + prefix + "_test.fvecs";
        string gt_file = base_dir + prefix + "_neighbors.ivecs";
        
        cout << "\n" << string(60, '*') << endl;
        cout << "Cache Ratio Analysis for: " << prefix << " (DeXOR)" << endl;
        cout << "Number of rounds per configuration: " << num_rounds << endl;
        cout << string(60, '*') << endl;
        
        for (float ratio : cache_ratios) {
            cout << "\n--- Testing cache_ratio=" << fixed << setprecision(2) << ratio << "% (" 
                 << num_rounds << " rounds) ---" << endl;
            
            size_t cache_size = (size_t)(num_vectors * ratio / 100.0);
            
            // LOAD INDEX FRESH for each cache_ratio
            L2SpaceDouble l2space(dim);
            HierarchicalNSWCABFRAMEWORK<double, codecs::DeXORCodecPolicy>* appr_alg = nullptr;
            
            try {
                cout << "Loading DeXOR index from " << index_path << " with cache_size=" << cache_size << "..." << endl;
                appr_alg = new HierarchicalNSWCABFRAMEWORK<double, codecs::DeXORCodecPolicy>(
                    &l2space, index_path, true, cache_size
                );
                cout << "Index loaded successfully." << endl;
            } catch (std::exception& e) {
                cerr << "Failed to load index: " << e.what() << endl;
                continue;
            }
            
            // Configure search parameters
            appr_alg->setEf(200);
            appr_alg->setUseTls(true);
            appr_alg->setTlsRatio(0.2);
            
            // Run multiple rounds of SEARCH ONLY
            vector<CacheAnalysisResult> round_results;
            vector<double> qps_values;
            
            for (int round = 0; round < num_rounds; ++round) {
                CacheAnalysisResult res = perform_search_on_compressed_index(
                    prefix, query_file, gt_file,
                    appr_alg, ratio, num_vectors, dim
                );
                if (res.qps > 0) {
                    round_results.push_back(res);
                    qps_values.push_back(res.qps);
                    cout << "  Round " << (round + 1) << ": QPS=" << fixed << setprecision(2) 
                         << res.qps << ", Recall=" << setprecision(4) << res.recall 
                         << ", DecodingCalls=" << res.decoding_calls << endl;
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
                long sum_getOriginalData = 0;
                long sum_backtrack = 0;
                
                for (const auto& res : round_results) {
                    sum_qps += res.qps;
                    sum_qps_sq += res.qps * res.qps;
                    sum_recall += res.recall;
                    sum_decoding += res.decoding_calls;
                    sum_distance += res.distance_computations;
                    sum_getOriginalData += res.getOriginalData_calls;
                    sum_backtrack += res.backtrack_hops;
                }
                
                aggregated.qps = sum_qps / num_rounds;
                aggregated.recall = sum_recall / num_rounds;
                aggregated.decoding_calls = sum_decoding / num_rounds;
                aggregated.distance_computations = sum_distance / num_rounds;
                aggregated.getOriginalData_calls = sum_getOriginalData / num_rounds;
                aggregated.backtrack_hops = sum_backtrack / num_rounds;
                
                // Calculate standard deviation
                double variance = (sum_qps_sq / num_rounds) - (aggregated.qps * aggregated.qps);
                aggregated.qps_std_dev = std::sqrt(variance);
                
                // Recalculate derived metrics
                aggregated.avg_decode_per_query = (double)aggregated.decoding_calls / num_vectors;
                aggregated.cache_efficiency_ratio = aggregated.qps / (ratio + 0.01);
                
                cout << "  Average QPS: " << fixed << setprecision(2) << aggregated.qps 
                     << " ± " << setprecision(2) << aggregated.qps_std_dev 
                     << ", GetOriginalData calls: " << aggregated.getOriginalData_calls
                     << ", Backtrack hops: " << aggregated.backtrack_hops << endl;
                
                all_results.push_back(aggregated);
            }
            
            // Clean up index for this cache_ratio
            delete appr_alg;
        }
    }
    
    // Output results to CSV
    write_results_to_csv("cache_ratio_analysis_dexor_results.csv", all_results);
    
    cout << "\n" << string(60, '=') << endl;
    cout << "Cache ratio analysis (DeXOR) completed!" << endl;
    cout << "Results saved to cache_ratio_analysis_dexor_results.csv" << endl;
    cout << "Ran " << num_rounds << " round(s) per configuration for stability" << endl;
    cout << string(60, '=') << endl;
    
    return 0;
}
