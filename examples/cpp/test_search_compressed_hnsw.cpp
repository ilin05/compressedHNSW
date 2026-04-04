#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <unordered_set>
#include <omp.h>
#include <iomanip>
#include <map>

#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/compressed_hnsw_framework.h"

using namespace std;
using namespace hnswlib;

namespace {
    const std::map<std::string, int> cache_sizes = {
        {"fashion-mnist-784-euclidean", 600},
        {"mnist-784-euclidean", 600},
        {"sift-128-euclidean", 10000},
        {"gist-960-euclidean", 10000}
    };
}

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

template <typename CodecPolicy>
void test_search_dataset_tmpl(const std::string& dataset_name, const std::string& base_dir, const std::string& algo_name, std::ofstream& csv_file) {
    std::string prefix = dataset_name.substr(0, dataset_name.find_last_of('_'));
    
    std::string query_file = base_dir + dataset_name + "_test.fvecs";
    std::string gt_file = base_dir + dataset_name + "_neighbors.ivecs";
    std::string index_path = dataset_name + "_train.fvecs_" + algo_name + "_pq.bin";
    
    cout << "\n==============================================" << endl;
    cout << "Testing search on dataset: " << dataset_name << " using " << algo_name << endl;
    
    size_t qsize = 0, qdim = 0;
    double* massQ = load_fvecs_as_double(query_file, qsize, qdim);
    if (!massQ) return;
    cout << "Loaded " << qsize << " queries of dimension " << qdim << endl;

    size_t gt_num = 0, gt_dim = 0;
    unsigned int* massQA = load_ivecs(gt_file, gt_num, gt_dim);
    if (!massQA) {
        delete[] massQ;
        return;
    }
    cout << "Loaded " << gt_num << " ground truth records, top-" << gt_dim << " per query" << endl;
    
    if (qsize != gt_num) {
        cerr << "Query size and Ground Truth size mismatch!" << endl;
        return;
    }

    L2SpaceDouble l2space(qdim);
    HierarchicalNSWCABFRAMEWORK<double, CodecPolicy>* appr_alg = nullptr;
    
    try {
        cout << "Loading index from " << index_path << "..." << endl;
        // int cache_sz = 600;
        int cache_sz = cache_sizes.at(dataset_name);
        if(cache_sizes.find(prefix) != cache_sizes.end()) {
            cache_sz = cache_sizes.at(prefix);
        }
        appr_alg = new HierarchicalNSWCABFRAMEWORK<double, CodecPolicy>(&l2space, index_path, true, cache_sz);
        cout << "Index successfully loaded." << endl;
    } catch (std::exception& e) {
        cerr << "Failed to load index: " << e.what() << endl;
        delete[] massQ;
        delete[] massQA;
        return;
    }

    size_t k = 10;
    if (k > gt_dim) k = gt_dim;

    cout << "Testing Recall@" << k << " by varying ef..." << endl;

    vector<size_t> efs = {10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 300, 400};
    
    for (size_t ef : efs) {
        appr_alg->setEf(ef);
        size_t correct = 0;
        StopW stopw;

        // #pragma omp parallel for reduction(+:correct)
        for (long i = 0; i < (long)qsize; ++i) {
            std::priority_queue<std::pair<double, labeltype>> result = appr_alg->searchKnn(massQ + i * qdim, k);
            std::unordered_set<labeltype> gt_set;
            for (size_t j = 0; j < k; ++j) {
                gt_set.insert(massQA[i * gt_dim + j]);
            }
            
            while (!result.empty()) {
                if (gt_set.find(result.top().second) != gt_set.end()) {
                    correct++;
                }
                result.pop();
            }
        }
        
        float time_us_per_query = stopw.getElapsedTimeMicro() / qsize;
        double recall = static_cast<double>(correct) / (k * qsize);
        
        cout << "ef = " << setw(4) << ef << " | Recall@" << k << ": " << fixed << setprecision(4) << recall
             << " | Time/Query: " << time_us_per_query << " us" << endl;
             
        csv_file << dataset_name << "," << algo_name << "," << k << "," << ef << ","
                 << recall << "," << time_us_per_query << "\\n";
    
        if(recall >= 0.99){
            break;
        }
    }
    
    delete[] massQ;
    delete[] massQA;
    delete appr_alg;
}

void test_search_dataset(const std::string& dataset_name, const std::string& base_dir, const std::string& algo_name, std::ofstream& csv_file) {
    if (algo_name == "DeXOR") test_search_dataset_tmpl<codecs::DeXORCodecPolicy>(dataset_name, base_dir, algo_name, csv_file);
    else if (algo_name == "Gorilla") test_search_dataset_tmpl<codecs::GorillaCodecPolicy>(dataset_name, base_dir, algo_name, csv_file);
    else if (algo_name == "Elf") test_search_dataset_tmpl<codecs::ElfCodecPolicy>(dataset_name, base_dir, algo_name, csv_file);
    else if (algo_name == "Camel") test_search_dataset_tmpl<codecs::CamelCodecPolicy>(dataset_name, base_dir, algo_name, csv_file);
    else throw std::runtime_error("Unknown algorithm: " + algo_name);
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

    vector<string> base_datasets = {
        // "fashion-mnist-784-euclidean",
        // "gist-960-euclidean",
        // "mnist-784-euclidean",
        "sift-128-euclidean"
    };

    vector<string> algorithms = {
        // "DeXOR"
        "Gorilla", 
        "Elf", 
        "Camel"
    };

    std::string csv_file_path = "compressed_hnsw_search_recall_results.csv";
    std::ofstream csv_file(csv_file_path);
    if (!csv_file.is_open()) {
        cerr << "Failed to open CSV for writing!" << endl;
        return -1;
    }
    csv_file << "Dataset,Algorithm,K,ef,Recall,TimePerQuery(us)\n";

    for (const auto& ds : base_datasets) {
        for (const auto& algo : algorithms) {
            test_search_dataset(ds, base_dir, algo, csv_file);
        }
    }
    
    csv_file.close();
    cout << "\nAll search tests completed. Results written to " << csv_file_path << endl;
    return 0;
}