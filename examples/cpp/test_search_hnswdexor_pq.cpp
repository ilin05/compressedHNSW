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
#include "../../hnswlib/hnswalg_simplified_alp_PQ.h"

using namespace std;
using namespace hnswlib;

namespace {
    const std::map<std::string, int> cache_sizes = {
        {"fashion-mnist-784-euclidean", 600}, // fashion-mnist-784-euclidean: cache_size = rows * 1%
        {"mnist-784-euclidean", 600}, // mnist-784-euclidean: cache_size = rows * 1%
        {"sift-128-euclidean", 10000}, // sift-128-euclidean: cache_size = rows * 1%
        {"gist-960-euclidean", 10000} // gist-960-euclidean: cache_size = rows * 1%
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

// 读取 fvecs 文件 (原数据为float)，将其存储为 double 数组以匹配查询
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

// 读取 ivecs 文件，作为 Ground Truth (neighbors.ivecs 是 int32 类型)
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

struct SearchResult {
    size_t ef;
    double recall;
    double time_us_per_query;
};

void test_search_dataset(const std::string& dataset_name, const std::string& base_dir, std::ofstream& csv_file) {
    std::string prefix = dataset_name.substr(0, dataset_name.find_last_of('_')); // 去掉 _train.fvecs 等后缀，如果命名不同请按需调整
    
    std::string query_file = base_dir + dataset_name + "_test.fvecs";
    std::string gt_file = base_dir + dataset_name + "_neighbors.ivecs";
    std::string index_path = dataset_name + "_train.fvecs_hnswcableann_pq.bin";

    
    cout << "\n==============================================" << endl;
    cout << "Testing search on dataset: " << dataset_name << endl;
    
    // Load Queries
    size_t qsize = 0, qdim = 0;
    double* massQ = load_fvecs_as_double(query_file, qsize, qdim);
    if (!massQ) return;
    cout << "Loaded " << qsize << " queries of dimension " << qdim << endl;

    // Load Ground Truth
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

    // Load Index
    L2SpaceDouble l2space(qdim);
    HierarchicalNSWCABLEANNPQ<double>* appr_alg = nullptr;
    
    try {
        cout << "Loading index from " << index_path << "..." << endl;
        appr_alg = new HierarchicalNSWCABLEANNPQ<double>(&l2space, index_path, true, cache_sizes.at(prefix));
        cout << "Index successfully loaded." << endl;
    } catch (std::exception& e) {
        cerr << "Failed to load index: " << e.what() << endl;
        delete[] massQ;
        delete[] massQA;
        return;
    }

    size_t k = 10; // 默认测 recall@10
    if (k > gt_dim) k = gt_dim;

    cout << "Testing Recall@" << k << " by varying ef..." << endl;

    vector<size_t> efs = {10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 300, 400, 500};
    // vector<size_t> efs = {10, 20, 30, 40, 50};
    
    for (size_t ef : efs) {
        appr_alg->setEf(ef);
        size_t correct = 0;
        size_t total = qsize * k;
        
        StopW stopw;
        
        // 可选开启多线程测QPS，如果是测单线程Latency这里请去掉 #pragma omp parallel for
        // #pragma omp parallel for reduction(+:correct)
        for (long i = 0; i < (long)qsize; i++) {
            std::priority_queue<std::pair<double, labeltype>> result = appr_alg->searchKnn(massQ + qdim * i, k);
            
            unordered_set<labeltype> g;
            for (size_t j = 0; j < k; j++) {
                g.insert(massQA[i * gt_dim + j]);
            }

            while (result.size()) {
                if (g.find(result.top().second) != g.end()) {
                    correct++;
                }
                result.pop();
            }
        }
        
        double time_ms_per_query = stopw.getElapsedTimeMicro() / 1000.0 / qsize;
        double recall = 1.0 * correct / total;
        
        cout << "ef: " << setw(3) << ef 
             << " | recall: " << fixed << setprecision(4) << recall 
             << " | time/query: " << fixed << setprecision(2) << time_ms_per_query << " ms" << endl;
             
        csv_file << dataset_name << "," << k << "," << ef << "," << recall << "," << time_ms_per_query << "\n";
        
        if (recall >= 0.99) break; // 如果 recall 已经接近 1 就不需要测更大的 ef 了
    }
    
    delete[] massQ;
    delete[] massQA;
    delete appr_alg;
}

int main(int argc, char** argv) {
    // 根据 HDF5 处理出的前缀名推断
    std::string base_dir = "../datasets/hdf5files/";
    
    // if(argc > 1) {
    //     base_dir = argv[1];
    //     if (base_dir.back() != '/' && base_dir.back() != '\\') {
    //         base_dir += "/";
    //     }
    // }

    // 这里填入 HDF5 提取出来的数据集前缀基础名称
    vector<string> base_datasets = {
        "fashion-mnist-784-euclidean",
        "gist-960-euclidean",
        "mnist-784-euclidean",
        "sift-128-euclidean"
    };

    // 解析命令行参数，允许用户指定要测试的数据集
    for(int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--dataset" && i + 1 < argc) {
            base_datasets.clear();
            while(i + 1 < argc && argv[i + 1][0] != '-') {
                base_datasets.push_back(argv[++i]);
            }
        }
    }

    std::string csv_file_path = "hnswdexor_pq_search_recall_results.csv";
    std::ofstream csv_file(csv_file_path);
    csv_file << "Dataset,K,ef,Recall,TimePerQuery(ms)\n";

    for (const auto& ds : base_datasets) {
        test_search_dataset(ds, base_dir, csv_file);
    }
    
    csv_file.close();
    cout << "\nAll search tests completed. Results written to " << csv_file_path << endl;
    return 0;
}