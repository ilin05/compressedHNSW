#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <omp.h>
#include <iomanip>

#include "../../hnswlib/hnswlib.h"

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

// 辅助函数：读取 fvecs 文件 (原数据为float)，但将其存储为 double 数组方便我们构建 double 类型索引
float* load_fvecs(const std::string& filename, size_t& num_vectors, size_t& dim) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        std::cerr << "Cannot open " << filename << std::endl;
        return nullptr;
    }
    
    // 读取第一条数据的维度
    int32_t d;
    input.read((char*)&d, 4);
    dim = d;
    
    // 计算包含多少个向量
    input.seekg(0, std::ios::end);
    size_t file_size = input.tellg();
    num_vectors = file_size / (4 + dim * 4);
    
    float* data = new float[num_vectors * dim];
    float* tmp = new float[dim];
    input.seekg(0, std::ios::beg);
    
    for (size_t i = 0; i < num_vectors; ++i) {
        input.read((char*)&d, 4); // 跳过维度d
        input.read((char*)tmp, dim * 4);
        for(size_t j = 0; j < dim; ++j) {
            data[i * dim + j] = static_cast<float>(tmp[j]);
        }
    }
    
    delete[] tmp;
    return data;
}

// 记录测试结果的结构体
struct TestResult {
    std::string dataset_name;
    double build_time;
    double compress_time;
    double data_compression_ratio;
    double index_compression_ratio;
};

void write_results_to_csv(const std::string& csv_file_path, const std::vector<TestResult>& results) {
    std::ofstream file(csv_file_path);
    if (file.is_open()) {
        file << "Dataset,Build time (s),Compress time (s),Data compression ratio,Index compression ratio\n";
        for (const auto& res : results) {
            file << res.dataset_name << ","
                 << res.build_time << ","
                 << res.compress_time << ","
                 << res.data_compression_ratio << ","
                 << res.index_compression_ratio << "\n";
        }
        file.close();
        std::cout << "\nResults successfully written to " << csv_file_path << std::endl;
    } else {
        std::cerr << "Failed to open CSV file for writing." << std::endl;
    }
}

TestResult test_build_index(const std::string& dataset_name, const std::string& base_dir) {
    TestResult res;
    res.dataset_name = dataset_name;
    res.build_time = 0.0;
    
    // 传统 HNSW 没有数据压缩
    res.compress_time = 0.0;
    res.data_compression_ratio = 1.0;
    res.index_compression_ratio = 1.0;

    std::string filepath = base_dir + dataset_name;
    cout << "\n==============================================" << endl;
    cout << "Testing dataset construction: " << dataset_name << endl;
    
    size_t num_vectors = 0, dim = 0;
    // 强制读取为 float 数组
    float* data = load_fvecs(filepath, num_vectors, dim);
    if (!data) return res;
    
    cout << "Loaded " << num_vectors << " vectors of dimension " << dim << " (converted to float)" << endl;
    
    // 初始化 L2 空间 (数据现为 float)
    L2Space l2space(dim);
    
    // HNSW 基本参数
    int M = 16;
    int efConstruction = 200;
    
    cout << "Allocating memory for index..." << endl;
    HierarchicalNSW<float>* appr_alg = 
        new HierarchicalNSW<float>(&l2space, num_vectors, M, efConstruction);
        
    StopW stopw;
    
    cout << "Inserting elements into HNSW... " << endl;
    // 多线程并发插入
    #pragma omp parallel for
    for (long i = 0; i < (long)num_vectors; ++i) {
        appr_alg->addPoint(data + i * dim, i);
    }
    
    res.build_time = 1e-6 * stopw.getElapsedTimeMicro();
    cout << "Graph Construction Time: " << res.build_time << " seconds" << endl;
    
    // 原版 HNSW 中获取索引大小的方式
    size_t original_index_size = appr_alg->indexFileSize();
    cout << "Index size: " << original_index_size << " bytes." << endl;

    // 保存索引
    std::string index_path = dataset_name + "_hnsw_float.bin";
    cout << "Saving index to " << index_path << "..." << endl;
    appr_alg->saveIndex(index_path);
    cout << "Index successfully saved." << endl;
    
    delete[] data;
    delete appr_alg;

    return res;
}

int main(int argc, char** argv) {
    omp_set_num_threads(32);

    // 假设从 build/ 目录执行，故默认相对路径如下
    // 用户可根据需要修改
    std::string base_dir = "../datasets/hdf5files/";
    
    // 允许通过命令行传参自定义基础路径
    // if(argc > 1) {
    //     base_dir = argv[1];
    //     if (base_dir.back() != '/' && base_dir.back() != '\\') {
    //         base_dir += "/";
    //     }
    // }

    vector<string> datasets = {
        "fashion-mnist-784-euclidean_train.fvecs",
        "gist-960-euclidean_train.fvecs",
        "mnist-784-euclidean_train.fvecs",
        "sift-128-euclidean_train.fvecs",
        "deep-image-96-angular_train.fvecs"
    };

    // 解析命令行参数，允许用户指定要测试的数据集
    for(int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--dataset" && i + 1 < argc) {
            datasets.clear();
            while(i + 1 < argc && argv[i + 1][0] != '-') {
                datasets.push_back(argv[++i]);
            }
        }
    }
    
    std::vector<TestResult> all_results;

    for (const auto& ds : datasets) {
        TestResult res = test_build_index(ds, base_dir);
        if (res.build_time > 0) { // 过滤掉加载失败的记录
            all_results.push_back(res);
        }
    }
    
    // 输出到单独的 CSV 文件，防止覆盖 ALP 测试结果
    write_results_to_csv("hnsw_float_build_results.csv", all_results);
    
    cout << "\nAll build tests (Original HNSW) completed." << endl;
    return 0;
}