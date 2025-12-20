#include "../../hnswlib/hnswlib.h"
#include "../data_processor/data_loader.h"

namespace {
    const double EPS[] = {1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12,
            1e-13, 1e-14, 1e-15, 1e-16, 1e-17, 1e-18, 1e-19, 1e-20, 1e-21, 1e-22, 1e-23};

    // file names list
    const std::vector<std::string> file_names = {
        "winequality-red",
        "winequality-white",
        "Stress-Lysis",
        "siftsmall_base",
        "SaYoPillow",
        "emotional_monitoring_dataset_with_target",
        "simulated_highdim_physical",
        "hair_loss",
        "sift1m",
        "fordTest",
        "fordTrain"
    };

}

static int getDecimalPlace(double value) {
    std::string valueStr = std::to_string(value);

    // 移除末尾的 '0'
    valueStr.erase(valueStr.find_last_not_of('0') + 1, std::string::npos);

    // 如果移除零后末尾是小数点（例如 3.000000 -> 3.），也移除小数点
    if (valueStr.back() == '.') {
        valueStr.pop_back();
    }

    int index = valueStr.find('.');
    if (index == std::string::npos) {
        return 0;
    } else {
        return static_cast<int>(valueStr.length() - index - 1); 
    }
}

void test_save_and_load_hnsw(std::string data_path, std::string file_name){
    std::string file_path = data_path + "/" + file_name + ".csv";
    std::vector<std::vector<double>> data = data_loader::loadData(file_path);
    if (data.empty()) {
        std::cerr << "Failed to load data or data is empty." << std::endl;
        return;
    }

    const int rows = static_cast<int>(data.size());
    const int cols = static_cast<int>(data.front().size());

    int dim = cols;               // Dimension of the elements
    int max_elements = rows;   // Maximum number of elements, should be known beforehand
    int M = 16;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff

    // Initing index
    hnswlib::L2SpaceDouble space(dim);
    // hnswlib::HierarchicalNSWCW<double>* alg_hnsw = new hnswlib::HierarchicalNSWCW<double>(&space, max_elements, encoding_algorithm_name, M, ef_construction);
    hnswlib::HierarchicalNSW<double>* alg_hnsw = new hnswlib::HierarchicalNSW<double>(&space, max_elements, M, ef_construction);

    double* data_ptr = new double[dim * max_elements];
    for (int i = 0; i < std::min(rows, max_elements); i++) {
        for (int j = 0; j < cols; j++) {
            data_ptr[i * dim + j] = data[i][j];
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < max_elements; i++) {
        alg_hnsw->addPoint(data_ptr + i * dim, i);
        if(i > 0 && i % 500 == 0){
            std::cout << "Added " << i << " points." << std::endl;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> build_duration = end - start;
    std::cout << "Index built in " << build_duration.count() << " seconds." << std::endl;

    // Serialize index
    std::string hnsw_path = "storage/" + file_name + "_hnsw.bin";
    alg_hnsw->saveIndex(hnsw_path);
    delete alg_hnsw;
    delete[] data_ptr;
}

void test_save_and_load_hnswcw(std::string data_path, std::string file_name) {
    std::string file_path = data_path + "/" + file_name + ".csv";
    std::vector<std::vector<double>> data = data_loader::loadData(file_path);
    if (data.empty()) {
        std::cerr << "Failed to load data or data is empty." << std::endl;
        return;
    }

    const int rows = static_cast<int>(data.size());
    const int cols = static_cast<int>(data.front().size());

    int dim = cols;               // Dimension of the elements
    int max_elements = rows;   // Maximum number of elements, should be known beforehand
    int M = 16;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff

    std::string encoding_algorithm_name = "DeXOR"; // Compression algorithm name

    // Initing index
    hnswlib::L2SpaceDouble space(dim);
    hnswlib::HierarchicalNSWCW<double>* alg_hnsw = new hnswlib::HierarchicalNSWCW<double>(&space, max_elements, encoding_algorithm_name, M, ef_construction);

    double* data_ptr = new double[dim * max_elements];
    for (int i = 0; i < std::min(rows, max_elements); i++) {
        for (int j = 0; j < cols; j++) {
            data_ptr[i * dim + j] = data[i][j];
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < max_elements; i++) {
        alg_hnsw->addPoint(data_ptr + i * dim, i);
        if(i > 0 && i % 500 == 0){
            std::cout << "Added " << i << " points." << std::endl;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> build_duration = end - start;
    std::cout << "Index built in " << build_duration.count() << " seconds." << std::endl;

    alg_hnsw -> printCompressionTree("storage/" + file_name + "_encoding_tree.txt");

    // Serialize index
    std::string hnswcw_path = "storage/" + file_name + "_hnswcw.bin";
    alg_hnsw->saveIndex(hnswcw_path);

    // 压缩的data部分总大小
    size_t compressed_data_size = alg_hnsw->getCompressedDataSize();
    std::cout << "Total compressed data size: " << compressed_data_size << " bytes." << std::endl;

    delete alg_hnsw;
    delete[] data_ptr;
}

int main() {
    std::string file_path = "../datasets/";
    
    for(const auto& file_name : file_names){
        std::cout << "Processing file: " << file_name << std::endl;
        test_save_and_load_hnsw(file_path, file_name);
        test_save_and_load_hnswcw(file_path, file_name);
    }
    return 0;
}
