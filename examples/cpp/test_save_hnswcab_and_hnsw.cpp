#include "../../hnswlib/hnswlib.h"
#include "../data_processor/data_loader.h"
#include <map>

namespace {
    const double EPS[] = {1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12,
            1e-13, 1e-14, 1e-15, 1e-16, 1e-17, 1e-18, 1e-19, 1e-20, 1e-21, 1e-22, 1e-23};

    // file names list
    const std::vector<std::string> file_names = {
        "winequality-red",
        "winequality-white",
        // "Stress-Lysis",
        "siftsmall_base",
        "gist_small_base",
        "mnist-784-euclidean_small_base",
        "fashion-mnist-784-euclidean_small_base"
        // "SaYoPillow",
        // "emotional_monitoring_dataset_with_target",
        // "simulated_highdim_physical"
        // "hair_loss"
        // "sift1m"
        // "fordTest",
        // "fordTrain"
    };

    const std::map<std::string, int> cache_sizes = {
        {"winequality-red", 16}, // winequality-red: cahce_size = rows * 1%
        {"winequality-white", 48}, // winequality-white: cahce_size = rows * 1%
        // {"Stress-Lysis", 20}, // Stress-Lysis: cahce_size = rows * 1%
        {"siftsmall_base", 100}, // siftsmall_base: cahce_size = rows * 1%
        {"gist_small_base", 100}, // gist_small_base: cahce_size = rows * 1%
        {"mnist-784-euclidean_small_base", 100}, // mnist-784-euclidean_small_base: cahce_size = rows * 1%
        {"fashion-mnist-784-euclidean_small_base", 100} // fashion-mnist-784-euclidean_small_base: cahce_size = rows * 1%
        // {"SaYoPillow", 6}, // SaYoPillow: cahce_size = rows * 1%
        // {"emotional_monitoring_dataset_with_target", 10}, // emotional_monitoring_dataset_with_target: cahce_size = rows * 1%
        // {"simulated_highdim_physical", 50}, // simulated_highdim_physical: cahce_size = rows * 1%
        // {"hair_loss", 500} // hair_loss: cahce_size = rows * 0.5%
    };

    // 结果以csv表格形式保存。表中每一行记录一种数据集的测试结果。列包括：
    // 0. file name
    // 1. hnswcab index build time (seconds)
    // 2. hnswcab compression ratio
    // 3. hnswcab recall
    // 4. hnswcab query time per query (milliseconds)
    // 5. hnsw index build time (seconds)
    // 6. hnsw recall
    // 7. hnsw query time per query (milliseconds)
    std::map<std::string, std::vector<double>> test_results;
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

std::vector<double> test_save_hnsw(std::string data_path, std::string file_name) {
    std::string file_path = data_path + "/" + file_name + ".csv";
    std::vector<std::vector<double>> data = data_loader::loadData(file_path);
    std::vector<double> results;
    if (data.empty()) {
        std::cerr << "Failed to load data or data is empty." << std::endl;
        return results;
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
        if(i > 0 && i % 5000 == 0){
            std::cout << "Added " << i << " points." << std::endl;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> build_duration = end - start;
    std::cout << "Index built in " << build_duration.count() << " seconds." << std::endl;
    results.push_back(build_duration.count());

    // Serialize index
    std::string hnsw_path = "storage/" + file_name + "_hnsw.bin";
    alg_hnsw->saveIndex(hnsw_path);
    delete alg_hnsw;
    delete[] data_ptr;
    return results;
}

std::vector<double> test_save_hnswcab(std::string data_path, std::string file_name) {
    std::string file_path = data_path + "/" + file_name + ".csv";
    std::vector<std::vector<double>> data = data_loader::loadData(file_path);
    std::vector<double> results;
    if (data.empty()) {
        std::cerr << "Failed to load data or data is empty." << std::endl;
        return results;
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
    hnswlib::HierarchicalNSWCAB<double>* alg_hnsw = new hnswlib::HierarchicalNSWCAB<double>(&space, max_elements, encoding_algorithm_name, M, ef_construction, true);

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
    alg_hnsw->compress_dataset();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> build_duration = end - start;
    std::cout << "Index built in " << build_duration.count() << " seconds." << std::endl;
    results.push_back(build_duration.count());

    alg_hnsw -> printCompressionTree("storage/" + file_name + "_encoding_tree.txt");

    // Serialize index
    std::string hnswcw_path = "storage/" + file_name + "_hnswcw.bin";
    alg_hnsw->saveIndex(hnswcw_path);

    // 原始大小
    size_t original_data_size = static_cast<size_t>(max_elements) * static_cast<size_t>(dim) * sizeof(double);
    // 压缩的data部分总大小
    size_t compressed_data_size = alg_hnsw->getCompressedDataSize();
    std::cout << "Total compressed data size: " << compressed_data_size << " bytes." << std::endl;
    double compression_ratio = static_cast<double>(original_data_size) / static_cast<double>(compressed_data_size);
    std::cout << "Compression ratio: " << compression_ratio << std::endl;
    results.push_back(compression_ratio);

    delete alg_hnsw;
    delete[] data_ptr;
    return results;
}

std::vector<double> test_load_hnsw(std::string data_path, std::string file_name){
    std::string file_path = data_path + "/" + file_name + ".csv";
    std::vector<std::vector<double>> data = data_loader::loadData(file_path);
    std::vector<double> results;
    if (data.empty()) {
        std::cerr << "Failed to load data or data is empty." << std::endl;
        return results;
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
    std::string hnsw_path = "storage/" + file_name + "_hnsw.bin";
    hnswlib::HierarchicalNSW<double>* alg_hnsw = new hnswlib::HierarchicalNSW<double>(&space, hnsw_path, false, false, max_elements);

    double* data_ptr = new double[dim * max_elements];
    for (int i = 0; i < std::min(rows, max_elements); i++) {
        for (int j = 0; j < cols; j++) {
            data_ptr[i * dim + j] = data[i][j];
        }
    }

    // Query the elements for themselves and measure recall
    float correct = 0;
    auto query_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < max_elements; i++) {
        std::priority_queue<std::pair<double, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data_ptr + i * dim, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i) correct++;
    }
    auto query_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> query_duration = query_end - query_start;
    float avg_query_time = query_duration.count() / max_elements; // milliseconds per query
    float recall = correct / max_elements;
    std::cout << "HNSW recall: " << recall << "\n";
    std::cout << "HNSW average query time: " << avg_query_time << " ms" << std::endl;
    results.push_back(recall);
    results.push_back(avg_query_time);
    
    delete alg_hnsw;
    delete[] data_ptr;
    return results;
}

std::vector<double> test_load_hnswcab(std::string data_path, std::string file_name, size_t cache_size = 0) {
    std::string file_path = data_path + "/" + file_name + ".csv";
    std::vector<std::vector<double>> data = data_loader::loadData(file_path);
    std::vector<double> results;
    if (data.empty()) {
        std::cerr << "Failed to load data or data is empty." << std::endl;
        return results;
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
    std::string hnswcw_path = "storage/" + file_name + "_hnswcw.bin";
    hnswlib::HierarchicalNSWCAB<double>* alg_hnsw = new hnswlib::HierarchicalNSWCAB<double>(&space, hnswcw_path, true, cache_size, false, max_elements);

    double* data_ptr = new double[dim * max_elements];
    for (int i = 0; i < std::min(rows, max_elements); i++) {
        for (int j = 0; j < cols; j++) {
            data_ptr[i * dim + j] = data[i][j];
        }
    }

    // Query the elements for themselves and measure recall
    float correct = 0;
    auto query_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < max_elements; i++) {
        std::priority_queue<std::pair<double, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data_ptr + i * dim, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i) correct++;
    }
    auto query_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> query_duration = query_end - query_start;
    float avg_query_time = query_duration.count() / max_elements; // milliseconds per query
    float recall = correct / max_elements;
    std::cout << "HNSWCW recall: " << recall << "\n";
    std::cout << "HNSWCW average query time: " << avg_query_time << " ms" << std::endl;
    results.push_back(recall);
    results.push_back(avg_query_time);

    // long total_decoding_time = alg_hnsw->getTotalTimeDecoding();
    // float avg_decoding_time = static_cast<float>(total_decoding_time) / max_elements / 1e3; // milliseconds per query
    // std::cout << "HNSWCW average decoding time: " << avg_decoding_time << " ms" << std::endl;

    delete alg_hnsw;
    delete[] data_ptr;
    return results;
}

void test_save_and_load_hnswcab(){
    for(const auto& file_name : file_names){
        std::cout << "Processing file: " << file_name << std::endl;
        std::vector<double> save_results = test_save_hnswcab("../datasets/", file_name);
        std::vector<double> load_results = test_load_hnswcab("../datasets/", file_name, cache_sizes.at(file_name));
        test_results[file_name][0] = save_results[0]; // hnswcab index build time
        test_results[file_name][1] = save_results[1]; // hnswcab compression ratio
        test_results[file_name][2] = load_results[0]; // hnswcab recall
        test_results[file_name][3] = load_results[1]; // hnswcab query time
    }
}

void test_save_and_load_hnsw(){
    for(const auto& file_name : file_names){
        std::cout << "Processing file: " << file_name << std::endl;
        std::vector<double> save_results = test_save_hnsw("../datasets/", file_name);
        std::vector<double> load_results = test_load_hnsw("../datasets/", file_name);
        test_results[file_name][4] = save_results[0]; // hnsw index build time
        test_results[file_name][5] = load_results[0]; // hnsw recall
        test_results[file_name][6] = load_results[1]; // hnsw query
    }
}

void initialize_test_results(){
    for(const auto& file_name : file_names){
        test_results[file_name] = std::vector<double>(7, 0.0); // 0: hnswcab build time, 1: hnswcab compression ratio, 2: hnswcab recall, 3: hnswcab query time, 4: hnsw build time, 5: hnsw recall, 6: hnsw query time
    }
}

void write_results_to_csv(const std::string& csv_file_path){
    std::ofstream csv_file(csv_file_path);
    if (!csv_file.is_open()) {
        std::cerr << "Failed to open CSV file for writing: " << csv_file_path << std::endl;
        return;
    }

    // Write header
    csv_file << "file_name,hnswcab_build_time(s),hnswcab_compression_ratio,hnswcab_recall,hnswcab_query_time(ms),"
             << "hnsw_build_time(s),hnsw_recall,hnsw_query_time(ms)\n";

    // Write data
    for (const auto& entry : test_results) {
        const std::string& file_name = entry.first;
        const std::vector<double>& results = entry.second;
        csv_file << file_name;
        for (const auto& value : results) {
            csv_file << "," << value;
        }
        csv_file << "\n";
    }

    csv_file.close();
    std::cout << "Results written to " << csv_file_path << std::endl;
}

int main() {

    std::string file_path = "../datasets/";
    initialize_test_results();
    test_save_and_load_hnswcab();
    test_save_and_load_hnsw();

    write_results_to_csv("hnswcab_hnsw_test_results.csv");

    return 0;
}
