#include "../../hnswlib/hnswlib.h"
#include "../data_processor/data_loader.h"
#include <map>

#ifdef __linux__
#include <sys/mman.h>
#endif


namespace {
    const double EPS[] = {1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12,
            1e-13, 1e-14, 1e-15, 1e-16, 1e-17, 1e-18, 1e-19, 1e-20, 1e-21, 1e-22, 1e-23};

    // file names list
    const std::vector<std::string> file_names = {
        // "winequality-red",
        // "winequality-white",
        // // "Stress-Lysis",
        // "siftsmall_base",
        // "gist_small_base",
        // "mnist-784-euclidean_small_base",
        // "fashion-mnist-784-euclidean_small_base",
        // "mnist-784-euclidean",
        // "fashion-mnist-784-euclidean",
        "sift1m"
        // "gist_base"
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
        {"fashion-mnist-784-euclidean_small_base", 100}, // fashion-mnist-784-euclidean_small_base: cahce_size = rows * 1%
        {"mnist-784-euclidean", 600}, // mnist-784-euclidean: cahce_size = rows * 1%
        {"fashion-mnist-784-euclidean", 600}, // fashion-mnist-784-euclidean: cahce_size = rows * 1%
        {"sift1m", 10000}, // sift1m: cahce_size = rows * 1%
        {"gist_base", 10000} // gist_base: cahce_size = rows * 1%
        // {"SaYoPillow", 6}, // SaYoPillow: cahce_size = rows * 1%
        // {"emotional_monitoring_dataset_with_target", 10}, // emotional_monitoring_dataset_with_target: cahce_size = rows * 1%
        // {"simulated_highdim_physical", 50}, // simulated_highdim_physical: cahce_size = rows * 1%
        // {"hair_loss", 500} // hair_loss: cahce_size = rows * 0.5%
    };

    // 结果以csv表格形式保存。表中每一行记录一种数据集的测试结果。列包括：
    // 0. file name
    // 1. hnswcab_leann index build time (seconds)
    // 2. hnswcab_leann compression ratio
    // 3. hnswcab_leann recall
    // 4. hnswcab_leann query time per query (milliseconds)
    // 5. hnswcab_leann decoding time per query (milliseconds)
    // 6. hnsw index build time (seconds)
    // 7. hnsw recall
    // 8. hnsw query time per query (milliseconds)
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
    std::vector<double> results;
    int rows = 0, cols = 0, dim = 0, max_elements = 0;
    double* data_ptr = nullptr;

    {
        std::string file_path = data_path + "/" + file_name + ".csv";
        std::vector<std::vector<double>> data = data_loader::loadData(file_path);
        if (data.empty()) {
            std::cerr << "Failed to load data or data is empty." << std::endl;
            return results;
        }

        rows = static_cast<int>(data.size());
        cols = static_cast<int>(data.front().size());
        dim = cols;
        max_elements = rows;

        data_ptr = new double[static_cast<size_t>(dim) * max_elements];
        for (int i = 0; i < std::min(rows, max_elements); i++) {
            for (int j = 0; j < cols; j++) {
                data_ptr[i * dim + j] = data[i][j];
            }
        }
    }

    int M = 16;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff

    // Initing index
    hnswlib::L2SpaceDouble space(dim);
    // hnswlib::HierarchicalNSWCW<double>* alg_hnsw = new hnswlib::HierarchicalNSWCW<double>(&space, max_elements, encoding_algorithm_name, M, ef_construction);
    hnswlib::HierarchicalNSW<double>* alg_hnsw = new hnswlib::HierarchicalNSW<double>(&space, max_elements, M, ef_construction);

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

std::vector<double> test_save_hnswcab_leann(std::string data_path, std::string file_name) {
    std::vector<double> results;
    int rows = 0, cols = 0, dim = 0, max_elements = 0;
    double* data_ptr = nullptr;

    {
        std::string file_path = data_path + "/" + file_name + ".csv";
        std::vector<std::vector<double>> data = data_loader::loadData(file_path);
        if (data.empty()) {
            std::cerr << "Failed to load data or data is empty." << std::endl;
            return results;
        }

        rows = static_cast<int>(data.size());
        cols = static_cast<int>(data.front().size());
        dim = cols;
        max_elements = rows;

        data_ptr = new double[static_cast<size_t>(dim) * max_elements];
        for (int i = 0; i < std::min(rows, max_elements); i++) {
            for (int j = 0; j < cols; j++) {
                data_ptr[i * dim + j] = data[i][j];
            }
        }
    }

    int M = 16;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff

    std::string encoding_algorithm_name = "DeXOR"; // Compression algorithm name

    // Initing index
    hnswlib::L2SpaceDouble space(dim);
    hnswlib::HierarchicalNSWCABLEANN<double>* alg_hnsw = new hnswlib::HierarchicalNSWCABLEANN<double>(&space, max_elements, encoding_algorithm_name, M, ef_construction, true);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < max_elements; i++) {
        alg_hnsw->addPoint(data_ptr + i * dim, i);
        if(i > 0 && i % 5000 == 0){
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
    std::vector<double> results;
    int rows = 0, cols = 0, dim = 0, max_elements = 0;
    double* data_ptr = nullptr;

    {
        std::string file_path = data_path + "/" + file_name + ".csv";
        data_ptr = data_loader::loadDataForSearch(file_path, rows, cols);
        if (data_ptr == nullptr) {
            std::cerr << "Failed to load data or data is empty." << std::endl;
            return results;
        }

        dim = cols;
        max_elements = rows;
    }

#ifdef __linux__
    if (mlock(data_ptr, static_cast<size_t>(dim) * max_elements * sizeof(double)) != 0) {
        perror("mlock failed");
    } else {
        std::cout << "Pinned data_ptr to memory." << std::endl;
    }
#endif

    int M = 16;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff

    // Initing index
    hnswlib::L2SpaceDouble space(dim);
    // hnswlib::HierarchicalNSWCW<double>* alg_hnsw = new hnswlib::HierarchicalNSWCW<double>(&space, max_elements, encoding_algorithm_name, M, ef_construction);
    std::string hnsw_path = "storage/" + file_name + "_hnsw.bin";
    hnswlib::HierarchicalNSW<double>* alg_hnsw = new hnswlib::HierarchicalNSW<double>(&space, hnsw_path, false, false, max_elements);

    // Query the elements for themselves and measure recall
    float correct = 0;
    int step = std::max(1, max_elements / 50000);
    int query_count = 0;
    auto query_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < max_elements; i+=step) {
        std::priority_queue<std::pair<double, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data_ptr + i * dim, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i) correct++;
        query_count++;
    }
    auto query_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> query_duration = query_end - query_start;
    float avg_query_time = query_duration.count() / query_count; // milliseconds per query
    float recall = correct / query_count;
    std::cout << "HNSW recall: " << recall << "\n";
    std::cout << "HNSW average query time: " << avg_query_time << " ms" << std::endl;
    results.push_back(recall);
    results.push_back(avg_query_time);
    
    delete alg_hnsw;
#ifdef __linux__
    munlock(data_ptr, static_cast<size_t>(dim) * max_elements * sizeof(double));
#endif
    delete[] data_ptr;
    return results;
}

std::vector<double> test_load_hnswcab_leann(std::string data_path, std::string file_name, size_t cache_size = 0) {
    std::vector<double> results;
    int rows = 0, cols = 0, dim = 0, max_elements = 0;
    double* data_ptr = nullptr;

    {
        std::string file_path = data_path + "/" + file_name + ".csv";
        data_ptr = data_loader::loadDataForSearch(file_path, rows, cols);
        if (data_ptr == nullptr) {
            std::cerr << "Failed to load data or data is empty." << std::endl;
            return results;
        }

        dim = cols;
        max_elements = rows;
    }

#ifdef __linux__
    if (mlock(data_ptr, static_cast<size_t>(dim) * max_elements * sizeof(double)) != 0) {
        perror("mlock failed");
    } else {
        std::cout << "Pinned data_ptr to memory." << std::endl;
    }
#endif

    int M = 16;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff

    std::string encoding_algorithm_name = "DeXOR"; // Compression algorithm name

    // Initing index
    hnswlib::L2SpaceDouble space(dim);
    std::string hnswcw_path = "storage/" + file_name + "_hnswcw.bin";
    hnswlib::HierarchicalNSWCABLEANN<double>* alg_hnsw = new hnswlib::HierarchicalNSWCABLEANN<double>(&space, hnswcw_path, true, cache_size, false, max_elements);

    // Query the elements for themselves and measure recall
    float correct = 0;
    int query_count = 0;
    int step = std::max(1, max_elements / 50000);
    auto query_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < max_elements; i += step) {
        std::priority_queue<std::pair<double, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data_ptr + i * dim, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i) correct++;
        query_count++;
    }
    auto query_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> query_duration = query_end - query_start;
    float avg_query_time = query_duration.count() / query_count; // milliseconds per query
    float recall = correct / query_count;
    float decoding_time_per_query = static_cast<float>(alg_hnsw->getTotalTimeDecoding()) / query_count / 1e3; // milliseconds per query
    float decoding_call_count_per_query = static_cast<float>(alg_hnsw->getDecodingCallCount()) / query_count;
    std::cout << "HNSWCW recall: " << recall << "\n";
    std::cout << "HNSWCW average query time: " << avg_query_time << " ms" << std::endl;
    std::cout << "HNSWCW average decoding time: " << decoding_time_per_query << " ms" << std::endl;
    results.push_back(recall);
    results.push_back(avg_query_time);
    results.push_back(decoding_time_per_query);
    results.push_back(decoding_call_count_per_query);

    // long total_decoding_time = alg_hnsw->getTotalTimeDecoding();
    // float avg_decoding_time = static_cast<float>(total_decoding_time) / max_elements / 1e3; // milliseconds per query
    // std::cout << "HNSWCW average decoding time: " << avg_decoding_time << " ms" << std::endl;

    delete alg_hnsw;
#ifdef __linux__
    munlock(data_ptr, static_cast<size_t>(dim) * max_elements * sizeof(double));
#endif
    delete[] data_ptr;
    return results;
}

void collect_save_hnswcab_leann_results(){
    for(const auto& file_name : file_names){
        std::cout << "Processing file: " << file_name << std::endl;
        std::vector<double> save_results = test_save_hnswcab_leann("../datasets/", file_name);
        test_results[file_name][0] = save_results[0]; // hnswcab_leann index build time
        test_results[file_name][1] = save_results[1]; // hnswcab_leann compression ratio
    }
}

void collect_load_hnswcab_leann_results(){
    for(const auto& file_name : file_names){
        std::cout << "Processing file: " << file_name << std::endl;
        std::vector<double> load_results = test_load_hnswcab_leann("../datasets/", file_name, cache_sizes.at(file_name));
        test_results[file_name][2] = load_results[0]; // hnswcab_leann recall
        test_results[file_name][3] = load_results[1]; // hnswcab_leann query time
        test_results[file_name][4] = load_results[2]; // hnswcab_leann decoding time
        test_results[file_name][5] = load_results[3]; // hnswcab_leann decoding call count
    }
}

void collect_save_hnsw_results(){
    for(const auto& file_name : file_names){
        std::cout << "Processing file: " << file_name << std::endl;
        std::vector<double> save_results = test_save_hnsw("../datasets/", file_name);
        test_results[file_name][6] = save_results[0]; // hnsw index build time
    }
}

void collect_load_hnsw_results(){
    for(const auto& file_name : file_names){
        std::cout << "Processing file: " << file_name << std::endl;
        std::vector<double> load_results = test_load_hnsw("../datasets/", file_name);
        test_results[file_name][7] = load_results[0]; // hnsw recall
        test_results[file_name][8] = load_results[1]; // hnsw query
    }
}

void initialize_test_results(){
    for(const auto& file_name : file_names){
        test_results[file_name] = std::vector<double>(9, 0.0); // 0: hnswcab_leann build time, 1: hnswcab_leann compression ratio, 2: hnswcab_leann recall, 3: hnswcab_leann query time, 4: hnswcab_leann decoding time, 5: hnswcab_leann decoding call count, 6: hnsw build time, 7: hnsw recall, 8: hnsw query time
    }
}

void write_results_to_csv(const std::string& csv_file_path){
    std::ofstream csv_file(csv_file_path);
    if (!csv_file.is_open()) {
        std::cerr << "Failed to open CSV file for writing: " << csv_file_path << std::endl;
        return;
    }

    // Write header
    csv_file << "file_name,hnswcab_leann_build_time(s),hnswcab_leann_compression_ratio,hnswcab_leann_recall,hnswcab_leann_query_time(ms),hnswcab_leann_decoding_time(ms),hnswcab_leann_decoding_calls_per_query,"
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
    // collect_save_hnswcab_leann_results();
    // collect_save_hnsw_results();
    // collect_load_hnswcab_leann_results();
    collect_load_hnsw_results();

    write_results_to_csv("hnswcab_leann_hnsw_test_results.csv");

    return 0;
}
