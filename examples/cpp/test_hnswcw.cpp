#include "../../hnswlib/hnswlib.h"
#include "../data_processor/data_loader.h"

namespace {
    const double EPS[] = {1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12,
            1e-13, 1e-14, 1e-15, 1e-16, 1e-17, 1e-18, 1e-19, 1e-20, 1e-21, 1e-22, 1e-23};

    // file names list
    const std::vector<std::string> file_names = {
        // "winequality-red",
        // "winequality-white",
        // "Stress-Lysis",
        // "siftsmall_base",
        // "SaYoPillow",
        // "emotional_monitoring_dataset_with_target",
        "hair_loss",
        "simulated_highdim_physical"
        // "fordTest",
        // "fordTrain"
    };

    // encoding algorithm names list
    const std::vector<std::string> encoding_algorithm_names = {
        "DeXOR",
        "Camel",
        "Elf",
        "Gorilla"
    };

    // 结果以csv表格形式保存。每个数据集对应一个csv表。表中的每一行记录一中encoding algorithm在该数据集上的测试结果，列包括：
    // encoding algorithm name, hnswcw index build time (seconds), hnswcw getOriginalData time in building index (seconds), hnsw index build time (seconds), hnswcw index size (bytes), hnsw index size (bytes), hnsw graph size (bytes), hnswcw recall, hnsw recall, hnswcw query time per query (milliseconds), hnswcw getOriginalData time per query (milliseconds), hnsw query time per query (milliseconds)
    // 1. hnswcw index build time: 构建hnswcw索引所需时间
    // 2. hnswcw getOriginalData time in building index: 构建hnswcw索引过程中解压数据所需时间
    // 3. hnsw index build time: 构建hnsw索引所需时间
    // 4. hnswcw index size: hnswcw索引大小
    // 5. hnswcw graph size: hnswcw图结构大小
    // 6. hnsw index size: hnsw索引大小
    // 7. hnsw graph size: hnsw图结构大小
    // 8. hnswcw recall: hnswcw索引的召回率
    // 9. hnsw recall: hnsw索引的召回率
    // 10. hnswcw query time per query: hnswcw索引每次查询所需时间
    // 11. hnswcw getOriginalData time per query: hnswcw索引每次查询解压数据所需时间
    // 12. hnsw query time per query: hnsw索引每次查询所需时间
    std::map<std::string, std::map<std::string, std::vector<double>>> test_results;
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

bool test_hnswcw(std::string data_path, std::string file_name, std::string encoding_algorithm_name) {
    std::string file_path = data_path + "/" + file_name + ".csv";
    std::vector<std::vector<double>> data = data_loader::loadData(file_path);
    if (data.empty()) {
        std::cerr << "Failed to load data or data is empty." << std::endl;
        return false;
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
    hnswlib::HierarchicalNSWCW<double>* alg_hnsw = new hnswlib::HierarchicalNSWCW<double>(&space, max_elements, encoding_algorithm_name, M, ef_construction);

    double* data_ptr = new double[dim * max_elements];
    for (int i = 0; i < std::min(rows, max_elements); i++) {
        for (int j = 0; j < cols; j++) {
            data_ptr[i * dim + j] = data[i][j];
        }
    }

    // Add data to index
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < max_elements; i++) {
        alg_hnsw->addPoint(data_ptr + i * dim, i);
    }
    alg_hnsw->compactLevel0();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> build_duration = end - start;
    test_results[file_name][encoding_algorithm_name][0] = build_duration.count();

    long getOriginalData_time_in_building_index = alg_hnsw->getTotalTimeGetOriginalData();
    test_results[file_name][encoding_algorithm_name][1] = getOriginalData_time_in_building_index / 1e6; // microseconds to seconds

    size_t hnsw_index_size = alg_hnsw->getIndexSize();
    test_results[file_name][encoding_algorithm_name][3] = hnsw_index_size;

    // Output the data part of all elements in level 0
    int total_compressed_data_size = alg_hnsw->getCompressedDataSize();
    // std::cout << "Total compressed data size: " << total_compressed_data_size << " bytes" << std::endl;
    size_t hnsw_graph_size = hnsw_index_size - total_compressed_data_size;
    test_results[file_name][encoding_algorithm_name][4] = hnsw_graph_size;

    // Query the elements for themselves and measure recall
    int query_count = std::min(100, max_elements);
    int step = max_elements / query_count;
    alg_hnsw->resetTotalTimeGetOriginalData();
    float correct = 0;
    auto start_query = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < max_elements; i += step) {
        std::priority_queue<std::pair<double, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data_ptr + i * dim, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i) correct++;
    }
    auto end_query = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> query_duration = end_query - start_query;
    
    long getOriginalData_time_in_queries = alg_hnsw->getTotalTimeGetOriginalData();
    float avg_getOriginalData_time_per_query = static_cast<float>(getOriginalData_time_in_queries) / query_count / 1e3; // microseconds to milliseconds

    float recall = correct / query_count;
    test_results[file_name][encoding_algorithm_name][7] = recall;

    float avg_query_time = query_duration.count() / query_count; // milliseconds per query
    test_results[file_name][encoding_algorithm_name][9] = avg_query_time;

    test_results[file_name][encoding_algorithm_name][10] = avg_getOriginalData_time_per_query;

    // 检查数据正确性
    bool mismatch_found = false;
    for(int i = 0; i < rows; i++) {
        std::vector<double> decompressedVec = alg_hnsw->getOriginalDataByInternalId(i);
        for(int j = 0; j < cols; j++) {
            double originalValue = data[i][j];
            double decompressedValue = decompressedVec[j];
            int place = getDecimalPlace(originalValue);
            double eps = EPS[place];
            if (std::abs(originalValue - decompressedValue) > eps && place < 13) {
                // std::cerr << "Data mismatch at row " << i << ", col " << j
                //           << ": original=" << originalValue
                //           << ", decompressed=" << decompressedValue
                //           << ", eps=" << eps << std::endl;
                mismatch_found = true;}
        }
    }

    delete[] data_ptr;
    delete alg_hnsw;
    return !mismatch_found;
}

void test_hnsw(std::string data_path, std::string file_name) {
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
    hnswlib::HierarchicalNSW<double>* alg_hnsw = new hnswlib::HierarchicalNSW<double>(&space, max_elements, M, ef_construction);

    double* data_ptr = new double[dim * max_elements];
    for (int i = 0; i < std::min(rows, max_elements); i++) {
        for (int j = 0; j < cols; j++) {
            data_ptr[i * dim + j] = data[i][j];
        }
    }

    // Add data to index
    auto start_building_index = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < max_elements; i++) {
        alg_hnsw->addPoint(data_ptr + i * dim, i);
    }
    auto end_building_index = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> build_duration = end_building_index - start_building_index;
    for(const auto& encoding_algorithm_name : encoding_algorithm_names) {
        if(test_results[file_name].find(encoding_algorithm_name) == test_results[file_name].end()) {
            continue;
        }
        test_results[file_name][encoding_algorithm_name][2] = build_duration.count();
    }

    size_t hnsw_index_size = alg_hnsw->getIndexSize();
    for(const auto& encoding_algorithm_name : encoding_algorithm_names) {
        if(test_results[file_name].find(encoding_algorithm_name) == test_results[file_name].end()) {
            continue;
        }
        test_results[file_name][encoding_algorithm_name][5] = hnsw_index_size;
    }

    size_t hnsw_graph_size = hnsw_index_size - (max_elements * dim * sizeof(double));
    for(const auto& encoding_algorithm_name : encoding_algorithm_names) {
        if(test_results[file_name].find(encoding_algorithm_name) == test_results[file_name].end()) {
            continue;
        }
        test_results[file_name][encoding_algorithm_name][6] = hnsw_graph_size;
    }

    float correct = 0;
    auto start_query =  std::chrono::high_resolution_clock::now();
    for (int i = 0; i < max_elements; i++) {
        std::priority_queue<std::pair<double, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data_ptr + i * dim, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i) correct++;
    }
    auto end_query = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> query_duration = end_query - start_query;
    float avg_query_time = query_duration.count() / max_elements; // milliseconds per query

    float recall = correct / max_elements;

    for(const auto& encoding_algorithm_name : encoding_algorithm_names) {
        if(test_results[file_name].find(encoding_algorithm_name) == test_results[file_name].end()) {
            continue;
        }
        test_results[file_name][encoding_algorithm_name][8] = recall;
        test_results[file_name][encoding_algorithm_name][11] = avg_query_time;
    }

    delete[] data_ptr;
    delete alg_hnsw;
}

void write_results_to_csv(const std::string& output_dir, const std::string& file_name) {
    std::string output_file = output_dir + "/" + file_name + "_hnswcw_test_results.csv";
    std::ofstream ofs(output_file);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open output file: " << output_file << std::endl;
        return;
    }

    // Write header
    ofs << "Encoding Algorithm,HNSWCW Index Build Time (s),HNSWCW getOriginalData Time in Building Index (s),"
            "HNSW Index Build Time (s),HNSWCW Index Size (bytes),HNSWCW Graph Size (bytes),"
            "HNSW Index Size (bytes),HNSW Graph Size (bytes),HNSWCW Recall,HNSW Recall,"
            "HNSWCW Query Time per Query (ms),HNSWCW getOriginalData Time per Query (ms),HNSW Query Time per Query (ms)\n";

    for (const auto& encoding_algorithm_name : encoding_algorithm_names) {
        const auto& results = test_results[file_name][encoding_algorithm_name];
        ofs << encoding_algorithm_name;
        for (const auto& value : results) {
            ofs << "," << value;
        }
        ofs << "\n";
    }

    ofs.close();
    std::cout << "Results written to " << output_file << std::endl;
}

int main() {
    std::string data_path = "../datasets";
    std::string output_dir = "../test_results";

    // 设置结果集
    for(const auto& file_name : file_names) {
        test_results[file_name] = std::map<std::string, std::vector<double>>();
        for (const auto& encoding_algorithm_name : encoding_algorithm_names) {
            test_results[file_name][encoding_algorithm_name] = std::vector<double>();
            // vector 预留12个位置
            for (int i = 0; i < 12; i++) {
                test_results[file_name][encoding_algorithm_name].push_back(0.0);
            }
        }
    }

    // 运行所有hnswcw测试
    for(const auto& file_name : file_names) {
        test_hnsw(data_path, file_name);

        for(const auto& encoding_algorithm_name : encoding_algorithm_names) {
            bool result = test_hnswcw(data_path, file_name, encoding_algorithm_name);
            if(!result) {
                std::cerr << "Test FAILED for " << file_name << " with encoding " << encoding_algorithm_name << std::endl;
            }else{
                std::cout << "Test PASSED for " << file_name << " with encoding " << encoding_algorithm_name << std::endl;
            }
        }
        std::cout << "Completed tests for dataset: " << file_name << std::endl;
        write_results_to_csv(output_dir, file_name);
    }

    return 0;
}
