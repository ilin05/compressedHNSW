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
        "hair_loss"
        // "fordTest",
        // "fordTrain"
    };

    // encoding algorithm names list
    const std::string encoding_algorithm_name = "DeXOR"; // Compression algorithm name
    // const std::vector<std::string> encoding_algorithm_names = {
    //     "DeXOR",
    //     // "Camel",
    //     // "Elf",
    //     // "Gorilla"
    // };

    // 结果以csv表格形式保存。每个数据集对应一个csv表。表中的每一行记录一种cache_size在该数据集上的测试结果，列包括：
    // 1. hnswcw index build time: 构建hnswcw索引所需时间
    // 2. hnswcw getOriginalData time in building index: 构建hnswcw索引过程中getOriginalData所需时间
    // 3. hnswcw recall: hnswcw索引的召回率
    // 4. hnswcw query time per query: hnswcw索引每次查询所需时间
    // 5. hnswcw getOriginalData time per query: hnswcw索引每次查询解压数据所需时间
    // 6. hnswcw index size: 索引总大小
    // 7. hnswcw graph size: 图结构大小
    // 8. cache pop count: 缓存淘汰次数
    // 9. getOriginalData times called: getOriginalData调用次数
    std::map<std::string, std::map<size_t, std::vector<double>>> test_results;

    std::vector<size_t> cache_sizes;
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

bool test_hnswcw(std::vector<std::vector<double>> data, std::string data_path, std::string file_name, std::string encoding_algorithm_name, size_t cache_max_size) {
    const int rows = static_cast<int>(data.size());
    const int cols = static_cast<int>(data.front().size());


    int dim = cols;               // Dimension of the elements
    int max_elements = rows;   // Maximum number of elements, should be known beforehand
    int M = 16;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff

    // Initing index
    hnswlib::L2SpaceDouble space(dim);
    hnswlib::HierarchicalNSWCW<double>* alg_hnsw = new hnswlib::HierarchicalNSWCW<double>(&space, max_elements, encoding_algorithm_name, M, ef_construction, cache_max_size);

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
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> build_duration = end - start;
    test_results[file_name][cache_max_size][0] = build_duration.count();

    long getOriginalData_time_in_building_index = alg_hnsw->getTotalTimeGetOriginalData();
    test_results[file_name][cache_max_size][1] = getOriginalData_time_in_building_index / 1e6; // microseconds to seconds

    size_t hnsw_index_size = alg_hnsw->getIndexSize();
    test_results[file_name][cache_max_size][5] = hnsw_index_size;

    // Output the data part of all elements in level 0
    int total_compressed_data_size = alg_hnsw->getCompressedDataSize();
    // std::cout << "Total compressed data size: " << total_compressed_data_size << " bytes" << std::endl;
    size_t hnsw_graph_size = hnsw_index_size - total_compressed_data_size;
    test_results[file_name][cache_max_size][6] = hnsw_graph_size;

    // Query the elements for themselves and measure recall
    int query_count = std::min(500, max_elements);
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
    test_results[file_name][cache_max_size][2] = recall;

    float avg_query_time = query_duration.count() / query_count; // milliseconds per query
    test_results[file_name][cache_max_size][3] = avg_query_time;

    test_results[file_name][cache_max_size][4] = avg_getOriginalData_time_per_query;

    test_results[file_name][cache_max_size][7] = static_cast<double>(alg_hnsw->getCachePopCount());

    test_results[file_name][cache_max_size][8] = static_cast<double>(alg_hnsw->getGetOriginalDataCallCount());

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

void write_results_to_csv(const std::string& output_dir, const std::string& file_name, const std::vector<size_t>& cache_sizes) {
    std::string output_file = output_dir + "/" + file_name + "_hnswcw_test_results.csv";
    std::ofstream ofs(output_file);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open output file: " << output_file << std::endl;
        return;
    }

    // Write header
    ofs << "CacheSize,BuildTime(s),GetOriginalDataTimeInBuildingIndex(s),Recall,QueryTimePerQuery(ms),GetOriginalDataTimePerQuery(ms),IndexSize(bytes),GraphSize(bytes),CachePopCount,GetOriginalDataCallCount\n";

    for (const auto& cache_size : cache_sizes) {
        const auto& results = test_results[file_name][cache_size];
        ofs << cache_size;
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
    std::string output_dir = "../test_hnsw_cached";

    // 设置结果集
    for(const auto& file_name : file_names) {
        test_results[file_name] = std::map<size_t, std::vector<double>>();

        std::string file_path = data_path + "/" + file_name + ".csv";
        std::vector<std::vector<double>> data = data_loader::loadData(file_path);
        size_t rows = data.size();
        if (data.empty()) {
            std::cerr << "Failed to load data or data is empty." << std::endl;
            return false;
        }

        // 测试6种cache_size，从0开始，每次加1/300th的data size
        cache_sizes.clear();
        for (int i = 0; i <= 5; i++) {
            size_t cache_size = std::min((rows / 500) * i, (size_t)100);
            cache_sizes.push_back(cache_size);
        }

        for (const auto& cache_size : cache_sizes) {
            test_results[file_name][cache_size] = std::vector<double>();
            // vector 预留9个位置
            for (int i = 0; i < 9; i++) {
                test_results[file_name][cache_size].push_back(0.0);
            }
        }

        for (const auto& cache_size : cache_sizes) {
            bool result = test_hnswcw(data, data_path, file_name, encoding_algorithm_name, cache_size);
            if(!result) {
                std::cerr << "Test FAILED for " << file_name << " with cache size " << cache_size << std::endl;
            }else{
                std::cout << "Test PASSED for " << file_name << " with cache size " << cache_size << std::endl;
        
            }
        }

        std::cout << "Completed tests for dataset: " << file_name << std::endl;
        write_results_to_csv(output_dir, file_name, cache_sizes);
    }

    return 0;
}
