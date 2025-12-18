#include "../../hnswlib/hnswlib.h"
#include "../data_processor/data_loader.h"

namespace {
    const double EPS[] = {1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12,
            1e-13, 1e-14, 1e-15, 1e-16, 1e-17, 1e-18, 1e-19, 1e-20, 1e-21, 1e-22, 1e-23};
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
    int M = 32;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 300;  // Controls index search speed/build speed tradeoff

    std::string encoding_algorithm_name = "DeXOR"; // Compression algorithm name

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

    // Query the elements for themselves and measure recall
    float correct = 0;
    auto query_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < max_elements; i++) {
        std::priority_queue<std::pair<double, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data_ptr + i * dim, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i) correct++;
        if(label % 10000 == 0){
            std::cout << "Queried " << i << " points." << std::endl;
        }
    }
    auto query_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> query_duration = query_end - query_start;
    float avg_query_time = query_duration.count() / max_elements; // milliseconds per query
    float recall = correct / max_elements;
    std::cout << "Recall: " << recall << "\n";
    std::cout << "Average query time: " << avg_query_time << " ms" << std::endl;

    // Serialize index
    std::string hnsw_path = "hnswcw.bin";
    alg_hnsw->saveIndex(hnsw_path);
    delete alg_hnsw;

    // Deserialize index and check data
    alg_hnsw = new hnswlib::HierarchicalNSW<double>(&space, hnsw_path);
    bool mismatch_found = false;
    for(int i = 0; i < rows; i++) {
        char* reloadedData = alg_hnsw->getDataByInternalId(i);
        std::vector<double> decompressedVec(dim);
        std::memcpy(decompressedVec.data(), reloadedData, dim * sizeof(double));
        for(int j = 0; j < cols; j++) {
            double originalValue = data[i][j];
            double decompressedValue = decompressedVec[j];
            int place = getDecimalPlace(originalValue);
            double eps = EPS[place];
            if (std::abs(originalValue - decompressedValue) > eps && place < 13) {
                std::cerr << "Data mismatch at row " << i << ", col " << j
                          << ": original=" << originalValue
                          << ", decompressed=" << decompressedValue
                          << ", eps=" << eps << std::endl;
                mismatch_found = true;}
        }
    }
    if(!mismatch_found) {
        std::cout << "Deserialized index passed data integrity check." << std::endl;
    }

    delete[] data_ptr;
    delete alg_hnsw;
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
    int M = 32;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 300;  // Controls index search speed/build speed tradeoff

    std::string encoding_algorithm_name = "DeXOR"; // Compression algorithm name

    // Initing index
    hnswlib::L2SpaceDouble space(dim);
    // hnswlib::HierarchicalNSWCW<double>* alg_hnsw = new hnswlib::HierarchicalNSWCW<double>(&space, max_elements, encoding_algorithm_name, M, ef_construction);
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

    // Query the elements for themselves and measure recall
    float correct = 0;
    auto query_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < max_elements; i++) {
        std::priority_queue<std::pair<double, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data_ptr + i * dim, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i) correct++;
        if(label % 10000 == 0){
            std::cout << "Queried " << i << " points." << std::endl;
        }
    }
    auto query_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> query_duration = query_end - query_start;
    float avg_query_time = query_duration.count() / max_elements; // milliseconds per query
    float recall = correct / max_elements;
    std::cout << "Recall: " << recall << "\n";
    std::cout << "Average query time: " << avg_query_time << " ms" << std::endl;

    alg_hnsw -> printCompressionTree();

    // Serialize index
    std::string hnswcw_path = "hnswcw.bin";
    alg_hnsw->saveIndex(hnswcw_path);

    std::vector<size_t> level0_element_start_positions = alg_hnsw->getLevel0ElementStartPositions();
    std::vector<char> data_level0_memory = alg_hnsw->getDataLevel0Memory();
    std::vector<std::vector<unsigned int>> encoding_chains;
    for(int i = 0; i < max_elements; i++) {
        encoding_chains.push_back(alg_hnsw->getEncodingChain(i));
    }

    delete alg_hnsw;

    // Deserialize index and check data
    alg_hnsw = new hnswlib::HierarchicalNSWCW<double>(&space, hnswcw_path, false, rows);

    bool mismatch_found = false;
    for(int i = 0; i < max_elements; i++) {
        std::vector<double> decompressedVec = alg_hnsw->getOriginalDataByInternalId(i);
        for(int j = 0; j < cols; j++) {
            double originalValue = data[i][j];
            double decompressedValue = decompressedVec[j];
            int place = getDecimalPlace(originalValue);
            double eps = EPS[place];
            if (std::abs(originalValue - decompressedValue) > eps && place < 13) {
                std::cerr << "Data mismatch at row " << i << ", col " << j
                          << ": original=" << originalValue
                          << ", decompressed=" << decompressedValue
                          << ", eps=" << eps << std::endl;
                mismatch_found = true;}
        }
    }
    if(!mismatch_found) {
        std::cout << "Deserialized index passed data integrity check." << std::endl;
    }

    // test recall after deserialization
    correct = 0;
    query_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < max_elements; i++) {
        std::priority_queue<std::pair<double, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data_ptr + i * dim, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i) correct++;
        if(label % 10000 == 0){
            std::cout << "Queried " << i << " points." << std::endl;
        }
    }
    query_end = std::chrono::high_resolution_clock::now();
    query_duration = query_end - query_start;
    avg_query_time = query_duration.count() / max_elements; // milliseconds per query
    recall = correct / max_elements;
    std::cout << "Recall: " << recall << "\n";
    std::cout << "Average query time: " << avg_query_time << " ms" << std::endl;

    delete[] data_ptr;
    delete alg_hnsw;
}

int main() {

    std::string file_name = "winequality-red";
    std::string file_path = "../datasets/";
    
    // test_save_and_load_hnsw(file_path, file_name);
    test_save_and_load_hnswcw(file_path, file_name);

    return 0;
}
