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

int main() {

    std::vector<std::vector<double>> data = data_loader::loadData("../datasets/simulated_highdim_physical.csv");
    if (data.empty()) {
        std::cerr << "Failed to load data or data is empty." << std::endl;
        return 1;
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
    hnswlib::HierarchicalNSW<double>* alg_hnsw = new hnswlib::HierarchicalNSW<double>(&space, max_elements, encoding_algorithm_name, M, ef_construction);

    double* data_ptr = new double[dim * max_elements];
    for (int i = 0; i < std::min(rows, max_elements); i++) {
        for (int j = 0; j < cols; j++) {
            data_ptr[i * dim + j] = data[i][j];
        }
    }

    // std::cout << "Data loaded into array." << std::endl;

    // Add data to index
    for (int i = 0; i < max_elements; i++) {
        alg_hnsw->addPoint(data_ptr + i * dim, i);
        if(i > 0 && i % 500 == 0){
            std::cout << "Added " << i << " points." << std::endl;
        }
        // std::cout << "Added point " << i << std::endl;
    }

    // Output the data part of all elements in level 0
    int total_compressed_data_size = alg_hnsw->getCompressedDataSize();
    std::cout << "Total compressed data size: " << total_compressed_data_size << " bytes" << std::endl;

    // 第0层数据总大小：
    std::cout << "Level 0 data size: " << alg_hnsw->data_level0_memory_.size() << " bytes" << std::endl;

    // Output the compression tree structure
    alg_hnsw->printCompressionTree();

    // std::cout << "Index built with " << max_elements << " elements." << std::endl;

    // Query the elements for themselves and measure recall
    float correct = 0;
    for (int i = 0; i < max_elements; i++) {
        std::priority_queue<std::pair<double, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data_ptr + i * dim, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i) correct++;
    }
    float recall = correct / max_elements;
    std::cout << "Recall: " << recall << "\n";

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
                std::cerr << "Data mismatch at row " << i << ", col " << j
                          << ": original=" << originalValue
                          << ", decompressed=" << decompressedValue
                          << ", eps=" << eps << std::endl;
                mismatch_found = true;}
        }
    }
    if(!mismatch_found) {
        std::cout << encoding_algorithm_name << " compression algorithm passed data integrity check." << std::endl;
    }

    // // Serialize index
    // std::string hnsw_path = "hnsw.bin";
    // alg_hnsw->saveIndex(hnsw_path);
    // delete alg_hnsw;

    // // Deserialize index and check recall
    // alg_hnsw = new hnswlib::HierarchicalNSW<double>(&space, hnsw_path);
    // correct = 0;
    // for (int i = 0; i < max_elements; i++) {
    //     std::priority_queue<std::pair<double, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data_ptr + i * dim, 1);
    //     hnswlib::labeltype label = result.top().second;
    //     if (label == i) correct++;
    // }
    // recall = (float)correct / max_elements;
    // std::cout << "Recall of deserialized index: " << recall << "\n";

    delete[] data_ptr;
    delete alg_hnsw;
    return 0;
}
