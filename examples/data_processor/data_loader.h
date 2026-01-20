#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <iostream>

namespace data_loader {

/**
 * @brief 从 CSV 文件加载数据.
 * 
 * @param filePath CSV文件的路径.
 * @return std::vector<std::vector<double>> 包含加载数据的二维向量.
 * @throws std::runtime_error 如果文件无法打开或数据不规整.
 */
static std::vector<std::vector<double>> loadData(const std::string& filePath) {
    std::vector<std::vector<double>> rows_vec;
    std::ifstream file(filePath);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filePath);
    }

    // Check for UTF-8 BOM
    char bom[3] = {0};
    file.read(bom, 3);
    if (file.gcount() == 3 && 
        static_cast<unsigned char>(bom[0]) == 0xEF && 
        static_cast<unsigned char>(bom[1]) == 0xBB && 
        static_cast<unsigned char>(bom[2]) == 0xBF) {
        // BOM found, skip it (file pointer is already at position 3)
    } else {
        // No BOM, rewind
        file.seekg(0, std::ios::beg);
    }

    std::string line;
    size_t num_cols = 0;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        std::vector<double> row;
        std::stringstream ss(line);
        std::string value;

        while (std::getline(ss, value, ',')) {
            // Trim whitespace
            const std::string whitespace = " \t\r\n";
            size_t start = value.find_first_not_of(whitespace);
            if (start != std::string::npos) {
                size_t end = value.find_last_not_of(whitespace);
                value = value.substr(start, end - start + 1);
            } else {
                value = "";
            }

            if (value.empty()) continue;

            try {
                row.push_back(std::stod(value));
            } catch (const std::invalid_argument& ia) {
                std::cerr << "Invalid argument: " << ia.what() << " for value: " << value << std::endl;
            } catch (const std::out_of_range& oor) {
                std::cerr << "Out of range: " << oor.what() << " for value: " << value << std::endl;
            }
        }

        if (!rows_vec.empty() && row.size() != num_cols) {
            file.close();
            throw std::runtime_error("Inconsistent number of columns in CSV file.");
        }
        
        if (rows_vec.empty() && !row.empty()) {
            num_cols = row.size();
        }

        rows_vec.push_back(row);
    }
    file.close();

    return rows_vec;
}

/**
 * @brief 从 CSV 文件加载数据并直接返回一维数组指针（优化内存占用）.
 * 
 * @param filePath CSV文件的路径.
 * @param rows 输出参数：行数.
 * @param dim 输出参数：维度（列数）.
 * @return double* 指向分配的一维数组的指针 (new double[]). 调用者负责 delete[].
 */
static double* loadDataForSearch(const std::string& filePath, int& rows, int& dim) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filePath);
    }

    // Check for UTF-8 BOM
    char bom[3] = {0};
    file.read(bom, 3);
    if (!(file.gcount() == 3 && 
        static_cast<unsigned char>(bom[0]) == 0xEF && 
        static_cast<unsigned char>(bom[1]) == 0xBB && 
        static_cast<unsigned char>(bom[2]) == 0xBF)) {
        file.seekg(0, std::ios::beg);
    }

    // Pass 1: Count rows and cols
    std::string line;
    size_t num_rows = 0;
    size_t num_cols = 0;
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::stringstream ss(line);
        std::string value;
        size_t current_row_cols = 0;

        while (std::getline(ss, value, ',')) {
             const std::string whitespace = " \t\r\n";
             size_t start = value.find_first_not_of(whitespace);
             if (start != std::string::npos) {
                 current_row_cols++;
             }
        }
        
        if (current_row_cols > 0) {
            if (num_cols == 0) num_cols = current_row_cols;
            num_rows++;
        }
    }

    rows = static_cast<int>(num_rows);
    dim = static_cast<int>(num_cols);

    if (num_rows == 0 || num_cols == 0) return nullptr;

    // Pass 2: Load data
    file.clear(); // Reset eof
    file.seekg(0, std::ios::beg);
    file.read(bom, 3);
    if (!(file.gcount() == 3 && 
        static_cast<unsigned char>(bom[0]) == 0xEF && 
        static_cast<unsigned char>(bom[1]) == 0xBB && 
        static_cast<unsigned char>(bom[2]) == 0xBF)) {
        file.seekg(0, std::ios::beg);
    }

    double* data_ptr = new double[num_rows * num_cols];
    size_t current_idx = 0;

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string value;

        while (std::getline(ss, value, ',')) {
            const std::string whitespace = " \t\r\n";
            size_t start = value.find_first_not_of(whitespace);
            if (start != std::string::npos) {
                size_t end = value.find_last_not_of(whitespace);
                value = value.substr(start, end - start + 1);
                try {
                    data_ptr[current_idx++] = std::stod(value);
                } catch (...) {
                    data_ptr[current_idx++] = 0.0;
                }
            }
        }
    }
    file.close();
    return data_ptr;
}

/**
 * @brief 从 .bvecs 文件加载数据并返回 double* 数组.
 * 格式: [dim (4 bytes)] [byte_0] [byte_1] ... [byte_dim-1] ... 重复
 * 
 * @param filePath bvecs文件路径
 * @param rows 输出: 向量数量
 * @param dim 输出: 维度
 * @return double* 数据指针, 需要 delete[] 释放
 */
static double* loadBvecs(const std::string& filePath, int& rows, int& dim) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filePath);
    }

    // 读取第一个向量的维度
    int d;
    if (!file.read(reinterpret_cast<char*>(&d), sizeof(int))) {
        throw std::runtime_error("Empty file or error reading dimension");
    }
    dim = d;

    // 计算总向量数
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t vecSize = sizeof(int) + dim;
    if (fileSize % vecSize != 0) {
        std::cerr << "Warning: File size is not a multiple of vector size. Data might be truncated." << std::endl;
    }

    size_t num_vectors = fileSize / vecSize;
    if(num_vectors > 15000000){
        num_vectors = 15000000; // 限制最大加载量15M
    }
    rows = static_cast<int>(num_vectors);

    std::cout << "Loading " << rows << " vectors of dimension " << dim << " from " << filePath << "..." << std::endl;

    double* data = new double[(size_t)rows * (size_t)dim];
    
    // 分块读取以提高效率
    size_t buffer_vecs = 100000; // 每次读取10w个
    std::vector<unsigned char> buffer(buffer_vecs * vecSize);

    size_t vectors_read = 0;
    while (vectors_read < num_vectors) {
        size_t to_read = std::min(buffer_vecs, num_vectors - vectors_read);
        file.read(reinterpret_cast<char*>(buffer.data()), to_read * vecSize);
        
        // 并行转换数据
        #pragma omp parallel for
        for (long i = 0; i < static_cast<long>(to_read); ++i) {
            unsigned char* ptr = buffer.data() + i * vecSize;
            // 跳过维度(4 bytes)
            ptr += sizeof(int);
            
            size_t data_idx = (vectors_read + i) * (size_t)dim;
            for (int j = 0; j < dim; ++j) {
                data[data_idx + j] = static_cast<double>(ptr[j]);
            }
        }
        vectors_read += to_read;
        if (vectors_read % 10000000 == 0) {
            std::cout << "Loaded " << vectors_read / 1000000 << "M vectors..." << std::endl;
        }
    }
    
    file.close();
    return data;
}

/**
 * @brief 从 .bvecs 文件加载部分数据（分块加载）.
 * 
 * @param filePath bvecs文件路径
 * @param offset_vectors 起始向量索引
 * @param count_vectors 加载的向量数量
 * @param dim 输出: 维度
 * @return double* 数据指针, 大小为 count_vectors * dim, 需要 delete[] 释放. 如果文件结束可能返回较少数据(需调用者检查逻辑，此处未返回实际读取数，假设调用者知道界限).
 */
static double* loadBvecsChunk(const std::string& filePath, size_t offset_vectors, size_t count_vectors, int& dim) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filePath);
    }

    // 读取维度
    int d;
    if (!file.read(reinterpret_cast<char*>(&d), sizeof(int))) {
        throw std::runtime_error("Empty file or error reading dimension");
    }
    dim = d;
    
    size_t vecSize = sizeof(int) + dim;
    
    // 跳转到偏移位置
    file.seekg(offset_vectors * vecSize, std::ios::beg);
    if (!file.good()) {
        file.close();
         // Return empty buffer or null? 
         // For safety let's allocate but warn
         std::cerr << "Warning: Seek past end of file in loadBvecsChunk" << std::endl;
         return new double[0];
    }

    double* data = new double[count_vectors * (size_t)dim];
    
    // 分块读取以提高效率
    size_t buffer_vecs = std::min((size_t)100000, count_vectors);
    std::vector<unsigned char> buffer(buffer_vecs * vecSize);

    size_t vectors_read = 0;
    while (vectors_read < count_vectors) {
        size_t to_read = std::min(buffer_vecs, count_vectors - vectors_read);
        
        file.read(reinterpret_cast<char*>(buffer.data()), to_read * vecSize);
        size_t read_count = file.gcount() / vecSize;
        
        if (read_count == 0) break;

        #pragma omp parallel for
        for (long i = 0; i < static_cast<long>(read_count); ++i) {
            unsigned char* ptr = buffer.data() + i * vecSize;
            ptr += sizeof(int); // skip dim
            
            size_t data_idx = (vectors_read + i) * (size_t)dim;
            for (int j = 0; j < dim; ++j) {
                data[data_idx + j] = static_cast<double>(ptr[j]);
            }
        }
        vectors_read += read_count;
        if (read_count < to_read) break; // EOF
    }
    
    file.close();
    
    if (vectors_read < count_vectors) {
        std::cerr << "Warning: Requested " << count_vectors << " vectors but only read " << vectors_read << std::endl;
    }
    
    return data;
}

/**
 * @brief 计算两个向量之间的欧氏距离.
 * 
 * @param a 第一个向量.
 * @param b 第二个向量.
 * @return double 两个向量之间的欧氏距离.
 */
static double distance(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Vectors must have the same dimension to calculate distance.");
    }
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += std::pow(a[i] - b[i], 2);
    }
    return std::sqrt(sum);
}

} // namespace data_loader