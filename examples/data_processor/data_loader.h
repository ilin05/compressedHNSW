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