#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace data_writer {

/**
 * @brief 将二维向量数据写入 CSV 文件.
 * 
 * @param data 要写入的二维向量 (std::vector<std::vector<double>>).
 * @param outputPath 输出CSV文件的路径.
 */
static void writeCsv(const std::vector<std::vector<double>>& data, const std::string& outputPath) {
    std::ofstream file(outputPath);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + outputPath);
    }

    for (const auto& row : data) {
        for (size_t j = 0; j < row.size(); ++j) {
            file << std::fixed << std::setprecision(1) << row[j];
            if (j < row.size() - 1) {
                file << ",";
            }
        }
        file << "\n";
    }

    file.close();
}

} // namespace data_writer