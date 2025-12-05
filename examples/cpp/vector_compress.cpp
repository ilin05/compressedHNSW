#include "../data_processor/data_loader.h"
#include "../../encoding_algorithms/algorithms_manager.h"
#include "../utils/block_stream_reader.h"
#include "../utils/stream_writer.h"
#include "../../encoding_algorithms/encoder.h"
#include "../../encoding_algorithms/decoder.h"

#include <cerrno>
#include <cstdio>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

namespace {
    const std::vector<std::string> kAlgorithms = {
        encoding_algorithm::EncodingAlgorithmEnums::DEXOR,
        encoding_algorithm::EncodingAlgorithmEnums::GORILLA,
        encoding_algorithm::EncodingAlgorithmEnums::ELF,
        encoding_algorithm::EncodingAlgorithmEnums::CAMEL
    };

    const std::string kDataType = "Double";
    const std::string kOutputDir = "storage";

    const double EPS[] = {1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12,
            1e-13, 1e-14, 1e-15, 1e-16, 1e-17, 1e-18, 1e-19, 1e-20, 1e-21, 1e-22, 1e-23};

    bool ensureDirectory(const std::string& dir) {
#if defined(_WIN32)
        int result = _mkdir(dir.c_str());
#else
        mode_t mode = 0755;
        int result = mkdir(dir.c_str(), mode);
#endif
        if (result == 0) {
            return true;
        }
        return errno == EEXIST;
    }
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
    try {
        std::vector<std::vector<double>> data = data_loader::loadData("../datasets/winequality-red.csv");
        if (data.empty()) {
            std::cerr << "Failed to load data or data is empty." << std::endl;
            return 1;
        }

        const int rows = static_cast<int>(data.size());
        const int cols = static_cast<int>(data.front().size());

        if (!ensureDirectory(kOutputDir)) {
            std::cerr << "Failed to create output directory: " << kOutputDir << std::endl;
            return 1;
        }

        std::map<std::string, int> algorithmBits;

        for (const std::string& algorithm : kAlgorithms) {
            int totalBits = 0;

            std::map<int, int> compressedBits;
            std::vector<std::unique_ptr<encoding_algorithm::Encoder>> encoders;
            encoders.reserve(static_cast<size_t>(cols));

            const std::string sharedOutputPath = kOutputDir + "/" + algorithm + ".bin";
            std::remove(sharedOutputPath.c_str());
            auto sharedWriter = std::make_shared<utils::StreamWriter>(sharedOutputPath);

            for (int col = 0; col < cols; ++col) {
                encoders.emplace_back(encoding_algorithm::AlgorithmsManager::getEncoder(kDataType, algorithm, sharedWriter));
            }

            // 按行写入，复用各列对应的 encoder
            for (int row = 0; row < rows; ++row) {
                compressedBits[row] = totalBits;
                for (int col = 0; col < cols; ++col) {
                    const int bitsWritten = encoders[col]->encode(data[row][col]);
                    totalBits += bitsWritten;
                }
            }

            const int closeBits = encoders[cols - 1]->close();
            totalBits += closeBits;
            compressedBits[rows] = totalBits;

            sharedWriter->clear();
            algorithmBits[algorithm] = totalBits;

            // decompress
            std::vector<std::vector<double>> decompressedData(rows, std::vector<double>(cols, 0.0));
            const int byteNum = cols * 8;
            // shared_ptr<utils::BlockStreamReader> reader = std::make_shared<utils::BlockStreamReader>(sharedOutputPath, static_cast<size_t>(byteNum));
            std::shared_ptr<utils::BlockStreamReader> reader = std::make_shared<utils::BlockStreamReader>(sharedOutputPath, static_cast<size_t>(byteNum));
            // 每列需要一个decoder
            std::vector<std::unique_ptr<encoding_algorithm::Decoder>> decoders;
            for (int col = 0; col < cols; ++col) {
                decoders.emplace_back(encoding_algorithm::AlgorithmsManager::getDecoder(kDataType, algorithm, reader));
            }
            for (int row = 0; row < rows; ++row) {
                int beginBit = compressedBits[row];
                int endBit = compressedBits[row + 1];
                // std::cout << "beginBit: " << beginBit << ", endBit: " << endBit << std::endl;
                reader->cacheData(beginBit, endBit);
                for (int col = 0; col < cols; ++col) {
                    double value = decoders[col]->decodeDouble();
                    decompressedData[row][col] = value;
                }
            }

            // 验证解压结果
            int errorCount = 0;
            for (int row = 0; row < rows; ++row) {
                for (int col = 0; col < cols; ++col) {
                    double v = data[row][col];
                    int place = getDecimalPlace(v);
                    double eps = EPS[place];
                    // std::cout << "Decompressed Value: " << decompressedData[row][col] << ", Original Value: " << v << ", Decimal Places: " << place << ", Eps: " << eps << std::endl;
                    if (std::abs(data[row][col] - decompressedData[row][col]) > eps && place < 13) {
                        errorCount++;
                        std::cerr << "Mismatch at row " << row << ", col " << col
                                  << ": original=" << data[row][col]
                                  << ", decompressed=" << decompressedData[row][col]
                                  << ", eps=" << eps << std::endl;
                        // break;
                    }
                }
            }
            if (errorCount == 0) {
                std::cout << "Algorithm " << algorithm << " passed verification." << std::endl;
            } else {
                std::cout << "Algorithm " << algorithm << " failed verification with " << errorCount << " errors." << std::endl;
            }
        }

        std::cout << "Compression results (bits per algorithm)" << std::endl;
        std::cout << "Rows: " << rows << ", Columns: " << cols << std::endl;
        for (const auto& entry : algorithmBits) {
            std::cout << "  " << entry.first << ": " << entry.second << " bits" << std::endl;
        }
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
