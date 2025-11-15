#include "../data_processor/data_loader.h"
#include "../../encoding_algorithms/algorithms_manager.h"

#include <cerrno>
#include <iostream>
#include <map>
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

int main() {
    try {
        std::vector<std::vector<double>> data = data_loader::loadData("../datasets/siftsmall_base.csv");
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

        std::map<std::string, long long> algorithmBits;

        for (const std::string& algorithm : kAlgorithms) {
            long long totalBits = 0;
            const std::string outputPath = kOutputDir + "/" + algorithm + ".bin";

            for (int col = 0; col < cols; ++col) {
                auto encoder = encoding_algorithm::AlgorithmsManager::getEncoder(kDataType, algorithm, outputPath);

                for (int row = 0; row < rows; ++row) {
                    totalBits += encoder->encode(data[row][col]);
                }

                totalBits += encoder->close();
                encoder->flush();
            }

            algorithmBits[algorithm] = totalBits;
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
