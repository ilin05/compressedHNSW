#pragma once

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace bigann_test_utils {

inline bool parse_subset_token(const std::string& token, size_t& subset_million) {
    // Expected format: bigann_10M
    const std::string prefix = "bigann_";
    if (token.size() <= prefix.size() + 1) return false;
    if (token.rfind(prefix, 0) != 0) return false;
    if (token.back() != 'M' && token.back() != 'm') return false;

    std::string num_part = token.substr(prefix.size(), token.size() - prefix.size() - 1);
    if (num_part.empty()) return false;
    for (char c : num_part) {
        if (!std::isdigit(static_cast<unsigned char>(c))) return false;
    }
    subset_million = static_cast<size_t>(std::stoull(num_part));
    return subset_million > 0;
}

inline std::vector<size_t> parse_subsets_from_cli(int argc, char** argv, const std::vector<size_t>& defaults) {
    std::vector<size_t> subsets = defaults;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--dataset" && i + 1 < argc) {
            subsets.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                size_t m = 0;
                if (parse_subset_token(argv[i + 1], m)) {
                    subsets.push_back(m);
                } else {
                    std::cerr << "Invalid dataset token: " << argv[i + 1]
                              << " (expected format like bigann_10M)" << std::endl;
                }
                ++i;
            }
        }
    }

    std::sort(subsets.begin(), subsets.end());
    subsets.erase(std::unique(subsets.begin(), subsets.end()), subsets.end());
    return subsets;
}

inline bool load_gt_ivecs_topk(const std::string& gt_path,
                               size_t expected_queries,
                               size_t topk,
                               std::vector<std::vector<unsigned int>>& gt) {
    std::ifstream input(gt_path, std::ios::binary);
    if (!input) {
        std::cerr << "Cannot open GT file: " << gt_path << std::endl;
        return false;
    }

    gt.clear();
    gt.reserve(expected_queries);

    for (size_t i = 0; i < expected_queries; ++i) {
        int32_t d = 0;
        if (!input.read(reinterpret_cast<char*>(&d), sizeof(int32_t))) {
            std::cerr << "GT file ended early at query " << i << std::endl;
            return false;
        }
        if (d <= 0) {
            std::cerr << "Invalid GT row width at query " << i << std::endl;
            return false;
        }

        std::vector<unsigned int> row(static_cast<size_t>(d));
        if (!input.read(reinterpret_cast<char*>(row.data()), static_cast<size_t>(d) * sizeof(unsigned int))) {
            std::cerr << "Failed to read GT row at query " << i << std::endl;
            return false;
        }

        if (row.size() > topk) row.resize(topk);
        gt.push_back(std::move(row));
    }

    return true;
}

inline std::string subset_label(size_t subset_million) {
    return "bigann_" + std::to_string(subset_million) + "M";
}

}  // namespace bigann_test_utils
