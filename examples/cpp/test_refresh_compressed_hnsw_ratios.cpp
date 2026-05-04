#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/compressed_hnsw_framework.h"

using namespace hnswlib;

namespace {

struct NormalRow {
    std::string dataset;
    std::string algorithm;
    std::string build_time;
    std::string compress_time;
    double data_ratio = 0.0;
    double index_ratio = 0.0;
};

struct BigannRow {
    std::string dataset;
    std::string subset_million;
    std::string algorithm;
    std::string vectors;
    std::string dim;
    std::string build_time;
    std::string compress_time;
    double data_ratio = 0.0;
    double index_ratio = 0.0;
    std::string index_file;
};

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> cols;
    std::stringstream ss(line);
    std::string col;
    while (std::getline(ss, col, ',')) {
        cols.push_back(col);
    }
    return cols;
}

size_t read_fvecs_num_and_dim(const std::string& file_path, size_t& dim_out) {
    std::ifstream in(file_path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open dataset file: " + file_path);
    }

    int32_t dim_i32 = 0;
    in.read(reinterpret_cast<char*>(&dim_i32), sizeof(int32_t));
    if (!in.good() || dim_i32 <= 0) {
        throw std::runtime_error("Invalid fvecs header: " + file_path);
    }

    dim_out = static_cast<size_t>(dim_i32);

    in.seekg(0, std::ios::end);
    const std::streampos file_size = in.tellg();
    const size_t stride = sizeof(int32_t) + dim_out * sizeof(float);
    if (stride == 0) {
        throw std::runtime_error("Invalid fvecs stride: " + file_path);
    }
    return static_cast<size_t>(file_size) / stride;
}

template <typename CodecPolicy>
std::pair<double, double> compute_ratios_with_codec(const std::string& index_path,
                                                    size_t dim,
                                                    size_t original_vectors,
                                                    size_t max_elements_for_load) {
    L2SpaceDouble space(dim);
    HierarchicalNSWCABFRAMEWORK<double, CodecPolicy> index(&space, index_path, true, 0, false, max_elements_for_load, false);

    const size_t original_data_size = original_vectors * dim * sizeof(double);
    const size_t compressed_data_size = index.getCompressedDataSize();
    const size_t original_index_size = index.getIndexSize();
    const size_t compressed_index_size = index.getCompressedIndexSize();

    if (compressed_data_size == 0 || compressed_index_size == 0) {
        throw std::runtime_error("Compressed size is zero for index: " + index_path);
    }

    const double data_ratio = static_cast<double>(original_data_size) / static_cast<double>(compressed_data_size);
    const double index_ratio = static_cast<double>(original_index_size) / static_cast<double>(compressed_index_size);
    return {data_ratio, index_ratio};
}

std::pair<double, double> compute_ratios_by_algo(const std::string& algo,
                                                 const std::string& index_path,
                                                 size_t dim,
                                                 size_t original_vectors,
                                                 size_t max_elements_for_load) {
    if (algo == "DeXOR") {
        return compute_ratios_with_codec<codecs::DeXORCodecPolicy>(index_path, dim, original_vectors, max_elements_for_load);
    }
    if (algo == "Gorilla") {
        return compute_ratios_with_codec<codecs::GorillaCodecPolicy>(index_path, dim, original_vectors, max_elements_for_load);
    }
    if (algo == "Elf") {
        return compute_ratios_with_codec<codecs::ElfCodecPolicy>(index_path, dim, original_vectors, max_elements_for_load);
    }
    if (algo == "Camel") {
        return compute_ratios_with_codec<codecs::CamelCodecPolicy>(index_path, dim, original_vectors, max_elements_for_load);
    }
    if (algo == "DeXORPlus") {
        return compute_ratios_with_codec<codecs::DeXORPlusCodecPolicy>(index_path, dim, original_vectors, max_elements_for_load);
    }
    throw std::runtime_error("Unknown algorithm: " + algo);
}

void process_normal_csv(const std::string& csv_path,
                        const std::string& dataset_base_dir,
                        const std::string& index_dir) {
    std::ifstream in(csv_path);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open CSV: " + csv_path);
    }

    std::string header;
    std::getline(in, header);

    std::vector<NormalRow> rows;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> cols = split_csv_line(line);
        if (cols.size() < 6) {
            continue;
        }

        NormalRow row;
        row.dataset = cols[0];
        row.algorithm = cols[1];
        row.build_time = cols[2];
        row.compress_time = cols[3];

        const std::string dataset_path = dataset_base_dir + "/" + row.dataset;
        const std::string index_path = index_dir + "/" + row.dataset + "_" + row.algorithm + "_pq.bin";

        size_t dim = 0;
        const size_t vectors = read_fvecs_num_and_dim(dataset_path, dim);
        const auto ratios = compute_ratios_by_algo(row.algorithm, index_path, dim, vectors, 0);
        row.data_ratio = ratios.first;
        row.index_ratio = ratios.second;

        rows.push_back(row);
    }
    in.close();

    std::ofstream out(csv_path);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot write CSV: " + csv_path);
    }

    out << "Dataset,Algorithm,BuildTime(s),CompressTime(s),DataCompressionRatio,IndexCompressionRatio\n";
    for (const auto& row : rows) {
        out << row.dataset << ","
            << row.algorithm << ","
            << row.build_time << ","
            << row.compress_time << ","
            << std::fixed << std::setprecision(6) << row.data_ratio << ","
            << std::fixed << std::setprecision(6) << row.index_ratio << "\n";
    }
}

void process_bigann_csv(const std::string& csv_path,
                        const std::string& index_dir) {
    std::ifstream in(csv_path);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open CSV: " + csv_path);
    }

    std::string header;
    std::getline(in, header);

    std::vector<BigannRow> rows;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> cols = split_csv_line(line);
        if (cols.size() < 10) {
            continue;
        }

        BigannRow row;
        row.dataset = cols[0];
        row.subset_million = cols[1];
        row.algorithm = cols[2];
        row.vectors = cols[3];
        row.dim = cols[4];
        row.build_time = cols[5];
        row.compress_time = cols[6];
        row.index_file = cols[9];

        const size_t vectors = static_cast<size_t>(std::stoull(row.vectors));
        const size_t dim = static_cast<size_t>(std::stoull(row.dim));
        const std::string index_path = index_dir + "/" + row.index_file;

        const auto ratios = compute_ratios_by_algo(row.algorithm, index_path, dim, vectors, vectors);
        row.data_ratio = ratios.first;
        row.index_ratio = ratios.second;

        rows.push_back(row);
    }
    in.close();

    std::ofstream out(csv_path);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot write CSV: " + csv_path);
    }

    out << "Dataset,SubsetMillion,Algorithm,Vectors,Dim,BuildTime(s),CompressTime(s),DataCompressionRatio,IndexCompressionRatio,IndexFile\n";
    for (const auto& row : rows) {
        out << row.dataset << ","
            << row.subset_million << ","
            << row.algorithm << ","
            << row.vectors << ","
            << row.dim << ","
            << row.build_time << ","
            << row.compress_time << ","
            << std::fixed << std::setprecision(6) << row.data_ratio << ","
            << std::fixed << std::setprecision(6) << row.index_ratio << ","
            << row.index_file << "\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string normal_csv = "script/compressed_hnsw_build_results.csv";
    std::string bigann_csv = "script/bigann_compressed_hnsw_build_results.csv";
    std::string dataset_base_dir = "datasets/hdf5files";
    std::string index_dir = "script";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--normal_csv" && i + 1 < argc) {
            normal_csv = argv[++i];
        } else if (arg == "--bigann_csv" && i + 1 < argc) {
            bigann_csv = argv[++i];
        } else if (arg == "--dataset_base_dir" && i + 1 < argc) {
            dataset_base_dir = argv[++i];
        } else if (arg == "--index_dir" && i + 1 < argc) {
            index_dir = argv[++i];
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << std::endl;
            return -1;
        }
    }

    try {
        process_normal_csv(normal_csv, dataset_base_dir, index_dir);
        process_bigann_csv(bigann_csv, index_dir);
    } catch (const std::exception& e) {
        std::cerr << "Failed to refresh compression ratios: " << e.what() << std::endl;
        return -1;
    }

    std::cout << "Updated compression ratios in: " << normal_csv << std::endl;
    std::cout << "Updated compression ratios in: " << bigann_csv << std::endl;
    return 0;
}
