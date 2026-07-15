#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "../../hnswlib/compressed_hnsw_framework.h"
#include "../../hnswlib/hnswalg.h"
#include "../../hnswlib/hnswalg_simplified_alp_PQ.h"

using namespace hnswlib;

namespace {

template <typename T>
std::vector<T> load_fvecs(const std::string& path, size_t& count, size_t& dim) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot open fvecs file: " + path);
    }

    int32_t file_dim = 0;
    input.read(reinterpret_cast<char*>(&file_dim), sizeof(file_dim));
    if (!input || file_dim <= 0) {
        throw std::runtime_error("Invalid fvecs header: " + path);
    }
    dim = static_cast<size_t>(file_dim);

    input.seekg(0, std::ios::end);
    const size_t bytes = static_cast<size_t>(input.tellg());
    const size_t record_bytes = sizeof(int32_t) + dim * sizeof(float);
    if (bytes % record_bytes != 0) {
        throw std::runtime_error("Unexpected fvecs size: " + path);
    }
    count = bytes / record_bytes;
    input.seekg(0, std::ios::beg);

    std::vector<T> data(count * dim);
    std::vector<float> buffer(dim);
    for (size_t i = 0; i < count; ++i) {
        int32_t record_dim = 0;
        input.read(reinterpret_cast<char*>(&record_dim), sizeof(record_dim));
        input.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(dim * sizeof(float)));
        if (!input || record_dim != file_dim) {
            throw std::runtime_error("Malformed fvecs record in: " + path);
        }
        for (size_t j = 0; j < dim; ++j) {
            data[i * dim + j] = static_cast<T>(buffer[j]);
        }
    }
    return data;
}

template <typename T>
std::unique_ptr<HierarchicalNSW<T>> build_raw_index(
    SpaceInterface<T>* space,
    const std::vector<T>& base,
    size_t count,
    size_t dim,
    size_t M,
    size_t ef_construction) {
    auto index = std::make_unique<HierarchicalNSW<T>>(space, count, M, ef_construction, 100);
    for (size_t i = 0; i < count; ++i) {
        index->addPoint(base.data() + i * dim, static_cast<labeltype>(i));
    }
    return index;
}

template <typename T, typename CodecPolicy>
void copy_raw_index_to_differential(
    const HierarchicalNSW<T>& source,
    HierarchicalNSWCABFRAMEWORK<T, CodecPolicy>& destination) {
    if (source.max_elements_ != destination.max_elements_ ||
        source.M_ != destination.M_ ||
        source.data_size_ != destination.data_size_ ||
        source.size_links_level0_ != destination.size_links_level0_) {
        throw std::runtime_error("Incompatible raw/differential HNSW layouts");
    }

    const size_t count = source.cur_element_count;
    destination.cur_element_count = count;
    destination.num_deleted_ = source.num_deleted_.load();
    destination.ef_ = source.ef_;
    destination.maxlevel_ = source.maxlevel_;
    destination.enterpoint_node_ = source.enterpoint_node_;
    destination.element_levels_ = source.element_levels_;
    destination.label_lookup_ = source.label_lookup_;
    destination.deleted_elements = source.deleted_elements;
    destination.is_compacted_ = false;
    destination.pq_data_.clear();
    destination.pq_centroids_.clear();
    destination.root_state_cache_.clear();
    std::fill(destination.level0_element_start_positions_.begin(),
              destination.level0_element_start_positions_.end(), 0);
    std::fill(destination.data_level0_memory_.begin(), destination.data_level0_memory_.end(), 0);
    std::memset(destination.linkLists_, 0, sizeof(char*) * destination.max_elements_);

    for (tableint id = 0; id < count; ++id) {
        char* dst_record = destination.data_level0_memory_.data() +
                           static_cast<size_t>(id) * destination.size_data_per_element_;
        std::memcpy(dst_record + destination.offsetLevel0_, source.get_linklist0(id),
                    source.size_links_level0_);

        const tableint no_predecessor = static_cast<tableint>(-1);
        std::memcpy(dst_record + destination.prenode_offset_, &no_predecessor,
                    sizeof(no_predecessor));
        const labeltype label = source.getExternalLabel(id);
        std::memcpy(dst_record + destination.label_offset_, &label, sizeof(label));
        std::memcpy(dst_record + destination.offsetData_, source.getDataByInternalId(id),
                    source.data_size_);

        if (source.element_levels_[id] > 0) {
            const size_t bytes = static_cast<size_t>(source.element_levels_[id]) *
                                 source.size_links_per_element_;
            destination.linkLists_[id] = static_cast<char*>(std::malloc(bytes));
            if (destination.linkLists_[id] == nullptr) {
                throw std::bad_alloc();
            }
            std::memcpy(destination.linkLists_[id], source.linkLists_[id], bytes);
        }
    }
}

template <typename T>
void copy_raw_index_to_alp(
    const HierarchicalNSW<T>& source,
    HierarchicalNSWALPSIMPLIFIEDPQ<T>& destination) {
    if (source.max_elements_ != destination.max_elements_ ||
        source.M_ != destination.M_ ||
        source.size_data_per_element_ != destination.size_data_per_element_ ||
        source.size_links_per_element_ != destination.size_links_per_element_) {
        throw std::runtime_error("Incompatible raw/ALP HNSW layouts");
    }

    const size_t count = source.cur_element_count;
    destination.cur_element_count = count;
    destination.num_deleted_ = source.num_deleted_.load();
    destination.ef_ = source.ef_;
    destination.maxlevel_ = source.maxlevel_;
    destination.enterpoint_node_ = source.enterpoint_node_;
    destination.element_levels_ = source.element_levels_;
    destination.label_lookup_ = source.label_lookup_;
    destination.deleted_elements = source.deleted_elements;
    destination.is_compacted_ = false;
    destination.pq_data_.clear();
    destination.pq_centroids_.clear();
    destination.uncompressed_mask_.clear();
    std::fill(destination.level0_element_start_positions_.begin(),
              destination.level0_element_start_positions_.end(), 0);
    std::memcpy(destination.data_level0_memory_.data(), source.data_level0_memory_,
                count * source.size_data_per_element_);
    std::memset(destination.linkLists_, 0, sizeof(char*) * destination.max_elements_);

    for (tableint id = 0; id < count; ++id) {
        if (source.element_levels_[id] > 0) {
            const size_t bytes = static_cast<size_t>(source.element_levels_[id]) *
                                 source.size_links_per_element_;
            destination.linkLists_[id] = static_cast<char*>(std::malloc(bytes));
            if (destination.linkLists_[id] == nullptr) {
                throw std::bad_alloc();
            }
            std::memcpy(destination.linkLists_[id], source.linkLists_[id], bytes);
        }
    }
}

struct PayloadStats {
    size_t vectors_checked = 0;
    size_t coordinates_checked = 0;
    size_t vector_mismatches = 0;
    size_t coordinate_mismatches = 0;
    size_t first_vector = std::numeric_limits<size_t>::max();
    size_t first_coordinate = std::numeric_limits<size_t>::max();
};

template <typename T, typename Index>
PayloadStats validate_payloads(const Index& index, const std::vector<T>& original,
                               size_t count, size_t dim) {
    PayloadStats stats;
    for (size_t id = 0; id < count; ++id) {
        const auto decoded = index.getOriginalDataByInternalId(static_cast<tableint>(id));
        ++stats.vectors_checked;
        stats.coordinates_checked += dim;
        if (decoded.size() != dim) {
            ++stats.vector_mismatches;
            stats.coordinate_mismatches += dim;
            if (stats.first_vector == std::numeric_limits<size_t>::max()) {
                stats.first_vector = id;
                stats.first_coordinate = 0;
            }
            continue;
        }

        bool vector_mismatch = false;
        for (size_t coord = 0; coord < dim; ++coord) {
            if (std::memcmp(&original[id * dim + coord], &decoded[coord], sizeof(T)) != 0) {
                ++stats.coordinate_mismatches;
                vector_mismatch = true;
                if (stats.first_vector == std::numeric_limits<size_t>::max()) {
                    stats.first_vector = id;
                    stats.first_coordinate = coord;
                }
            }
        }
        if (vector_mismatch) {
            ++stats.vector_mismatches;
        }
    }
    return stats;
}

template <typename Queue>
std::vector<labeltype> ordered_labels(Queue queue) {
    std::vector<labeltype> labels(queue.size());
    for (size_t i = queue.size(); i > 0; --i) {
        labels[i - 1] = queue.top().second;
        queue.pop();
    }
    return labels;
}

struct SearchStats {
    size_t queries = 0;
    size_t ordered_matches = 0;
    size_t set_matches = 0;
    size_t first_mismatch_query = std::numeric_limits<size_t>::max();
    size_t first_mismatch_rank = std::numeric_limits<size_t>::max();
};

template <typename T, typename RawIndex, typename CompressedIndex>
SearchStats validate_search_equivalence(
    RawIndex& raw,
    CompressedIndex& compressed,
    const std::vector<T>& queries,
    size_t query_count,
    size_t dim,
    size_t ef,
    size_t k) {
    raw.setEf(ef);
    compressed.setEf(ef);

    SearchStats stats;
    for (size_t query_id = 0; query_id < query_count; ++query_id) {
        const T* query = queries.data() + query_id * dim;
        const auto raw_labels = ordered_labels(raw.searchKnn(query, k));
        const auto compressed_labels = ordered_labels(compressed.searchKnn(query, k));

        bool ordered_match = raw_labels == compressed_labels;
        auto raw_set = raw_labels;
        auto compressed_set = compressed_labels;
        std::sort(raw_set.begin(), raw_set.end());
        std::sort(compressed_set.begin(), compressed_set.end());

        ++stats.queries;
        if (ordered_match) {
            ++stats.ordered_matches;
        }
        if (raw_set == compressed_set) {
            ++stats.set_matches;
        }
        if (!ordered_match && stats.first_mismatch_query == std::numeric_limits<size_t>::max()) {
            stats.first_mismatch_query = query_id;
            const size_t shared = std::min(raw_labels.size(), compressed_labels.size());
            size_t rank = 0;
            while (rank < shared && raw_labels[rank] == compressed_labels[rank]) {
                ++rank;
            }
            stats.first_mismatch_rank = rank;
        }
    }
    return stats;
}

std::string csv_index(size_t value) {
    return value == std::numeric_limits<size_t>::max() ? "-1" : std::to_string(value);
}

struct Config {
    std::string base_dir = "../datasets/hdf5files/";
    std::string output_prefix = "exact_lvc_correctness";
    std::vector<std::string> datasets = {"sift-128-euclidean", "gist-960-euclidean"};
    std::vector<std::string> differential_codecs = {"DeXOR", "Gorilla", "Camel", "Elf"};
    std::vector<size_t> efs = {80};
    std::vector<size_t> ks = {1, 10};
    double cache_ratio = 0.01;
    int chain_max = 2;
    size_t M = 16;
    size_t ef_construction = 200;
    size_t max_queries = 0;
};

template <typename T>
void write_payload_row(std::ofstream& csv, const std::string& dataset, const std::string& backend,
                       const std::string& scalar_type, const Config& config,
                       const PayloadStats& stats) {
    csv << dataset << ',' << backend << ',' << scalar_type << ',' << config.chain_max << ','
        << config.cache_ratio << ',' << stats.vectors_checked << ',' << stats.coordinates_checked << ','
        << stats.vector_mismatches << ',' << stats.coordinate_mismatches << ','
        << csv_index(stats.first_vector) << ',' << csv_index(stats.first_coordinate) << '\n';
    csv.flush();
}

void write_search_row(std::ofstream& csv, const std::string& dataset, const std::string& backend,
                      const std::string& scalar_type, const Config& config,
                      size_t ef, size_t k, const SearchStats& stats) {
    csv << dataset << ',' << backend << ',' << scalar_type << ',' << config.chain_max << ','
        << config.cache_ratio << ',' << ef << ',' << k << ',' << stats.queries << ','
        << stats.ordered_matches << ',' << (stats.queries - stats.ordered_matches) << ','
        << stats.set_matches << ',' << (stats.queries - stats.set_matches) << ','
        << csv_index(stats.first_mismatch_query) << ',' << csv_index(stats.first_mismatch_rank) << '\n';
    csv.flush();
}

template <typename CodecPolicy>
void run_differential_codec(const std::string& codec_name,
                            const std::string& dataset,
                            const Config& config,
                            HierarchicalNSW<double>& raw,
                            L2SpaceDouble& space,
                            const std::vector<double>& base,
                            const std::vector<double>& queries,
                            size_t dim,
                            std::ofstream& payload_csv,
                            std::ofstream& search_csv) {
    const size_t cache_size = static_cast<size_t>(raw.cur_element_count * config.cache_ratio);
    HierarchicalNSWCABFRAMEWORK<double, CodecPolicy> compressed(
        &space, raw.max_elements_, config.M, config.ef_construction, true, cache_size, 100, false);
    copy_raw_index_to_differential(raw, compressed);
    compressed.setUseTls(false);
    compressed.compress_dataset(config.chain_max);
    if (cache_size > 0) {
        compressed.loadCacheByLevel();
    }

    const PayloadStats payload = validate_payloads<double>(
        compressed, base, raw.cur_element_count, dim);
    write_payload_row<double>(payload_csv, dataset, codec_name, "float64", config, payload);

    const size_t query_count = config.max_queries == 0
        ? queries.size() / dim
        : std::min(config.max_queries, queries.size() / dim);
    for (size_t ef : config.efs) {
        for (size_t k : config.ks) {
            const SearchStats search = validate_search_equivalence(
                raw, compressed, queries, query_count, dim, ef, k);
            write_search_row(search_csv, dataset, codec_name, "float64", config, ef, k, search);
        }
    }
}

void run_alp(const std::string& dataset,
             const Config& config,
             HierarchicalNSW<float>& raw,
             L2Space& space,
             const std::vector<float>& base,
             const std::vector<float>& queries,
             size_t dim,
             std::ofstream& payload_csv,
             std::ofstream& search_csv) {
    const size_t cache_size = static_cast<size_t>(raw.cur_element_count * config.cache_ratio);
    HierarchicalNSWALPSIMPLIFIEDPQ<float> compressed(
        &space, raw.max_elements_, config.M, config.ef_construction, 100, false);
    copy_raw_index_to_alp(raw, compressed);
    compressed.setUseTLS(false);
    compressed.compress_dataset(cache_size);

    const PayloadStats payload = validate_payloads<float>(
        compressed, base, raw.cur_element_count, dim);
    write_payload_row<float>(payload_csv, dataset, "ALP", "float32", config, payload);

    const size_t query_count = config.max_queries == 0
        ? queries.size() / dim
        : std::min(config.max_queries, queries.size() / dim);
    for (size_t ef : config.efs) {
        for (size_t k : config.ks) {
            const SearchStats search = validate_search_equivalence(
                raw, compressed, queries, query_count, dim, ef, k);
            write_search_row(search_csv, dataset, "ALP", "float32", config, ef, k, search);
        }
    }
}

template <typename ValueParser>
void parse_values(int argc, char** argv, int& index, std::vector<typename std::invoke_result<ValueParser, const std::string&>::type>& output,
                  ValueParser parser) {
    output.clear();
    while (index + 1 < argc && std::string(argv[index + 1]).rfind("--", 0) != 0) {
        output.push_back(parser(argv[++index]));
    }
    if (output.empty()) {
        throw std::runtime_error("Expected at least one value after " + std::string(argv[index]));
    }
}

Config parse_config(int argc, char** argv) {
    Config config;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--base_dir" && i + 1 < argc) {
            config.base_dir = argv[++i];
        } else if (arg == "--output_prefix" && i + 1 < argc) {
            config.output_prefix = argv[++i];
        } else if (arg == "--dataset") {
            parse_values(argc, argv, i, config.datasets, [](const std::string& value) { return value; });
        } else if (arg == "--algorithm") {
            parse_values(argc, argv, i, config.differential_codecs,
                         [](const std::string& value) { return value; });
        } else if (arg == "--ef") {
            parse_values(argc, argv, i, config.efs,
                         [](const std::string& value) { return static_cast<size_t>(std::stoull(value)); });
        } else if (arg == "--k") {
            parse_values(argc, argv, i, config.ks,
                         [](const std::string& value) { return static_cast<size_t>(std::stoull(value)); });
        } else if (arg == "--cache_ratio" && i + 1 < argc) {
            config.cache_ratio = std::stod(argv[++i]);
        } else if (arg == "--chain_max" && i + 1 < argc) {
            config.chain_max = std::stoi(argv[++i]);
        } else if (arg == "--M" && i + 1 < argc) {
            config.M = static_cast<size_t>(std::stoull(argv[++i]));
        } else if (arg == "--ef_construction" && i + 1 < argc) {
            config.ef_construction = static_cast<size_t>(std::stoull(argv[++i]));
        } else if (arg == "--max_queries" && i + 1 < argc) {
            config.max_queries = static_cast<size_t>(std::stoull(argv[++i]));
        } else {
            throw std::runtime_error("Unknown or incomplete argument: " + arg);
        }
    }
    if (config.base_dir.back() != '/' && config.base_dir.back() != '\\') {
        config.base_dir.push_back('/');
    }
    if (config.cache_ratio < 0.0 || config.cache_ratio > 1.0) {
        throw std::runtime_error("--cache_ratio must be in [0, 1]");
    }
    return config;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Config config = parse_config(argc, argv);
        std::ofstream payload_csv(config.output_prefix + "_payload.csv");
        std::ofstream search_csv(config.output_prefix + "_search.csv");
        if (!payload_csv || !search_csv) {
            throw std::runtime_error("Cannot open correctness output CSV files");
        }
        payload_csv << "Dataset,Backend,ScalarType,ChainMaxLength,HubCacheRatio,VectorsChecked,CoordinatesChecked,"
                    << "VectorMismatches,CoordinateMismatches,FirstMismatchVector,FirstMismatchCoordinate\n";
        search_csv << "Dataset,Backend,ScalarType,ChainMaxLength,HubCacheRatio,EF,K,Queries,"
                   << "OrderedTopKMatches,OrderedTopKMismatches,SetTopKMatches,SetTopKMismatches,"
                   << "FirstMismatchQuery,FirstMismatchRank\n";

        for (const std::string& dataset : config.datasets) {
            size_t base_count = 0, base_dim = 0, query_count = 0, query_dim = 0;
            const auto base_double = load_fvecs<double>(config.base_dir + dataset + "_train.fvecs", base_count, base_dim);
            const auto queries_double = load_fvecs<double>(config.base_dir + dataset + "_test.fvecs", query_count, query_dim);
            if (base_dim != query_dim) {
                throw std::runtime_error("Base/query dimension mismatch for " + dataset);
            }

            std::cout << "[Exact-LVC] Building raw float64 HNSW for " << dataset
                      << " (vectors=" << base_count << ", dim=" << base_dim << ")" << std::endl;
            L2SpaceDouble double_space(base_dim);
            auto raw_double = build_raw_index<double>(
                &double_space, base_double, base_count, base_dim, config.M, config.ef_construction);

            for (const std::string& codec : config.differential_codecs) {
                std::cout << "  validating " << codec << " (TLS off)" << std::endl;
                if (codec == "DeXOR") {
                    run_differential_codec<codecs::DeXORCodecPolicy>(codec, dataset, config, *raw_double,
                        double_space, base_double, queries_double, base_dim, payload_csv, search_csv);
                } else if (codec == "Gorilla") {
                    run_differential_codec<codecs::GorillaCodecPolicy>(codec, dataset, config, *raw_double,
                        double_space, base_double, queries_double, base_dim, payload_csv, search_csv);
                } else if (codec == "Camel") {
                    run_differential_codec<codecs::CamelCodecPolicy>(codec, dataset, config, *raw_double,
                        double_space, base_double, queries_double, base_dim, payload_csv, search_csv);
                } else if (codec == "Elf") {
                    run_differential_codec<codecs::ElfCodecPolicy>(codec, dataset, config, *raw_double,
                        double_space, base_double, queries_double, base_dim, payload_csv, search_csv);
                } else {
                    throw std::runtime_error("Unsupported differential codec: " + codec);
                }
            }

            size_t base_float_count = 0, base_float_dim = 0, query_float_count = 0, query_float_dim = 0;
            const auto base_float = load_fvecs<float>(config.base_dir + dataset + "_train.fvecs", base_float_count, base_float_dim);
            const auto queries_float = load_fvecs<float>(config.base_dir + dataset + "_test.fvecs", query_float_count, query_float_dim);
            std::cout << "  validating ALP (TLS off; payload check is reported, not used as a gate)" << std::endl;
            L2Space float_space(base_float_dim);
            auto raw_float = build_raw_index<float>(
                &float_space, base_float, base_float_count, base_float_dim, config.M, config.ef_construction);
            run_alp(dataset, config, *raw_float, float_space, base_float, queries_float,
                    base_float_dim, payload_csv, search_csv);
        }

        std::cout << "Payload validation: " << config.output_prefix << "_payload.csv\n"
                  << "Search equivalence: " << config.output_prefix << "_search.csv" << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Exact-LVC correctness test failed: " << error.what() << std::endl;
        return 1;
    }
}
