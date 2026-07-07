#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/stat.h>

#include <faiss/IndexFlatCodes.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexPQ.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/index_io.h>

#include "../../hnswlib/hnswalg.h"
#include "../../hnswlib/hnswalg_simplified_alp_PQ.h"
#include "../../hnswlib/compressed_hnsw_framework.h"
#include "../../hnswlib/hnswlib.h"

using namespace hnswlib;

struct ComponentRow {
    std::string dataset;
    std::string index_type;
    std::string actual_variant;
    std::string index_file;
    std::string component;
    uint64_t bytes = 0;
    uint64_t file_size_bytes = 0;
    size_t ntotal = 0;
    size_t dim = 0;
    std::string note;
};

struct LvcCandidateRow {
    std::string dataset;
    std::string variant;
    std::string index_file;
    uint64_t bytes = 0;
    bool selected = false;
};

static std::string join_path(const std::string& dir, const std::string& name) {
    if (dir.empty() || dir == ".") return name;
    const char back = dir.back();
    if (back == '/' || back == '\\') return dir + name;
    return dir + "/" + name;
}

static bool exists_file(const std::string& path) {
    struct stat st {};
    return stat(path.c_str(), &st) == 0 && ((st.st_mode & S_IFMT) == S_IFREG);
}

static uint64_t file_size_bytes(const std::string& path) {
    struct stat st {};
    if (stat(path.c_str(), &st) != 0) return 0;
    return static_cast<uint64_t>(st.st_size);
}

static std::string filename_only(const std::string& path) {
    const size_t pos = path.find_last_of("/\\");
    return pos == std::string::npos ? path : path.substr(pos + 1);
}

static std::string stem_only(const std::string& path) {
    std::string name = filename_only(path);
    const size_t pos = name.find_last_of('.');
    return pos == std::string::npos ? name : name.substr(0, pos);
}

static size_t infer_dim_from_dataset(const std::string& dataset) {
    const std::string marker = "-";
    for (size_t i = 0; i < dataset.size(); ++i) {
        if (!std::isdigit(static_cast<unsigned char>(dataset[i]))) {
            continue;
        }
        size_t j = i;
        while (j < dataset.size() && std::isdigit(static_cast<unsigned char>(dataset[j]))) {
            ++j;
        }
        if (j < dataset.size() && dataset[j] == '-') {
            return static_cast<size_t>(std::stoull(dataset.substr(i, j - i)));
        }
        i = j;
    }
    throw std::runtime_error("Cannot infer dimension from dataset name: " + dataset);
}

static size_t read_fvecs_dim_or_infer(const std::string& base_dir, const std::string& dataset) {
    std::string train_file = join_path(base_dir, dataset + "_train.fvecs");
    std::ifstream input(train_file, std::ios::binary);
    if (input) {
        int32_t dim = 0;
        input.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        if (input.good() && dim > 0) {
            return static_cast<size_t>(dim);
        }
    }
    return infer_dim_from_dataset(dataset);
}

static std::string first_existing(const std::string& index_dir,
                                  const std::vector<std::string>& names) {
    for (const auto& name : names) {
        std::string path = join_path(index_dir, name);
        if (exists_file(path)) {
            return path;
        }
    }
    return "";
}

static void add_row(std::vector<ComponentRow>& rows,
                    const std::string& dataset,
                    const std::string& index_type,
                    const std::string& actual_variant,
                    const std::string& path,
                    const std::string& component,
                    uint64_t bytes,
                    uint64_t file_size,
                    size_t ntotal,
                    size_t dim,
                    const std::string& note = "") {
    ComponentRow row;
    row.dataset = dataset;
    row.index_type = index_type;
    row.actual_variant = actual_variant;
    row.index_file = filename_only(path);
    row.component = component;
    row.bytes = bytes;
    row.file_size_bytes = file_size;
    row.ntotal = ntotal;
    row.dim = dim;
    row.note = note;
    rows.push_back(std::move(row));
}

static uint64_t upper_links_bytes(size_t cur_element_count,
                                  const std::vector<int>& element_levels,
                                  size_t size_links_per_element) {
    uint64_t total = 0;
    for (size_t i = 0; i < cur_element_count; ++i) {
        if (element_levels[i] > 0) {
            total += static_cast<uint64_t>(size_links_per_element) *
                     static_cast<uint64_t>(element_levels[i]);
        }
    }
    return total;
}

static uint64_t upper_link_size_headers_bytes(size_t cur_element_count) {
    return static_cast<uint64_t>(cur_element_count) * sizeof(unsigned int);
}

static void add_residual_as_metadata(std::vector<ComponentRow>& rows,
                                     const std::string& dataset,
                                     const std::string& index_type,
                                     const std::string& actual_variant,
                                     const std::string& path,
                                     uint64_t file_size,
                                     uint64_t serialized_subtotal,
                                     size_t ntotal,
                                     size_t dim) {
    if (file_size > serialized_subtotal) {
        add_row(rows, dataset, index_type, actual_variant, path,
                "Original Metadata", file_size - serialized_subtotal,
                file_size, ntotal, dim, "serialized headers and small fields");
    }
}

static void profile_hnswlib_hnsw(const std::string& dataset,
                                 const std::string& path,
                                 size_t dim,
                                 std::vector<ComponentRow>& rows) {
    L2Space l2(dim);
    HierarchicalNSW<float> index(&l2, path, false);
    const size_t n = index.cur_element_count.load();
    const uint64_t file_size = file_size_bytes(path);

    const uint64_t vector_payload = static_cast<uint64_t>(n) * index.data_size_;
    const uint64_t graph =
        static_cast<uint64_t>(n) * index.size_links_level0_ +
        upper_links_bytes(n, index.element_levels_, index.size_links_per_element_) +
        upper_link_size_headers_bytes(n);
    const uint64_t metadata = static_cast<uint64_t>(n) * sizeof(labeltype);

    uint64_t subtotal = vector_payload + graph + metadata;
    add_row(rows, dataset, "HNSW", "hnswlib-float", path,
            "Vector Payloads", vector_payload, file_size, n, dim);
    add_row(rows, dataset, "HNSW", "hnswlib-float", path,
            "Graph Structure", graph, file_size, n, dim,
            "level-0 links, upper-level links, and serialized upper-link size fields");
    add_row(rows, dataset, "HNSW", "hnswlib-float", path,
            "Original Metadata", metadata, file_size, n, dim, "labels");
    add_residual_as_metadata(rows, dataset, "HNSW", "hnswlib-float",
                             path, file_size, subtotal, n, dim);
}

static uint64_t faiss_hnsw_metadata_bytes(const faiss::HNSW& hnsw) {
    return static_cast<uint64_t>(hnsw.assign_probas.size()) * sizeof(double) +
           static_cast<uint64_t>(hnsw.cum_nneighbor_per_level.size()) * sizeof(int) +
           static_cast<uint64_t>(hnsw.levels.size()) * sizeof(int) +
           static_cast<uint64_t>(hnsw.offsets.size()) * sizeof(size_t) +
           sizeof(hnsw.entry_point) + sizeof(hnsw.max_level) +
           sizeof(hnsw.efConstruction) + sizeof(hnsw.efSearch);
}

static void profile_faiss_hnsw_quantized(const std::string& dataset,
                                         const std::string& index_type,
                                         const std::string& actual_variant,
                                         const std::string& path,
                                         size_t dim,
                                         std::vector<ComponentRow>& rows) {
    std::unique_ptr<faiss::Index> base(faiss::read_index(path.c_str()));
    auto* hnsw = dynamic_cast<faiss::IndexHNSW*>(base.get());
    if (!hnsw) {
        throw std::runtime_error("Loaded Faiss index is not IndexHNSW: " + path);
    }
    auto* flat_codes = dynamic_cast<faiss::IndexFlatCodes*>(hnsw->storage);
    if (!flat_codes) {
        throw std::runtime_error("Faiss HNSW storage is not IndexFlatCodes: " + path);
    }

    const size_t n = static_cast<size_t>(base->ntotal);
    const uint64_t file_size = file_size_bytes(path);
    const uint64_t codes = static_cast<uint64_t>(flat_codes->codes.size()) * sizeof(uint8_t);
    const uint64_t graph =
        static_cast<uint64_t>(hnsw->hnsw.neighbors.size()) *
        sizeof(faiss::HNSW::storage_idx_t);
    const uint64_t metadata = faiss_hnsw_metadata_bytes(hnsw->hnsw);

    uint64_t codebook = 0;
    if (auto* pq = dynamic_cast<faiss::IndexPQ*>(hnsw->storage)) {
        codebook += static_cast<uint64_t>(pq->pq.centroids.size()) * sizeof(float);
    } else if (auto* sq = dynamic_cast<faiss::IndexScalarQuantizer*>(hnsw->storage)) {
        codebook += static_cast<uint64_t>(sq->sq.trained.size()) * sizeof(float);
        codebook += sizeof(sq->sq.qtype) + sizeof(sq->sq.rangestat) +
                    sizeof(sq->sq.rangestat_arg) + sizeof(sq->sq.d) +
                    sizeof(sq->sq.code_size) + sizeof(sq->sq.bits);
    }

    uint64_t subtotal = codes + graph + metadata + codebook;
    add_row(rows, dataset, index_type, actual_variant, path,
            "Quantized Vector Codes", codes, file_size, n, dim);
    add_row(rows, dataset, index_type, actual_variant, path,
            "Graph Structure", graph, file_size, n, dim, "Faiss HNSW neighbors array");
    add_row(rows, dataset, index_type, actual_variant, path,
            "Original Metadata", metadata, file_size, n, dim,
            "Faiss HNSW levels, offsets, probabilities, and scalar fields");
    add_row(rows, dataset, index_type, actual_variant, path,
            "Quantizer Codebook", codebook, file_size, n, dim,
            "PQ centroids or SQ trained ranges");
    add_residual_as_metadata(rows, dataset, index_type, actual_variant,
                             path, file_size, subtotal, n, dim);
}

template <typename IndexT>
static uint64_t compact_record_end(const IndexT& index, tableint i) {
    if (i + 1 < index.cur_element_count.load() &&
        index.level0_element_start_positions_[i + 1] > 0) {
        return index.level0_element_start_positions_[i + 1];
    }
    return index.data_level0_memory_.size();
}

template <typename CodecPolicy>
static void profile_differential_lvc_impl(const std::string& dataset,
                                          const std::string& actual_variant,
                                          const std::string& path,
                                          size_t dim,
                                          size_t cache_ratio_percent,
                                          std::vector<ComponentRow>& rows) {
    L2SpaceDouble l2(dim);

    // Load with a 1% root/hub state cache by default. The constructor invokes
    // loadCacheByLevel() when cache_max_size_ is positive.
    size_t provisional_cache_size = 1;
    {
        HierarchicalNSWCABFRAMEWORK<double, CodecPolicy> probe(&l2, path, true, 0);
        provisional_cache_size = std::max<size_t>(
            1, probe.cur_element_count.load() * cache_ratio_percent / 100);
    }
    HierarchicalNSWCABFRAMEWORK<double, CodecPolicy> index(
        &l2, path, true, provisional_cache_size);

    const size_t n = index.cur_element_count.load();
    const uint64_t file_size = file_size_bytes(path);
    uint64_t level0_graph = 0;
    uint64_t vector_payload = 0;

    for (tableint i = 0; i < n; ++i) {
        const size_t start = index.level0_element_start_positions_[i];
        const size_t linklist_size_offset = sizeof(tableint) + sizeof(labeltype);
        const auto degree = *reinterpret_cast<const unsigned short int*>(
            index.data_level0_memory_.data() + start + linklist_size_offset);
        const size_t data_offset =
            linklist_size_offset + sizeof(linklistsizeint) +
            static_cast<size_t>(degree) * sizeof(tableint);
        const size_t payload_start = start + data_offset;
        const size_t end = compact_record_end(index, i);
        level0_graph += sizeof(linklistsizeint) +
                        static_cast<uint64_t>(degree) * sizeof(tableint);
        if (end > payload_start) {
            vector_payload += end - payload_start;
        }
    }

    const uint64_t upper_graph =
        upper_links_bytes(n, index.element_levels_, index.size_links_per_element_) +
        upper_link_size_headers_bytes(n);
    const uint64_t graph = level0_graph + upper_graph;
    const uint64_t original_metadata = static_cast<uint64_t>(n) * sizeof(labeltype);
    const uint64_t compression_metadata =
        static_cast<uint64_t>(n) * sizeof(size_t) +
        static_cast<uint64_t>(n) * sizeof(tableint);

    uint64_t codebook = static_cast<uint64_t>(index.pq_data_.size()) * sizeof(uint8_t);
    for (const auto& c : index.pq_centroids_) {
        codebook += static_cast<uint64_t>(c.size()) * sizeof(double);
    }

    uint64_t cache = 0;
    for (const auto& entry : index.root_state_cache_) {
        cache += sizeof(tableint);
        cache += sizeof(std::vector<typename CodecPolicy::StateType>);
        cache += static_cast<uint64_t>(entry.second.capacity()) *
                 sizeof(typename CodecPolicy::StateType);
    }

    const uint64_t serialized_subtotal =
        vector_payload + graph + original_metadata + compression_metadata + codebook;
    add_row(rows, dataset, "LVC", actual_variant, path,
            "Vector Payloads", vector_payload, file_size, n, dim,
            "compressed differential payloads");
    add_row(rows, dataset, "LVC", actual_variant, path,
            "Graph Structure", graph, file_size, n, dim,
            "compacted level-0 links and upper-level links");
    add_row(rows, dataset, "LVC", actual_variant, path,
            "Original Metadata", original_metadata, file_size, n, dim, "labels");
    add_row(rows, dataset, "LVC", actual_variant, path,
            "Compression Metadata", compression_metadata, file_size, n, dim,
            "level0 start positions and prenode ids");
    add_row(rows, dataset, "LVC", actual_variant, path,
            "Lightweight Codebook", codebook, file_size, n, dim,
            "two-level filtering PQ data and centroids");
    add_row(rows, dataset, "LVC", actual_variant, path,
            "Cache", cache, file_size, n, dim, "runtime 1% root/hub state cache");
    add_residual_as_metadata(rows, dataset, "LVC", actual_variant,
                             path, file_size, serialized_subtotal, n, dim);
}

static std::string lvc_variant_from_path(const std::string& path) {
    const std::string name = filename_only(path);
    if (name.find("hnswalp") != std::string::npos) return "ALP";
    if (name.find("DeXOR") != std::string::npos || name.find("hnswdexor") != std::string::npos) return "DeXOR";
    if (name.find("Gorilla") != std::string::npos) return "Gorilla";
    if (name.find("Elf") != std::string::npos) return "Elf";
    if (name.find("Camel") != std::string::npos) return "Camel";
    return "Unknown";
}

static void profile_alp_lvc(const std::string& dataset,
                            const std::string& actual_variant,
                            const std::string& path,
                            size_t dim,
                            std::vector<ComponentRow>& rows) {
    L2Space l2(dim);
    HierarchicalNSWALPSIMPLIFIEDPQ<float> index(&l2, path, false);
    const size_t n = index.cur_element_count.load();
    const uint64_t file_size = file_size_bytes(path);
    uint64_t level0_graph = 0;
    uint64_t vector_payload = 0;
    uint64_t cache = 0;

    for (tableint i = 0; i < n; ++i) {
        const size_t start = index.level0_element_start_positions_[i];
        const size_t linklist_size_offset = sizeof(labeltype);
        const auto degree = *reinterpret_cast<const unsigned short int*>(
            index.data_level0_memory_.data() + start + linklist_size_offset);
        const size_t data_offset =
            linklist_size_offset + sizeof(linklistsizeint) +
            static_cast<size_t>(degree) * sizeof(tableint);
        const size_t payload_start = start + data_offset;
        const size_t end = compact_record_end(index, i);
        const uint64_t payload_bytes = end > payload_start ? end - payload_start : 0;

        level0_graph += sizeof(linklistsizeint) +
                        static_cast<uint64_t>(degree) * sizeof(tableint);
        if (i < index.uncompressed_mask_.size() && index.uncompressed_mask_[i]) {
            cache += payload_bytes;
        } else {
            vector_payload += payload_bytes;
        }
    }

    const uint64_t upper_graph =
        upper_links_bytes(n, index.element_levels_, index.size_links_per_element_) +
        upper_link_size_headers_bytes(n);
    const uint64_t graph = level0_graph + upper_graph;
    const uint64_t original_metadata = static_cast<uint64_t>(n) * sizeof(labeltype);
    const uint64_t compression_metadata =
        static_cast<uint64_t>(index.level0_element_start_positions_.size()) * sizeof(size_t) +
        static_cast<uint64_t>(index.uncompressed_mask_.size()) * sizeof(uint8_t);

    uint64_t codebook = static_cast<uint64_t>(index.pq_data_.size()) * sizeof(uint8_t);
    for (const auto& c : index.pq_centroids_) {
        codebook += static_cast<uint64_t>(c.size()) * sizeof(float);
    }

    const uint64_t serialized_subtotal =
        vector_payload + graph + original_metadata + compression_metadata + codebook + cache;
    add_row(rows, dataset, "LVC", actual_variant, path,
            "Vector Payloads", vector_payload, file_size, n, dim,
            "compressed scaling-based payloads");
    add_row(rows, dataset, "LVC", actual_variant, path,
            "Graph Structure", graph, file_size, n, dim,
            "compacted level-0 links and upper-level links");
    add_row(rows, dataset, "LVC", actual_variant, path,
            "Original Metadata", original_metadata, file_size, n, dim, "labels");
    add_row(rows, dataset, "LVC", actual_variant, path,
            "Compression Metadata", compression_metadata, file_size, n, dim,
            "level0 start positions and uncompressed mask");
    add_row(rows, dataset, "LVC", actual_variant, path,
            "Lightweight Codebook", codebook, file_size, n, dim,
            "two-level filtering PQ data and centroids");
    add_row(rows, dataset, "LVC", actual_variant, path,
            "Cache", cache, file_size, n, dim,
            "uncompressed vectors stored in the compacted index");
    add_residual_as_metadata(rows, dataset, "LVC", actual_variant,
                             path, file_size, serialized_subtotal, n, dim);
}

static void profile_differential_lvc(const std::string& dataset,
                                     const std::string& actual_variant,
                                     const std::string& path,
                                     size_t dim,
                                     size_t cache_ratio_percent,
                                     std::vector<ComponentRow>& rows) {
    if (actual_variant == "DeXOR") {
        profile_differential_lvc_impl<codecs::DeXORCodecPolicy>(
            dataset, actual_variant, path, dim, cache_ratio_percent, rows);
    } else if (actual_variant == "Gorilla") {
        profile_differential_lvc_impl<codecs::GorillaCodecPolicy>(
            dataset, actual_variant, path, dim, cache_ratio_percent, rows);
    } else if (actual_variant == "Elf") {
        profile_differential_lvc_impl<codecs::ElfCodecPolicy>(
            dataset, actual_variant, path, dim, cache_ratio_percent, rows);
    } else if (actual_variant == "Camel") {
        profile_differential_lvc_impl<codecs::CamelCodecPolicy>(
            dataset, actual_variant, path, dim, cache_ratio_percent, rows);
    } else {
        throw std::runtime_error("Unsupported LVC codec variant: " + actual_variant);
    }
}

static std::vector<std::string> lvc_candidate_names(const std::string& dataset) {
    std::vector<std::string> names;
    names.push_back(dataset + "_train.fvecs_hnswalp_simplified_pq.bin");
    names.push_back(dataset + "_hnswalp_pq.bin");
    const std::vector<std::string> algos = {"DeXOR", "Gorilla", "Elf", "Camel"};
    const std::vector<std::string> chain_tags = {
        "ch2", "ch3", "ch4", "ch5", "ch6", "ch7", "ch8", "chunlim"
    };
    for (const auto& algo : algos) {
        for (const auto& chain : chain_tags) {
            names.push_back(dataset + "_" + algo + "_" + chain + "_pq.bin");
        }
        names.push_back(dataset + "_train.fvecs_" + algo + "_pq.bin");
    }
    names.push_back(dataset + "_hnswdexor_pq.bin");
    return names;
}

static std::vector<LvcCandidateRow> collect_lvc_candidates(
        const std::string& dataset,
        const std::string& index_dir) {
    std::vector<LvcCandidateRow> candidates;
    for (const auto& name : lvc_candidate_names(dataset)) {
        std::string path = join_path(index_dir, name);
        if (!exists_file(path)) continue;
        LvcCandidateRow row;
        row.dataset = dataset;
        row.variant = lvc_variant_from_path(path);
        row.index_file = filename_only(path);
        row.bytes = file_size_bytes(path);
        if (row.variant != "Unknown") {
            candidates.push_back(std::move(row));
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        return a.bytes < b.bytes;
    });
    if (!candidates.empty()) {
        candidates.front().selected = true;
    }
    return candidates;
}

static std::string selected_lvc_path_from_candidates(
        const std::string& index_dir,
        const std::vector<LvcCandidateRow>& candidates) {
    std::string best;
    for (const auto& candidate : candidates) {
        if (candidate.selected) {
            best = join_path(index_dir, candidate.index_file);
            break;
        }
    }
    return best;
}

static void write_csv(const std::string& path, const std::vector<ComponentRow>& rows) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Cannot open output CSV: " + path);
    }
    out << "Dataset,IndexType,ActualVariant,IndexFile,Component,Bytes,GiB,FileSizeBytes,N,Dim,Note\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& r : rows) {
        out << r.dataset << ','
            << r.index_type << ','
            << r.actual_variant << ','
            << r.index_file << ','
            << r.component << ','
            << r.bytes << ','
            << static_cast<double>(r.bytes) / (1024.0 * 1024.0 * 1024.0) << ','
            << r.file_size_bytes << ','
            << r.ntotal << ','
            << r.dim << ','
            << '"' << r.note << '"' << '\n';
    }
}

static void write_lvc_overview_csv(const std::string& path,
                                   const std::vector<LvcCandidateRow>& rows) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Cannot open LVC overview CSV: " + path);
    }
    out << "Dataset,Variant,IndexFile,Bytes,GiB,Selected\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& r : rows) {
        out << r.dataset << ','
            << r.variant << ','
            << r.index_file << ','
            << r.bytes << ','
            << static_cast<double>(r.bytes) / (1024.0 * 1024.0 * 1024.0) << ','
            << (r.selected ? 1 : 0) << '\n';
    }
}

int main(int argc, char** argv) {
    std::string index_dir = ".";
    std::string base_dir = "../datasets/hdf5files";
    std::string output_csv = "index_storage_breakdown.csv";
    std::string lvc_overview_csv = "lvc_codec_index_size_overview.csv";
    size_t diff_cache_ratio_percent = 1;
    std::vector<std::string> datasets = {
        "deep-image-96-angular",
        "fashion-mnist-784-euclidean",
        "gist-960-euclidean",
        "mnist-784-euclidean",
        "sift-128-euclidean"
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--index_dir" && i + 1 < argc) {
            index_dir = argv[++i];
        } else if (arg == "--base_dir" && i + 1 < argc) {
            base_dir = argv[++i];
        } else if (arg == "--output_csv" && i + 1 < argc) {
            output_csv = argv[++i];
        } else if (arg == "--lvc_overview_csv" && i + 1 < argc) {
            lvc_overview_csv = argv[++i];
        } else if (arg == "--diff_cache_percent" && i + 1 < argc) {
            diff_cache_ratio_percent = static_cast<size_t>(std::stoull(argv[++i]));
        } else if (arg == "--dataset" && i + 1 < argc) {
            datasets.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                datasets.push_back(argv[++i]);
            }
        }
    }

    std::vector<ComponentRow> rows;
    std::vector<LvcCandidateRow> lvc_overview_rows;
    for (const auto& dataset : datasets) {
        const size_t dim = read_fvecs_dim_or_infer(base_dir, dataset);
        std::cout << "\n[Storage Breakdown] dataset=" << dataset
                  << " dim=" << dim << std::endl;

        const std::string hnsw_path = first_existing(index_dir, {
            dataset + "_train.fvecs_hnsw_float.bin",
            dataset + "_train.fvecs_hnsw.bin",
            dataset + "_hnsw.bin"
        });
        if (!hnsw_path.empty()) {
            std::cout << "  HNSW: " << hnsw_path << std::endl;
            profile_hnswlib_hnsw(dataset, hnsw_path, dim, rows);
        } else {
            std::cerr << "  [skip] HNSW index not found for " << dataset << std::endl;
        }

        const std::string pq_path = first_existing(index_dir, {
            dataset + "_train.fvecs_HNSWPQ_1.bin",
            dataset + "_HNSWPQ_1.bin"
        });
        if (!pq_path.empty()) {
            std::cout << "  HNSW+PQ: " << pq_path << std::endl;
            profile_faiss_hnsw_quantized(dataset, "HNSW+PQ",
                                         stem_only(pq_path),
                                         pq_path, dim, rows);
        } else {
            std::cerr << "  [skip] HNSW+PQ index not found for " << dataset << std::endl;
        }

        const std::string sq_path = first_existing(index_dir, {
            dataset + "_train.fvecs_HNSWSQ_8.bin",
            dataset + "_HNSWSQ_8.bin"
        });
        if (!sq_path.empty()) {
            std::cout << "  HNSW+SQ: " << sq_path << std::endl;
            profile_faiss_hnsw_quantized(dataset, "HNSW+SQ",
                                         stem_only(sq_path),
                                         sq_path, dim, rows);
        } else {
            std::cerr << "  [skip] HNSW+SQ index not found for " << dataset << std::endl;
        }

        auto lvc_candidates = collect_lvc_candidates(dataset, index_dir);
        if (!lvc_candidates.empty()) {
            std::cout << "  LVC codec size overview:" << std::endl;
            for (const auto& c : lvc_candidates) {
                std::cout << "    " << (c.selected ? "* " : "  ")
                          << c.variant << " " << c.index_file
                          << " " << c.bytes << " bytes" << std::endl;
                lvc_overview_rows.push_back(c);
            }
        }
        const std::string lvc_path = selected_lvc_path_from_candidates(index_dir, lvc_candidates);
        if (!lvc_path.empty()) {
            const std::string variant = lvc_variant_from_path(lvc_path);
            std::cout << "  LVC(" << variant << "): " << lvc_path << std::endl;
            if (variant == "ALP") {
                profile_alp_lvc(dataset, variant, lvc_path, dim, rows);
            } else {
                profile_differential_lvc(dataset, variant, lvc_path, dim,
                                         diff_cache_ratio_percent, rows);
            }
        } else {
            std::cerr << "  [skip] LVC index not found for " << dataset << std::endl;
        }
    }

    write_csv(output_csv, rows);
    write_lvc_overview_csv(lvc_overview_csv, lvc_overview_rows);
    std::cout << "\nSaved storage breakdown CSV: " << output_csv << std::endl;
    std::cout << "Saved LVC codec overview CSV: " << lvc_overview_csv << std::endl;
    return 0;
}
