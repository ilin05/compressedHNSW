#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <omp.h>

#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/compressed_hnsw_framework.h"

using namespace std;
using namespace hnswlib;

class StopW {
    std::chrono::steady_clock::time_point begin_;
public:
    StopW() : begin_(std::chrono::steady_clock::now()) {}
    void reset() { begin_ = std::chrono::steady_clock::now(); }
    double elapsed_us() const {
        return static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - begin_).count());
    }
};

double* load_fvecs_as_double(const std::string& filename, size_t& num_vectors, size_t& dim) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        std::cerr << "Cannot open " << filename << std::endl;
        return nullptr;
    }

    int32_t d = 0;
    input.read((char*)&d, 4);
    dim = static_cast<size_t>(d);

    input.seekg(0, std::ios::end);
    size_t file_size = static_cast<size_t>(input.tellg());
    num_vectors = file_size / (4 + dim * 4);

    double* data = new double[num_vectors * dim];
    float* tmp = new float[dim];
    input.seekg(0, std::ios::beg);

    for (size_t i = 0; i < num_vectors; ++i) {
        input.read((char*)&d, 4);
        input.read((char*)tmp, dim * 4);
        for (size_t j = 0; j < dim; ++j) {
            data[i * dim + j] = static_cast<double>(tmp[j]);
        }
    }

    delete[] tmp;
    return data;
}

struct BuildResult {
    std::string dataset;
    std::string algorithm;
    int chain_max_length;
    size_t vectors;
    size_t dim;
    double build_time_s;
    double compress_time_s;
    double data_ratio;
    double index_ratio;
    std::string index_file;
    std::string root_policy;
    size_t roots = 0;
    size_t max_depth = 0;
    double mean_reference_l2 = 0;
    size_t payload_bytes = 0;
    size_t graph_metadata_bytes = 0;
    size_t file_bytes = 0;
    double full_index_ratio_f32 = 0;
    size_t f32_patch_bytes = 0;
    size_t codec_bytes = 0;
    size_t pq_code_bytes = 0;
    size_t pq_centroid_bytes = 0;
    std::string shared_graph_file;
};

static std::string chain_tag(int chain_max_length) {
    if (chain_max_length < 0) return "chunlim";
    return "ch" + std::to_string(chain_max_length);
}

template <typename CodecPolicy>
BuildResult build_one(const std::string& dataset,
                      const std::string& base_dir,
                      const std::string& algo,
                      int chain_max_length,
                      int M,
                      int ef_construction,
                      CompressionRootPolicy root_policy, bool verify_vectors, bool f32_lossless,
                      const std::string& shared_graph_file, double shared_graph_build_s) {
    BuildResult r;
    r.dataset = dataset;
    r.algorithm = algo;
    r.chain_max_length = chain_max_length;
    r.root_policy = root_policy == CompressionRootPolicy::Level0Degree ? "degree" :
                    root_policy == CompressionRootPolicy::Random ? "random" : "level";
    r.shared_graph_file = shared_graph_file;
    r.vectors = 0;
    r.dim = 0;
    r.build_time_s = 0.0;
    r.compress_time_s = 0.0;
    r.data_ratio = 0.0;
    r.index_ratio = 0.0;

    const std::string train_file = base_dir + dataset + "_train.fvecs";
    double* data = load_fvecs_as_double(train_file, r.vectors, r.dim);
    if (!data) {
        return r;
    }

    L2SpaceDouble l2space(r.dim);
    size_t cache_max_size = r.vectors / 100;
    auto* index = shared_graph_file.empty()
        ? new HierarchicalNSWCABFRAMEWORK<double, CodecPolicy>(
            &l2space, r.vectors, M, ef_construction, true, cache_max_size, 100, false)
        : new HierarchicalNSWCABFRAMEWORK<double, CodecPolicy>(
            &l2space, shared_graph_file, true, 0);
    index->setF32Lossless(f32_lossless);

    std::cout << "\n=== Build dataset=" << dataset
              << " algo=" << algo
              << " chain_max=" << chain_max_length << " ===" << std::endl;

    StopW timer;
    if (shared_graph_file.empty()) {
        #pragma omp parallel for
        for (long long i = 0; i < static_cast<long long>(r.vectors); ++i) {
            index->addPoint(data + i * r.dim, static_cast<labeltype>(i));
        }
        r.build_time_s = timer.elapsed_us() * 1e-6;
    } else {
        if (index->cur_element_count != r.vectors)
            throw std::runtime_error("Shared HNSW graph vector count mismatch");
        r.build_time_s = shared_graph_build_s;
    }

    timer.reset();
    index->compress_dataset(chain_max_length, 0.01, root_policy);
    r.compress_time_s = timer.elapsed_us() * 1e-6;

    size_t original_data_size = r.vectors * r.dim * sizeof(double);
    size_t compressed_data_size = static_cast<size_t>(index->getCompressedDataSize());
    size_t original_index_size = index->getIndexSize();
    size_t compressed_index_size = index->getCompressedIndexSize();

    if (compressed_data_size > 0) {
        r.data_ratio = static_cast<double>(original_data_size) / static_cast<double>(compressed_data_size);
    }
    if (compressed_index_size > 0) {
        r.index_ratio = static_cast<double>(original_index_size) / static_cast<double>(compressed_index_size);
    }

    r.index_file = dataset + "_" + algo + "_" + chain_tag(chain_max_length) +
                   (r.root_policy == "level" ? "" : "_" + r.root_policy) + "_pq.bin";
    index->exportCompressionForest(r.index_file + ".forest.csv");
    for (size_t i = 0; i < r.vectors; ++i) {
        if (index->compression_parent_export_[i] == static_cast<tableint>(-1)) ++r.roots;
        r.max_depth = std::max(r.max_depth, size_t(index->compression_depth_export_[i]));
        r.mean_reference_l2 += index->compression_reference_l2_export_[i];
    }
    r.mean_reference_l2 /= std::max<size_t>(1, r.vectors - r.roots);
    r.payload_bytes = compressed_data_size;
    r.f32_patch_bytes = index->getF32PatchBytes();
    r.codec_bytes = compressed_data_size - r.f32_patch_bytes - (f32_lossless ? r.vectors * sizeof(uint32_t) : 0);
    r.pq_code_bytes = index->pq_data_.size();
    for (const auto& centroids : index->pq_centroids_) r.pq_centroid_bytes += centroids.size() * sizeof(double);
    r.graph_metadata_bytes = compressed_index_size - compressed_data_size;
    r.full_index_ratio_f32 = compressed_index_size ?
        double(original_index_size - original_data_size + r.vectors * r.dim * sizeof(float)) /
            compressed_index_size : 0;
    if (verify_vectors) {
        for (tableint i = 0; i < r.vectors; ++i) {
            const auto decoded = index->getOriginalDataByInternalId(i);
            const auto label = index->getExternalLabel(i);
            size_t first = 0;
            if (label < r.vectors && decoded.size() == r.dim) {
                while (first < r.dim) {
                    const float actual = static_cast<float>(decoded[first]);
                    const float expected = static_cast<float>(data[label * r.dim + first]);
                    if (std::memcmp(&actual, &expected, sizeof(float)) != 0) break;
                    ++first;
                }
            }
            if (label >= r.vectors || decoded.size() != r.dim || first != r.dim) {
                throw std::runtime_error("HNSW F32 decode failed at internal node " + std::to_string(i) +
                    " label=" + std::to_string(label) + " coordinate=" + std::to_string(first) +
                    " expected=" + (first < r.dim ? std::to_string(data[label * r.dim + first]) : "missing") +
                    " decoded=" + (first < decoded.size() ? std::to_string(decoded[first]) : "missing"));
            }
        }
        std::cout << "PASS HNSW F32 bitwise vectors=" << r.vectors << '/' << r.vectors << '\n';
    }
    index->saveIndex(r.index_file);
    r.file_bytes = std::filesystem::file_size(r.index_file);
    if (verify_vectors) {
        HierarchicalNSWCABFRAMEWORK<double, CodecPolicy> reloaded(
            &l2space, r.index_file, true, cache_max_size);
        for (tableint i = 0; i < r.vectors; ++i) {
            const auto decoded = reloaded.getOriginalDataByInternalId(i);
            const auto label = reloaded.getExternalLabel(i);
            if (label >= r.vectors || decoded.size() != r.dim)
                throw std::runtime_error("Reloaded HNSW vector shape mismatch at " + std::to_string(i));
            for (size_t j = 0; j < r.dim; ++j) {
                const float actual = static_cast<float>(decoded[j]);
                const float expected = static_cast<float>(data[label * r.dim + j]);
                if (std::memcmp(&actual, &expected, sizeof(float)) != 0)
                    throw std::runtime_error("Reloaded HNSW F32 decode failed at node " + std::to_string(i) +
                                             " coordinate=" + std::to_string(j));
            }
        }
        std::cout << "PASS reloaded HNSW F32 bitwise vectors=" << r.vectors << '/' << r.vectors << '\n';
    }

    delete index;
    delete[] data;
    return r;
}

BuildResult dispatch_build(const std::string& dataset,
                           const std::string& base_dir,
                           const std::string& algo,
                           int chain_max_length,
                           int M,
                           int ef_construction, CompressionRootPolicy root_policy, bool verify_vectors,
                           bool f32_lossless, const std::string& shared_graph_file, double shared_graph_build_s) {
    if (algo == "DeXOR") return build_one<codecs::DeXORCodecPolicy>(dataset, base_dir, algo, chain_max_length, M, ef_construction, root_policy, verify_vectors, f32_lossless, shared_graph_file, shared_graph_build_s);
    if (algo == "Gorilla") return build_one<codecs::GorillaCodecPolicy>(dataset, base_dir, algo, chain_max_length, M, ef_construction, root_policy, verify_vectors, f32_lossless, shared_graph_file, shared_graph_build_s);
    if (algo == "Elf") return build_one<codecs::ElfCodecPolicy>(dataset, base_dir, algo, chain_max_length, M, ef_construction, root_policy, verify_vectors, f32_lossless, shared_graph_file, shared_graph_build_s);
    if (algo == "Camel") return build_one<codecs::CamelCodecPolicy>(dataset, base_dir, algo, chain_max_length, M, ef_construction, root_policy, verify_vectors, f32_lossless, shared_graph_file, shared_graph_build_s);
    if (algo == "DeXORPlus") return build_one<codecs::DeXORPlusCodecPolicy>(dataset, base_dir, algo, chain_max_length, M, ef_construction, root_policy, verify_vectors, f32_lossless, shared_graph_file, shared_graph_build_s);
    throw std::runtime_error("Unknown algorithm: " + algo);
}

int main(int argc, char** argv) {
    std::string base_dir = "../datasets/hdf5files/";
    std::string out_csv = "compressed_hnsw_ablation_build_results.csv";
    int threads = 32;
    int M = 16;
    int ef_construction = 200;
    CompressionRootPolicy root_policy = CompressionRootPolicy::Level;
    bool verify_vectors = false;
    bool f32_lossless = false;
    bool reuse_graph = false;

    std::vector<std::string> datasets = {
        "fashion-mnist-784-euclidean",
        "gist-960-euclidean",
        "mnist-784-euclidean",
        "sift-128-euclidean",
        "deep-image-96-angular"
    };

    std::vector<std::string> algorithms = {
        "DeXOR",
        "Gorilla",
        "Elf",
        "Camel",
        "DeXORPlus"
    };

    std::vector<int> chain_max_list = {2, 5, -1};

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--base_dir" && i + 1 < argc) {
            base_dir = argv[++i];
        } else if (arg == "--output_csv" && i + 1 < argc) {
            out_csv = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            threads = std::stoi(argv[++i]);
        } else if (arg == "--M" && i + 1 < argc) {
            M = std::stoi(argv[++i]);
        } else if (arg == "--ef_construction" && i + 1 < argc) {
            ef_construction = std::stoi(argv[++i]);
        } else if (arg == "--root_policy" && i + 1 < argc) {
            const std::string policy = argv[++i];
            if (policy == "degree") root_policy = CompressionRootPolicy::Level0Degree;
            else if (policy == "level") root_policy = CompressionRootPolicy::Level;
            else if (policy == "random") root_policy = CompressionRootPolicy::Random;
            else throw std::runtime_error("Unknown root policy: " + policy);
        } else if (arg == "--verify_vectors") {
            verify_vectors = true;
        } else if (arg == "--f32_lossless") {
            f32_lossless = true;
        } else if (arg == "--reuse_graph") {
            reuse_graph = true;
        } else if (arg == "--dataset" && i + 1 < argc) {
            datasets.clear();
            while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                datasets.push_back(argv[++i]);
            }
        } else if (arg == "--algorithm" && i + 1 < argc) {
            algorithms.clear();
            while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                algorithms.push_back(argv[++i]);
            }
        } else if (arg == "--chain_max" && i + 1 < argc) {
            chain_max_list.clear();
            while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                chain_max_list.push_back(std::stoi(argv[++i]));
            }
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << std::endl;
            return -1;
        }
    }

    if (!base_dir.empty() && base_dir.back() != '/' && base_dir.back() != '\\') {
        base_dir += '/';
    }

    omp_set_num_threads(threads);

    std::ofstream csv(out_csv);
    if (!csv.is_open()) {
        std::cerr << "Failed to open csv: " << out_csv << std::endl;
        return -1;
    }

    csv << "Dataset,Algorithm,ChainMaxLength,Vectors,Dim,BuildTime(s),CompressTime(s),DataCompressionRatio,IndexCompressionRatio,IndexFile,RootPolicy,Roots,MaxDepth,MeanReferenceL2,PayloadBytes,GraphMetadataBytes,IndexFileBytes,RawF32Bytes,PayloadRatioF32,FullIndexRatioF32,F32PatchBytes,CodecBytes,PQCodeBytes,PQCentroidBytes,SharedGraphFile\n";
    for (const auto& ds : datasets) {
        std::string shared_graph_file;
        double shared_graph_build_s = 0;
        if (reuse_graph) {
            size_t n = 0, dim = 0;
            double* data = load_fvecs_as_double(base_dir + ds + "_train.fvecs", n, dim);
            if (!data) throw std::runtime_error("Cannot load shared HNSW graph input");
            L2SpaceDouble space(dim);
            HierarchicalNSWCABFRAMEWORK<double, codecs::DeXORCodecPolicy> graph(
                &space, n, M, ef_construction, true, 0, 100, false);
            StopW graph_timer;
            #pragma omp parallel for
            for (long long i = 0; i < static_cast<long long>(n); ++i)
                graph.addPoint(data + i * dim, static_cast<labeltype>(i));
            shared_graph_build_s = graph_timer.elapsed_us() * 1e-6;
            shared_graph_file = ds + "_shared_raw_graph.bin";
            graph.saveIndex(shared_graph_file);
            delete[] data;
            std::cout << "Shared HNSW graph=" << shared_graph_file
                      << " build_s=" << shared_graph_build_s << '\n';
        }
        for (const auto& algo : algorithms) {
            for (int chain_max : chain_max_list) {
                BuildResult r = dispatch_build(ds, base_dir, algo, chain_max, M, ef_construction,
                    root_policy, verify_vectors, f32_lossless, shared_graph_file, shared_graph_build_s);
                csv << r.dataset << ','
                    << r.algorithm << ','
                    << r.chain_max_length << ','
                    << r.vectors << ','
                    << r.dim << ','
                    << std::fixed << std::setprecision(6)
                    << r.build_time_s << ','
                    << r.compress_time_s << ','
                    << r.data_ratio << ','
                    << r.index_ratio << ','
                    << r.index_file << ',' << r.root_policy << ',' << r.roots << ','
                    << r.max_depth << ',' << r.mean_reference_l2 << ',' << r.payload_bytes << ','
                    << r.graph_metadata_bytes << ',' << r.file_bytes << ','
                    << r.vectors * r.dim * sizeof(float) << ','
                    << (r.payload_bytes ? double(r.vectors * r.dim * sizeof(float)) / r.payload_bytes : 0) << ','
                    << r.full_index_ratio_f32 << ',' << r.f32_patch_bytes << ','
                    << r.codec_bytes << ',' << r.pq_code_bytes << ',' << r.pq_centroid_bytes << ','
                    << r.shared_graph_file << '\n';
            }
        }
    }

    csv.close();
    std::cout << "\nSaved CSV: " << out_csv << std::endl;
    return 0;
}
