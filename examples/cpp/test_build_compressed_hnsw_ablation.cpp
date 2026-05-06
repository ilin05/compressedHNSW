#include <chrono>
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
                      int ef_construction) {
    BuildResult r;
    r.dataset = dataset;
    r.algorithm = algo;
    r.chain_max_length = chain_max_length;
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
    auto* index = new HierarchicalNSWCABFRAMEWORK<double, CodecPolicy>(
        &l2space, r.vectors, M, ef_construction, true, cache_max_size, 100, false);

    std::cout << "\n=== Build dataset=" << dataset
              << " algo=" << algo
              << " chain_max=" << chain_max_length << " ===" << std::endl;

    StopW timer;
    #pragma omp parallel for
    for (long long i = 0; i < static_cast<long long>(r.vectors); ++i) {
        index->addPoint(data + i * r.dim, static_cast<labeltype>(i));
    }
    r.build_time_s = timer.elapsed_us() * 1e-6;

    timer.reset();
    index->compress_dataset(chain_max_length);
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

    r.index_file = dataset + "_" + algo + "_" + chain_tag(chain_max_length) + "_pq.bin";
    index->saveIndex(r.index_file);

    delete index;
    delete[] data;
    return r;
}

BuildResult dispatch_build(const std::string& dataset,
                           const std::string& base_dir,
                           const std::string& algo,
                           int chain_max_length,
                           int M,
                           int ef_construction) {
    if (algo == "DeXOR") return build_one<codecs::DeXORCodecPolicy>(dataset, base_dir, algo, chain_max_length, M, ef_construction);
    if (algo == "Gorilla") return build_one<codecs::GorillaCodecPolicy>(dataset, base_dir, algo, chain_max_length, M, ef_construction);
    if (algo == "Elf") return build_one<codecs::ElfCodecPolicy>(dataset, base_dir, algo, chain_max_length, M, ef_construction);
    if (algo == "Camel") return build_one<codecs::CamelCodecPolicy>(dataset, base_dir, algo, chain_max_length, M, ef_construction);
    if (algo == "DeXORPlus") return build_one<codecs::DeXORPlusCodecPolicy>(dataset, base_dir, algo, chain_max_length, M, ef_construction);
    throw std::runtime_error("Unknown algorithm: " + algo);
}

int main(int argc, char** argv) {
    std::string base_dir = "../datasets/hdf5files/";
    std::string out_csv = "compressed_hnsw_ablation_build_results.csv";
    int threads = 32;
    int M = 16;
    int ef_construction = 200;

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

    csv << "Dataset,Algorithm,ChainMaxLength,Vectors,Dim,BuildTime(s),CompressTime(s),DataCompressionRatio,IndexCompressionRatio,IndexFile\n";

    for (const auto& ds : datasets) {
        for (const auto& algo : algorithms) {
            for (int chain_max : chain_max_list) {
                BuildResult r = dispatch_build(ds, base_dir, algo, chain_max, M, ef_construction);
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
                    << r.index_file << '\n';
            }
        }
    }

    csv.close();
    std::cout << "\nSaved CSV: " << out_csv << std::endl;
    return 0;
}
