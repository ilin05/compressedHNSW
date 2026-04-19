#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>

#include <faiss/IndexHNSW.h>
#include <faiss/index_factory.h>
#include <faiss/index_io.h>

#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/hnswalg_simplified_alp_PQ.h"
#include "../../examples/data_processor/data_loader.h"

using namespace hnswlib;
namespace fs = std::filesystem;

class StopW {
    std::chrono::steady_clock::time_point begin_;
public:
    StopW() : begin_(std::chrono::steady_clock::now()) {}
    void reset() { begin_ = std::chrono::steady_clock::now(); }
    double elapsed_s() const {
        return static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - begin_).count()) * 1e-6;
    }
};

struct BuildRecord {
    std::string dataset;
    std::string index_type;
    std::string index_file;
    double build_time_s = 0.0;
    uint64_t index_size_bytes = 0;
};

static double* load_fvecs_as_double(const std::string& filename, size_t& num_vectors, size_t& dim) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot open fvecs file: " + filename);
    }

    int32_t d = 0;
    input.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
    if (!input.good() || d <= 0) {
        throw std::runtime_error("Invalid fvecs header: " + filename);
    }
    dim = static_cast<size_t>(d);

    input.seekg(0, std::ios::end);
    size_t file_size = static_cast<size_t>(input.tellg());
    num_vectors = file_size / (sizeof(int32_t) + dim * sizeof(float));

    input.seekg(0, std::ios::beg);
    double* data = new double[num_vectors * dim];
    std::vector<float> tmp(dim);

    for (size_t i = 0; i < num_vectors; ++i) {
        input.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
        input.read(reinterpret_cast<char*>(tmp.data()), dim * sizeof(float));
        for (size_t j = 0; j < dim; ++j) {
            data[i * dim + j] = static_cast<double>(tmp[j]);
        }
    }

    return data;
}

static std::vector<float> to_float_buffer(const double* src, size_t n) {
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = static_cast<float>(src[i]);
    }
    return out;
}

static uint64_t file_size_bytes(const std::string& path) {
    if (!fs::exists(path)) {
        return 0;
    }
    return static_cast<uint64_t>(fs::file_size(path));
}

static BuildRecord build_mnist_hnsw_float(const std::string& dataset_name,
                                          const std::vector<float>& data,
                                          size_t n,
                                          size_t dim,
                                          int M,
                                          int ef_construction) {
    BuildRecord rec;
    rec.dataset = dataset_name;
    rec.index_type = "HNSW";
    rec.index_file = "mnist-784_hnsw_float.bin";

    L2Space l2(dim);
    auto* index = new HierarchicalNSW<float>(&l2, n, M, ef_construction);

    StopW sw;
    #pragma omp parallel for
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
        index->addPoint((void*)(data.data() + i * dim), static_cast<labeltype>(i));
    }
    rec.build_time_s = sw.elapsed_s();

    index->saveIndex(rec.index_file);
    rec.index_size_bytes = file_size_bytes(rec.index_file);

    delete index;
    return rec;
}

static BuildRecord build_mnist_compressed_dexor(const std::string& dataset_name,
                                                const double* data,
                                                size_t n,
                                                size_t dim,
                                                int M,
                                                int ef_construction,
                                                size_t cache_size) {
    BuildRecord rec;
    rec.dataset = dataset_name;
    rec.index_type = "Compressed-HNSW-DeXOR";
    rec.index_file = "mnist-784_compressed_hnsw_dexor.bin";

    L2SpaceDouble l2(dim);
    auto* index = new HierarchicalNSWCABLEANNPQ<double>(
        &l2,
        n,
        "DeXOR",
        M,
        ef_construction,
        true,
        cache_size
    );

    StopW sw;
    #pragma omp parallel for
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
        index->addPoint(data + i * dim, static_cast<labeltype>(i));
    }
    index->compress_dataset();
    rec.build_time_s = sw.elapsed_s();

    index->saveIndex(rec.index_file);
    rec.index_size_bytes = file_size_bytes(rec.index_file);

    delete index;
    return rec;
}

static BuildRecord build_mnist_faiss(const std::string& dataset_name,
                                     const std::vector<float>& data,
                                     size_t n,
                                     size_t dim,
                                     const std::string& index_type,
                                     const std::string& factory,
                                     const std::string& out_file,
                                     int ef_construction) {
    BuildRecord rec;
    rec.dataset = dataset_name;
    rec.index_type = index_type;
    rec.index_file = out_file;

    faiss::Index* idx = faiss::index_factory(static_cast<int>(dim), factory.c_str(), faiss::METRIC_L2);
    if (auto* hnsw = dynamic_cast<faiss::IndexHNSW*>(idx)) {
        hnsw->hnsw.efConstruction = ef_construction;
    }

    StopW sw;
    idx->train(n, data.data());
    idx->add(n, data.data());
    rec.build_time_s = sw.elapsed_s();

    faiss::write_index(idx, out_file.c_str());
    rec.index_size_bytes = file_size_bytes(out_file);

    delete idx;
    return rec;
}

static BuildRecord build_bigann_hnsw_float(const std::string& dataset_name,
                                           const std::string& base_file,
                                           size_t n,
                                           size_t dim,
                                           size_t chunk,
                                           int M,
                                           int ef_construction) {
    BuildRecord rec;
    rec.dataset = dataset_name;
    rec.index_type = "HNSW";
    rec.index_file = "bigann_100M_hnsw_float.bin";

    L2Space l2(dim);
    auto* index = new HierarchicalNSW<float>(&l2, n, M, ef_construction);

    StopW sw;
    size_t loaded = 0;
    while (loaded < n) {
        const size_t take = std::min(chunk, n - loaded);
        int chunk_dim = 0;
        double* raw = data_loader::loadBvecsChunk(base_file, loaded, take, chunk_dim);
        std::vector<float> fbuf = to_float_buffer(raw, take * dim);
        delete[] raw;

        #pragma omp parallel for
        for (long long i = 0; i < static_cast<long long>(take); ++i) {
            index->addPoint((void*)(fbuf.data() + i * dim), static_cast<labeltype>(loaded + static_cast<size_t>(i)));
        }
        loaded += take;
    }
    rec.build_time_s = sw.elapsed_s();

    index->saveIndex(rec.index_file);
    rec.index_size_bytes = file_size_bytes(rec.index_file);

    delete index;
    return rec;
}

static BuildRecord build_bigann_compressed_alp(const std::string& dataset_name,
                                               const std::string& base_file,
                                               size_t n,
                                               size_t dim,
                                               size_t chunk,
                                               int M,
                                               int ef_construction) {
    BuildRecord rec;
    rec.dataset = dataset_name;
    rec.index_type = "Compressed-HNSW-ALP";
    rec.index_file = "bigann_100M_compressed_hnsw_alp.bin";

    L2SpaceDouble l2(dim);
    auto* index = new HierarchicalNSWALPSIMPLIFIEDPQ<double>(&l2, n, M, ef_construction);

    StopW sw;
    size_t loaded = 0;
    while (loaded < n) {
        const size_t take = std::min(chunk, n - loaded);
        int chunk_dim = 0;
        double* raw = data_loader::loadBvecsChunk(base_file, loaded, take, chunk_dim);

        #pragma omp parallel for
        for (long long i = 0; i < static_cast<long long>(take); ++i) {
            index->addPoint(raw + i * dim, static_cast<labeltype>(loaded + static_cast<size_t>(i)));
        }
        delete[] raw;
        loaded += take;
    }
    index->compress_dataset();
    rec.build_time_s = sw.elapsed_s();

    index->saveIndex(rec.index_file);
    rec.index_size_bytes = file_size_bytes(rec.index_file);

    delete index;
    return rec;
}

static BuildRecord build_bigann_faiss(const std::string& dataset_name,
                                      const std::string& base_file,
                                      size_t n,
                                      size_t dim,
                                      size_t train_size,
                                      size_t add_chunk,
                                      const std::string& index_type,
                                      const std::string& factory,
                                      const std::string& out_file,
                                      int ef_construction) {
    BuildRecord rec;
    rec.dataset = dataset_name;
    rec.index_type = index_type;
    rec.index_file = out_file;

    const size_t train_count = std::min(train_size, n);
    int dim_detect = 0;
    double* train_raw = data_loader::loadBvecsChunk(base_file, 0, train_count, dim_detect);
    std::vector<float> train = to_float_buffer(train_raw, train_count * dim);
    delete[] train_raw;

    faiss::Index* idx = faiss::index_factory(static_cast<int>(dim), factory.c_str(), faiss::METRIC_L2);
    if (auto* hnsw = dynamic_cast<faiss::IndexHNSW*>(idx)) {
        hnsw->hnsw.efConstruction = ef_construction;
    }

    StopW sw;
    idx->train(train_count, train.data());

    size_t loaded = 0;
    while (loaded < n) {
        const size_t take = std::min(add_chunk, n - loaded);
        int chunk_dim = 0;
        double* raw = data_loader::loadBvecsChunk(base_file, loaded, take, chunk_dim);
        std::vector<float> fbuf = to_float_buffer(raw, take * dim);
        delete[] raw;

        idx->add(take, fbuf.data());
        loaded += take;
    }

    rec.build_time_s = sw.elapsed_s();

    faiss::write_index(idx, out_file.c_str());
    rec.index_size_bytes = file_size_bytes(out_file);

    delete idx;
    return rec;
}

static void write_csv(const std::string& csv_path, const std::vector<BuildRecord>& rows) {
    std::ofstream out(csv_path);
    out << "Dataset,IndexType,BuildTime(s),IndexFile,IndexSize(Bytes)\n";
    for (const auto& r : rows) {
        out << r.dataset << ','
            << r.index_type << ','
            << std::fixed << std::setprecision(6) << r.build_time_s << ','
            << r.index_file << ','
            << r.index_size_bytes << "\n";
    }
}

int main(int argc, char** argv) {
    omp_set_num_threads(16);

    std::string mnist_file = "../datasets/hdf5files/mnist-784-euclidean_train.fvecs";
    std::string bigann_file = "../bigann/bigann_base.bvecs";
    std::string out_csv = "index_size_comparison_results.csv";

    size_t bigann_vectors = 100000000ULL;
    size_t chunk = 500000ULL;
    size_t train_size = 1000000ULL;
    int M = 16;
    int ef_construction = 200;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mnist_file" && i + 1 < argc) {
            mnist_file = argv[++i];
        } else if (arg == "--bigann_file" && i + 1 < argc) {
            bigann_file = argv[++i];
        } else if (arg == "--out_csv" && i + 1 < argc) {
            out_csv = argv[++i];
        } else if (arg == "--bigann_vectors" && i + 1 < argc) {
            bigann_vectors = static_cast<size_t>(std::stoull(argv[++i]));
        } else if (arg == "--chunk" && i + 1 < argc) {
            chunk = static_cast<size_t>(std::stoull(argv[++i]));
        } else if (arg == "--train_size" && i + 1 < argc) {
            train_size = static_cast<size_t>(std::stoull(argv[++i]));
        } else if (arg == "--M" && i + 1 < argc) {
            M = std::stoi(argv[++i]);
        } else if (arg == "--ef_construction" && i + 1 < argc) {
            ef_construction = std::stoi(argv[++i]);
        }
    }

    std::vector<BuildRecord> rows;

    // MNIST-784: compressed HNSW uses DeXOR.
    size_t mnist_n = 0;
    size_t mnist_dim = 0;
    std::cout << "Loading MNIST from " << mnist_file << std::endl;
    std::unique_ptr<double[]> mnist_double(load_fvecs_as_double(mnist_file, mnist_n, mnist_dim));
    std::vector<float> mnist_float = to_float_buffer(mnist_double.get(), mnist_n * mnist_dim);

    rows.push_back(build_mnist_compressed_dexor("mnist-784", mnist_double.get(), mnist_n, mnist_dim, M, ef_construction, mnist_n / 100));
    rows.push_back(build_mnist_hnsw_float("mnist-784", mnist_float, mnist_n, mnist_dim, M, ef_construction));
    rows.push_back(build_mnist_faiss("mnist-784", mnist_float, mnist_n, mnist_dim, "IVF", "IVF1024,Flat", "mnist-784_ivf_float.bin", ef_construction));
    rows.push_back(build_mnist_faiss("mnist-784", mnist_float, mnist_n, mnist_dim, "HNSW+PQ", "HNSW16,PQ784", "mnist-784_hnsw_pq_float.bin", ef_construction));
    rows.push_back(build_mnist_faiss("mnist-784", mnist_float, mnist_n, mnist_dim, "HNSW+SQ", "HNSW16,SQ8", "mnist-784_hnsw_sq_float.bin", ef_construction));

    // BIGANN-100M: compressed HNSW uses ALP.
    const size_t bigann_dim = 128;
    rows.push_back(build_bigann_compressed_alp("bigann_100M", bigann_file, bigann_vectors, bigann_dim, chunk, M, ef_construction));
    rows.push_back(build_bigann_hnsw_float("bigann_100M", bigann_file, bigann_vectors, bigann_dim, chunk, M, ef_construction));
    rows.push_back(build_bigann_faiss("bigann_100M", bigann_file, bigann_vectors, bigann_dim, train_size, chunk, "IVF", "IVF4096,Flat", "bigann_100M_ivf_float.bin", ef_construction));
    rows.push_back(build_bigann_faiss("bigann_100M", bigann_file, bigann_vectors, bigann_dim, train_size, chunk, "HNSW+PQ", "HNSW16,PQ128", "bigann_100M_hnsw_pq_float.bin", ef_construction));
    rows.push_back(build_bigann_faiss("bigann_100M", bigann_file, bigann_vectors, bigann_dim, train_size, chunk, "HNSW+SQ", "HNSW16,SQ8", "bigann_100M_hnsw_sq_float.bin", ef_construction));

    write_csv(out_csv, rows);
    std::cout << "Saved CSV: " << out_csv << std::endl;
    std::cout << "DiskANN rows are intentionally not included." << std::endl;

    return 0;
}
