#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQFastScan.h>
#include <faiss/IndexNSG.h>
#include <faiss/index_io.h>

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Matrix {
    size_t rows = 0, dim = 0;
    std::vector<uint32_t> values;
};

Matrix read_vectors(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open " + path);
    const auto bytes = input.tellg();
    input.seekg(0);
    int32_t dim = 0;
    input.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    if (!input || dim <= 0 || bytes % (4 + 4 * std::streamoff(dim)))
        throw std::runtime_error("invalid vector file " + path);
    Matrix result;
    result.dim = size_t(dim);
    result.rows = size_t(bytes / (4 + 4 * std::streamoff(dim)));
    result.values.resize(result.rows * result.dim);
    input.seekg(0);
    for (size_t row = 0; row < result.rows; ++row) {
        int32_t width = 0;
        input.read(reinterpret_cast<char*>(&width), 4);
        if (!input || width != dim) throw std::runtime_error("inconsistent vector width");
        input.read(reinterpret_cast<char*>(result.values.data() + row * result.dim),
                   4 * result.dim);
        if (!input) throw std::runtime_error("truncated vector file");
    }
    return result;
}

double recall_at_k(const Matrix& truth, const std::vector<faiss::idx_t>& ids,
                   size_t k) {
    if (k > truth.dim) throw std::runtime_error("K exceeds ground-truth width");
    size_t hits = 0;
    for (size_t row = 0; row < truth.rows; ++row) {
        for (size_t col = 0; col < k; ++col) {
            const auto id = ids[row * k + col];
            if (id < 0) continue;
            for (size_t j = 0; j < k; ++j) {
                if (uint64_t(id) == truth.values[row * truth.dim + j]) {
                    ++hits;
                    break;
                }
            }
        }
    }
    return double(hits) / double(truth.rows * k);
}
} // namespace

int main(int argc, char** argv) {
    try {
        std::string dataset, algorithm, index_path, query_path, truth_path, output;
        std::vector<int> ks{1, 10};
        std::vector<int> params{1, 2, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100};
        int rounds = 3;
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--k" || option == "--params") {
                auto& values = option == "--k" ? ks : params;
                values.clear();
                while (i + 1 < argc && argv[i + 1][0] != '-')
                    values.push_back(std::stoi(argv[++i]));
                continue;
            }
            if (i + 1 >= argc) throw std::invalid_argument("missing option value");
            const std::string value = argv[++i];
            if (option == "--dataset") dataset = value;
            else if (option == "--algorithm") algorithm = value;
            else if (option == "--index") index_path = value;
            else if (option == "--queries") query_path = value;
            else if (option == "--ground-truth") truth_path = value;
            else if (option == "--output") output = value;
            else if (option == "--rounds") rounds = std::stoi(value);
            else throw std::invalid_argument("unknown option " + option);
        }
        if (dataset.empty() || index_path.empty() || query_path.empty() ||
            truth_path.empty() || output.empty() || rounds < 1 || ks.empty() ||
            params.empty() || (algorithm != "ivf_flat" &&
                               algorithm != "ivfpq_fastscan" && algorithm != "nsg"))
            throw std::invalid_argument("invalid search arguments");
        for (int k : ks) if (k < 1) throw std::invalid_argument("K must be positive");
        for (int p : params) if (p < 1) throw std::invalid_argument("search parameter must be positive");

        omp_set_num_threads(1);
        auto queries = read_vectors(query_path);
        auto truth = read_vectors(truth_path);
        if (!queries.rows || queries.rows != truth.rows)
            throw std::runtime_error("query and ground-truth row counts differ");
        std::unique_ptr<faiss::Index> index(faiss::read_index(index_path.c_str()));
        if (size_t(index->d) != queries.dim)
            throw std::runtime_error("query dimension differs from index");
        auto* ivf = dynamic_cast<faiss::IndexIVF*>(index.get());
        auto* nsg = dynamic_cast<faiss::IndexNSG*>(index.get());
        if ((algorithm == "ivf_flat" && !dynamic_cast<faiss::IndexIVFFlat*>(ivf)) ||
            (algorithm == "ivfpq_fastscan" &&
             !dynamic_cast<faiss::IndexIVFPQFastScan*>(ivf)) ||
            (algorithm == "nsg" && !nsg))
            throw std::runtime_error("index type does not match --algorithm");

        const auto bytes = std::filesystem::file_size(index_path);
        const double index_kb = double(bytes) / 1024.0;
        const double vectors_per_kb = double(index->ntotal) / index_kb;
        std::ofstream csv(output);
        if (!csv) throw std::runtime_error("cannot open output CSV");
        csv << "Dataset,IndexType,Param,K,Recall,QPS,Latency(us),IndexSizeKB,VectorsPerKB,VQ\n";
        csv << std::fixed << std::setprecision(6);
        std::vector<float> query_values(queries.values.size());
        std::memcpy(query_values.data(), queries.values.data(),
                    queries.values.size() * sizeof(float));
        const float* query_data = query_values.data();
        const std::string type = algorithm == "ivf_flat" ? "IVF-Flat" :
                                 algorithm == "ivfpq_fastscan" ? "IVFPQ-FastScan" : "NSG";
        // With one OpenMP thread, Faiss FastScan does not slice its 3D PQ LUT.
        // Bound its temporary memory without changing the query set or timing scope.
        constexpr size_t fastscan_query_batch = 64;
        auto search_queries = [&](size_t count, int k, float* out_distances,
                                  faiss::idx_t* out_ids) {
            const size_t batch = algorithm == "ivfpq_fastscan" ? fastscan_query_batch : count;
            for (size_t offset = 0; offset < count; offset += batch) {
                const size_t take = std::min(batch, count - offset);
                index->search(take, query_data + offset * queries.dim, k,
                              out_distances + offset * k, out_ids + offset * k);
            }
        };
        for (int k : ks) {
            if (size_t(k) > truth.dim) throw std::runtime_error("K exceeds ground-truth width");
            std::vector<float> distances(queries.rows * k);
            std::vector<faiss::idx_t> ids(queries.rows * k);
            for (int param : params) {
                if (ivf) {
                    if (size_t(param) > ivf->nlist) continue;
                    ivf->nprobe = param;
                } else {
                    if (param < k) continue;
                    nsg->setSearchL(param);
                }
                std::cerr << "Searching " << dataset << ' ' << type << " K=" << k
                          << ' ' << (ivf ? "nprobe=" : "search_L=") << param
                          << " batch=" << (algorithm == "ivfpq_fastscan" ? fastscan_query_batch : queries.rows)
                          << '\n';
                search_queries(std::min<size_t>(queries.rows, 64), k,
                               distances.data(), ids.data()); // warmup outside timing
                double elapsed_us = 0;
                double recall = 0;
                for (int round = 0; round < rounds; ++round) {
                    const auto start = std::chrono::steady_clock::now();
                    search_queries(queries.rows, k, distances.data(), ids.data());
                    elapsed_us += std::chrono::duration<double, std::micro>(
                                      std::chrono::steady_clock::now() - start).count();
                    if (round == 0) recall = recall_at_k(truth, ids, k);
                }
                const double latency_us = elapsed_us / (rounds * queries.rows);
                const double qps = 1e6 / latency_us;
                const double vq = vectors_per_kb * qps;
                const std::string setting = (ivf ? "nprobe=" : "search_L=") +
                                            std::to_string(param);
                csv << dataset << ',' << type << ',' << setting << ',' << k << ','
                    << recall << ',' << qps << ',' << latency_us << ','
                    << index_kb << ',' << vectors_per_kb << ',' << vq << '\n';
                csv.flush();
                std::cout << dataset << ' ' << type << ' ' << setting << " K=" << k
                          << " recall=" << recall << " QPS=" << qps << '\n';
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "revision Faiss search failed: " << error.what() << '\n';
        return 1;
    }
}
