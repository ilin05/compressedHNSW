#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQFastScan.h>
#include <faiss/index_io.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>

namespace {
using Clock = std::chrono::steady_clock;

double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::vector<float> read_fvecs(const std::string& path, size_t& n, size_t& dim) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open " + path);
    const auto bytes = input.tellg();
    input.seekg(0);
    int32_t width = 0;
    input.read(reinterpret_cast<char*>(&width), 4);
    if (!input || width <= 0) throw std::runtime_error("invalid fvecs header");
    dim = size_t(width);
    const auto row_bytes = std::streamoff(4 + 4 * dim);
    if (bytes <= 0 || bytes % row_bytes) throw std::runtime_error("invalid fvecs file size");
    n = size_t(bytes / row_bytes);
    input.seekg(0);
    std::vector<float> data(n * dim);
    for (size_t i = 0; i < n; ++i) {
        input.read(reinterpret_cast<char*>(&width), 4);
        if (!input || size_t(width) != dim) throw std::runtime_error("inconsistent fvecs dimension");
        input.read(reinterpret_cast<char*>(data.data() + i * dim), 4 * dim);
        if (!input) throw std::runtime_error("truncated fvecs file");
    }
    return data;
}
} // namespace

int main(int argc, char** argv) {
    try {
        std::string input, output, metrics, dataset, algorithm;
        double train_fraction = 0.15;
        size_t train_cap = 1500000, nlist = 0, add_chunk = 100000;
        int threads = 32, by_residual = 1;
        uint64_t seed = 20261010;
        for (int i = 1; i < argc; ++i) {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for argument");
            const std::string key = argv[i];
            const std::string value = argv[++i];
            if (key == "--input") input = value;
            else if (key == "--output") output = value;
            else if (key == "--metrics") metrics = value;
            else if (key == "--dataset") dataset = value;
            else if (key == "--algorithm") algorithm = value;
            else if (key == "--train-fraction") train_fraction = std::stod(value);
            else if (key == "--train-cap") train_cap = std::stoull(value);
            else if (key == "--nlist") nlist = std::stoull(value);
            else if (key == "--add-chunk") add_chunk = std::stoull(value);
            else if (key == "--threads") threads = std::stoi(value);
            else if (key == "--by-residual") by_residual = std::stoi(value);
            else if (key == "--seed") seed = std::stoull(value);
            else throw std::invalid_argument("unknown argument: " + key);
        }
        if (input.empty() || output.empty() || metrics.empty() || dataset.empty() ||
            (algorithm != "ivf_flat" && algorithm != "ivfpq_fastscan") ||
            !(train_fraction > 0 && train_fraction <= 1) || !train_cap ||
            !add_chunk || threads < 1 || (by_residual != 0 && by_residual != 1))
            throw std::invalid_argument("invalid baseline build arguments");
        omp_set_num_threads(threads);
        size_t n = 0, dim = 0;
        auto data = read_fvecs(input, n, dim);
        if (!nlist) nlist = size_t(std::floor(std::sqrt(double(n))));
        if (!nlist || nlist >= n || (algorithm == "ivfpq_fastscan" && dim % 4))
            throw std::invalid_argument("invalid nlist or FastScan dimension");

        const size_t train_count = std::min(n, std::max<size_t>(
            40 * nlist, std::min(train_cap, size_t(std::ceil(train_fraction * n)))));
        const auto sample_start = Clock::now();
        std::vector<size_t> ids(n);
        std::iota(ids.begin(), ids.end(), 0);
        std::mt19937_64 rng(seed);
        std::shuffle(ids.begin(), ids.end(), rng);
        std::vector<float> training(train_count * dim);
        for (size_t i = 0; i < train_count; ++i)
            std::copy_n(data.data() + ids[i] * dim, dim, training.data() + i * dim);
        ids.clear();
        ids.shrink_to_fit();
        const double sample_s = seconds(sample_start);

        std::unique_ptr<faiss::IndexIVF> index;
        const size_t pq_m = algorithm == "ivfpq_fastscan" ? dim / 4 : 0;
        if (algorithm == "ivf_flat") {
            auto* flat = new faiss::IndexIVFFlat(new faiss::IndexFlatL2(dim), dim, nlist);
            flat->own_fields = true;
            index.reset(flat);
        } else {
            auto* fast = new faiss::IndexIVFPQFastScan(
                new faiss::IndexFlatL2(dim), dim, nlist, pq_m, 4,
                faiss::METRIC_L2, 32);
            fast->own_fields = true;
            fast->by_residual = by_residual;
            index.reset(fast);
        }
        const auto train_start = Clock::now();
        index->train(train_count, training.data());
        const double train_s = seconds(train_start);
        training.clear();
        training.shrink_to_fit();
        const auto add_start = Clock::now();
        for (size_t offset = 0; offset < n; offset += add_chunk) {
            const size_t take = std::min(add_chunk, n - offset);
            index->add(take, data.data() + offset * dim);
        }
        const double add_s = seconds(add_start);
        if (size_t(index->ntotal) != n) throw std::runtime_error("incomplete Faiss index");
        std::filesystem::create_directories(std::filesystem::path(output).parent_path());
        const auto save_start = Clock::now();
        faiss::write_index(index.get(), output.c_str());
        const double save_s = seconds(save_start);
        const auto file_bytes = std::filesystem::file_size(output);
        std::unique_ptr<faiss::Index> reloaded(faiss::read_index(output.c_str()));
        if (size_t(reloaded->ntotal) != n) throw std::runtime_error("reloaded index size differs");
        float test_distance = 0;
        faiss::idx_t test_id = -1;
        reloaded->search(1, data.data(), 1, &test_distance, &test_id);
        if (test_id < 0) throw std::runtime_error("reloaded index returned no result");
        std::ofstream csv(metrics);
        if (!csv) throw std::runtime_error("cannot create metrics CSV");
        csv << "dataset,algorithm,nodes,dim,nlist,train_vectors,pq_m,pq_bits,by_residual,bbs,"
               "sample_seconds,train_seconds,add_seconds,serialize_seconds,build_seconds,"
               "index_file_bytes,index_path\n";
        csv << dataset << ',' << algorithm << ',' << n << ',' << dim << ',' << nlist << ','
            << train_count << ',' << pq_m << ',' << (pq_m ? 4 : 0) << ','
            << (pq_m ? by_residual : 0) << ',' << (pq_m ? 32 : 0) << ','
            << std::setprecision(9) << sample_s << ',' << train_s << ',' << add_s
            << ',' << save_s << ',' << sample_s + train_s + add_s << ','
            << file_bytes << ',' << output << '\n';
        std::cout << algorithm << ' ' << dataset << " nodes=" << n << " dim=" << dim
                  << " nlist=" << nlist << " train=" << train_count
                  << " bytes=" << file_bytes << " build_seconds="
                  << sample_s + train_s + add_s << " serialize_seconds=" << save_s << '\n';
    } catch (const std::exception& error) {
        std::cerr << "FAIL baseline build: " << error.what() << '\n';
        return 1;
    }
}
