#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>

#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/compressed_hnsw_framework.h"
#include "../../hnswlib/compressed_codecs.h"

using namespace hnswlib;

static std::vector<double> read_fvecs(const std::string& path, size_t& count, size_t& dim) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("Cannot open " + path);
    const auto bytes = static_cast<size_t>(in.tellg());
    in.seekg(0);
    int32_t width = 0;
    in.read(reinterpret_cast<char*>(&width), sizeof(width));
    if (width <= 0) throw std::runtime_error("Invalid fvecs dimension");
    dim = static_cast<size_t>(width);
    const size_t row_bytes = sizeof(width) + dim * sizeof(float);
    if (bytes % row_bytes) throw std::runtime_error("Invalid fvecs length");
    count = bytes / row_bytes;
    std::vector<double> values(count * dim);
    std::vector<float> row(dim);
    in.seekg(0);
    for (size_t i = 0; i < count; ++i) {
        int32_t actual = 0;
        in.read(reinterpret_cast<char*>(&actual), sizeof(actual));
        in.read(reinterpret_cast<char*>(row.data()), dim * sizeof(float));
        if (!in || actual != width) throw std::runtime_error("Malformed fvecs row");
        for (size_t j = 0; j < dim; ++j) values[i * dim + j] = row[j];
    }
    return values;
}

int main(int argc, char** argv) {
    std::string train, queries, output;
    int threads = 32, M = 16, ef_construction = 200, ef = 100;
    uint32_t seed = 20261012;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (i + 1 >= argc) throw std::runtime_error("Missing value for " + arg);
        if (arg == "--train") train = argv[++i];
        else if (arg == "--queries") queries = argv[++i];
        else if (arg == "--output") output = argv[++i];
        else if (arg == "--threads") threads = std::stoi(argv[++i]);
        else if (arg == "--M") M = std::stoi(argv[++i]);
        else if (arg == "--ef-construction") ef_construction = std::stoi(argv[++i]);
        else if (arg == "--ef") ef = std::stoi(argv[++i]);
        else if (arg == "--seed") seed = static_cast<uint32_t>(std::stoul(argv[++i]));
        else throw std::runtime_error("Unknown option " + arg);
    }
    if (train.empty() || queries.empty() || output.empty() || threads < 1 || ef < 1)
        throw std::runtime_error("Required: --train --queries --output; threads and ef must be positive");
    size_t n = 0, dim = 0, nq = 0, qdim = 0;
    auto base = read_fvecs(train, n, dim);
    auto query = read_fvecs(queries, nq, qdim);
    if (dim != qdim || nq < 2) throw std::runtime_error("Invalid query shape");
    std::filesystem::create_directories(output);
    omp_set_num_threads(threads);
    L2SpaceDouble space(dim);
    HierarchicalNSWCABFRAMEWORK<double, codecs::DeXORCodecPolicy> graph(
        &space, n, M, ef_construction, true, 0, 100, false);
    #pragma omp parallel for
    for (long long i = 0; i < static_cast<long long>(n); ++i)
        graph.addPoint(base.data() + i * dim, static_cast<labeltype>(i));
    const std::string graph_file = output + "/shared_raw_graph.bin";
    graph.saveIndex(graph_file);
    std::vector<size_t> ids(nq);
    std::iota(ids.begin(), ids.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(ids.begin(), ids.end(), rng);
    const size_t trace_count = nq / 2;
    graph.setEf(static_cast<size_t>(ef));
    graph.startAccessTrace();
    for (size_t i = 0; i < trace_count; ++i)
        graph.searchKnn(query.data() + ids[i] * dim, 1);
    graph.stopAccessTrace();
    std::ofstream counts(output + "/access_counts.txt");
    const auto& values = graph.getAccessTraceCounts();
    uint64_t total = 0;
    for (size_t i = 0; i < values.size(); ++i) {
        counts << i << ' ' << values[i] << '\n';
        total += values[i];
    }
    std::ofstream eval(output + "/eval_query_ids.txt");
    for (size_t i = trace_count; i < nq; ++i) eval << ids[i] << '\n';
    std::cout << "graph=" << graph_file << " nodes=" << n << " dim=" << dim
              << " trace_queries=" << trace_count << " eval_queries=" << nq - trace_count
              << " access_events=" << total << " ef=" << ef << '\n';
}
