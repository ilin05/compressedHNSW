#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/compressed_hnsw_framework.h"
#include "../../hnswlib/hnswalg_simplified_alp_PQ.h"

using namespace hnswlib;

static std::vector<float> load_fvecs(const std::string& path, size_t count, size_t& dim) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open " + path);
    int32_t d = 0;
    in.read(reinterpret_cast<char*>(&d), 4);
    if (d <= 0) throw std::runtime_error("Invalid fvecs dimension");
    dim = static_cast<size_t>(d);
    in.seekg(0);
    std::vector<float> values(count * dim);
    for (size_t i = 0; i < count; ++i) {
        in.read(reinterpret_cast<char*>(&d), 4);
        if (!in || static_cast<size_t>(d) != dim ||
            !in.read(reinterpret_cast<char*>(values.data() + i * dim), dim * 4))
            throw std::runtime_error("Missing or inconsistent fvecs row " + std::to_string(i));
    }
    return values;
}

static std::vector<double> as_double(const float* src, size_t dim) {
    return std::vector<double>(src, src + dim);
}

static void check_f32(const std::vector<double>& actual, const float* expected, size_t dim) {
    if (actual.size() != dim) throw std::runtime_error("Wrong decoded dimension");
    for (size_t j = 0; j < dim; ++j) {
        const float value = static_cast<float>(actual[j]);
        if (std::memcmp(&value, expected + j, 4) != 0)
            throw std::runtime_error("LVC F32 reconstruction mismatch at coordinate " + std::to_string(j));
    }
}

static std::vector<std::vector<uint32_t>> exact_gt(const std::vector<float>& train,
        const std::vector<float>& queries, size_t dim, size_t active, size_t qcount,
        const std::unordered_set<uint32_t>& deleted, size_t k) {
    std::vector<std::vector<uint32_t>> gt(qcount);
    for (size_t qi = 0; qi < qcount; ++qi) {
        std::vector<std::pair<double, uint32_t>> distances;
        distances.reserve(active - deleted.size());
        for (uint32_t id = 0; id < active; ++id) {
            if (deleted.count(id)) continue;
            double dist = 0;
            for (size_t j = 0; j < dim; ++j) {
                const double delta = double(queries[qi * dim + j]) - train[size_t(id) * dim + j];
                dist += delta * delta;
            }
            distances.emplace_back(dist, id);
        }
        std::partial_sort(distances.begin(), distances.begin() + k, distances.end());
        for (size_t i = 0; i < k; ++i) gt[qi].push_back(distances[i].second);
    }
    return gt;
}

template <typename Index, typename Query>
static void report_search(const char* name, Index& index, const std::vector<float>& queries,
        size_t dim, const std::vector<std::vector<uint32_t>>& gt,
        const std::unordered_set<uint32_t>& deleted, size_t k) {
    const auto begin = std::chrono::steady_clock::now();
    size_t hits = 0;
    for (size_t qi = 0; qi < gt.size(); ++qi) {
        std::vector<Query> query(queries.begin() + qi * dim, queries.begin() + (qi + 1) * dim);
        auto result = index.searchKnn(query.data(), k);
        std::unordered_set<uint32_t> truth(gt[qi].begin(), gt[qi].end());
        while (!result.empty()) {
            const uint32_t id = static_cast<uint32_t>(result.top().second);
            if (deleted.count(id)) throw std::runtime_error(std::string(name) + " returned deleted node");
            hits += truth.count(id);
            result.pop();
        }
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    std::cout << name << " recall@" << k << '=' << double(hits) / (gt.size() * k)
              << " qps=" << gt.size() / seconds << " mean_us=" << seconds * 1e6 / gt.size() << '\n';
}

template <typename Policy>
static void test_lvc(const char* name, const std::vector<float>& train,
        const std::vector<float>& queries, size_t dim, size_t base, size_t inserted,
        size_t qcount, const std::string& output) {
    L2SpaceDouble space(dim);
    using Index = HierarchicalNSWCABFRAMEWORK<double, Policy>;
    Index index(&space, base + inserted + 1, 16, 200, true, 0, 100, false, true);
    for (size_t id = 0; id < base; ++id) {
        auto row = as_double(train.data() + id * dim, dim);
        index.addPoint(row.data(), id);
    }
    index.compress_dataset(2);
    std::vector<tableint> parents(base);
    for (size_t id = 0; id < base; ++id) parents[id] = index.getPrenodeId(id);
    const auto start = std::chrono::steady_clock::now();
    for (size_t id = base; id < base + inserted; ++id) {
        auto row = as_double(train.data() + id * dim, dim);
        index.addPoint(row.data(), id);
        check_f32(index.template getDataByLabel<double>(id), train.data() + id * dim, dim);
        const tableint parent = index.getPrenodeId(id);
        if (parent != static_cast<tableint>(-1) && index.getPrenodeId(parent) != static_cast<tableint>(-1))
            throw std::runtime_error("chain_max_depth=2 violated");
    }
    const double insert_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    for (size_t id = 0; id < base; ++id)
        if (parents[id] != index.getPrenodeId(id)) throw std::runtime_error("Existing predecessor changed");
    uint32_t root = 0, child = 0;
    bool found = false;
    for (uint32_t id = 1; id < base && !found; ++id)
        if (parents[id] != static_cast<tableint>(-1)) { root = parents[id]; child = id; found = true; }
    if (!found) throw std::runtime_error("No root with child for deletion test");
    const std::unordered_set<uint32_t> deleted{0, static_cast<uint32_t>(base)};
    const auto del_start = std::chrono::steady_clock::now();
    for (uint32_t id : deleted) index.markDelete(id);
    const double delete_us = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - del_start).count();
    auto gt = exact_gt(train, queries, dim, base + inserted, qcount, deleted, 10);
    index.setEf(50);
    index.setUseTls(true);
    index.setTlsRatio(0.2);
    std::cout << name << " insert_us=" << insert_s * 1e6 / inserted
              << " graph_us=" << index.dynamic_graph_us_ / inserted
              << " encoding_us=" << index.dynamic_encoding_us_ / inserted
              << " delete_us=" << delete_us / deleted.size()
              << " index_bytes=" << index.getCompressedIndexSize() << '\n';
    report_search<Index, double>(name, index, queries, dim, gt, deleted, 10);
    size_t expected_deleted = deleted.size();
    if (!deleted.count(root)) {
        index.markDelete(root);
        ++expected_deleted;
    }
    check_f32(index.template getDataByLabel<double>(child), train.data() + size_t(child) * dim, dim);
    {
        auto query = as_double(queries.data(), dim);
        auto results = index.searchKnn(query.data(), 10);
        while (!results.empty()) {
            if (results.top().second == root) throw std::runtime_error("Deleted root returned");
            results.pop();
        }
    }
    const std::string path = output + "/" + name + ".bin";
    index.saveIndex(path);
    Index loaded(&space, path, true);
    if (loaded.getDeletedCount() != expected_deleted) throw std::runtime_error("Deleted count lost on reload");
    auto last = as_double(train.data() + (base + inserted) * dim, dim);
    loaded.addPoint(last.data(), base + inserted);
    check_f32(loaded.template getDataByLabel<double>(base + inserted),
              train.data() + (base + inserted) * dim, dim);
    if (loaded.pq_data_.size() != (base + inserted + 1) * loaded.pq_m_)
        throw std::runtime_error("LVC PQ code count did not grow after reload");
    loaded.setUseTls(true);
    loaded.setTlsRatio(0.2);
    auto post_reload = loaded.searchKnn(last.data(), 10);
    while (!post_reload.empty()) {
        if (post_reload.top().second == root || post_reload.top().second == base)
            throw std::runtime_error("Reloaded LVC returned deleted node");
        post_reload.pop();
    }
}

static void test_alp(const std::vector<float>& train, const std::vector<float>& queries,
        size_t dim, size_t base, size_t inserted, size_t qcount, const std::string& output) {
    L2Space space(dim);
    using Index = HierarchicalNSWALPSIMPLIFIEDPQ<float>;
    Index index(&space, base + inserted + 1, 16, 200, 100, false, 1e-4f, true);
    for (size_t id = 0; id < base; ++id) index.addPoint(train.data() + id * dim, id);
    index.compress_dataset();
    const auto start = std::chrono::steady_clock::now();
    double max_error = 0;
    for (size_t id = base; id < base + inserted; ++id) {
        index.addPoint(train.data() + id * dim, id);
        const auto decoded = index.getDataByLabel<float>(id);
        for (size_t j = 0; j < dim; ++j)
            max_error = std::max(max_error, double(std::abs(decoded[j] - train[id * dim + j])));
    }
    const double insert_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const std::unordered_set<uint32_t> deleted{0, static_cast<uint32_t>(base)};
    const auto del_start = std::chrono::steady_clock::now();
    for (uint32_t id : deleted) index.markDelete(id);
    const double delete_us = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - del_start).count();
    auto gt = exact_gt(train, queries, dim, base + inserted, qcount, deleted, 10);
    index.setEf(50);
    index.setUseTLS(true);
    index.setTLSRatio(0.2f);
    std::cout << "alp insert_us=" << insert_s * 1e6 / inserted
              << " graph_us=" << index.dynamic_graph_us_ / inserted
              << " encoding_us=" << index.dynamic_encoding_us_ / inserted
              << " delete_us=" << delete_us / deleted.size()
              << " max_insert_abs_error=" << max_error
              << " index_bytes=" << index.getCompressedIndexSize() << '\n';
    report_search<Index, float>("alp", index, queries, dim, gt, deleted, 10);
    const std::string path = output + "/alp.bin";
    index.saveIndex(path);
    Index loaded(&space, path);
    if (loaded.getDeletedCount() != deleted.size()) throw std::runtime_error("ALP deletion lost on reload");
    loaded.addPoint(train.data() + (base + inserted) * dim, base + inserted);
    const auto decoded = loaded.getDataByLabel<float>(base + inserted);
    for (size_t j = 0; j < dim; ++j)
        if (std::abs(decoded[j] - train[(base + inserted) * dim + j]) > 1e-2f)
            throw std::runtime_error("ALP reloaded insertion error exceeds 1e-2");
    if (loaded.pq_data_.size() != (base + inserted + 1) * loaded.pq_m_)
        throw std::runtime_error("ALP PQ code count did not grow after reload");
    loaded.setUseTLS(true);
    loaded.setTLSRatio(0.2f);
    auto post_reload = loaded.searchKnn(train.data() + (base + inserted) * dim, 10);
    while (!post_reload.empty()) {
        if (post_reload.top().second == 0 || post_reload.top().second == base)
            throw std::runtime_error("Reloaded ALP returned deleted node");
        post_reload.pop();
    }
}

static void test_raw(const std::vector<float>& train, const std::vector<float>& queries,
        size_t dim, size_t base, size_t inserted, size_t qcount, const std::string& output) {
    L2SpaceDouble space(dim);
    HierarchicalNSW<double> index(&space, base + inserted + 1, 16, 200);
    for (size_t id = 0; id < base; ++id) {
        auto row = as_double(train.data() + id * dim, dim);
        index.addPoint(row.data(), id);
    }
    const auto start = std::chrono::steady_clock::now();
    for (size_t id = base; id < base + inserted; ++id) {
        auto row = as_double(train.data() + id * dim, dim);
        index.addPoint(row.data(), id);
    }
    const double insert_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const std::unordered_set<uint32_t> deleted{0, static_cast<uint32_t>(base)};
    const auto del_start = std::chrono::steady_clock::now();
    for (uint32_t id : deleted) index.markDelete(id);
    const double delete_us = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - del_start).count();
    index.setEf(50);
    const std::string path = output + "/raw.bin";
    index.saveIndex(path);
    std::cout << "raw insert_us=" << insert_s * 1e6 / inserted
              << " delete_us=" << delete_us / deleted.size()
              << " index_bytes=" << std::filesystem::file_size(path) << '\n';
    auto gt = exact_gt(train, queries, dim, base + inserted, qcount, deleted, 10);
    report_search<HierarchicalNSW<double>, double>("raw", index, queries, dim, gt, deleted, 10);
}

static void test_raw_f32(const std::vector<float>& train, const std::vector<float>& queries,
        size_t dim, size_t base, size_t inserted, size_t qcount, const std::string& output) {
    L2Space space(dim);
    HierarchicalNSW<float> index(&space, base + inserted + 1, 16, 200);
    for (size_t id = 0; id < base; ++id) index.addPoint(train.data() + id * dim, id);
    const auto start = std::chrono::steady_clock::now();
    for (size_t id = base; id < base + inserted; ++id) index.addPoint(train.data() + id * dim, id);
    const double insert_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const std::unordered_set<uint32_t> deleted{0, static_cast<uint32_t>(base)};
    const auto del_start = std::chrono::steady_clock::now();
    for (uint32_t id : deleted) index.markDelete(id);
    const double delete_us = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - del_start).count();
    index.setEf(50);
    const std::string path = output + "/raw_f32.bin";
    index.saveIndex(path);
    std::cout << "raw_f32 insert_us=" << insert_s * 1e6 / inserted
              << " delete_us=" << delete_us / deleted.size()
              << " index_bytes=" << std::filesystem::file_size(path) << '\n';
    auto gt = exact_gt(train, queries, dim, base + inserted, qcount, deleted, 10);
    report_search<HierarchicalNSW<float>, float>("raw_f32", index, queries, dim, gt, deleted, 10);
}

int main(int argc, char** argv) {
    try {
        if (argc != 7) throw std::runtime_error(
            "Usage: test_dynamic_compressed_hnsw TRAIN.fvecs TEST.fvecs BASE INSERT QUERIES OUTPUT_DIR");
        const size_t base = std::stoul(argv[3]), inserted = std::stoul(argv[4]);
        const size_t qcount = std::stoul(argv[5]);
        if (base < 256 || inserted < 2 || qcount == 0)
            throw std::runtime_error("Need BASE>=256, INSERT>=2, QUERIES>=1");
        std::filesystem::create_directories(argv[6]);
        size_t dim = 0, query_dim = 0;
        auto train = load_fvecs(argv[1], base + inserted + 1, dim);
        auto queries = load_fvecs(argv[2], qcount, query_dim);
        if (dim != query_dim) throw std::runtime_error("Train/query dimensions differ");
        test_raw(train, queries, dim, base, inserted, qcount, argv[6]);
        test_raw_f32(train, queries, dim, base, inserted, qcount, argv[6]);
        test_alp(train, queries, dim, base, inserted, qcount, argv[6]);
        test_lvc<codecs::DeXORCodecPolicy>("dexor", train, queries, dim, base, inserted, qcount, argv[6]);
        test_lvc<codecs::GorillaCodecPolicy>("gorilla", train, queries, dim, base, inserted, qcount, argv[6]);
        test_lvc<codecs::ElfCodecPolicy>("elf", train, queries, dim, base, inserted, qcount, argv[6]);
        std::cout << "PASS dynamic LVC SIFT updates\n";
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
