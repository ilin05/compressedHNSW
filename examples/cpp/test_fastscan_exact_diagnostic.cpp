#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQFastScan.h>
#include <faiss/index_io.h>
#include <faiss/utils/distances.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
double micros(Clock::time_point start) {
    return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}
template <typename T> struct Matrix {
    size_t n = 0, d = 0;
    std::vector<T> values;
};
template <typename T> Matrix<T> read_vecs(const std::string &path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("Cannot open " + path);
    const auto bytes = input.tellg();
    input.seekg(0);
    int32_t d = 0;
    input.read(reinterpret_cast<char *>(&d), 4);
    if (!input || d <= 0 || bytes % (4 + std::streamoff(d) * sizeof(T)))
        throw std::runtime_error("Invalid vecs shape: " + path);
    Matrix<T> result;
    result.d = d; result.n = size_t(bytes / (4 + std::streamoff(d) * sizeof(T)));
    result.values.resize(result.n * result.d);
    input.seekg(0);
    for (size_t i = 0; i < result.n; ++i) {
        int32_t width;
        input.read(reinterpret_cast<char *>(&width), 4);
        input.read(reinterpret_cast<char *>(result.values.data() + i * result.d), result.d * sizeof(T));
        if (!input || width != d) throw std::runtime_error("Invalid/truncated vecs row: " + path);
    }
    return result;
}
template <typename T> void write_vecs(const std::string &path, const Matrix<T> &matrix) {
    std::ofstream output(path, std::ios::binary);
    const int32_t d = static_cast<int32_t>(matrix.d);
    for (size_t i = 0; i < matrix.n; ++i) {
        output.write(reinterpret_cast<const char *>(&d), 4);
        output.write(reinterpret_cast<const char *>(matrix.values.data() + i * matrix.d), matrix.d * sizeof(T));
    }
    if (!output) throw std::runtime_error("Cannot write fixture: " + path);
}
void make_fixture(const std::string &directory) {
    std::filesystem::create_directories(directory);
    Matrix<float> base{1024, 8, std::vector<float>(1024 * 8)};
    Matrix<float> queries{64, 8, std::vector<float>(64 * 8)};
    std::mt19937 random(20261022);
    std::uniform_real_distribution<float> sample(-5, 5);
    for (float &v : base.values) v = sample(random);
    std::copy_n(base.values.data(), base.d, base.values.data() + base.d);
    for (float &v : queries.values) v = sample(random);
    std::copy_n(base.values.data(), base.d, queries.values.data());
    faiss::IndexFlatL2 quantizer(base.d);
    faiss::IndexIVFPQFastScan fast(&quantizer, base.d, 4, base.d, 4);
    fast.by_residual = true;
    fast.train(base.n, base.values.data()); fast.add(base.n, base.values.data());
    faiss::write_index(&fast, (directory + "/index.bin").c_str());
    faiss::IndexFlatL2 exact(base.d);
    exact.add(base.n, base.values.data());
    std::vector<float> distances(queries.n);
    std::vector<faiss::idx_t> ids(queries.n);
    exact.search(queries.n, queries.values.data(), 1, distances.data(), ids.data());
    Matrix<int32_t> gt{queries.n, 1, std::vector<int32_t>(ids.begin(), ids.end())};
    gt.values[0] = 1; // IDs 0 and 1 are equal-distance true nearest neighbors.
    write_vecs(directory + "/base.fvecs", base);
    write_vecs(directory + "/query.fvecs", queries);
    write_vecs(directory + "/gt.ivecs", gt);
}
} // namespace

int main(int argc, char **argv) {
    try {
        omp_set_num_threads(1);
        if (argc == 3 && std::string(argv[1]) == "--make-fixture") {
            make_fixture(argv[2]); return 0;
        }
        std::string index_path, base_path, query_path, gt_path, output, dataset = "sift-128-euclidean";
        size_t rounds = 3, batch = 32, fullscan_queries = 100, query_count = 0;
        std::vector<size_t> probes{1, 2, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100};
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--nprobe") {
                probes.clear();
                while (i + 1 < argc && argv[i + 1][0] != '-') probes.push_back(std::stoul(argv[++i]));
                continue;
            }
            if (++i >= argc) throw std::invalid_argument("Missing option value");
            const std::string value = argv[i];
            if (option == "--index") index_path = value;
            else if (option == "--base") base_path = value;
            else if (option == "--queries") query_path = value;
            else if (option == "--ground-truth") gt_path = value;
            else if (option == "--output") output = value;
            else if (option == "--dataset") dataset = value;
            else if (option == "--rounds") rounds = std::stoul(value);
            else if (option == "--batch") batch = std::stoul(value);
            else if (option == "--query-count") query_count = std::stoul(value);
            else if (option == "--fullscan-queries") fullscan_queries = std::stoul(value);
            else throw std::invalid_argument("Unknown option " + option);
        }
        if (index_path.empty() || base_path.empty() || query_path.empty() || gt_path.empty() || output.empty() ||
            !rounds || !batch || probes.empty() || std::find(probes.begin(), probes.end(), 0) != probes.end())
            throw std::invalid_argument("Invalid diagnostic arguments");
        const auto base = read_vecs<float>(base_path);
        const auto queries = read_vecs<float>(query_path);
        const auto gt = read_vecs<int32_t>(gt_path);
        std::unique_ptr<faiss::Index> index(faiss::read_index(index_path.c_str()));
        auto *fast = dynamic_cast<faiss::IndexIVFPQFastScan *>(index.get());
        if (!fast || fast->metric_type != faiss::METRIC_L2 || fast->pq.M != base.d || fast->pq.nbits != 4 ||
            size_t(fast->d) != base.d || size_t(fast->ntotal) != base.n || base.d != queries.d ||
            queries.n != gt.n || !queries.n || !fast->is_trained || fast->skip != 0)
            throw std::runtime_error("Expected matching F32/L2 data and trained M=dim, 4-bit FastScan index");
        // Validate that stored IDs are exactly the original dataset row IDs.
        std::vector<int32_t> owner(base.n, -1);
        size_t total_ids = 0;
        for (size_t list = 0; list < fast->nlist; ++list) {
            faiss::InvertedLists::ScopedIds ids(fast->invlists, list);
            for (size_t j = 0; j < fast->invlists->list_size(list); ++j) {
                const auto id = ids[j];
                if (id < 0 || size_t(id) >= base.n || owner[id] != -1)
                    throw std::runtime_error("Index IDs are not a permutation of dataset row IDs");
                owner[id] = static_cast<int32_t>(list); ++total_ids;
            }
        }
        if (total_ids != base.n) throw std::runtime_error("Index does not cover the full dataset");
        const size_t all_queries = query_count ? std::min(query_count, queries.n) : queries.n;
        for (size_t i = 0; i < all_queries; ++i)
            if (gt.values[i * gt.d] < 0 || size_t(gt.values[i * gt.d]) >= base.n)
                throw std::runtime_error("Invalid ground-truth ID");
        if (fullscan_queries) probes.push_back(fast->nlist);
        std::sort(probes.begin(), probes.end()); probes.erase(std::unique(probes.begin(), probes.end()), probes.end());
        std::ofstream csv(output);
        if (!csv) throw std::runtime_error("Cannot open output CSV");
        csv << "Dataset,Mode,NProbe,QueryCount,Rounds,RecallAt1,GTInListsAt1,DistanceRecallAt1,"
               "CoarseLatencyUS,ScanLatencyUS,LatencyUS,QPS,MeanCandidates,NList,Dim,PQM,PQBits,ByResidual,CodeBytes\n";
        csv << std::setprecision(10);
        for (size_t nprobe : probes) {
            if (nprobe > fast->nlist) continue;
            fast->nprobe = nprobe;
            const size_t nq = nprobe == fast->nlist && fullscan_queries
                ? std::min(all_queries, fullscan_queries) : all_queries;
            std::vector<faiss::idx_t> lists(nq * nprobe);
            std::vector<float> centroid_distances(nq * nprobe);
            double coarse_us = 0;
            // Identical query batches for native and preassigned coarse search.
            for (size_t start = 0; start < nq; start += batch) {
                const size_t take = std::min(batch, nq - start);
                const auto tick = Clock::now();
                fast->quantizer->search(take, queries.values.data() + start * base.d, nprobe,
                    centroid_distances.data() + start * nprobe, lists.data() + start * nprobe);
                coarse_us += micros(tick);
            }
            uint64_t candidates = 0, covered = 0;
            for (size_t i = 0; i < nq; ++i) {
                bool contains_gt = false;
                for (size_t j = 0; j < nprobe; ++j) {
                    const auto list = lists[i * nprobe + j];
                    if (list < 0) continue;
                    candidates += fast->invlists->list_size(list);
                    contains_gt |= owner[gt.values[i * gt.d]] == list;
                }
                covered += contains_gt;
            }
            std::vector<float> distances(nq);
            std::vector<faiss::idx_t> ids(nq);
            auto scan = [&](bool exact, size_t count) {
                if (!exact) {
                    for (size_t start = 0; start < count; start += batch) {
                        const size_t take = std::min(batch, count - start);
                        fast->search_preassigned(take, queries.values.data() + start * base.d, 1,
                            lists.data() + start * nprobe, centroid_distances.data() + start * nprobe,
                            distances.data() + start, ids.data() + start, false);
                    }
                } else {
                    for (size_t i = 0; i < count; ++i) {
                        float best = std::numeric_limits<float>::infinity();
                        faiss::idx_t chosen = -1;
                        for (size_t j = 0; j < nprobe; ++j) {
                            const auto list = lists[i * nprobe + j];
                            if (list < 0) continue;
                            faiss::InvertedLists::ScopedIds members(fast->invlists, list);
                            for (size_t k = 0; k < fast->invlists->list_size(list); ++k) {
                                const auto candidate = members[k];
                                const float d = faiss::fvec_L2sqr(queries.values.data() + i * base.d,
                                    base.values.data() + candidate * base.d, base.d);
                                if (d < best || (d == best && candidate < chosen)) { best = d; chosen = candidate; }
                            }
                        }
                        distances[i] = best; ids[i] = chosen;
                    }
                }
            };
            for (bool exact : {false, true}) {
                scan(exact, std::min(nq, batch)); // warmup, outside timing
                if (!exact) {
                    const size_t take = std::min(nq, batch);
                    std::vector<float> native_d(take);
                    std::vector<faiss::idx_t> native_ids(take);
                    fast->search(take, queries.values.data(), 1, native_d.data(), native_ids.data());
                    for (size_t i = 0; i < take; ++i)
                        if (native_ids[i] != ids[i] || native_d[i] != distances[i])
                            throw std::runtime_error("Native/preassigned FastScan mismatch");
                }
                double scan_us = 0;
                for (size_t r = 0; r < rounds; ++r) {
                    const auto tick = Clock::now(); scan(exact, nq); scan_us += micros(tick);
                }
                uint64_t hits = 0, distance_hits = 0;
                for (size_t i = 0; i < nq; ++i) {
                    hits += ids[i] == gt.values[i * gt.d];
                    if (ids[i] < 0) continue;
                    const float truth_d = faiss::fvec_L2sqr(queries.values.data() + i * base.d,
                        base.values.data() + size_t(gt.values[i * gt.d]) * base.d, base.d);
                    const float actual_d = faiss::fvec_L2sqr(queries.values.data() + i * base.d,
                        base.values.data() + size_t(ids[i]) * base.d, base.d);
                    distance_hits += std::abs(actual_d - truth_d) <= 1e-5f * std::max(1.0f, std::abs(truth_d));
                }
                const double latency = coarse_us / nq + scan_us / (rounds * nq);
                const char *mode = exact ? "SharedPartition-Exact-F32" : "IVFPQ-FastScan";
                csv << dataset << ',' << mode << ',' << nprobe << ',' << nq << ',' << rounds << ','
                    << double(hits) / nq << ',' << double(covered) / nq << ',' << double(distance_hits) / nq << ','
                    << coarse_us / nq << ',' << scan_us / (rounds * nq) << ',' << latency << ',' << 1e6 / latency << ','
                    << double(candidates) / nq << ',' << fast->nlist << ',' << base.d << ',' << fast->pq.M << ','
                    << fast->pq.nbits << ',' << fast->by_residual << ',' << fast->code_size << '\n';
                csv.flush();
                std::cout << mode << " nprobe=" << nprobe << " queries=" << nq << " recall@1="
                          << double(hits) / nq << " coverage=" << double(covered) / nq << std::endl;
                if (exact && nprobe == fast->nlist && distance_hits != nq)
                    throw std::runtime_error("Full-scan Exact-F32 disagrees with GT distances; check data/metric/IDs");
            }
        }
    } catch (const std::exception &error) {
        std::cerr << "FastScan diagnostic failed: " << error.what() << '\n'; return 1;
    }
}
