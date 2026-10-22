// Independent insertion decode-cache / upper-TLS ablation. Stable logical source IDs.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/hnswalg_simplified_alp_PQ.h"

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using ALP = hnswlib::HierarchicalNSWALPSIMPLIFIEDPQ<float>;
using Raw = hnswlib::HierarchicalNSW<float>;
using Label = hnswlib::labeltype;
static constexpr Label invalid_label = std::numeric_limits<Label>::max();

struct Options {
    std::string input, queries, gt, output;
    size_t count = 0, query_count = 0, rounds = 10, search_rounds = 3;
    uint64_t seed = 20261015;
    double delete_ratio = 0.05;
    float alpha = 0.2f, query_alpha = 1.0f;
    std::vector<int> efs{40};
    bool decode_cache = false, upper_tls = false;
    float upper_alpha = 0.2f;
    bool insert_filter = true;
    bool keep_index = false;
};

struct Vectors { size_t count, dim; std::vector<float> values;
    const float* row(size_t id) const { return values.data() + id * dim; }
};

static Vectors load_vectors(const std::string& path, size_t limit) {
    std::ifstream in(path, std::ios::binary);
    int32_t dim = 0;
    if (!in.read(reinterpret_cast<char*>(&dim), 4) || dim <= 0)
        throw std::runtime_error("Invalid fvecs: " + path);
    const size_t row_bytes = (size_t(dim) + 1) * 4, bytes = fs::file_size(path);
    if (bytes % row_bytes) throw std::runtime_error("Truncated fvecs: " + path);
    size_t count = bytes / row_bytes;
    if (limit) count = std::min(count, limit);
    Vectors result{count, size_t(dim), std::vector<float>(count * dim)};
    in.seekg(0);
    for (size_t id = 0; id < count; ++id) {
        int32_t d;
        if (!in.read(reinterpret_cast<char*>(&d), 4) || d != dim ||
            !in.read(reinterpret_cast<char*>(result.values.data() + id * dim), dim * 4))
            throw std::runtime_error("Invalid fvecs row: " + std::to_string(id));
    }
    for (float value : result.values)
        if (!std::isfinite(value)) throw std::runtime_error("Non-finite fvecs coordinate");
    return result;
}

static std::vector<uint32_t> load_truth(const std::string& path, size_t queries, size_t nodes) {
    std::ifstream in(path, std::ios::binary);
    std::vector<uint32_t> truth(queries);
    for (size_t q = 0; q < queries; ++q) {
        int32_t width;
        if (!in.read(reinterpret_cast<char*>(&width), 4) || width <= 0 ||
            !in.read(reinterpret_cast<char*>(&truth[q]), 4) || truth[q] >= nodes)
            throw std::runtime_error("Invalid GT row / wrong dataset prefix: " + std::to_string(q));
        in.seekg(int64_t(width - 1) * 4, std::ios::cur);
    }
    return truth;
}

static double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

static double l2(const float* a, const float* b, size_t dim) {
    double sum = 0;
    for (size_t j = 0; j < dim; ++j) { double d = double(a[j]) - b[j]; sum += d * d; }
    return sum;
}

static ALP::InsertFilterStats stats(const Raw&) { return {}; }
static ALP::InsertFilterStats stats(const ALP& index) { return index.getInsertFilterStats(); }
static void configure(Raw&, bool, float, float) {}
static void configure(ALP& index, bool enabled, float alpha, float query_alpha) {
    index.setInsertUsePQFilter(enabled);
    index.setInsertPQAlpha(alpha);
    index.setUseTLS(query_alpha < 1);
    index.setTLSRatio(query_alpha);
}
static void check_pq(const Raw&) {}
static void check_pq(const ALP& index) {
    if (index.pq_data_.size() != index.cur_element_count * index.pq_m_)
        throw std::runtime_error("PQ codes did not grow with inserted records");
}

template<class Index>
static void run(Index& index, const std::string& mode, bool filter, float alpha,
                const Options& opt, const Vectors& train, const Vectors& queries,
                const std::vector<uint32_t>& truth,
                const std::vector<std::vector<uint32_t>>& updates, std::ofstream& csv) {
    configure(index, filter, alpha, opt.query_alpha);
    if constexpr (std::is_same<Index, ALP>::value) {
        index.setInsertDecodeCache(opt.decode_cache);
        index.setInsertUpperTLS(opt.upper_tls);
        index.setInsertUpperAlpha(opt.upper_alpha);
        index.resetInsertFilterStats();
    }
    std::vector<Label> current(train.count);
    std::iota(current.begin(), current.end(), Label(0));
    std::vector<uint32_t> sources(train.count);
    std::iota(sources.begin(), sources.end(), uint32_t(0));
    const size_t batch_size = updates.front().size();
    const fs::path snapshot = fs::path(opt.output) / (mode + ".bin");
    auto validate_label = [&](Label label) {
        if (label >= sources.size() || current[sources[label]] != label)
            throw std::runtime_error("Deleted/unknown version returned: " + std::to_string(label));
        return sources[label];
    };
    for (size_t batch = 0; batch <= opt.rounds; ++batch) {
        double delete_s = 0, insert_s = 0, max_error = 0;
        if constexpr (std::is_same<Index, ALP>::value) index.resetInsertFilterStats();
        const auto before = stats(index);
        const double graph_before = [&]() {
            if constexpr (std::is_same<Index, ALP>::value) return index.dynamic_graph_us_;
            else return 0.0;
        }();
        const double encoding_before = [&]() {
            if constexpr (std::is_same<Index, ALP>::value) return index.dynamic_encoding_us_;
            else return 0.0;
        }();
        if (batch) {
            auto started = Clock::now();
            for (uint32_t source : updates[batch - 1]) index.markDelete(current[source]);
            delete_s = seconds(started);
            for (uint32_t source : updates[batch - 1]) {
                const Label old = current[source];
                if (!index.isMarkedDeleted(index.label_lookup_.at(old)))
                    throw std::runtime_error("Deletion mark was not set");
                current[source] = invalid_label;
            }
            // Check result exclusion before any reinsertion restores the logical dataset.
            index.setEf(80);
            for (size_t q = 0; q < std::min<size_t>(32, queries.count); ++q) {
                auto result = index.searchKnn(queries.row(q), 1);
                if (result.empty()) throw std::runtime_error("Empty post-delete result");
                validate_label(result.top().second);
            }
            const size_t old_size = sources.size();
            started = Clock::now();
            for (size_t j = 0; j < batch_size; ++j)
                index.addPoint(train.row(updates[batch - 1][j]), Label(old_size + j));
            insert_s = seconds(started);
            if constexpr (std::is_same<Index, ALP>::value) {
                if (index.insert_decode_cache_active_ || !index.insert_decoded_vectors_.empty() ||
                    index.insert_decoded_bytes_ != 0)
                    throw std::runtime_error("Insertion decode cache escaped its lifecycle");
            }
            for (size_t j = 0; j < batch_size; ++j) {
                const uint32_t source = updates[batch - 1][j];
                const Label label = old_size + j;
                sources.push_back(source);
                current[source] = label;
                const auto decoded = index.template getDataByLabel<float>(label);
                for (size_t d = 0; d < train.dim; ++d) {
                    double error = std::abs(double(decoded[d]) - train.row(source)[d]);
                    if (!std::isfinite(error) || error > 1.1e-4)
                        throw std::runtime_error("Inserted vector reconstruction failure at label=" +
                            std::to_string(label) + " coordinate=" + std::to_string(d));
                    max_error = std::max(max_error, error);
                }
            }
        }
        const auto after = stats(index);
        if (index.getCurrentElementCount() != train.count + batch * batch_size ||
            index.getDeletedCount() != batch * batch_size)
            throw std::runtime_error("Physical/deleted record count mismatch");
        check_pq(index);
        // Validate every stored edge, including links of tombstones.
        for (size_t id = 0; id < index.getCurrentElementCount(); ++id) {
            for (int level = 0; level <= index.element_levels_[id]; ++level) {
                auto* links = index.get_linklist_at_level(id, level);
                const size_t count = index.getListCount(links);
                if (count > (level ? index.maxM_ : index.maxM0_))
                    throw std::runtime_error("Neighbor capacity exceeded");
                const auto* neighbors = reinterpret_cast<const hnswlib::tableint*>(links + 1);
                std::unordered_set<hnswlib::tableint> unique;
                for (size_t j = 0; j < count; ++j)
                    if (neighbors[j] >= index.getCurrentElementCount() || neighbors[j] == id ||
                        index.element_levels_[neighbors[j]] < level || !unique.insert(neighbors[j]).second)
                        throw std::runtime_error("Invalid graph edge");
            }
        }
        // Validate all live payloads outside update/query timers, including old
        // records whose neighbor lists may have been rewritten during insertion.
        double max_live_error = 0;
        for (size_t source = 0; source < train.count; ++source) {
            const auto decoded = index.template getDataByLabel<float>(current[source]);
            for (size_t d = 0; d < train.dim; ++d) {
                const double error = std::abs(double(decoded[d]) - train.row(source)[d]);
                if (!std::isfinite(error) || error > 1.1e-4)
                    throw std::runtime_error("Live payload reconstruction failure at source=" +
                        std::to_string(source) + " coordinate=" + std::to_string(d));
                max_live_error = std::max(max_live_error, error);
            }
        }
        index.saveIndex(snapshot.string());
        const auto bytes = fs::file_size(snapshot);
        // Compare complete serialization, not just nearest-neighbor outputs.
        const std::string reference = mode == "alp_cache" ? "alp_base" :
            mode == "alp_joint" ? "alp_upper" :
            mode == "alp_alpha1" ? "alp_exact" : "";
        const fs::path reference_path = fs::path(opt.output) /
            ((reference.empty() ? mode : reference) + "_b" + std::to_string(batch) + ".index");
        if (mode == "alp_base" || mode == "alp_upper" || mode == "alp_exact") {
            fs::copy_file(snapshot, reference_path, fs::copy_options::overwrite_existing);
        } else if (!reference.empty()) {
            if (fs::file_size(reference_path) != bytes) throw std::runtime_error("Index size mismatch: " + mode);
            std::ifstream a(snapshot, std::ios::binary), b(reference_path, std::ios::binary);
            std::vector<char> x(1 << 20), y(1 << 20);
            while (a) {
                a.read(x.data(), x.size()); b.read(y.data(), y.size());
                if (a.gcount() != b.gcount() || !std::equal(x.begin(), x.begin() + a.gcount(), y.begin()))
                    throw std::runtime_error("Serialized index differs with equivalent optimization: " + mode);
            }
            if (!opt.keep_index) fs::remove(reference_path);
        }
        const double graph_us = [&]() {
            if constexpr (std::is_same<Index, ALP>::value) return index.dynamic_graph_us_ - graph_before;
            else return insert_s * 1e6;
        }();
        const double encoding_us = [&]() {
            if constexpr (std::is_same<Index, ALP>::value) return index.dynamic_encoding_us_ - encoding_before;
            else return 0.0;
        }();
        for (int ef : opt.efs) {
            index.setEf(ef);
            for (size_t q = 0; q < std::min<size_t>(100, queries.count); ++q)
                index.searchKnn(queries.row(q), 1);
            for (size_t round = 0; round < opt.search_rounds; ++round) {
                std::vector<Label> labels(queries.count);
                std::vector<float> distances(queries.count);
                auto started = Clock::now();
                for (size_t q = 0; q < queries.count; ++q) {
                    const auto result = index.searchKnn(queries.row(q), 1);
                    if (result.size() != 1) throw std::runtime_error("Empty search result");
                    distances[q] = result.top().first;
                    labels[q] = result.top().second;
                }
                const double search_s = seconds(started);
                size_t hits = 0, tie_hits = 0;
                for (size_t q = 0; q < queries.count; ++q) {
                    const auto source = validate_label(labels[q]);
                    hits += source == truth[q];
                    const double actual = l2(queries.row(q), train.row(source), train.dim);
                    const double target = l2(queries.row(q), train.row(truth[q]), train.dim);
                    if (actual + 1e-8 < target)
                        throw std::runtime_error("GT is not nearest; use GT for this dataset prefix");
                    tie_hits += actual <= target + 1e-8;
                }
                const auto prefix = fs::path(opt.output) / (mode + "_b" + std::to_string(batch) +
                    "_ef" + std::to_string(ef) + "_r" + std::to_string(round));
                std::ofstream label_out(prefix.string() + ".labels", std::ios::binary);
                label_out.write(reinterpret_cast<const char*>(labels.data()), labels.size() * sizeof(Label));
                std::ofstream distance_out(prefix.string() + ".distances", std::ios::binary);
                distance_out.write(reinterpret_cast<const char*>(distances.data()), distances.size() * sizeof(float));
                csv << mode << ',' << opt.decode_cache << ',' << opt.upper_tls << ',' << opt.upper_alpha << ',' << filter << ',' << alpha << ',' << opt.query_alpha << ',' << batch << ','
                    << train.count << ',' << index.getCurrentElementCount() << ',' << index.getDeletedCount() << ','
                    << bytes << ',' << delete_s << ',' << insert_s << ','
                    << (batch ? delete_s * 1e6 / batch_size : 0) << ','
                    << (batch ? insert_s * 1e6 / batch_size : 0) << ','
                    << after.searches - before.searches << ','
                    << after.pq_candidates - before.pq_candidates << ','
                    << after.exact_candidates - before.exact_candidates << ','
                    << after.fallback_searches - before.fallback_searches << ','
                    << after.payload_decodes - before.payload_decodes << ','
                    << after.decode_cache_hits << ',' << after.decode_cache_peak_bytes << ','
                    << after.decode_cache_peak_nodes << ',' << after.upper_searches << ','
                    << after.upper_pq_candidates << ',' << after.upper_exact_candidates << ','
                    << graph_us << ',' << encoding_us << ','
                    << max_error << ',' << train.count << ',' << max_live_error << ','
                    << ef << ',' << round << ',' << queries.count << ','
                    << double(hits) / queries.count << ',' << double(tie_hits) / queries.count << ','
                    << queries.count / search_s << ',' << search_s * 1e6 / queries.count << '\n';
                csv.flush();
            }
        }
        std::cout << "PASS " << mode << " batch=" << batch << " nodes=" << index.getCurrentElementCount()
                  << " deleted=" << index.getDeletedCount() << " bytes=" << bytes
                  << " insert_us=" << (batch ? insert_s * 1e6 / batch_size : 0)
                  << " pq_candidates=" << after.pq_candidates - before.pq_candidates
                  << " exact_candidates=" << after.exact_candidates - before.exact_candidates
                  << " payload_decodes=" << after.payload_decodes - before.payload_decodes
                  << " verified_vectors=" << train.count << " max_error=" << max_live_error << std::endl;
    }
    if (!opt.keep_index) fs::remove(snapshot);
}

int main(int argc, char** argv) {
    try {
        Options opt;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            auto next = [&]() -> std::string {
                if (++i >= argc) throw std::runtime_error("Missing value for " + arg);
                return argv[i];
            };
            if (arg == "--input") opt.input = next();
            else if (arg == "--queries") opt.queries = next();
            else if (arg == "--gt") opt.gt = next();
            else if (arg == "--output") opt.output = next();
            else if (arg == "--count") opt.count = std::stoull(next());
            else if (arg == "--query-count") opt.query_count = std::stoull(next());
            else if (arg == "--rounds") opt.rounds = std::stoull(next());
            else if (arg == "--search-rounds") opt.search_rounds = std::stoull(next());
            else if (arg == "--seed") opt.seed = std::stoull(next());
            else if (arg == "--delete-ratio") opt.delete_ratio = std::stod(next());
            else if (arg == "--upper-alpha") opt.upper_alpha = std::stof(next());
            else if (arg == "--insert-pq-alpha") opt.alpha = std::stof(next());
            else if (arg == "--insert-use-pq-filter") {
                const auto value = next();
                if (value != "0" && value != "1") throw std::runtime_error("Insert filter must be 0 or 1");
                opt.insert_filter = value == "1";
            }
            else if (arg == "--query-pq-alpha") opt.query_alpha = std::stof(next());
            else if (arg == "--keep-index") opt.keep_index = true;
            else if (arg == "--ef") {
                opt.efs.clear();
                while (i + 1 < argc && argv[i + 1][0] != '-') opt.efs.push_back(std::stoi(argv[++i]));
            } else throw std::runtime_error("Unknown option: " + arg);
        }
        if (opt.input.empty() || opt.queries.empty() || opt.gt.empty() || opt.output.empty() ||
            !opt.rounds || !opt.search_rounds || !(opt.delete_ratio > 0 && opt.delete_ratio < 1) ||
            !(opt.upper_alpha > 0 && opt.upper_alpha <= 1) ||
            !(opt.alpha > 0 && opt.alpha <= 1) || !(opt.query_alpha > 0 && opt.query_alpha <= 1) ||
            opt.efs.empty() || *std::min_element(opt.efs.begin(), opt.efs.end()) < 1)
            throw std::runtime_error("Invalid arguments; --input --queries --gt --output are required");
        auto train = load_vectors(opt.input, opt.count), queries = load_vectors(opt.queries, opt.query_count);
        if (train.dim != queries.dim || train.count < 256 || !queries.count)
            throw std::runtime_error("Need >=256 vectors, nonempty queries and matching dimensions");
        const auto truth = load_truth(opt.gt, queries.count, train.count);
        const size_t batch_size = std::max<size_t>(1, size_t(train.count * opt.delete_ratio));
        const size_t capacity = train.count + opt.rounds * batch_size;
        if (capacity >= std::numeric_limits<hnswlib::tableint>::max())
            throw std::runtime_error("Internal ID overflow");
        fs::create_directories(opt.output);
        std::vector<std::vector<uint32_t>> updates(opt.rounds);
        std::mt19937_64 rng(opt.seed);
        std::vector<uint32_t> ids(train.count);
        std::iota(ids.begin(), ids.end(), 0);
        std::ofstream update_csv(fs::path(opt.output) / "updates.csv");
        update_csv << "batch,source_id,new_label\n";
        for (size_t b = 0; b < opt.rounds; ++b) {
            std::shuffle(ids.begin(), ids.end(), rng);
            updates[b].assign(ids.begin(), ids.begin() + batch_size);
            for (size_t j = 0; j < batch_size; ++j)
                update_csv << b + 1 << ',' << updates[b][j] << ',' << train.count + b * batch_size + j << '\n';
        }
        std::ofstream csv(fs::path(opt.output) / "metrics.csv");
        csv << std::setprecision(12);
        csv << "mode,decode_cache,upper_tls,upper_alpha,insert_filter,insert_alpha,query_alpha,batch,active_nodes,physical_nodes,deleted_nodes,"
               "index_file_bytes,delete_seconds,insert_seconds,delete_mean_us,insert_mean_us,filter_searches,"
               "pq_candidates,exact_candidates,fallback_searches,insert_payload_decodes,decode_cache_hits,"
               "decode_cache_peak_bytes,decode_cache_peak_nodes,upper_searches,upper_pq_candidates,upper_exact_candidates,"
               "graph_us,encoding_us,max_insert_abs_error,verified_live_vectors,"
               "max_live_abs_error,ef_search,search_round,"
               "queries,recall_at_1,recall_at_1_ties,qps,mean_query_us\n";
        hnswlib::L2Space space(train.dim);
        {
            Raw index(&space, capacity, 16, 200, 100);
            for (size_t id = 0; id < train.count; ++id) index.addPoint(train.row(id), id);
            run(index, "raw_f32", false, 1, opt, train, queries, truth, updates, csv);
        }
        const auto base_path = fs::path(opt.output) / "alp_initial.bin";
        {
            ALP index(&space, capacity, 16, 200, 100, false, 1e-4f, true);
            for (size_t id = 0; id < train.count; ++id) index.addPoint(train.row(id), id);
            index.compress_dataset(std::max<size_t>(1, train.count / 100));
            index.saveIndex(base_path.string());
        }
        struct Mode { const char* name; bool cache, upper, base; float alpha; };
        for (const Mode mode : {
                Mode{"alp_base", false, false, opt.insert_filter, opt.alpha},
                Mode{"alp_cache", true, false, opt.insert_filter, opt.alpha},
                Mode{"alp_upper", false, true, opt.insert_filter, opt.alpha},
                Mode{"alp_joint", true, true, opt.insert_filter, opt.alpha},
                Mode{"alp_exact", true, false, false, 1},
                Mode{"alp_alpha1", true, true, true, 1}}) {
            ALP index(&space, base_path.string());
            Options configured = opt;
            configured.decode_cache = mode.cache;
            configured.upper_tls = mode.upper;
            if (std::string(mode.name) == "alp_alpha1") configured.upper_alpha = 1;
            run(index, mode.name, mode.base, mode.alpha,
                configured, train, queries, truth, updates, csv);
            bool rejected = false;
            try { index.addPoint(train.row(0), Label(0)); }
            catch (const std::runtime_error&) { rejected = true; }
            if (!rejected || index.insert_decode_cache_active_ || !index.insert_decoded_vectors_.empty())
                throw std::runtime_error("Exception did not clean up insertion cache");
        }
        if (!opt.keep_index) fs::remove(base_path);
        if (!opt.keep_index)
            for (const char* mode : {"alp_base", "alp_upper", "alp_exact"})
                for (size_t batch = 0; batch <= opt.rounds; ++batch)
                    fs::remove(fs::path(opt.output) / (std::string(mode) + "_b" + std::to_string(batch) + ".index"));
        std::cout << "PASS payloads, deletion exclusion, complete-index cache/alpha1 equivalence" << std::endl;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << std::endl;
        return 1;
    }
}
