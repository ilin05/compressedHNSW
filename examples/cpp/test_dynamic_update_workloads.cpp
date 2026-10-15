// Full-dataset dynamic ALP workloads. One process owns one insertion policy.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#include <omp.h>
#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/hnswalg_simplified_alp_PQ.h"

namespace fs = std::filesystem;
using ALP = hnswlib::HierarchicalNSWALPSIMPLIFIEDPQ<float>;
using Raw = hnswlib::HierarchicalNSW<float>;
using Label = hnswlib::labeltype;
using Clock = std::chrono::steady_clock;
static constexpr Label invalid_label = std::numeric_limits<Label>::max();
static double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

// Source data remain mapped rather than copied into another full vector array.
struct Fvecs {
    int fd = -1;
    void* memory = MAP_FAILED;
    size_t bytes = 0, count = 0, dim = 0;
    explicit Fvecs(const std::string& path, size_t limit = 0) {
        bytes = fs::file_size(path);
        fd = open(path.c_str(), O_RDONLY);
        if (fd < 0 || bytes < 4) throw std::runtime_error("Cannot open fvecs: " + path);
        memory = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
        if (memory == MAP_FAILED) throw std::runtime_error("Cannot mmap fvecs: " + path);
        int32_t d;
        std::memcpy(&d, memory, 4);
        if (d <= 0 || bytes % ((size_t(d) + 1) * 4))
            throw std::runtime_error("Invalid fvecs shape: " + path);
        dim = d;
        count = bytes / ((dim + 1) * 4);
        if (limit > count) throw std::runtime_error("Requested fvecs limit exceeds file rows");
        if (limit) count = limit;
    }
    Fvecs(const Fvecs&) = delete;
    ~Fvecs() { if (memory != MAP_FAILED) munmap(memory, bytes); if (fd >= 0) close(fd); }
    const float* row(size_t id) const {
        if (id >= count) throw std::runtime_error("Source ID outside dataset");
        const char* ptr = static_cast<const char*>(memory) + id * (dim + 1) * 4;
        int32_t d;
        std::memcpy(&d, ptr, 4);
        if (size_t(d) != dim) throw std::runtime_error("Inconsistent fvecs row dimension");
        return reinterpret_cast<const float*>(ptr + 4);
    }
};

struct Options {
    std::string dataset, experiment, mode, input, queries, gt, gt_dir, initial_index, output;
    size_t count = 0, query_count = 0, batches = 50, search_rounds = 3;
    size_t M = 16, ef_construction = 200;
    uint64_t seed = 20261015;
    std::vector<int> efs{20, 40, 80};
    bool verify_size = false;
};

static std::vector<uint32_t> load_truth(const fs::path& path, size_t queries, size_t active) {
    std::ifstream in(path, std::ios::binary);
    std::vector<uint32_t> result(queries);
    for (size_t q = 0; q < queries; ++q) {
        int32_t width;
        if (!in.read(reinterpret_cast<char*>(&width), 4) || width <= 0 ||
            !in.read(reinterpret_cast<char*>(&result[q]), 4) || result[q] >= active)
            throw std::runtime_error("Invalid/wrong-prefix GT: " + path.string());
        in.seekg(int64_t(width - 1) * 4, std::ios::cur);
        if (!in) throw std::runtime_error("Truncated GT: " + path.string());
    }
    return result;
}

static double l2(const float* a, const float* b, size_t dim) {
    double sum = 0;
    for (size_t j = 0; j < dim; ++j) { const double d = double(a[j]) - b[j]; sum += d * d; }
    return sum;
}

template<class Index> static size_t header_bytes(const Index& i) {
    return sizeof(i.offsetLevel0_) + sizeof(i.max_elements_) + sizeof(i.cur_element_count) +
        sizeof(i.size_data_per_element_) + sizeof(i.label_offset_) + sizeof(i.offsetData_) +
        sizeof(i.maxlevel_) + sizeof(i.enterpoint_node_) + sizeof(i.maxM_) + sizeof(i.maxM0_) +
        sizeof(i.M_) + sizeof(i.mult_) + sizeof(i.ef_construction_);
}
static size_t serialized_bytes(const Raw& index) { return index.indexFileSize(); }
static size_t serialized_bytes(const ALP& i) {
    // Match saveIndex() exactly, including offsets for allocated ID capacity.
    size_t bytes = header_bytes(i) + sizeof(i.is_compacted_) + sizeof(i.pq_m_) + sizeof(i.pq_d_sub_);
    for (const auto& c : i.pq_centroids_) bytes += sizeof(size_t) + c.size() * sizeof(float);
    bytes += sizeof(size_t) + i.pq_data_.size();
    bytes += sizeof(size_t) + i.level0_element_start_positions_.size() * sizeof(size_t);
    bytes += sizeof(size_t) + i.uncompressed_mask_.size();
    bytes += sizeof(size_t) + i.data_level0_memory_.size();
    for (size_t id = 0; id < i.cur_element_count; ++id)
        bytes += sizeof(unsigned int) + i.size_links_per_element_ * i.element_levels_[id];
    return bytes + sizeof(uint32_t) + sizeof(i.enable_dynamic_updates_) + sizeof(i.reconstruction_tolerance_);
}
static ALP::InsertFilterStats counters(const Raw&) { return {}; }
static ALP::InsertFilterStats counters(const ALP& i) { return i.getInsertFilterStats(); }
static std::pair<double, double> timers(const Raw&) { return {}; }
static std::pair<double, double> timers(const ALP& i) { return {i.dynamic_graph_us_, i.dynamic_encoding_us_}; }
static size_t payload_capacity(const Raw& i) { return i.max_elements_ * i.size_data_per_element_; }
static size_t payload_capacity(const ALP& i) { return i.data_level0_memory_.capacity(); }
static void query_config(Raw&, bool) {}
static void query_config(ALP& i, bool tls) { i.setUseTLS(tls); i.setTLSRatio(0.2f); }
static void check_pq(const Raw&) {}
static void check_pq(const ALP& i) {
    if (i.pq_data_.size() != i.cur_element_count * i.pq_m_)
        throw std::runtime_error("PQ record count mismatch");
}
static std::pair<size_t, size_t> rss_kib() {
    struct rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    size_t total = 0, resident = 0;
    std::ifstream in("/proc/self/statm");
    in >> total >> resident;
    return {resident * size_t(sysconf(_SC_PAGESIZE)) / 1024, size_t(usage.ru_maxrss)};
}

template<class Index>
static void verify_serialization(Index& index, const fs::path& path) {
    index.saveIndex(path.string());
    if (fs::file_size(path) != serialized_bytes(index))
        throw std::runtime_error("Serialized-size accounting disagrees with saved file");
    fs::remove(path);
}

template<class Index>
static void workload(Index& index, const Options& o, const Fvecs& train, const Fvecs& queries,
                     size_t initial, double build_seconds, double compression_seconds) {
    const bool repeated = o.experiment == "repeated";
    const size_t steps = repeated ? o.batches : 10;
    const size_t repeat_batch = std::max<size_t>(1, train.count * 5 / 100);
    std::vector<Label> current(initial);
    std::iota(current.begin(), current.end(), Label(0));
    std::vector<uint32_t> sources(initial);
    std::iota(sources.begin(), sources.end(), uint32_t(0));
    std::vector<uint32_t> shuffled;
    if (repeated) { shuffled.resize(train.count); std::iota(shuffled.begin(), shuffled.end(), 0); }
    std::mt19937_64 random(o.seed);
    fs::create_directories(o.output);
    std::ofstream updates(fs::path(o.output) / "updates.csv"), search(fs::path(o.output) / "search.csv");
    updates.exceptions(std::ios::failbit | std::ios::badbit);
    search.exceptions(std::ios::failbit | std::ios::badbit);
    updates << std::setprecision(12);
    search << std::setprecision(12);
    const std::string identity = o.dataset + ',' + o.experiment + ',' + o.mode + ',';
    updates << "dataset,experiment,insert_mode,batch,prefix_percent,active_nodes,physical_nodes,deleted_nodes,"
        "capacity_nodes,inserted_nodes,marked_nodes,index_file_bytes,record_buffer_capacity_bytes,"
        "insert_seconds,delete_seconds,resize_seconds,insert_mean_us,delete_mean_us,graph_mean_us,"
        "encoding_mean_us,filter_searches,pq_candidates,exact_candidates,fallback_searches,payload_decodes,"
        "verified_vectors,max_abs_error,update_hash,current_process_rss_kib,peak_process_rss_kib,"
        "initial_build_seconds,initial_compression_seconds,threads\n";
    search << "dataset,experiment,insert_mode,batch,prefix_percent,query_mode,query_alpha,ef_search,"
        "search_round,queries,recall_at_1,recall_at_1_ties,qps,mean_query_us,threads\n";
    const auto full_truth = repeated ? load_truth(o.gt, queries.count, train.count) : std::vector<uint32_t>{};
    auto valid_source = [&](Label label) {
        if (label >= sources.size() || sources[label] >= current.size() || current[sources[label]] != label)
            throw std::runtime_error("Deleted/unknown version returned: " + std::to_string(label));
        return sources[label];
    };
    for (size_t batch = 0; batch <= steps; ++batch) {
        const int percent = repeated ? 100 : 50 + int(batch) * 5;
        const size_t active = repeated ? train.count : train.count * size_t(percent) / 100;
        std::vector<uint32_t> selected;
        if (batch) {
            if (repeated) {
                std::shuffle(shuffled.begin(), shuffled.end(), random);
                selected.assign(shuffled.begin(), shuffled.begin() + repeat_batch);
            } else {
                const size_t previous = train.count * size_t(percent - 5) / 100;
                selected.resize(active - previous);
                std::iota(selected.begin(), selected.end(), uint32_t(previous));
            }
        }
        uint64_t hash = 1469598103934665603ULL;
        for (uint32_t source : selected) { hash ^= source; hash *= 1099511628211ULL; }
        const auto before = counters(index);
        const auto stage_before = timers(index);
        double insert_s = 0, delete_s = 0, resize_s = 0;
        if (!selected.empty()) {
            if (repeated) {
                auto start = Clock::now();
                for (uint32_t source : selected) index.markDelete(current[source]);
                delete_s = seconds(start);
                for (uint32_t source : selected) {
                    if (!index.isMarkedDeleted(index.label_lookup_.at(current[source])))
                        throw std::runtime_error("Missing deletion mark");
                    current[source] = invalid_label;
                }
                query_config(index, false);
                index.setEf(o.efs.back());
                for (size_t q = 0; q < std::min<size_t>(32, queries.count); ++q) {
                    auto result = index.searchKnn(queries.row(q), 1);
                    if (result.empty()) throw std::runtime_error("No post-delete result");
                    valid_source(result.top().second);
                }
            }
            // Same capacity policy for both implementations. Expansion is part
            // of end-to-end insertion time and is also reported separately.
            auto start = Clock::now();
            if (index.getMaxElements() < sources.size() + selected.size())
                index.resizeIndex(sources.size() + selected.size());
            resize_s = seconds(start);
            const size_t old_records = sources.size();
            start = Clock::now();
            for (size_t j = 0; j < selected.size(); ++j)
                index.addPoint(train.row(selected[j]), Label(old_records + j));
            insert_s = resize_s + seconds(start);
            if (!repeated) current.resize(active, invalid_label);
            for (size_t j = 0; j < selected.size(); ++j) {
                sources.push_back(selected[j]);
                current[selected[j]] = Label(old_records + j);
            }
        }
        const auto after = counters(index);
        const auto stage_after = timers(index);
        if (index.getCurrentElementCount() != sources.size() ||
            index.getDeletedCount() != (repeated ? batch * repeat_batch : 0) ||
            index.getCurrentElementCount() - index.getDeletedCount() != active)
            throw std::runtime_error("Active/physical/deleted count mismatch");
        check_pq(index);
        double max_error = 0;
        // All inserted records, plus a deterministic old-record sample; no per
        // vector timing is mixed into update measurements.
        auto verify = [&](uint32_t source) {
            const auto values = index.template getDataByLabel<float>(current[source]);
            for (size_t d = 0; d < train.dim; ++d) {
                const double error = std::abs(double(values[d]) - train.row(source)[d]);
                if (!std::isfinite(error) || error > 1.1e-4)
                    throw std::runtime_error("Reconstruction failure source=" + std::to_string(source) +
                        " coordinate=" + std::to_string(d));
                max_error = std::max(max_error, error);
            }
        };
        for (uint32_t source : selected) verify(source);
        const size_t old_samples = std::min<size_t>(1024, active);
        for (size_t j = 0; j < old_samples; ++j) verify(uint32_t(j * active / old_samples));
        if (o.verify_size && (batch == 0 || batch == steps))
            verify_serialization(index, fs::path(o.output) / "size_check.bin");
        const auto memory = rss_kib();
        const double graph_us = o.mode == "raw_f32" ? (insert_s - resize_s) * 1e6 :
            stage_after.first - stage_before.first;
        updates << identity << batch << ',' << percent << ',' << active << ',' << sources.size() << ','
            << index.getDeletedCount() << ',' << index.getMaxElements() << ',' << selected.size() << ','
            << (repeated ? selected.size() : 0) << ',' << serialized_bytes(index) << ',' << payload_capacity(index) << ','
            << insert_s << ',' << delete_s << ',' << resize_s << ','
            << (selected.empty() ? 0 : insert_s * 1e6 / selected.size()) << ','
            << (selected.empty() || !repeated ? 0 : delete_s * 1e6 / selected.size()) << ','
            << (selected.empty() ? 0 : graph_us / selected.size()) << ','
            << (selected.empty() ? 0 : (stage_after.second - stage_before.second) / selected.size()) << ','
            << after.searches - before.searches << ',' << after.pq_candidates - before.pq_candidates << ','
            << after.exact_candidates - before.exact_candidates << ','
            << after.fallback_searches - before.fallback_searches << ','
            << after.payload_decodes - before.payload_decodes << ',' << selected.size() + old_samples << ','
            << max_error << ',' << hash << ',' << memory.first << ',' << memory.second << ','
            << build_seconds << ',' << compression_seconds << ",1\n";
        updates.flush();
        char gt_name[64];
        std::snprintf(gt_name, sizeof(gt_name), "gt_prefix%03d.ivecs", percent);
        const auto truth = repeated ? full_truth : load_truth(fs::path(o.gt_dir) / gt_name, queries.count, active);
        for (bool tls : {false, true}) {
            if (tls && o.mode == "raw_f32") continue;
            query_config(index, tls);
            for (int ef : o.efs) {
                index.setEf(ef);
                for (size_t q = 0; q < std::min<size_t>(100, queries.count); ++q)
                    index.searchKnn(queries.row(q), 1);
                for (size_t round = 0; round < o.search_rounds; ++round) {
                    std::vector<Label> labels(queries.count);
                    const auto start = Clock::now();
                    for (size_t q = 0; q < queries.count; ++q) {
                        const auto result = index.searchKnn(queries.row(q), 1);
                        if (result.size() != 1) throw std::runtime_error("No search result");
                        labels[q] = result.top().second;
                    }
                    const double elapsed = seconds(start);
                    size_t hits = 0, ties = 0;
                    for (size_t q = 0; q < queries.count; ++q) {
                        const uint32_t source = valid_source(labels[q]);
                        hits += source == truth[q];
                        const double actual = l2(queries.row(q), train.row(source), train.dim);
                        const double target = l2(queries.row(q), train.row(truth[q]), train.dim);
                        if (actual + 1e-8 < target) throw std::runtime_error("GT is not nearest for active prefix");
                        ties += actual <= target + 1e-8;
                    }
                    search << identity << batch << ',' << percent << ',' << (tls ? "tls_0.2" : "tls_off") << ','
                        << (tls ? 0.2 : 1.0) << ',' << ef << ',' << round << ',' << queries.count << ','
                        << double(hits) / queries.count << ',' << double(ties) / queries.count << ','
                        << queries.count / elapsed << ',' << elapsed * 1e6 / queries.count << ",1\n";
                    search.flush();
                }
            }
        }
        std::cout << "PASS dataset=" << o.dataset << " experiment=" << o.experiment << " mode=" << o.mode
            << " batch=" << batch << " prefix=" << percent << " active=" << active
            << " physical=" << sources.size() << " index_bytes=" << serialized_bytes(index)
            << " insert_s=" << insert_s << " decodes=" << after.payload_decodes - before.payload_decodes
            << " max_error=" << max_error << std::endl;
    }
}

int main(int argc, char** argv) {
    try {
        omp_set_dynamic(0);
        omp_set_num_threads(1);
        Options o;
        for (int a = 1; a < argc; ++a) {
            const std::string arg = argv[a];
            auto next = [&]() { if (++a >= argc) throw std::runtime_error("Missing " + arg); return std::string(argv[a]); };
            if (arg == "--dataset") o.dataset = next();
            else if (arg == "--experiment") o.experiment = next();
            else if (arg == "--mode") o.mode = next();
            else if (arg == "--input") o.input = next();
            else if (arg == "--queries") o.queries = next();
            else if (arg == "--gt") o.gt = next();
            else if (arg == "--gt-dir") o.gt_dir = next();
            else if (arg == "--initial-index") o.initial_index = next();
            else if (arg == "--output") o.output = next();
            else if (arg == "--count") o.count = std::stoull(next());
            else if (arg == "--query-count") o.query_count = std::stoull(next());
            else if (arg == "--batches") o.batches = std::stoull(next());
            else if (arg == "--search-rounds") o.search_rounds = std::stoull(next());
            else if (arg == "--seed") o.seed = std::stoull(next());
            else if (arg == "--M") o.M = std::stoull(next());
            else if (arg == "--ef-construction") o.ef_construction = std::stoull(next());
            else if (arg == "--verify-size") o.verify_size = true;
            else if (arg == "--ef") {
                o.efs.clear();
                while (a + 1 < argc && argv[a + 1][0] != '-') o.efs.push_back(std::stoi(argv[++a]));
            } else throw std::runtime_error("Unknown option " + arg);
        }
        if (o.dataset.empty() || o.output.empty() || o.input.empty() || o.queries.empty() ||
            (o.experiment != "repeated" && o.experiment != "growing") || !o.batches || !o.search_rounds ||
            o.M < 2 || o.ef_construction < o.M || o.efs.empty() ||
            *std::min_element(o.efs.begin(), o.efs.end()) < 1)
            throw std::runtime_error("Invalid workload arguments");
        Fvecs train(o.input, o.count), queries(o.queries, o.query_count);
        const size_t initial = o.experiment == "repeated" ? train.count : train.count / 2;
        const size_t max_records = o.experiment == "repeated" ?
            train.count + o.batches * std::max<size_t>(1, train.count * 5 / 100) : train.count;
        if (initial < 256 || !queries.count || train.dim != queries.dim ||
            max_records >= std::numeric_limits<hnswlib::tableint>::max())
            throw std::runtime_error("Invalid dimensions, too few PQ training rows, or ID overflow");
        fs::create_directories(o.output);
        hnswlib::L2Space space(train.dim);
        if (o.mode == "prepare_alp") {
            ALP index(&space, initial, o.M, o.ef_construction, 100, false, 1e-4f, true);
            auto start = Clock::now();
            for (size_t id = 0; id < initial; ++id) index.addPoint(train.row(id), id);
            const double build_s = seconds(start);
            start = Clock::now();
            index.compress_dataset(std::max<size_t>(1, initial / 100));
            const double compression_s = seconds(start);
            index.saveIndex(o.initial_index);
            if (fs::file_size(o.initial_index) != serialized_bytes(index))
                throw std::runtime_error("Initial ALP serialized-size verification failed");
            std::ofstream info(fs::path(o.output) / "initial.csv");
            info << "build_seconds,compression_seconds,initial_nodes,index_file_bytes\n"
                << std::setprecision(12) << build_s << ',' << compression_s << ',' << initial << ','
                << serialized_bytes(index) << '\n';
        } else if (o.mode == "raw_f32") {
            Raw index(&space, initial, o.M, o.ef_construction, 100);
            const auto start = Clock::now();
            for (size_t id = 0; id < initial; ++id) index.addPoint(train.row(id), id);
            workload(index, o, train, queries, initial, seconds(start), 0);
        } else if (o.mode == "alp_a01" || o.mode == "alp_a02" || o.mode == "alp_off") {
            ALP index(&space, o.initial_index);
            if (index.getCurrentElementCount() != initial || !index.enable_dynamic_updates_)
                throw std::runtime_error("Wrong initial ALP checkpoint");
            index.setInsertUsePQFilter(o.mode != "alp_off");
            index.setInsertPQAlpha(o.mode == "alp_a01" ? 0.1f : 0.2f);
            std::ifstream info(fs::path(o.initial_index).parent_path() / "initial.csv");
            std::string header, values;
            std::getline(info, header); std::getline(info, values);
            const auto comma = values.find(',');
            if (!info || comma == std::string::npos) throw std::runtime_error("Missing initial index timings");
            const double build_s = std::stod(values.substr(0, comma));
            const double compression_s = std::stod(values.substr(comma + 1));
            workload(index, o, train, queries, initial, build_s, compression_s);
        } else throw std::runtime_error("Unknown insertion mode");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << std::endl;
        return 1;
    }
}
