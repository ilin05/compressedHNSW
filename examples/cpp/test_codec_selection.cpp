// Sample real payload encoders on one shared, uncompressed HNSW graph.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>
#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/compressed_hnsw_framework.h"
#include "../../hnswlib/hnswalg_simplified_alp_PQ.h"

using namespace hnswlib;
using Graph = HierarchicalNSWCABFRAMEWORK<double, codecs::DeXORCodecPolicy>;
using Clock = std::chrono::steady_clock;
static double seconds(Clock::time_point begin) {
    return std::chrono::duration<double>(Clock::now() - begin).count();
}
struct Measurement { size_t bytes; double median_s, wall_s; };
static Measurement measure(size_t count, int rounds, const std::function<size_t(size_t)>& encode) {
    for (size_t i = 0; i < std::min<size_t>(16, count); ++i) encode(i);
    const auto wall = Clock::now();
    std::vector<double> times;
    size_t expected = 0;
    for (int r = 0; r < rounds; ++r) {
        size_t bytes = 0;
        const auto begin = Clock::now();
        for (size_t i = 0; i < count; ++i) bytes += encode(i);
        times.push_back(seconds(begin));
        if (r && bytes != expected) throw std::runtime_error("Non-deterministic sample payload");
        expected = bytes;
    }
    std::sort(times.begin(), times.end());
    return {expected, times[times.size() / 2], seconds(wall)};
}

template<class Codec>
static Measurement profile(Graph& graph, const std::vector<tableint>& sample,
                           const std::vector<tableint>& parent, size_t dim, int rounds) {
    return measure(sample.size(), rounds, [&](size_t position) {
        const tableint id = sample[position];
        std::vector<typename Codec::StateType> states(dim);
        std::vector<tableint> ancestors;
        for (tableint p = parent[id]; p != tableint(-1); p = parent[p]) {
            if (ancestors.size() >= parent.size()) throw std::runtime_error("Forest cycle");
            ancestors.push_back(p);
        }
        std::reverse(ancestors.begin(), ancestors.end());
        std::vector<char> scratch, payload;
        utils::MemoryStreamWriter replay(&scratch), writer(&payload);
        for (tableint p : ancestors) {
            const auto* values = reinterpret_cast<const double*>(graph.getDataByInternalId(p));
            for (size_t j = 0; j < dim; ++j) Codec::encode(values[j], states[j], replay);
        }
        const auto* values = reinterpret_cast<const double*>(graph.getDataByInternalId(id));
        for (size_t j = 0; j < dim; ++j) Codec::encode(values[j], states[j], writer);
        writer.align();
        return payload.size(); // Scratch bytes are NOT stored payload bytes.
    });
}

int main(int argc, char** argv) try {
    std::string train, graph_path, output, dataset, policy_name = "level";
    double sample_ratio = .01, root_ratio = .01, alp_hub_ratio = .01;
    float alp_tolerance = 1e-4f;
    int rounds = 3, threads = 1, M = 16, ef = 200;
    uint32_t seed = 20261016;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (i + 1 >= argc) throw std::runtime_error("Missing value: " + key);
        const std::string value = argv[++i];
        if (key == "--train") train = value;
        else if (key == "--graph") graph_path = value;
        else if (key == "--output") output = value;
        else if (key == "--dataset") dataset = value;
        else if (key == "--root-policy") policy_name = value;
        else if (key == "--sample-ratio") sample_ratio = std::stod(value);
        else if (key == "--root-ratio") root_ratio = std::stod(value);
        else if (key == "--alp-hub-ratio") alp_hub_ratio = std::stod(value);
        else if (key == "--alp-tolerance") alp_tolerance = std::stof(value);
        else if (key == "--rounds") rounds = std::stoi(value);
        else if (key == "--build-threads") threads = std::stoi(value);
        else if (key == "--M") M = std::stoi(value);
        else if (key == "--ef-construction") ef = std::stoi(value);
        else if (key == "--seed") seed = std::stoul(value);
        else throw std::runtime_error("Unknown argument: " + key);
    }
    if (train.empty() || graph_path.empty() || output.empty() || dataset.empty() ||
        dataset.find_first_of(",\n") != std::string::npos ||
        !(sample_ratio > 0 && sample_ratio <= 1) || !(root_ratio > 0 && root_ratio <= 1) ||
        !(alp_hub_ratio >= 0 && alp_hub_ratio <= 1) || rounds < 1 || threads < 1 || M < 2 || ef < 1)
        throw std::runtime_error("Required: --train --graph --output --dataset; invalid sampling/build parameters");
    CompressionRootPolicy policy;
    if (policy_name == "level") policy = CompressionRootPolicy::Level;
    else if (policy_name == "degree") policy = CompressionRootPolicy::Level0Degree;
    else throw std::runtime_error("root-policy must be level or degree");
    std::ifstream input(train, std::ios::binary);
    int32_t width = 0;
    input.read(reinterpret_cast<char*>(&width), sizeof(width));
    if (!input || width <= 0) throw std::runtime_error("Invalid fvecs input: " + train);
    const size_t dim = width, stride = sizeof(width) + dim * sizeof(float);
    const auto file_bytes = std::filesystem::file_size(train);
    if (file_bytes % stride || !file_bytes) throw std::runtime_error("Truncated fvecs input");
    const size_t n = file_bytes / stride;
    if (n > std::numeric_limits<tableint>::max()) throw std::runtime_error("Too many nodes");
    std::filesystem::create_directories(output);
    if (!std::filesystem::path(graph_path).parent_path().empty())
        std::filesystem::create_directories(std::filesystem::path(graph_path).parent_path());
    L2SpaceDouble space(dim);
    std::unique_ptr<Graph> graph;
    double build_s = 0;
    const bool reused = std::filesystem::exists(graph_path);
    if (reused) {
        graph.reset(new Graph(&space, graph_path, true, 0));
    } else {
        std::vector<double> values(n * dim);
        std::vector<float> row(dim);
        input.seekg(0);
        for (size_t i = 0; i < n; ++i) {
            int32_t actual;
            input.read(reinterpret_cast<char*>(&actual), sizeof(actual));
            input.read(reinterpret_cast<char*>(row.data()), dim * sizeof(float));
            if (!input || actual != width) throw std::runtime_error("Invalid fvecs row " + std::to_string(i));
            std::copy(row.begin(), row.end(), values.begin() + i * dim);
        }
        graph.reset(new Graph(&space, n, M, ef, true, 0, 100, false));
        omp_set_num_threads(threads);
        const auto start = Clock::now();
        #pragma omp parallel for
        for (long long i = 0; i < static_cast<long long>(n); ++i)
            graph->addPoint(values.data() + i * dim, i);
        build_s = seconds(start);
        graph->saveIndex(graph_path);
    }
    if (graph->is_compacted_ || graph->cur_element_count != n || graph->data_size_ != dim * sizeof(double) || graph->num_deleted_)
        throw std::runtime_error("Expected a raw F64 HNSW graph matching train count/dimension, with no deletions");
    if (graph->M_ != static_cast<size_t>(M) || graph->ef_construction_ != static_cast<size_t>(std::max(M, ef)))
        throw std::runtime_error("Existing raw graph M/efConstruction differ from requested parameters");
    omp_set_num_threads(1); // Forest and trial encoders are measured with one thread.
    const auto selection_start = Clock::now();
    std::vector<tableint> sample(n);
    std::iota(sample.begin(), sample.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(sample.begin(), sample.end(), rng);
    sample.resize(std::max<size_t>(1, static_cast<size_t>(std::ceil(n * sample_ratio))));
    std::sort(sample.begin(), sample.end());
    std::ofstream ids(output + "/sample_ids.csv");
    ids << "InternalId,Label\n";
    for (auto id : sample) ids << id << ',' << graph->getExternalLabel(id) << '\n';
    // ALP uses the same vectors, but no forest. Match its retained raw hub policy.
    std::vector<tableint> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](tableint a, tableint b) {
        return graph->element_levels_[a] > graph->element_levels_[b];
    });
    std::vector<bool> hubs(n, false);
    const size_t nhubs = alp_hub_ratio == 0 ? 0 : std::max<size_t>(1, n * alp_hub_ratio);
    for (size_t i = 0; i < nhubs; ++i) hubs[order[i]] = true;
    order.clear(); order.shrink_to_fit();
    std::vector<float> sample_f32(sample.size() * dim);
    for (size_t i = 0; i < sample.size(); ++i) {
        const auto* source = reinterpret_cast<const double*>(graph->getDataByInternalId(sample[i]));
        for (size_t j = 0; j < dim; ++j) sample_f32[i * dim + j] = static_cast<float>(source[j]);
    }
    L2Space alp_space(dim);
    HierarchicalNSWALPSIMPLIFIEDPQ<float> alp(&alp_space, 1, M, ef, 100, false, alp_tolerance);
    auto alp_measurement = measure(sample.size(), rounds, [&](size_t i) {
        if (hubs[sample[i]]) return dim * sizeof(float);
        std::vector<char> payload;
        utils::MemoryStreamWriter writer(&payload);
        alp.alp_encode_vector(sample_f32.data() + i * dim, dim, writer);
        writer.align();
        return payload.size();
    });
    std::ofstream csv(output + "/predictions.csv");
    if (!csv) throw std::runtime_error("Cannot open predictions output");
    csv << "Dataset,Codec,ChainMaxLength,Nodes,Dim,Seed,SampleCount,SampleRatio,RootPolicy,RootRatio,Roots,AverageDepth,SampleAverageDepth,SampleRoots,SamplePayloadBytes,PredictedPayloadBytes,PredictedPayloadRatioF32,ForestSeconds,SampleEncodeMedianSeconds,PredictedEncodeSeconds,PredictedForestEncodeSeconds,EncodeProfileWallSeconds,GraphBuildSeconds,GraphReused,ALPTolerance,ALPHubs,TimeScope\n";
    csv << std::setprecision(12);
    double forest_total_s = 0;
    for (int chain : {2, -1}) {
        const auto start = Clock::now();
        auto parent = graph->prepareCompressionForest(chain, root_ratio, policy);
        const double forest_s = seconds(start);
        forest_total_s += forest_s;
        size_t roots = 0, sample_roots = 0;
        double depth_sum = 0, sample_depth_sum = 0;
        const auto& depths = graph->compression_depth_export_;
        for (size_t i = 0; i < n; ++i) {
            depth_sum += depths[i];
            if (parent[i] == tableint(-1)) { ++roots; if (depths[i]) throw std::runtime_error("Root depth"); }
            else if (parent[i] >= n || depths[parent[i]] + 1 != depths[i])
                throw std::runtime_error("Invalid parent/depth or cycle");
            if (chain == 2 && depths[i] > 1) throw std::runtime_error("Depth bound");
        }
        for (auto id : sample) { sample_depth_sum += depths[id]; sample_roots += parent[id] == tableint(-1); }
        auto emit = [&](const std::string& codec, Measurement measurement) {
            const double scale = double(n) / sample.size();
            const double payload = measurement.bytes * scale;
            const double encode = measurement.median_s * scale;
            const double candidate_forest = codec == "ALP" ? 0 : forest_s;
            csv << dataset << ',' << codec << ',' << chain << ',' << n << ',' << dim << ',' << seed << ','
                << sample.size() << ',' << double(sample.size()) / n << ',' << policy_name << ',' << root_ratio << ','
                << (codec == "ALP" ? 0 : roots) << ',' << (codec == "ALP" ? 0 : depth_sum / n) << ','
                << (codec == "ALP" ? 0 : sample_depth_sum / sample.size()) << ',' << (codec == "ALP" ? 0 : sample_roots) << ','
                << measurement.bytes << ',' << payload << ',' << double(n) * dim * sizeof(float) / payload << ','
                << candidate_forest << ',' << measurement.median_s << ',' << encode << ',' << candidate_forest + encode << ','
                << measurement.wall_s << ',' << build_s << ',' << reused << ',' << alp_tolerance << ',' << nhubs
                << ",forest_and_payload_only_no_PQ_or_compaction\n";
            csv.flush();
            std::cout << "chain=" << chain << " codec=" << codec << " predicted_F32_ratio="
                      << double(n) * dim * sizeof(float) / payload << " predicted_forest_encode_s="
                      << candidate_forest + encode << std::endl;
        };
        emit("ALP", alp_measurement);
        emit("DeXOR", profile<codecs::DeXORCodecPolicy>(*graph, sample, parent, dim, rounds));
        emit("Gorilla", profile<codecs::GorillaCodecPolicy>(*graph, sample, parent, dim, rounds));
        emit("Elf", profile<codecs::ElfCodecPolicy>(*graph, sample, parent, dim, rounds));
    }
    std::ofstream timing(output + "/profiling.csv");
    timing << "GraphBuildSeconds,GraphReused,ForestTotalSeconds,SelectionWallSeconds,SamplingIncludesAncestors\n"
           << std::setprecision(12) << build_s << ',' << reused << ',' << forest_total_s << ','
           << seconds(selection_start) << ",1\n";
    std::cout << "PASS full forests, deterministic sample encoding; graph=" << graph_path << std::endl;
    return 0;
} catch (const std::exception& error) {
    std::cerr << "codec selection failed: " << error.what() << '\n';
    return 1;
}
