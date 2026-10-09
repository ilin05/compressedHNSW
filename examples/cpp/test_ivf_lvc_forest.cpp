#include "ivf_lvc/forest.h"

#include <faiss/IndexFlat.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<float> read_fvecs(const std::string& path, size_t limit, size_t& dim) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open: " + path);
    int32_t first_dim = 0;
    input.read(reinterpret_cast<char*>(&first_dim), sizeof(first_dim));
    if (!input || first_dim <= 0) throw std::runtime_error("invalid fvecs dimension");
    dim = static_cast<size_t>(first_dim);
    input.seekg(0, std::ios::end);
    const auto bytes = input.tellg();
    const auto row_bytes = static_cast<std::streamoff>(sizeof(int32_t) + dim * sizeof(float));
    if (bytes <= 0 || bytes % row_bytes != 0) throw std::runtime_error("invalid fvecs file length");
    const size_t count = std::min(limit, static_cast<size_t>(bytes / row_bytes));
    input.seekg(0);
    std::vector<float> data(count * dim);
    for (size_t i = 0; i < count; ++i) {
        int32_t row_dim = 0;
        input.read(reinterpret_cast<char*>(&row_dim), sizeof(row_dim));
        if (row_dim != first_dim) throw std::runtime_error("inconsistent fvecs dimension");
        input.read(reinterpret_cast<char*>(data.data() + i * dim), dim * sizeof(float));
        if (!input) throw std::runtime_error("truncated fvecs row");
    }
    return data;
}

double squared_l2(const float* a, const float* b, size_t dim) {
    double sum = 0;
    for (size_t j = 0; j < dim; ++j) {
        const double delta = double(a[j]) - double(b[j]);
        sum += delta * delta;
    }
    return sum;
}

void verify_root_selection(const faiss::IndexIVFLVC& index,
                           const ivf_lvc::Forest& forest, double ratio) {
    const size_t dim = static_cast<size_t>(index.d);
    std::vector<float> centroid(dim);
    size_t counted_roots = 0;
    for (size_t list = 0; list < index.nlist; ++list) {
        const size_t count = index.invlists->list_size(list);
        if (!count) continue;
        index.quantizer->reconstruct(list, centroid.data());
        faiss::InvertedLists::ScopedIds ids(index.invlists, list);
        faiss::InvertedLists::ScopedCodes codes(index.invlists, list);
        const auto* raw = reinterpret_cast<const float*>(codes.get());
        size_t roots = 0;
        std::vector<size_t> root_positions;
        size_t nearest_pos = 0;
        double best = squared_l2(raw, centroid.data(), dim);
        for (size_t pos = 0; pos < count; ++pos) {
            const auto& node = forest.nodes.at(static_cast<size_t>(ids.get()[pos]));
            if (node.list != static_cast<faiss::idx_t>(list))
                throw std::runtime_error("wrong forest list");
            if (node.parent == -1) {
                ++roots;
                root_positions.push_back(pos);
            }
            const double distance = squared_l2(raw + pos * dim, centroid.data(), dim);
            if (distance < best || (distance == best && ids.get()[pos] < ids.get()[nearest_pos])) {
                nearest_pos = pos;
                best = distance;
            }
        }
        const size_t expected = std::max<size_t>(1, static_cast<size_t>(std::floor(count * ratio)));
        if (roots != expected || forest.nodes.at(static_cast<size_t>(ids.get()[nearest_pos])).parent != -1)
            throw std::runtime_error("root count or centroid-nearest root differs");
        for (size_t pos = 0; pos < count; ++pos) {
            const auto& node = forest.nodes.at(static_cast<size_t>(ids.get()[pos]));
            if (node.parent == -1) continue;
            size_t closest = root_positions.front();
            double best_root_distance = squared_l2(raw + pos * dim, raw + closest * dim, dim);
            for (size_t ri = 1; ri < root_positions.size(); ++ri) {
                const size_t candidate = root_positions[ri];
                const double distance = squared_l2(raw + pos * dim, raw + candidate * dim, dim);
                if (distance < best_root_distance ||
                    (distance == best_root_distance && ids.get()[candidate] < ids.get()[closest])) {
                    closest = candidate;
                    best_root_distance = distance;
                }
            }
            if (node.parent != ids.get()[closest] ||
                std::abs(node.reference_l2 - std::sqrt(best_root_distance)) > 1e-9)
                throw std::runtime_error("child is not attached to closest table root");
        }
        counted_roots += roots;
    }
    if (counted_roots != forest.summary.roots)
        throw std::runtime_error("total root count differs");
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string input, output = "ivf_lvc_forest.csv";
        size_t nlist = 8, count = 512;
        uint64_t seed = 20261009;
        double root_ratio = 0.01;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
            if (arg == "--input") input = argv[++i];
            else if (arg == "--output") output = argv[++i];
            else if (arg == "--nlist") nlist = std::stoull(argv[++i]);
            else if (arg == "--count") count = std::stoull(argv[++i]);
            else if (arg == "--seed") seed = std::stoull(argv[++i]);
            else if (arg == "--root-ratio") root_ratio = std::stod(argv[++i]);
            else throw std::invalid_argument("unknown argument: " + arg);
        }
        if (input.empty() || nlist == 0 || count == 0)
            throw std::invalid_argument("provide --input, positive --nlist and --count");
        size_t dim = 0;
        const auto data = read_fvecs(input, count, dim);
        const size_t points = data.size() / dim;
        if (points < nlist * 39)
            throw std::invalid_argument("Faiss needs at least 39 training vectors per centroid");

        faiss::IndexIVFFlat baseline(new faiss::IndexFlatL2(dim), dim, nlist);
        baseline.own_fields = true;
        baseline.train(points, data.data());
        baseline.add(points, data.data());
        auto fork = ivf_lvc::clone_from_ivf_flat(baseline);

        auto forest = ivf_lvc::build_forest(*fork, seed, root_ratio);
        verify_root_selection(*fork, forest, root_ratio);
        auto repeat = ivf_lvc::build_forest(*fork, seed, root_ratio);
        for (size_t i = 0; i < forest.nodes.size(); ++i) {
            if (forest.nodes[i].parent != repeat.nodes[i].parent ||
                forest.nodes[i].root != repeat.nodes[i].root) {
                throw std::runtime_error("forest is not deterministic");
            }
        }
        forest.write_csv(output);

        // The copied IVF-Flat class must preserve the baseline search at this stage.
        const size_t queries = std::min<size_t>(10, points);
        std::vector<faiss::idx_t> base_ids(queries), fork_ids(queries);
        std::vector<float> base_dist(queries), fork_dist(queries);
        baseline.search(queries, data.data(), 1, base_dist.data(), base_ids.data());
        fork->search(queries, data.data(), 1, fork_dist.data(), fork_ids.data());
        for (size_t i = 0; i < queries; ++i) {
            if (base_ids[i] != fork_ids[i] ||
                std::memcmp(&base_dist[i], &fork_dist[i], sizeof(float))) {
                throw std::runtime_error("copied IVF search differs from baseline");
            }
        }
        std::cout << "PASS shared IVF partition and search: nodes=" << points
                  << " dim=" << dim << " nlist=" << nlist << '\n';
        std::cout << "PASS deterministic table-local forest: nonempty_lists="
                  << forest.summary.nonempty_lists << " roots=" << forest.summary.roots
                  << " max_depth=1 mean_reference_l2=" << forest.summary.mean_reference_l2
                  << " output=" << output << '\n';
    } catch (const std::exception& error) {
        std::cerr << "FAIL IVF-LVC forest: " << error.what() << '\n';
        return 1;
    }
}
