#include "ivf_lvc/forest.h"
#include <faiss/clone_index.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <random>
#include <stdexcept>

namespace ivf_lvc {
namespace {

double squared_l2(const float* a, const float* b, size_t dim) {
    double sum = 0;
    for (size_t j = 0; j < dim; ++j) {
        const double delta = double(a[j]) - double(b[j]);
        sum += delta * delta;
    }
    return sum;
}

const float* vector_at(const uint8_t* codes, size_t offset, size_t dim) {
    return reinterpret_cast<const float*>(codes + offset * dim * sizeof(float));
}

} // namespace

Forest build_forest(const faiss::IndexIVFLVC& index, uint64_t seed,
                    double root_fraction) {
    if (!index.is_trained || !index.invlists || index.metric_type != faiss::METRIC_L2 ||
        !(root_fraction > 0 && root_fraction <= 1)) {
        throw std::invalid_argument("forest needs a trained L2 IVF index and root_fraction in (0,1]");
    }

    Forest forest;
    forest.nodes.resize(static_cast<size_t>(index.ntotal));
    std::vector<uint8_t> seen(forest.nodes.size(), 0);
    double distance_sum = 0;
    const size_t dim = static_cast<size_t>(index.d);
    std::vector<float> centroid(dim);

    for (size_t list = 0; list < index.nlist; ++list) {
        const size_t count = index.invlists->list_size(list);
        if (!count) continue;
        ++forest.summary.nonempty_lists;
        faiss::InvertedLists::ScopedIds ids(index.invlists, list);
        faiss::InvertedLists::ScopedCodes codes(index.invlists, list);
        index.quantizer->reconstruct(static_cast<faiss::idx_t>(list), centroid.data());

        size_t first = 0;
        double best_centroid_distance = squared_l2(vector_at(codes.get(), 0, dim),
                                                   centroid.data(), dim);
        for (size_t pos = 1; pos < count; ++pos) {
            const double distance = squared_l2(vector_at(codes.get(), pos, dim),
                                               centroid.data(), dim);
            if (distance < best_centroid_distance ||
                (distance == best_centroid_distance && ids.get()[pos] < ids.get()[first])) {
                best_centroid_distance = distance;
                first = pos;
            }
        }

        const size_t root_count = std::min(count, std::max<size_t>(
            1, static_cast<size_t>(std::floor(double(count) * root_fraction))));
        std::vector<size_t> roots;
        roots.reserve(root_count);
        roots.push_back(first);
        std::vector<size_t> candidates;
        candidates.reserve(count - 1);
        for (size_t pos = 0; pos < count; ++pos) {
            if (pos != first) candidates.push_back(pos);
        }
        std::mt19937_64 rng(seed ^ (uint64_t(list) + 0x9e3779b97f4a7c15ULL));
        std::shuffle(candidates.begin(), candidates.end(), rng);
        roots.insert(roots.end(), candidates.begin(),
                     candidates.begin() + (root_count - 1));
        std::vector<uint8_t> is_root(count, 0);
        for (const size_t pos : roots) is_root[pos] = 1;

        for (size_t pos = 0; pos < count; ++pos) {
            const auto id = ids.get()[pos];
            if (id < 0 || static_cast<size_t>(id) >= seen.size() || seen[id]) {
                throw std::runtime_error("forest requires unique dense IDs 0..N-1");
            }
            seen[id] = 1;
            auto& node = forest.nodes[static_cast<size_t>(id)];
            node.id = id;
            node.list = static_cast<faiss::idx_t>(list);
            if (is_root[pos]) {
                node.root = id;
                ++forest.summary.roots;
                continue;
            }
            size_t best_root = roots.front();
            double best_distance = squared_l2(vector_at(codes.get(), pos, dim),
                                              vector_at(codes.get(), best_root, dim), dim);
            for (size_t ri = 1; ri < roots.size(); ++ri) {
                const size_t candidate = roots[ri];
                const double distance = squared_l2(vector_at(codes.get(), pos, dim),
                                                   vector_at(codes.get(), candidate, dim), dim);
                if (distance < best_distance ||
                    (distance == best_distance && ids.get()[candidate] < ids.get()[best_root])) {
                    best_distance = distance;
                    best_root = candidate;
                }
            }
            node.parent = ids.get()[best_root];
            node.root = node.parent;
            node.depth = 1;
            node.reference_l2 = std::sqrt(best_distance);
            distance_sum += node.reference_l2;
        }
    }

    if (std::find(seen.begin(), seen.end(), 0) != seen.end()) {
        throw std::runtime_error("some dense IDs were not assigned to an IVF list");
    }
    forest.summary.nodes = forest.nodes.size();
    if (forest.summary.nodes > forest.summary.roots) {
        forest.summary.mean_reference_l2 =
            distance_sum / double(forest.summary.nodes - forest.summary.roots);
    }
    forest.verify(index);
    return forest;
}

void Forest::verify(const faiss::IndexIVFLVC& index) const {
    if (nodes.size() != static_cast<size_t>(index.ntotal)) {
        throw std::runtime_error("forest node count differs from IVF index");
    }
    size_t counted_roots = 0;
    for (size_t id = 0; id < nodes.size(); ++id) {
        const auto& node = nodes[id];
        if (node.id != static_cast<faiss::idx_t>(id) || node.list < 0 ||
            static_cast<size_t>(node.list) >= index.nlist) {
            throw std::runtime_error("invalid forest ID or list");
        }
        if (node.parent == -1) {
            if (node.root != node.id || node.depth != 0) {
                throw std::runtime_error("invalid root record");
            }
            ++counted_roots;
        } else {
            if (node.parent < 0 || static_cast<size_t>(node.parent) >= nodes.size() ||
                node.parent == node.id || node.depth != 1 || node.root != node.parent ||
                nodes[static_cast<size_t>(node.parent)].parent != -1 ||
                nodes[static_cast<size_t>(node.parent)].list != node.list ||
                !std::isfinite(node.reference_l2)) {
                throw std::runtime_error("invalid child parent, depth, or reference distance");
            }
        }
    }
    if (counted_roots != summary.roots) throw std::runtime_error("root count mismatch");
}

void Forest::write_csv(const std::string& path) const {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot write forest CSV: " + path);
    output << "vector_id,list_id,parent_id,root_id,depth,reference_l2\n";
    output << std::setprecision(17);
    for (const auto& node : nodes) {
        output << node.id << ',' << node.list << ',' << node.parent << ','
               << node.root << ',' << node.depth << ',' << node.reference_l2 << '\n';
    }
}

void verify_shared_partition(const faiss::IndexIVFFlat& baseline,
                             const faiss::IndexIVFLVC& fork) {
    if (baseline.d != fork.d || baseline.nlist != fork.nlist ||
        baseline.ntotal != fork.ntotal || baseline.metric_type != fork.metric_type ||
        baseline.code_size != fork.code_size) {
        throw std::runtime_error("IVF index metadata differs");
    }
    std::vector<float> left(baseline.d), right(fork.d);
    for (size_t list = 0; list < baseline.nlist; ++list) {
        baseline.quantizer->reconstruct(list, left.data());
        fork.quantizer->reconstruct(list, right.data());
        if (std::memcmp(left.data(), right.data(), baseline.d * sizeof(float))) {
            throw std::runtime_error("IVF centroids differ");
        }
        const size_t count = baseline.invlists->list_size(list);
        if (count != fork.invlists->list_size(list)) {
            throw std::runtime_error("IVF list sizes differ");
        }
        faiss::InvertedLists::ScopedIds left_ids(baseline.invlists, list);
        faiss::InvertedLists::ScopedIds right_ids(fork.invlists, list);
        faiss::InvertedLists::ScopedCodes left_codes(baseline.invlists, list);
        faiss::InvertedLists::ScopedCodes right_codes(fork.invlists, list);
        if (count && (std::memcmp(left_ids.get(), right_ids.get(), count * sizeof(faiss::idx_t)) ||
                      std::memcmp(left_codes.get(), right_codes.get(), count * baseline.code_size))) {
            throw std::runtime_error("IVF posting IDs or F32 vectors differ");
        }
    }
}

std::unique_ptr<faiss::IndexIVFLVC> clone_from_ivf_flat(
    const faiss::IndexIVFFlat& baseline) {
    if (!baseline.is_trained || !baseline.quantizer || !baseline.invlists ||
        baseline.metric_type != faiss::METRIC_L2) {
        throw std::invalid_argument("expected a trained L2 IndexIVFFlat");
    }
    auto fork = std::make_unique<faiss::IndexIVFLVC>(
        faiss::clone_index(baseline.quantizer), baseline.d, baseline.nlist,
        baseline.metric_type);
    fork->own_fields = true;
    fork->is_trained = true;
    fork->nprobe = baseline.nprobe;
    fork->max_codes = baseline.max_codes;
    for (size_t list = 0; list < baseline.nlist; ++list) {
        const size_t count = baseline.invlists->list_size(list);
        if (!count) continue;
        faiss::InvertedLists::ScopedIds ids(baseline.invlists, list);
        faiss::InvertedLists::ScopedCodes codes(baseline.invlists, list);
        fork->invlists->add_entries(list, count, ids.get(), codes.get());
    }
    fork->ntotal = baseline.ntotal;
    verify_shared_partition(baseline, *fork);
    return fork;
}

} // namespace ivf_lvc
