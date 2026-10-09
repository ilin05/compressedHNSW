#pragma once

#include "ivf_lvc/IndexIVFLVC.h"
#include <faiss/IndexIVFFlat.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ivf_lvc {

struct ForestNode {
    faiss::idx_t id = -1;
    faiss::idx_t list = -1;
    faiss::idx_t parent = -1;
    faiss::idx_t root = -1;
    uint32_t depth = 0;
    double reference_l2 = 0;
};

struct ForestSummary {
    size_t nodes = 0;
    size_t nonempty_lists = 0;
    size_t roots = 0;
    double mean_reference_l2 = 0;
};

struct Forest {
    std::vector<ForestNode> nodes;
    ForestSummary summary;
    void verify(const faiss::IndexIVFLVC& index) const;
    void write_csv(const std::string& path) const;
};

// Phase 1 assumes the default dense IDs assigned by Index::add.
Forest build_forest(const faiss::IndexIVFLVC& index, uint64_t seed = 20261009,
                    double root_fraction = 0.01);

// The baseline and fork must share both the coarse quantizer and every posting.
void verify_shared_partition(const faiss::IndexIVFFlat& baseline,
                             const faiss::IndexIVFLVC& fork);

// Copy the already trained quantizer and exact postings. The original IVF-Flat
// remains usable as a baseline; the returned fork owns its independent copy.
std::unique_ptr<faiss::IndexIVFLVC> clone_from_ivf_flat(
    const faiss::IndexIVFFlat& baseline);

} // namespace ivf_lvc
