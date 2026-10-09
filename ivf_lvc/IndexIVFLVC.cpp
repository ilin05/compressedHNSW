/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include "ivf_lvc/IndexIVFLVC.h"
#include "ivf_lvc/compact_records.h"

#include <omp.h>

#include <cinttypes>
#include <cstdio>
#include <numeric>
#include <queue>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/IDSelector.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/expanded_scanners.h>
#include <faiss/utils/extra_distances.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/utils.h>

namespace faiss {

/*****************************************
 * IndexIVFLVC implementation
 ******************************************/

IndexIVFLVC::IndexIVFLVC(
        Index* quantizer,
        size_t d,
        size_t nlist,
        MetricType metric,
        bool own_invlists)
        : IndexIVF(
                  quantizer,
                  d,
                  nlist,
                  sizeof(float) * d,
                  metric,
                  own_invlists) {
    code_size = sizeof(float) * d;
    by_residual = false;
}

IndexIVFLVC::IndexIVFLVC() {
    by_residual = false;
}

IndexIVFLVC::~IndexIVFLVC() = default;

void IndexIVFLVC::add_core(
        idx_t n,
        const float* x,
        const idx_t* xids,
        const idx_t* coarse_idx,
        void* inverted_list_context) {
    if (compact_records_) FAISS_THROW_MSG("cannot add to a compacted IVF-LVC index");
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(coarse_idx);
    FAISS_THROW_IF_NOT(!by_residual);
    assert(invlists);
    direct_map.check_can_add(xids);

    int64_t n_add = 0;

    DirectMapAdd dm_adder(direct_map, n, xids);

#pragma omp parallel reduction(+ : n_add)
    {
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        // each thread takes care of a subset of lists
        for (size_t i = 0; i < n; i++) {
            idx_t list_no = coarse_idx[i];

            if (list_no >= 0 && list_no % nt == rank) {
                idx_t id = xids ? xids[i] : ntotal + i;
                const float* xi = x + i * d;
                size_t offset = invlists->add_entry(
                        list_no, id, (const uint8_t*)xi, inverted_list_context);
                dm_adder.add(i, list_no, offset);
                n_add++;
            } else if (rank == 0 && list_no == -1) {
                dm_adder.add(i, -1, 0);
            }
        }
    }

    if (verbose) {
        printf("IndexIVFLVC::add_core: added %" PRId64 " / %" PRId64
               " vectors\n",
               n_add,
               n);
    }
    ntotal += n;
}

void IndexIVFLVC::encode_vectors(
        idx_t n,
        const float* x,
        const idx_t* list_nos,
        uint8_t* codes,
        bool include_listnos) const {
    FAISS_THROW_IF_NOT(!by_residual);
    if (!include_listnos) {
        memcpy(codes, x, code_size * n);
    } else {
        size_t coarse_size = coarse_code_size();
        for (size_t i = 0; i < n; i++) {
            int64_t list_no = list_nos[i];
            uint8_t* code = codes + i * (code_size + coarse_size);
            const float* xi = x + i * d;
            if (list_no >= 0) {
                encode_listno(list_no, code);
                memcpy(code + coarse_size, xi, code_size);
            } else {
                memset(code, 0, code_size + coarse_size);
            }
        }
    }
}

void IndexIVFLVC::decode_vectors(
        idx_t n,
        const uint8_t* codes,
        const idx_t* /*listnos*/,
        float* x) const {
    for (size_t i = 0; i < n; i++) {
        const uint8_t* code = codes + i * code_size;
        float* xi = x + i * d;
        memcpy(xi, code, code_size);
    }
}

void IndexIVFLVC::sa_decode(idx_t n, const uint8_t* bytes, float* x) const {
    size_t coarse_size = coarse_code_size();
    for (size_t i = 0; i < n; i++) {
        const uint8_t* code = bytes + i * (code_size + coarse_size);
        float* xi = x + i * d;
        memcpy(xi, code + coarse_size, code_size);
    }
}

namespace {

template <typename VectorDistance>
struct IVFFlatScanner : InvertedListScanner {
    VectorDistance vd;
    using C = typename VectorDistance::C;

    IVFFlatScanner(
            const VectorDistance& vd,
            bool store_pairs,
            const IDSelector* sel)
            : InvertedListScanner(store_pairs, sel), vd(vd) {
        keep_max = vd.is_similarity;
        code_size = vd.d * sizeof(float);
    }

    const float* xi;
    void set_query(const float* query) override {
        this->xi = query;
    }

    void set_list(idx_t list_no, float /* coarse_dis */) override {
        this->list_no = list_no;
    }

    float distance_to_code(const uint8_t* code) const final {
        const float* yj = (float*)code;
        return vd(xi, yj);
    }

    // redefining the scan_codes allows to inline the distance_to_code
    size_t scan_codes(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            ResultHandler& handler) const {
        return run_scan_codes_fix_C<C>(*this, list_size, codes, ids, handler);
    }
};

} // anonymous namespace

InvertedListScanner* IndexIVFLVC::get_InvertedListScanner(
        bool store_pairs,
        const IDSelector* sel,
        const IVFSearchParameters*) const {
    return with_VectorDistance(
            d, metric_type, metric_arg, [&](auto vd) -> InvertedListScanner* {
                return new IVFFlatScanner<decltype(vd)>(vd, store_pairs, sel);
            });
}

void IndexIVFLVC::reconstruct_from_offset(
        int64_t list_no,
        int64_t offset,
        float* recons) const {
    if (compact_records_) {
        if (list_no < 0 || size_t(list_no) >= nlist || offset < 0 ||
            size_t(offset) >= compact_records_->list_offsets[list_no + 1] -
                                  compact_records_->list_offsets[list_no])
            FAISS_THROW_MSG("invalid compact IVF-LVC list or offset");
        compact_records_->decode_slot(compact_records_->list_offsets[list_no] + offset, recons);
        return;
    }
    memcpy(recons, invlists->get_single_code(list_no, offset), code_size);
}

void IndexIVFLVC::compact(ivf_lvc::PayloadMode mode, const ivf_lvc::Forest* forest) {
    if (compact_records_) FAISS_THROW_MSG("IVF-LVC index is already compacted");
    auto records = ivf_lvc::CompactRecords::build(*this, forest, mode);
    records->set_state_cache_ratio(0.01);
    replace_invlists(new ArrayInvertedLists(nlist, code_size), true);
    compact_records_ = std::move(records);
}

void IndexIVFLVC::set_decode_state_cache_ratio(double ratio) {
    if (!compact_records_) FAISS_THROW_MSG("IVF-LVC index is not compacted");
    compact_records_->set_state_cache_ratio(ratio);
}

void IndexIVFLVC::save_compact(const std::string& prefix) const {
    if (!compact_records_) FAISS_THROW_MSG("IVF-LVC index is not compacted");
    write_index(quantizer, (prefix + ".quantizer.faiss").c_str());
    compact_records_->save(prefix + ".records.bin");
}

std::unique_ptr<IndexIVFLVC> IndexIVFLVC::load_compact(const std::string& prefix) {
    auto records = ivf_lvc::CompactRecords::load(prefix + ".records.bin");
    std::unique_ptr<Index> quantizer(read_index((prefix + ".quantizer.faiss").c_str()));
    if (!quantizer || quantizer->d != int(records->dim) ||
        quantizer->ntotal != idx_t(records->list_offsets.size() - 1) ||
        quantizer->metric_type != METRIC_L2)
        FAISS_THROW_MSG("IVF-LVC saved quantizer does not match compact records");
    auto index = std::make_unique<IndexIVFLVC>(
        quantizer.get(), records->dim, records->list_offsets.size() - 1);
    quantizer.release();
    index->own_fields = true;
    index->is_trained = true;
    index->ntotal = records->ids.size();
    index->nprobe = records->nprobe;
    index->max_codes = records->max_codes;
    index->compact_records_ = std::move(records);
    index->compact_records_->set_state_cache_ratio(0.01);
    return index;
}

void IndexIVFLVC::search_preassigned(
        idx_t n, const float* x, idx_t k, const idx_t* assign,
        const float* centroid_dis, float* distances, idx_t* labels,
        bool store_pairs, const IVFSearchParameters* params,
        IndexIVFStats* stats) const {
    if (!compact_records_) {
        IndexIVF::search_preassigned(n, x, k, assign, centroid_dis, distances,
                                     labels, store_pairs, params, stats);
        return;
    }
    if (store_pairs || k <= 0) FAISS_THROW_MSG("unsupported compact IVF-LVC search option");
    const auto& records = *compact_records_;
    const size_t probes = std::min(nlist, params ? params->nprobe : nprobe);
    const size_t code_limit = params ? params->max_codes : max_codes;
    for (idx_t qi = 0; qi < n; ++qi) {
        std::priority_queue<std::pair<float, idx_t>> best;
        std::vector<float> decoded(d);
        size_t scanned = 0;
        for (size_t pi = 0; pi < probes; ++pi) {
            const idx_t list = assign[qi * probes + pi];
            if (list < 0 || size_t(list) >= nlist) continue;
            if (stats) ++stats->nlist;
            for (size_t slot = records.list_offsets[list];
                 slot < records.list_offsets[list + 1]; ++slot) {
                if (code_limit && scanned >= code_limit) break;
                records.decode_slot(slot, decoded.data());
                const float distance = fvec_L2sqr(x + qi * d, decoded.data(), d);
                const auto candidate = std::make_pair(distance, records.ids[slot]);
                if (best.size() < size_t(k)) best.push(candidate);
                else if (candidate < best.top()) { best.pop(); best.push(candidate); }
                ++scanned;
            }
            if (code_limit && scanned >= code_limit) break;
        }
        if (stats) stats->ndis += scanned;
        for (idx_t rank = k; rank-- > 0;) {
            const size_t result = size_t(qi * k + rank);
            if (!best.empty()) {
                distances[result] = best.top().first;
                labels[result] = best.top().second;
                best.pop();
            } else {
                distances[result] = std::numeric_limits<float>::infinity();
                labels[result] = -1;
            }
        }
    }
}


} // namespace faiss
