/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include "ivf_lvc/IndexIVFLVC.h"

#include <omp.h>

#include <cinttypes>
#include <cstdio>
#include <numeric>

#include <faiss/IndexFlat.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/IDSelector.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/expanded_scanners.h>
#include <faiss/utils/extra_distances.h>
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

void IndexIVFLVC::add_core(
        idx_t n,
        const float* x,
        const idx_t* xids,
        const idx_t* coarse_idx,
        void* inverted_list_context) {
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
    memcpy(recons, invlists->get_single_code(list_no, offset), code_size);
}


} // namespace faiss
