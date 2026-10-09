/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef FAISS_INDEX_IVF_LVC_H
#define FAISS_INDEX_IVF_LVC_H

#include <stdint.h>
#include <memory>
#include <string>
#include <unordered_map>

#include <faiss/IndexIVF.h>

namespace ivf_lvc {
struct Forest;
struct CompactRecords;
enum class PayloadMode;
}

namespace faiss {

/** Project-local copy of Faiss 1.14.1 IndexIVFFlat for the IVF-LVC extension.
 *  Stage 1 retains flat-vector storage and baseline search behavior.
 *
 *  Inverted file with stored vectors. Here the inverted file
 * pre-selects the vectors to be searched, but they are not otherwise
 * encoded, the code array just contains the raw float entries.
 */
struct IndexIVFLVC : IndexIVF {
    std::unique_ptr<ivf_lvc::CompactRecords> compact_records_;

    IndexIVFLVC(
            Index* quantizer,
            size_t d,
            size_t nlist_,
            MetricType = METRIC_L2,
            bool own_invlists = true);

    void add_core(
            idx_t n,
            const float* x,
            const idx_t* xids,
            const idx_t* precomputed_idx,
            void* inverted_list_context = nullptr) override;

    void encode_vectors(
            idx_t n,
            const float* x,
            const idx_t* list_nos,
            uint8_t* codes,
            bool include_listnos = false) const override;

    void decode_vectors(
            idx_t n,
            const uint8_t* codes,
            const idx_t* list_nos,
            float* x) const override;

    InvertedListScanner* get_InvertedListScanner(
            bool store_pairs,
            const IDSelector* sel,
            const IVFSearchParameters* params) const override;

    void search_preassigned(idx_t n, const float* x, idx_t k,
                            const idx_t* assign, const float* centroid_dis,
                            float* distances, idx_t* labels, bool store_pairs,
                            const IVFSearchParameters* params = nullptr,
                            IndexIVFStats* stats = nullptr) const override;

    void compact(ivf_lvc::PayloadMode mode, const ivf_lvc::Forest* forest = nullptr);
    void save_compact(const std::string& prefix) const;
    static std::unique_ptr<IndexIVFLVC> load_compact(const std::string& prefix);
    const ivf_lvc::CompactRecords* compact_records() const { return compact_records_.get(); }
    void set_decode_state_cache_ratio(double ratio);

    void reconstruct_from_offset(int64_t list_no, int64_t offset, float* recons)
            const override;

    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

    IndexIVFLVC();
    ~IndexIVFLVC() override;
};


} // namespace faiss

#endif
