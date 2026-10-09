#pragma once

#include <faiss/IndexIVF.h>
#include "hnswlib/compressed_codecs.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace faiss { struct IndexIVFLVC; }
namespace ivf_lvc { struct Forest; }

namespace ivf_lvc {

enum class PayloadMode { Raw, ALP, DeXOR, Gorilla, Elf };
PayloadMode parse_mode(const std::string& name);
const char* mode_name(PayloadMode mode);

struct CompactBytes {
    size_t payload = 0;
    size_t ids = 0;
    size_t offsets = 0;
    size_t parents = 0;
    size_t patches = 0;
    size_t total() const { return payload + ids + offsets + parents + patches; }
};

struct CompactRecords {
    template <typename State> struct CachedState {
        std::vector<State> states;
        std::vector<float> vector;
    };
    PayloadMode mode;
    size_t dim;
    size_t nprobe = 1;
    size_t max_codes = 0;
    std::vector<uint64_t> list_offsets;
    std::vector<faiss::idx_t> ids;
    std::vector<uint64_t> record_offsets;
    std::vector<uint8_t> payload;
    std::vector<faiss::idx_t> parents;      // Slot order, differential modes only.
    std::vector<uint64_t> slot_for_id;      // Dense ID -> slot, differential only.
    std::vector<uint64_t> patch_offsets;    // Slot order, differential only.
    std::vector<uint32_t> patch_coordinates;
    std::vector<uint32_t> patch_bits;
    double state_cache_ratio = 0;
    std::unordered_map<size_t, CachedState<hnswlib::codecs::DeXORCodecPolicy::StateType>> dexor_cache;
    std::unordered_map<size_t, CachedState<hnswlib::codecs::GorillaCodecPolicy::StateType>> gorilla_cache;
    std::unordered_map<size_t, CachedState<hnswlib::codecs::ElfCodecPolicy::StateType>> elf_cache;

    static std::unique_ptr<CompactRecords> build(const faiss::IndexIVFLVC& index,
                                                  const Forest* forest,
                                                  PayloadMode mode);
    void decode_slot(size_t slot, float* result) const;
    void set_state_cache_ratio(double ratio);
    size_t state_cache_nodes() const;
    size_t state_cache_bytes() const;
    void save(const std::string& path) const;
    static std::unique_ptr<CompactRecords> load(const std::string& path);
    CompactBytes bytes() const;
    size_t patch_count() const { return patch_coordinates.size(); }
};

} // namespace ivf_lvc
