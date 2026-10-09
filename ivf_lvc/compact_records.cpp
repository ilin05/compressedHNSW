#include "ivf_lvc/compact_records.h"
#include "ivf_lvc/IndexIVFLVC.h"
#include "ivf_lvc/forest.h"
#include "hnswlib/compressed_codecs.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>

namespace ivf_lvc {
namespace {

bool differential(PayloadMode mode) {
    return mode == PayloadMode::DeXOR || mode == PayloadMode::Gorilla ||
           mode == PayloadMode::Elf;
}

template <typename T>
void write_value(std::ostream& output, const T& value) {
    static_assert(std::is_trivially_copyable<T>::value, "binary field must be POD");
    output.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!output) throw std::runtime_error("cannot write IVF-LVC record file");
}

template <typename T>
T read_value(std::istream& input) {
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!input) throw std::runtime_error("truncated IVF-LVC record file");
    return value;
}

template <typename T>
void write_vector(std::ostream& output, const std::vector<T>& values) {
    write_value(output, uint64_t(values.size()));
    if (!values.empty()) output.write(reinterpret_cast<const char*>(values.data()),
                                       values.size() * sizeof(T));
    if (!output) throw std::runtime_error("cannot write IVF-LVC vector field");
}

template <typename T>
void read_vector(std::istream& input, std::vector<T>& values) {
    const uint64_t size = read_value<uint64_t>(input);
    if (size > (uint64_t(1) << 44) / sizeof(T))
        throw std::runtime_error("invalid IVF-LVC vector field length");
    values.resize(static_cast<size_t>(size));
    if (!values.empty()) input.read(reinterpret_cast<char*>(values.data()),
                                     values.size() * sizeof(T));
    if (!input) throw std::runtime_error("truncated IVF-LVC vector field");
}

uint32_t float_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

template <typename Policy>
void encode_differential(const float* value, const std::vector<const float*>& ancestors, size_t dim,
                         std::vector<char>& output) {
    std::vector<typename Policy::StateType> states(dim);
    for (size_t j = 0; j < dim; ++j) {
        if (!std::isfinite(value[j]))
            throw std::runtime_error("differential codec requires finite F32 input");
    }
    for (const float* ancestor : ancestors) {
        std::vector<char> discarded;
        utils::MemoryStreamWriter discard_writer(&discarded);
        for (size_t j = 0; j < dim; ++j) {
            if (!std::isfinite(ancestor[j]))
                throw std::runtime_error("differential codec requires finite F32 input");
            Policy::encode(static_cast<double>(ancestor[j]), states[j], discard_writer);
        }
    }
    utils::MemoryStreamWriter writer(&output);
    for (size_t j = 0; j < dim; ++j)
        Policy::encode(static_cast<double>(value[j]), states[j], writer);
    writer.align();
}

template <typename Policy>
void decode_differential(const CompactRecords& records, size_t slot,
                         float* result, std::vector<typename Policy::StateType>* final_states = nullptr) {
    std::vector<typename Policy::StateType> states(records.dim);
    const auto& cache = [&]() -> const auto& {
        if constexpr (std::is_same_v<Policy, hnswlib::codecs::DeXORCodecPolicy>) return records.dexor_cache;
        else if constexpr (std::is_same_v<Policy, hnswlib::codecs::GorillaCodecPolicy>) return records.gorilla_cache;
        else return records.elf_cache;
    }();
    std::vector<size_t> path;
    size_t current = slot;
    while (true) {
        const auto hit = cache.find(current);
        if (hit != cache.end()) {
            states = hit->second.states;
            if (current == slot) std::memcpy(result, hit->second.vector.data(), records.dim * sizeof(float));
            break;
        }
        path.push_back(current);
        const auto parent = records.parents[current];
        if (parent < 0) break;
        if (size_t(parent) >= records.slot_for_id.size() || path.size() > records.ids.size())
            throw std::runtime_error("invalid differential predecessor chain");
        current = records.slot_for_id[size_t(parent)];
        if (current >= records.ids.size()) throw std::runtime_error("invalid differential parent slot");
    }
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        const size_t record_slot = *it;
        const auto begin = records.record_offsets[record_slot];
        const auto end = records.record_offsets[record_slot + 1];
        utils::MemoryBlockStreamReader reader(records.payload.data());
        reader.resetBuffer(records.payload.data() + begin, end - begin);
        for (size_t j = 0; j < records.dim; ++j) {
            const double value = Policy::decode(states[j], reader);
            if (record_slot == slot) result[j] = static_cast<float>(value);
        }
    }
    if (final_states) *final_states = std::move(states);
}

int bit_width(uint64_t value) {
    int result = 0;
    while (value) { ++result; value >>= 1; }
    return result;
}

// Matches the current HNSW+ALP F32 exponent/FFOR record and 1e-4 tolerance.
void alp_encode(const float* value, size_t dim, std::vector<char>& output) {
    int exponent = 16;
    for (int exp = 0; exp <= 16; ++exp) {
        const float factor = std::pow(10.0f, float(exp));
        const float inverse = 1.0f / factor;
        bool acceptable = true;
        for (size_t j = 0; j < dim; ++j) {
            if (!std::isfinite(value[j])) throw std::runtime_error("ALP requires finite F32 input");
            const float restored = std::round(value[j] * factor) * inverse;
            if (std::abs(value[j] - restored) > 1e-4f) {
                acceptable = false;
                break;
            }
        }
        if (acceptable) { exponent = exp; break; }
    }
    const float factor = std::pow(10.0f, float(exponent));
    int64_t minimum = INT64_MAX, maximum = INT64_MIN;
    for (size_t j = 0; j < dim; ++j) {
        const double scaled = double(value[j]) * double(factor);
        if (!std::isfinite(scaled) || std::abs(scaled) > 9e18)
            throw std::runtime_error("ALP scaled value outside int64 range");
        const int64_t integer = std::llround(value[j] * factor);
        minimum = std::min(minimum, integer);
        maximum = std::max(maximum, integer);
    }
    const int width = bit_width(uint64_t(maximum) - uint64_t(minimum));
    utils::MemoryStreamWriter writer(&output);
    writer.write(exponent, 8);
    writer.write(width, 8);
    writer.write(static_cast<long long>(minimum), 64);
    for (size_t j = 0; j < dim; ++j) {
        const int64_t integer = std::llround(value[j] * factor);
        writer.write(static_cast<long long>(uint64_t(integer) - uint64_t(minimum)), width);
    }
    writer.align();
}

void alp_decode(const uint8_t* data, size_t bytes, size_t dim, float* result) {
    utils::MemoryBlockStreamReader reader(data);
    reader.resetBuffer(data, bytes);
    const int exponent = reader.readInt(8);
    const int width = reader.readInt(8);
    const int64_t minimum = reader.readLong(64);
    if (exponent < 0 || exponent > 16 || width < 0 || width > 64)
        throw std::runtime_error("invalid ALP record");
    const float factor = std::pow(10.0f, float(-exponent));
    for (size_t j = 0; j < dim; ++j) {
        const auto delta = static_cast<uint64_t>(reader.readLong(width));
        const auto integer = static_cast<int64_t>(uint64_t(minimum) + delta);
        result[j] = static_cast<float>(integer) * factor;
    }
}

} // namespace

PayloadMode parse_mode(const std::string& name) {
    if (name == "raw") return PayloadMode::Raw;
    if (name == "alp") return PayloadMode::ALP;
    if (name == "dexor") return PayloadMode::DeXOR;
    if (name == "gorilla") return PayloadMode::Gorilla;
    if (name == "elf") return PayloadMode::Elf;
    throw std::invalid_argument("unknown IVF-LVC mode: " + name);
}

const char* mode_name(PayloadMode mode) {
    switch (mode) {
        case PayloadMode::Raw: return "raw";
        case PayloadMode::ALP: return "alp";
        case PayloadMode::DeXOR: return "dexor";
        case PayloadMode::Gorilla: return "gorilla";
        case PayloadMode::Elf: return "elf";
    }
    return "unknown";
}

std::unique_ptr<CompactRecords> CompactRecords::build(
    const faiss::IndexIVFLVC& index, const Forest* forest, PayloadMode mode) {
    if (!index.invlists || !index.is_trained || index.metric_type != faiss::METRIC_L2 ||
        (differential(mode) && (!forest || forest->nodes.size() != size_t(index.ntotal))))
        throw std::invalid_argument("invalid IVF index or missing differential forest");
    if (differential(mode)) forest->verify(index);
    auto records = std::make_unique<CompactRecords>();
    records->mode = mode;
    records->dim = index.d;
    records->nprobe = index.nprobe;
    records->max_codes = index.max_codes;
    records->list_offsets.reserve(index.nlist + 1);
    records->list_offsets.push_back(0);
    for (size_t list = 0; list < index.nlist; ++list)
        records->list_offsets.push_back(records->list_offsets.back() + index.invlists->list_size(list));
    const size_t count = records->list_offsets.back();
    if (count != size_t(index.ntotal)) throw std::runtime_error("IVF posting count differs from ntotal");
    records->ids.reserve(count);
    if (differential(mode)) {
        records->parents.reserve(count);
        records->slot_for_id.assign(count, UINT64_MAX);
    }
    for (size_t list = 0; list < index.nlist; ++list) {
        faiss::InvertedLists::ScopedIds ids(index.invlists, list);
        for (size_t pos = 0; pos < index.invlists->list_size(list); ++pos) {
            const auto id = ids.get()[pos];
            if (id < 0 || size_t(id) >= count) throw std::runtime_error("expected dense IVF IDs");
            if (differential(mode)) {
                if (records->slot_for_id[id] != UINT64_MAX)
                    throw std::runtime_error("duplicate IVF ID");
                records->slot_for_id[id] = records->ids.size();
                records->parents.push_back(forest->nodes[id].parent);
            }
            records->ids.push_back(id);
        }
    }
    records->record_offsets.reserve(count + 1);
    records->record_offsets.push_back(0);
    for (size_t list = 0; list < index.nlist; ++list) {
        faiss::InvertedLists::ScopedIds ids(index.invlists, list);
        faiss::InvertedLists::ScopedCodes codes(index.invlists, list);
        const size_t list_size = index.invlists->list_size(list);
        for (size_t pos = 0; pos < list_size; ++pos) {
            const auto* value = reinterpret_cast<const float*>(codes.get() + pos * index.code_size);
            const auto id = ids.get()[pos];
            std::vector<const float*> ancestors;
            if (differential(mode)) {
                auto parent_id = forest->nodes[id].parent;
                while (parent_id >= 0) {
                    const auto parent_slot = records->slot_for_id.at(size_t(parent_id));
                    if (parent_slot == UINT64_MAX || parent_slot < records->list_offsets[list] ||
                        parent_slot >= records->list_offsets[list + 1])
                        throw std::runtime_error("differential parent not in same list");
                    const auto parent_pos = parent_slot - records->list_offsets[list];
                    ancestors.push_back(reinterpret_cast<const float*>(codes.get() + parent_pos * index.code_size));
                    parent_id = forest->nodes[size_t(parent_id)].parent;
                }
                std::reverse(ancestors.begin(), ancestors.end());
            }
            std::vector<char> encoded;
            switch (mode) {
                case PayloadMode::Raw:
                    encoded.assign(reinterpret_cast<const char*>(value),
                                   reinterpret_cast<const char*>(value) + index.code_size);
                    break;
                case PayloadMode::ALP: alp_encode(value, records->dim, encoded); break;
                case PayloadMode::DeXOR:
                    encode_differential<hnswlib::codecs::DeXORCodecPolicy>(value, ancestors, records->dim, encoded);
                    break;
                case PayloadMode::Gorilla:
                    encode_differential<hnswlib::codecs::GorillaCodecPolicy>(value, ancestors, records->dim, encoded);
                    break;
                case PayloadMode::Elf:
                    encode_differential<hnswlib::codecs::ElfCodecPolicy>(value, ancestors, records->dim, encoded);
                    break;
            }
            records->payload.insert(records->payload.end(), encoded.begin(), encoded.end());
            records->record_offsets.push_back(records->payload.size());
        }
    }

    if (differential(mode)) {
        std::vector<uint64_t> patch_offsets;
        patch_offsets.reserve(count + 1);
        patch_offsets.push_back(0);
        std::vector<float> decoded(records->dim);
        for (size_t list = 0; list < index.nlist; ++list) {
            faiss::InvertedLists::ScopedCodes codes(index.invlists, list);
            const size_t list_size = index.invlists->list_size(list);
            for (size_t pos = 0; pos < list_size; ++pos) {
                const size_t slot = records->list_offsets[list] + pos;
                const auto* raw = reinterpret_cast<const float*>(codes.get() + pos * index.code_size);
                records->decode_slot(slot, decoded.data());
                for (size_t j = 0; j < records->dim; ++j) {
                    if (float_bits(raw[j]) != float_bits(decoded[j])) {
                        records->patch_coordinates.push_back(j);
                        records->patch_bits.push_back(float_bits(raw[j]));
                    }
                }
                patch_offsets.push_back(records->patch_coordinates.size());
            }
        }
        records->patch_offsets = std::move(patch_offsets);
    }
    records->list_offsets.shrink_to_fit();
    records->ids.shrink_to_fit();
    records->record_offsets.shrink_to_fit();
    records->payload.shrink_to_fit();
    records->parents.shrink_to_fit();
    records->slot_for_id.shrink_to_fit();
    records->patch_offsets.shrink_to_fit();
    records->patch_coordinates.shrink_to_fit();
    records->patch_bits.shrink_to_fit();
    return records;
}

void CompactRecords::decode_slot(size_t slot, float* result) const {
    if (slot >= ids.size()) throw std::out_of_range("IVF-LVC slot out of range");
    const size_t begin = record_offsets[slot], end = record_offsets[slot + 1];
    switch (mode) {
        case PayloadMode::Raw:
            if (end - begin != dim * sizeof(float)) throw std::runtime_error("bad raw record size");
            std::memcpy(result, payload.data() + begin, end - begin);
            break;
        case PayloadMode::ALP:
            alp_decode(payload.data() + begin, end - begin, dim, result);
            break;
        case PayloadMode::DeXOR:
            decode_differential<hnswlib::codecs::DeXORCodecPolicy>(*this, slot, result);
            break;
        case PayloadMode::Gorilla:
            decode_differential<hnswlib::codecs::GorillaCodecPolicy>(*this, slot, result);
            break;
        case PayloadMode::Elf:
            decode_differential<hnswlib::codecs::ElfCodecPolicy>(*this, slot, result);
            break;
    }
    if (!patch_offsets.empty()) {
        for (size_t p = patch_offsets[slot]; p < patch_offsets[slot + 1]; ++p)
            std::memcpy(result + patch_coordinates[p], &patch_bits[p], sizeof(float));
    }
}

template <typename Policy, typename Cache>
void populate_state_cache(CompactRecords& records, const std::vector<size_t>& selected,
                          Cache& cache) {
    cache.reserve(selected.size());
    for (size_t slot : selected) {
        typename Cache::mapped_type entry;
        entry.vector.resize(records.dim);
        decode_differential<Policy>(records, slot, entry.vector.data(), &entry.states);
        for (size_t p = records.patch_offsets[slot]; p < records.patch_offsets[slot + 1]; ++p)
            std::memcpy(entry.vector.data() + records.patch_coordinates[p],
                        &records.patch_bits[p], sizeof(float));
        cache.emplace(slot, std::move(entry));
    }
}

void CompactRecords::set_state_cache_ratio(double ratio) {
    if (!std::isfinite(ratio) || ratio < 0 || ratio > 1)
        throw std::invalid_argument("IVF-LVC state cache ratio must be in [0,1]");
    state_cache_ratio = ratio;
    dexor_cache.clear();
    gorilla_cache.clear();
    elf_cache.clear();
    if (!differential(mode) || ratio == 0) return;
    const size_t count = ids.size();
    const size_t target = std::min(count, size_t(std::floor(double(count) * ratio)));
    if (!target) return;
    std::vector<int64_t> depths(count, -1);
    for (size_t slot = 0; slot < count; ++slot) {
        if (depths[slot] >= 0) continue;
        std::vector<size_t> path;
        size_t current = slot;
        while (depths[current] < 0) {
            if (path.size() > count) throw std::runtime_error("cyclic IVF-LVC predecessor chain");
            path.push_back(current);
            const auto parent = parents[current];
            if (parent < 0) { depths[current] = 0; break; }
            if (size_t(parent) >= slot_for_id.size() || slot_for_id[size_t(parent)] >= count)
                throw std::runtime_error("invalid IVF-LVC predecessor");
            current = slot_for_id[size_t(parent)];
        }
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            if (depths[*it] < 0) depths[*it] = depths[slot_for_id[size_t(parents[*it])]] + 1;
        }
    }
    std::vector<size_t> selected(count);
    std::iota(selected.begin(), selected.end(), 0);
    std::sort(selected.begin(), selected.end(), [&](size_t a, size_t b) {
        return depths[a] != depths[b] ? depths[a] < depths[b] : ids[a] < ids[b];
    });
    selected.resize(target);
    if (mode == PayloadMode::DeXOR)
        populate_state_cache<hnswlib::codecs::DeXORCodecPolicy>(*this, selected, dexor_cache);
    else if (mode == PayloadMode::Gorilla)
        populate_state_cache<hnswlib::codecs::GorillaCodecPolicy>(*this, selected, gorilla_cache);
    else
        populate_state_cache<hnswlib::codecs::ElfCodecPolicy>(*this, selected, elf_cache);
}

size_t CompactRecords::state_cache_nodes() const {
    return dexor_cache.size() + gorilla_cache.size() + elf_cache.size();
}

size_t CompactRecords::state_cache_bytes() const {
    size_t bytes = 0;
    auto add = [&](const auto& cache) {
        for (const auto& item : cache)
            bytes += sizeof(item) + item.second.states.size() * sizeof(item.second.states[0]) +
                     item.second.vector.size() * sizeof(float);
    };
    add(dexor_cache); add(gorilla_cache); add(elf_cache);
    return bytes;
}

CompactBytes CompactRecords::bytes() const {
    CompactBytes result;
    result.payload = payload.size();
    result.ids = ids.size() * sizeof(faiss::idx_t);
    result.offsets = (list_offsets.size() + record_offsets.size()) * sizeof(uint64_t);
    result.parents = (parents.size() * sizeof(faiss::idx_t) +
                      slot_for_id.size() * sizeof(uint64_t));
    result.patches = (patch_offsets.size() * sizeof(uint64_t) +
                      (patch_coordinates.size() + patch_bits.size()) * sizeof(uint32_t));
    return result;
}

void CompactRecords::save(const std::string& path) const {
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("cannot create IVF-LVC record file: " + path);
    write_value(output, uint64_t(0x3243564c46495649ULL)); // IVIFLVC2
    write_value(output, uint32_t(1));
    write_value(output, uint32_t(mode));
    write_value(output, uint64_t(dim));
    write_value(output, uint64_t(nprobe));
    write_value(output, uint64_t(max_codes));
    write_vector(output, list_offsets);
    write_vector(output, ids);
    write_vector(output, record_offsets);
    write_vector(output, payload);
    write_vector(output, parents);
    write_vector(output, slot_for_id);
    write_vector(output, patch_offsets);
    write_vector(output, patch_coordinates);
    write_vector(output, patch_bits);
}

std::unique_ptr<CompactRecords> CompactRecords::load(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open IVF-LVC record file: " + path);
    if (read_value<uint64_t>(input) != 0x3243564c46495649ULL ||
        read_value<uint32_t>(input) != 1)
        throw std::runtime_error("unsupported IVF-LVC record format");
    const auto mode = read_value<uint32_t>(input);
    if (mode > uint32_t(PayloadMode::Elf)) throw std::runtime_error("invalid IVF-LVC codec");
    auto records = std::make_unique<CompactRecords>();
    records->mode = PayloadMode(mode);
    records->dim = read_value<uint64_t>(input);
    records->nprobe = read_value<uint64_t>(input);
    records->max_codes = read_value<uint64_t>(input);
    read_vector(input, records->list_offsets);
    read_vector(input, records->ids);
    read_vector(input, records->record_offsets);
    read_vector(input, records->payload);
    read_vector(input, records->parents);
    read_vector(input, records->slot_for_id);
    read_vector(input, records->patch_offsets);
    read_vector(input, records->patch_coordinates);
    read_vector(input, records->patch_bits);
    const size_t count = records->ids.size();
    if (records->dim == 0 || records->list_offsets.empty() ||
        records->list_offsets.front() != 0 || records->list_offsets.back() != count ||
        records->record_offsets.size() != count + 1 ||
        records->record_offsets.front() != 0 ||
        records->record_offsets.back() != records->payload.size() ||
        !std::is_sorted(records->list_offsets.begin(), records->list_offsets.end()) ||
        !std::is_sorted(records->record_offsets.begin(), records->record_offsets.end()))
        throw std::runtime_error("invalid IVF-LVC record offsets");
    if (differential(records->mode)) {
        if (records->parents.size() != count || records->slot_for_id.size() != count ||
            records->patch_offsets.size() != count + 1 ||
            records->patch_offsets.front() != 0 ||
            records->patch_offsets.back() != records->patch_coordinates.size() ||
            records->patch_coordinates.size() != records->patch_bits.size() ||
            !std::is_sorted(records->patch_offsets.begin(), records->patch_offsets.end()))
            throw std::runtime_error("invalid IVF-LVC differential metadata");
        for (auto coordinate : records->patch_coordinates)
            if (coordinate >= records->dim) throw std::runtime_error("invalid IVF-LVC patch coordinate");
    } else if (!records->parents.empty() || !records->slot_for_id.empty() ||
               !records->patch_offsets.empty() || !records->patch_coordinates.empty() ||
               !records->patch_bits.empty()) {
        throw std::runtime_error("unexpected differential metadata in IVF-LVC record file");
    }
    return records;
}

} // namespace ivf_lvc
