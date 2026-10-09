#include "ivf_lvc/compact_records.h"
#include "ivf_lvc/forest.h"

#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<float> read_fvecs(const std::string& path, size_t limit, size_t& dim) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open " + path);
    int32_t first_dim = 0;
    input.read(reinterpret_cast<char*>(&first_dim), 4);
    if (!input || first_dim <= 0) throw std::runtime_error("invalid fvecs dimension");
    dim = first_dim;
    input.seekg(0, std::ios::end);
    const auto bytes = input.tellg();
    const auto row_bytes = std::streamoff(4 + 4 * dim);
    if (bytes <= 0 || bytes % row_bytes != 0) throw std::runtime_error("invalid fvecs file length");
    const size_t count = std::min(limit, size_t(bytes / row_bytes));
    input.seekg(0);
    std::vector<float> data(count * dim);
    for (size_t i = 0; i < count; ++i) {
        int32_t row_dim = 0;
        input.read(reinterpret_cast<char*>(&row_dim), 4);
        if (row_dim != first_dim) throw std::runtime_error("inconsistent fvecs row dimension");
        input.read(reinterpret_cast<char*>(data.data() + i * dim), 4 * dim);
        if (!input) throw std::runtime_error("truncated fvecs row");
    }
    return data;
}

uint32_t bits(float value) {
    uint32_t result;
    std::memcpy(&result, &value, 4);
    return result;
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string input, output = "ivf_lvc_compact_results.csv";
        size_t count = 2048, nlist = 8;
        uint64_t seed = 20261009;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
            if (arg == "--input") input = argv[++i];
            else if (arg == "--output") output = argv[++i];
            else if (arg == "--count") count = std::stoull(argv[++i]);
            else if (arg == "--nlist") nlist = std::stoull(argv[++i]);
            else if (arg == "--seed") seed = std::stoull(argv[++i]);
            else throw std::invalid_argument("unknown argument: " + arg);
        }
        if (input.empty() || nlist == 0) throw std::invalid_argument("provide --input and --nlist");
        size_t dim = 0;
        auto data = read_fvecs(input, count, dim);
        const size_t nodes = data.size() / dim;
        if (nodes < nlist * 39) throw std::invalid_argument("need >=39 training points per IVF centroid");
        faiss::IndexIVFFlat baseline(new faiss::IndexFlatL2(dim), dim, nlist);
        baseline.own_fields = true;
        baseline.train(nodes, data.data());
        baseline.add(nodes, data.data());
        baseline.nprobe = nlist;

        auto initial = ivf_lvc::clone_from_ivf_flat(baseline);
        const auto forest = ivf_lvc::build_forest(*initial, seed);
        const size_t raw_f32_bytes = nodes * dim * sizeof(float);
        const size_t centroid_bytes = nlist * dim * sizeof(float);
        const size_t baseline_bytes = raw_f32_bytes + nodes * sizeof(faiss::idx_t) + centroid_bytes;
        const std::string baseline_file = output + ".ivfflat.faiss";
        faiss::write_index(&baseline, baseline_file.c_str());
        const auto baseline_file_bytes = std::filesystem::file_size(baseline_file);
        std::ofstream csv(output);
        if (!csv) throw std::runtime_error("cannot write CSV: " + output);
        csv << "mode,nodes,dim,nlist,roots,exact_vectors,exact_coordinates,patches,"
               "max_abs_error,raw_f32_bytes,payload_bytes,id_bytes,offset_bytes,"
               "parent_bytes,patch_bytes,total_compact_bytes,centroid_bytes,"
               "payload_ratio,full_index_ratio,baseline_file_bytes,"
               "compact_file_bytes,serialized_ratio\n";
        const std::vector<ivf_lvc::PayloadMode> modes = {
            ivf_lvc::PayloadMode::Raw, ivf_lvc::PayloadMode::ALP,
            ivf_lvc::PayloadMode::DeXOR, ivf_lvc::PayloadMode::Gorilla,
            ivf_lvc::PayloadMode::Elf};
        for (auto mode : modes) {
            auto index = ivf_lvc::clone_from_ivf_flat(baseline);
            index->nprobe = nlist;
            index->compact(mode, &forest);
            const auto* records = index->compact_records();
            if (!records || records->ids.size() != nodes)
                throw std::runtime_error("compact record count mismatch");
            for (size_t list = 0; list < nlist; ++list) {
                if (index->invlists->list_size(list) != 0)
                    throw std::runtime_error("raw IVF posting remains resident after compaction");
            }
            size_t exact_vectors = 0, exact_coordinates = 0;
            double max_abs_error = 0;
            std::vector<float> decoded(dim);
            for (size_t list = 0; list < nlist; ++list) {
                for (size_t pos = 0; pos < records->list_offsets[list + 1] - records->list_offsets[list]; ++pos) {
                    const size_t slot = records->list_offsets[list] + pos;
                    index->reconstruct_from_offset(list, pos, decoded.data());
                    const auto id = size_t(records->ids[slot]);
                    bool vector_exact = true;
                    for (size_t j = 0; j < dim; ++j) {
                        const float original = data[id * dim + j];
                        if (bits(original) == bits(decoded[j])) ++exact_coordinates;
                        else vector_exact = false;
                        max_abs_error = std::max(max_abs_error,
                                                 std::abs(double(original) - double(decoded[j])));
                    }
                    exact_vectors += vector_exact;
                }
            }
            if (mode != ivf_lvc::PayloadMode::ALP && exact_coordinates != nodes * dim)
                throw std::runtime_error(std::string("F32 bitwise reconstruction failed for ") +
                                         ivf_lvc::mode_name(mode));
            if (mode == ivf_lvc::PayloadMode::ALP && max_abs_error > 1e-3)
                throw std::runtime_error("ALP reconstruction exceeds 1e-3 tolerance");

            const std::string compact_prefix = output + "." + ivf_lvc::mode_name(mode);
            index->save_compact(compact_prefix);
            auto reloaded = faiss::IndexIVFLVC::load_compact(compact_prefix);
            std::vector<float> restored(dim);
            for (size_t list = 0; list < nlist; ++list) {
                for (size_t pos = 0; pos < records->list_offsets[list + 1] - records->list_offsets[list]; ++pos) {
                    index->reconstruct_from_offset(list, pos, decoded.data());
                    reloaded->reconstruct_from_offset(list, pos, restored.data());
                    if (std::memcmp(decoded.data(), restored.data(), dim * sizeof(float)))
                        throw std::runtime_error("reloaded IVF-LVC vector differs from saved index");
                }
            }

            const size_t query_count = std::min<size_t>(32, nodes);
            std::vector<float> baseline_dist(query_count), compact_dist(query_count);
            std::vector<faiss::idx_t> baseline_ids(query_count), compact_ids(query_count);
            baseline.search(query_count, data.data(), 1, baseline_dist.data(), baseline_ids.data());
            index->search(query_count, data.data(), 1, compact_dist.data(), compact_ids.data());
            std::vector<float> reload_dist(query_count);
            std::vector<faiss::idx_t> reload_ids(query_count);
            reloaded->search(query_count, data.data(), 1, reload_dist.data(), reload_ids.data());
            if (compact_ids != reload_ids || compact_dist != reload_dist)
                throw std::runtime_error("reloaded IVF-LVC search differs from saved index");
            if (mode != ivf_lvc::PayloadMode::ALP) {
                for (size_t qi = 0; qi < query_count; ++qi) {
                    if (baseline_ids[qi] != compact_ids[qi] ||
                        baseline_dist[qi] != compact_dist[qi])
                        throw std::runtime_error("compact exact search differs from IVF-Flat");
                }
            }

            const auto memory = records->bytes();
            const double payload_ratio = double(raw_f32_bytes) / memory.payload;
            const double full_ratio = double(baseline_bytes) / (memory.total() + centroid_bytes);
            const auto compact_file_bytes =
                std::filesystem::file_size(compact_prefix + ".records.bin") +
                std::filesystem::file_size(compact_prefix + ".quantizer.faiss");
            const double serialized_ratio = double(baseline_file_bytes) / compact_file_bytes;
            csv << ivf_lvc::mode_name(mode) << ',' << nodes << ',' << dim << ',' << nlist << ','
                << forest.summary.roots << ',' << exact_vectors << ',' << exact_coordinates << ','
                << records->patch_count() << ',' << std::setprecision(9) << max_abs_error << ','
                << raw_f32_bytes << ',' << memory.payload << ',' << memory.ids << ','
                << memory.offsets << ',' << memory.parents << ',' << memory.patches << ','
                << memory.total() << ',' << centroid_bytes << ',' << std::fixed
                << std::setprecision(4) << payload_ratio << ',' << full_ratio << ','
                << baseline_file_bytes << ',' << compact_file_bytes << ','
                << serialized_ratio << '\n';
            std::cout << "PASS " << ivf_lvc::mode_name(mode) << " vectors=" << exact_vectors
                      << '/' << nodes << " coordinates=" << exact_coordinates << '/'
                      << nodes * dim << " max_abs_error=" << max_abs_error
                      << " patches=" << records->patch_count()
                      << " payload_bytes=" << memory.payload
                      << " metadata_bytes=" << memory.total() - memory.payload
                      << " payload_ratio=" << payload_ratio
                      << " full_index_ratio=" << full_ratio
                      << " serialized_ratio=" << serialized_ratio << '\n';
        }
        std::cout << "Results: " << output << '\n';
    } catch (const std::exception& error) {
        std::cerr << "FAIL IVF-LVC compaction: " << error.what() << '\n';
        return 1;
    }
}
