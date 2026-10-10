#include "ivf_lvc/pq_codes.h"
#include "ivf_lvc/IndexIVFLVC.h"

#include <faiss/impl/ProductQuantizer.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>

namespace ivf_lvc {
namespace {
constexpr uint64_t magic = 0x31515043564c4649ULL; // IFLVCPQ1

template <typename T> void write(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!out) throw std::runtime_error("cannot write IVF-LVC PQ data");
}
template <typename T> T read(std::istream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!in) throw std::runtime_error("truncated IVF-LVC PQ data");
    return value;
}
} // namespace

void PQCodes::distance_table(const float* query, float* table) const {
    const size_t subdim = dim / m;
    for (size_t sub = 0; sub < m; ++sub) {
        for (size_t code = 0; code < 256; ++code) {
            const float* center = centroids.data() + (sub * 256 + code) * subdim;
            float distance = 0;
            for (size_t j = 0; j < subdim; ++j) {
                const float delta = query[sub * subdim + j] - center[j];
                distance += delta * delta;
            }
            table[sub * 256 + code] = distance;
        }
    }
}

void PQCodes::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot create IVF-LVC PQ file");
    write(out, magic);
    write(out, uint64_t(dim)); write(out, uint64_t(m)); write(out, uint64_t(count));
    write(out, posting_hash);
    write(out, training_seconds); write(out, encoding_seconds);
    out.write(reinterpret_cast<const char*>(centroids.data()), codebook_bytes());
    out.write(reinterpret_cast<const char*>(codes.data()), code_bytes());
    if (!out) throw std::runtime_error("cannot write IVF-LVC PQ arrays");
}

std::shared_ptr<PQCodes> PQCodes::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in || read<uint64_t>(in) != magic) throw std::runtime_error("invalid IVF-LVC PQ file");
    auto result = std::make_shared<PQCodes>();
    result->dim = read<uint64_t>(in); result->m = read<uint64_t>(in);
    result->count = read<uint64_t>(in);
    result->posting_hash = read<uint64_t>(in);
    result->training_seconds = read<double>(in);
    result->encoding_seconds = read<double>(in);
    if (!result->dim || result->dim % 8 || result->m != result->dim / 8 ||
        result->count > (uint64_t(1) << 36))
        throw std::runtime_error("invalid IVF-LVC PQ dimensions");
    result->centroids.resize(result->m * 256 * 8);
    result->codes.resize(result->count * result->m);
    in.read(reinterpret_cast<char*>(result->centroids.data()), result->codebook_bytes());
    in.read(reinterpret_cast<char*>(result->codes.data()), result->code_bytes());
    if (!in) throw std::runtime_error("truncated IVF-LVC PQ arrays");
    return result;
}

std::shared_ptr<PQCodes> train_pq_codes(const faiss::IndexIVFLVC& index,
                                        const float* dense_vectors, size_t training_limit) {
    if (!index.is_trained || !index.invlists || !dense_vectors || index.d <= 0 ||
        index.d % 8 || index.ntotal < 256 || training_limit < 256)
        throw std::invalid_argument("PQ needs trained IVF, dense F32 vectors, dim divisible by 8");
    const size_t count = size_t(index.ntotal);
    const size_t dim = size_t(index.d);
    const size_t m = dim / 8;
    const size_t sample_count = std::min(count, training_limit);
    std::vector<float> samples(sample_count * dim);
    std::vector<size_t> sample_ids(count);
    std::iota(sample_ids.begin(), sample_ids.end(), 0);
    std::mt19937_64 rng(20261010);
    std::shuffle(sample_ids.begin(), sample_ids.end(), rng);
    for (size_t i = 0; i < sample_count; ++i)
        std::copy_n(dense_vectors + sample_ids[i] * dim, dim, samples.data() + i * dim);
    sample_ids.clear();
    sample_ids.shrink_to_fit();

    faiss::ProductQuantizer pq(dim, m, 8);
    const auto start = std::chrono::steady_clock::now();
    pq.train(sample_count, samples.data());
    const auto trained = std::chrono::steady_clock::now();
    auto result = std::make_shared<PQCodes>();
    result->dim = dim; result->m = m; result->count = count;
    result->training_seconds = std::chrono::duration<double>(trained - start).count();
    result->centroids = pq.centroids;
    std::vector<uint8_t> dense_codes(count * m);
    pq.compute_codes(dense_vectors, dense_codes.data(), count);
    result->codes.resize(count * m);
    result->posting_hash = 14695981039346656037ULL;
    size_t slots = 0;
    for (size_t list = 0; list < index.nlist; ++list) {
        faiss::InvertedLists::ScopedIds ids(index.invlists, list);
        for (size_t pos = 0; pos < index.invlists->list_size(list); ++pos) {
            const auto id = ids.get()[pos];
            if (id < 0 || size_t(id) >= count) throw std::runtime_error("PQ expects dense IVF IDs");
            result->posting_hash = (result->posting_hash ^ uint64_t(id)) * 1099511628211ULL;
            std::copy_n(dense_codes.data() + size_t(id) * m, m,
                        result->codes.data() + slots * m);
            ++slots;
        }
    }
    if (slots != count) throw std::runtime_error("PQ posting count mismatch");
    result->encoding_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - trained).count();
    return result;
}

} // namespace ivf_lvc
