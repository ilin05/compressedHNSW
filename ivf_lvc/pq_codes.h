#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace faiss { struct IndexIVFLVC; }

namespace ivf_lvc {

struct PQCodes {
    size_t dim = 0;
    size_t m = 0;
    size_t count = 0;
    uint64_t posting_hash = 0;
    double training_seconds = 0;
    double encoding_seconds = 0;
    std::vector<float> centroids;
    std::vector<uint8_t> codes; // IVF posting slot order, m bytes per vector.

    size_t codebook_bytes() const { return centroids.size() * sizeof(float); }
    size_t code_bytes() const { return codes.size(); }
    void distance_table(const float* query, float* table) const;
    void save(const std::string& path) const;
    static std::shared_ptr<PQCodes> load(const std::string& path);
};

std::shared_ptr<PQCodes> train_pq_codes(const faiss::IndexIVFLVC& index,
                                        const float* dense_vectors,
                                        size_t training_limit = 200000);

} // namespace ivf_lvc
