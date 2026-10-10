#include "ivf_lvc/compact_records.h"
#include "ivf_lvc/forest.h"
#include "ivf_lvc/pq_codes.h"

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/clone_index.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::vector<float> read_fvecs(const std::string& path, size_t limit, size_t& dim) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    int32_t width = 0;
    in.read(reinterpret_cast<char*>(&width), 4);
    if (!in || width <= 0) throw std::runtime_error("invalid fvecs dimension");
    dim = size_t(width);
    in.seekg(0, std::ios::end);
    const auto bytes = in.tellg();
    const auto row_bytes = std::streamoff(4 + 4 * dim);
    if (bytes < row_bytes || bytes % row_bytes) throw std::runtime_error("invalid fvecs file");
    const size_t count = std::min(limit, size_t(bytes / row_bytes));
    in.seekg(0);
    std::vector<float> data(count * dim);
    for (size_t i = 0; i < count; ++i) {
        int32_t row_width = 0;
        in.read(reinterpret_cast<char*>(&row_width), 4);
        if (row_width != width) throw std::runtime_error("inconsistent fvecs width");
        in.read(reinterpret_cast<char*>(data.data() + i * dim), 4 * dim);
        if (!in) throw std::runtime_error("truncated fvecs data");
    }
    return data;
}

std::vector<uint32_t> read_truth1(const std::string& path, size_t expected) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::vector<uint32_t> result(expected);
    int32_t width = 0;
    for (size_t i = 0; i < expected; ++i) {
        in.read(reinterpret_cast<char*>(&width), 4);
        if (!in || width <= 0) throw std::runtime_error("truncated ivecs header");
        in.read(reinterpret_cast<char*>(&result[i]), 4);
        in.seekg(std::streamoff(width - 1) * 4, std::ios::cur);
        if (!in) throw std::runtime_error("truncated ivecs row");
    }
    return result;
}

std::vector<size_t> parse_probes(const std::string& text) {
    std::vector<size_t> result;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        const size_t value = std::stoull(item);
        if (!value) throw std::invalid_argument("nprobe must be positive");
        result.push_back(value);
    }
    if (result.empty()) throw std::invalid_argument("empty nprobe list");
    return result;
}

std::vector<double> parse_alphas(const std::string& text) {
    std::vector<double> result;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        const double value = std::stod(item);
        if (!std::isfinite(value) || value <= 0 || value > 1)
            throw std::invalid_argument("pq_alpha must be in (0,1]");
        result.push_back(value);
    }
    if (result.empty()) throw std::invalid_argument("empty pq_alpha list");
    return result;
}

struct SearchResult {
    std::vector<faiss::idx_t> ids;
    std::vector<float> distances;
    double recall = 0;
    double qps = 0;
    double mean_us = 0;
};

SearchResult search(faiss::Index& index, const std::vector<float>& queries,
                    size_t dim, const std::vector<uint32_t>& truth, size_t rounds) {
    const size_t count = queries.size() / dim;
    SearchResult result;
    result.ids.resize(count);
    result.distances.resize(count);
    const auto start = std::chrono::steady_clock::now();
    for (size_t round = 0; round < rounds; ++round)
        index.search(count, queries.data(), 1, result.distances.data(), result.ids.data());
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    size_t correct = 0;
    for (size_t i = 0; i < count; ++i) correct += result.ids[i] == faiss::idx_t(truth[i]);
    result.recall = double(correct) / count;
    result.qps = double(count * rounds) / seconds;
    result.mean_us = seconds * 1e6 / double(count * rounds);
    return result;
}
} // namespace

int main(int argc, char** argv) {
    try {
        std::string train_file, query_file, truth_file, output_dir = "ivf_lvc_pq_results";
        std::string probe_text = "1,2,4,6,8,10,15,20", alpha_text = "0.05,0.1";
        size_t train_limit = SIZE_MAX, query_limit = SIZE_MAX, nlist = 0, rounds = 1;
        double cache_ratio = 0.01;
        bool use_pq_filter = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--use_pq_filter") { use_pq_filter = true; continue; }
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
            if (arg == "--train") train_file = argv[++i];
            else if (arg == "--queries") query_file = argv[++i];
            else if (arg == "--truth") truth_file = argv[++i];
            else if (arg == "--output-dir") output_dir = argv[++i];
            else if (arg == "--train-limit") train_limit = std::stoull(argv[++i]);
            else if (arg == "--query-limit") query_limit = std::stoull(argv[++i]);
            else if (arg == "--nlist") nlist = std::stoull(argv[++i]);
            else if (arg == "--nprobes") probe_text = argv[++i];
            else if (arg == "--rounds") rounds = std::stoull(argv[++i]);
            else if (arg == "--pq_alpha" || arg == "--pq_alphas") alpha_text = argv[++i];
            else if (arg == "--cache-ratio") cache_ratio = std::stod(argv[++i]);
            else throw std::invalid_argument("unknown argument: " + arg);
        }
        if (train_file.empty() || query_file.empty() || truth_file.empty() || !rounds)
            throw std::invalid_argument("provide train/queries/truth and positive rounds");
        const auto alphas = parse_alphas(alpha_text);
        size_t dim = 0, query_dim = 0;
        auto train = read_fvecs(train_file, train_limit, dim);
        auto queries = read_fvecs(query_file, query_limit, query_dim);
        if (dim != query_dim || dim % 8) throw std::invalid_argument("PQ needs matching dimensions divisible by 8");
        const size_t count = train.size() / dim;
        const auto truth = read_truth1(truth_file, queries.size() / dim);
        if (!nlist) nlist = size_t(std::sqrt(double(count)));
        if (count < nlist * 39) throw std::invalid_argument("not enough IVF training points per centroid");
        auto probes = parse_probes(probe_text);
        for (auto value : probes) if (value > nlist) throw std::invalid_argument("nprobe exceeds nlist");
        std::filesystem::create_directories(output_dir);

        faiss::IndexIVFFlat baseline(new faiss::IndexFlatL2(dim), dim, nlist);
        baseline.own_fields = true;
        const auto build_start = std::chrono::steady_clock::now();
        baseline.train(count, train.data());
        baseline.add(count, train.data());
        const double ivf_build_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - build_start).count();
        auto initial = ivf_lvc::clone_from_ivf_flat(baseline);
        const auto forest_start = std::chrono::steady_clock::now();
        const auto forest = ivf_lvc::build_forest(*initial, 20261010, 0.01, 2);
        const auto forest_unbounded = ivf_lvc::build_forest(*initial, 20261010, 0.01, -1);
        const double forest_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - forest_start).count();
        auto pq = ivf_lvc::train_pq_codes(*initial, train.data());
        pq->save(output_dir + "/shared_pq.bin");
        pq = ivf_lvc::PQCodes::load(output_dir + "/shared_pq.bin");
        auto make_ivfpq = [&](size_t m) {
            auto index = std::make_unique<faiss::IndexIVFPQ>(
                faiss::clone_index(baseline.quantizer), dim, nlist, m, 8);
            index->own_fields = true;
            index->pq.cp.max_points_per_centroid = 782;
            return index;
        };
        auto ivfpq = make_ivfpq(dim / 8);
        auto ivfpq_full = make_ivfpq(dim);
        const auto ivfpq_start = std::chrono::steady_clock::now();
        ivfpq->train(count, train.data());
        const double ivfpq_training_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - ivfpq_start).count();
        const auto ivfpq_add_start = std::chrono::steady_clock::now();
        ivfpq->add(count, train.data());
        const double ivfpq_encoding_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - ivfpq_add_start).count();
        const auto ivfpq_full_start = std::chrono::steady_clock::now();
        ivfpq_full->train(count, train.data());
        const double ivfpq_full_training_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - ivfpq_full_start).count();
        const auto ivfpq_full_add_start = std::chrono::steady_clock::now();
        ivfpq_full->add(count, train.data());
        const double ivfpq_full_encoding_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - ivfpq_full_add_start).count();
        if (ivfpq->code_size != pq->m || ivfpq->ntotal != faiss::idx_t(count) ||
            ivfpq->pq.M != pq->m || ivfpq_full->code_size != dim ||
            ivfpq_full->ntotal != faiss::idx_t(count))
            throw std::runtime_error("IVFPQ does not use the same bytes per vector");
        std::vector<float> left_centroid(dim), right_centroid(dim);
        for (size_t list = 0; list < nlist; ++list) {
            baseline.quantizer->reconstruct(list, left_centroid.data());
            ivfpq->quantizer->reconstruct(list, right_centroid.data());
            if (left_centroid != right_centroid)
                throw std::runtime_error("IVFPQ coarse centroids differ from IVF-Flat");
            ivfpq_full->quantizer->reconstruct(list, right_centroid.data());
            if (left_centroid != right_centroid)
                throw std::runtime_error("full-byte IVFPQ coarse centroids differ from IVF-Flat");
        }
        std::ofstream pq_csv(output_dir + "/pq_summary.csv");
        pq_csv << "variant,nodes,dim,nlist,pq_m,pq_bits,pq_training_seconds,pq_encoding_seconds,"
                  "pq_codebook_bytes,pq_code_bytes,pq_file_bytes,pq_precomputed_table_bytes,"
                  "ivf_build_seconds,forest_seconds,roots\n";
        pq_csv << "lvc_shared," << count << ',' << dim << ',' << nlist << ',' << pq->m << ",8,"
               << pq->training_seconds << ',' << pq->encoding_seconds << ','
               << pq->codebook_bytes() << ',' << pq->code_bytes() << ','
               << std::filesystem::file_size(output_dir + "/shared_pq.bin") << ",0,"
               << ivf_build_seconds << ',' << forest_seconds << ',' << forest.summary.roots << '\n';
        pq_csv << "ivfpq_m_div_8," << count << ',' << dim << ',' << nlist << ',' << ivfpq->pq.M << ",8,"
               << ivfpq_training_seconds << ',' << ivfpq_encoding_seconds << ','
               << ivfpq->pq.centroids.size() * sizeof(float) << ',' << count * ivfpq->code_size
               << ",0," << ivfpq->precomputed_table.nbytes() << ',' << ivf_build_seconds
               << ",0,0\n";
        pq_csv << "ivfpq_m," << count << ',' << dim << ',' << nlist << ',' << ivfpq_full->pq.M << ",8,"
               << ivfpq_full_training_seconds << ',' << ivfpq_full_encoding_seconds << ','
               << ivfpq_full->pq.centroids.size() * sizeof(float) << ',' << count * ivfpq_full->code_size
               << ",0," << ivfpq_full->precomputed_table.nbytes() << ',' << ivf_build_seconds
               << ",0,0\n";
        pq_csv.close();
        if (pq->code_bytes() != count * dim / 8)
            throw std::runtime_error("PQ code size is not N * dim/8");

        const size_t raw_payload_bytes = count * dim * sizeof(float);
        const size_t coarse_bytes = nlist * dim * sizeof(float);
        const size_t posting_metadata_bytes = count * sizeof(faiss::idx_t) +
                                              (nlist + 1) * sizeof(uint64_t);
        const size_t raw_index_bytes = raw_payload_bytes + posting_metadata_bytes + coarse_bytes;
        struct Footprint {
            size_t payload, metadata, codebook, codes, state, auxiliary;
            size_t total(size_t coarse) const {
                return payload + metadata + codebook + codes + state + auxiliary + coarse;
            }
        };
        std::ofstream compression_csv(output_dir + "/compression.csv");
        compression_csv << "mode,chain_max,nodes,dim,nlist,roots,average_chain_depth,max_chain_depth,"
                           "mean_reference_l2,payload_bytes,record_metadata_bytes,coarse_centroid_bytes,"
                           "pq_codebook_bytes,pq_code_bytes,state_cache_bytes,auxiliary_bytes,index_bytes,"
                           "raw_f32_payload_bytes,raw_ivf_index_bytes,payload_ratio,index_ratio,"
                           "compression_seconds\n";
        auto report_compression = [&](const std::string& mode, int chain_max,
                                      const ivf_lvc::Forest* used_forest,
                                      const Footprint& fp, double seconds) {
            const size_t total = fp.total(coarse_bytes);
            compression_csv << mode << ',' << chain_max << ',' << count << ',' << dim << ','
                << nlist << ',' << (used_forest ? used_forest->summary.roots : 0) << ','
                << (used_forest ? used_forest->summary.mean_depth : 0) << ','
                << (used_forest ? used_forest->summary.max_depth : 0) << ','
                << (used_forest ? used_forest->summary.mean_reference_l2 : 0) << ','
                << fp.payload << ',' << fp.metadata << ',' << coarse_bytes << ','
                << fp.codebook << ',' << fp.codes << ',' << fp.state << ','
                << fp.auxiliary << ',' << total << ',' << raw_payload_bytes << ','
                << raw_index_bytes << ',' << std::setprecision(9)
                << double(raw_payload_bytes) / (fp.payload ? fp.payload : fp.codes) << ','
                << double(raw_index_bytes) / total << ',' << seconds << '\n';
            compression_csv.flush();
        };
        std::ofstream csv(output_dir + "/search.csv");
        csv << "mode,chain_max,pq_filter,pq_alpha,nprobe,queries,rounds,recall_at_1,"
               "qps,mean_us,ids_changed_vs_exact,lost_truth_vs_exact,gained_truth_vs_exact,index_bytes\n";
        auto report_search = [&](const std::string& mode, int chain_max, bool filtered,
                                 double alpha, size_t nprobe, const SearchResult& result,
                                 const SearchResult* exact, size_t index_bytes) {
            size_t changed = 0, lost = 0, gained = 0;
            if (exact) for (size_t i = 0; i < result.ids.size(); ++i) {
                changed += result.ids[i] != exact->ids[i];
                lost += exact->ids[i] == faiss::idx_t(truth[i]) &&
                        result.ids[i] != faiss::idx_t(truth[i]);
                gained += exact->ids[i] != faiss::idx_t(truth[i]) &&
                          result.ids[i] == faiss::idx_t(truth[i]);
            }
            csv << mode << ',' << chain_max << ',' << (filtered ? 1 : 0) << ','
                << alpha << ',' << nprobe << ',' << truth.size() << ',' << rounds << ','
                << std::setprecision(9) << result.recall << ',' << result.qps << ','
                << result.mean_us << ',' << changed << ',' << lost << ',' << gained
                << ',' << index_bytes << '\n';
            csv.flush();
            std::cout << mode << " chain=" << chain_max << " filter=" << filtered
                      << " alpha=" << alpha << " nprobe=" << nprobe
                      << " recall@1=" << result.recall << " mean_us=" << result.mean_us << '\n';
        };
        const Footprint flat_fp{raw_payload_bytes, posting_metadata_bytes, 0, 0, 0, 0};
        report_compression("ivf_flat", 0, nullptr, flat_fp, 0);
        auto pq_footprint = [&](const faiss::IndexIVFPQ& index) {
            return Footprint{0, posting_metadata_bytes,
                index.pq.centroids.size() * sizeof(float), count * index.code_size,
                0, index.precomputed_table.nbytes()};
        };
        const auto pq_short_fp = pq_footprint(*ivfpq);
        const auto pq_full_fp = pq_footprint(*ivfpq_full);
        report_compression("ivfpq_m_div_8", 0, nullptr, pq_short_fp,
                           ivfpq_training_seconds + ivfpq_encoding_seconds);
        report_compression("ivfpq_m", 0, nullptr, pq_full_fp,
                           ivfpq_full_training_seconds + ivfpq_full_encoding_seconds);
        for (size_t nprobe : probes) {
            baseline.nprobe = nprobe;
            report_search("ivf_flat", 0, false, 1, nprobe,
                          search(baseline, queries, dim, truth, rounds), nullptr,
                          flat_fp.total(coarse_bytes));
            ivfpq->nprobe = nprobe;
            report_search("ivfpq_m_div_8", 0, false, 1, nprobe,
                          search(*ivfpq, queries, dim, truth, rounds), nullptr,
                          pq_short_fp.total(coarse_bytes));
            ivfpq_full->nprobe = nprobe;
            report_search("ivfpq_m", 0, false, 1, nprobe,
                          search(*ivfpq_full, queries, dim, truth, rounds), nullptr,
                          pq_full_fp.total(coarse_bytes));
        }
        for (auto mode : {ivf_lvc::PayloadMode::Raw, ivf_lvc::PayloadMode::ALP,
                          ivf_lvc::PayloadMode::DeXOR, ivf_lvc::PayloadMode::Gorilla,
                          ivf_lvc::PayloadMode::Elf}) {
            const bool differential = mode == ivf_lvc::PayloadMode::DeXOR ||
                mode == ivf_lvc::PayloadMode::Gorilla || mode == ivf_lvc::PayloadMode::Elf;
            for (int chain_max : (differential ? std::vector<int>{2, -1} : std::vector<int>{0})) {
                const ivf_lvc::Forest* used_forest = !differential ? nullptr :
                    chain_max == 2 ? &forest : &forest_unbounded;
                auto index = ivf_lvc::clone_from_ivf_flat(baseline);
                const auto start = std::chrono::steady_clock::now();
                index->compact(mode, used_forest);
                const double compression_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - start).count();
                index->set_decode_state_cache_ratio(cache_ratio);
                index->set_pq_codes(pq);
                const auto* records = index->compact_records();
                const auto bytes = records->bytes();
                const Footprint fp{bytes.payload, bytes.total() - bytes.payload,
                                   pq->codebook_bytes(), pq->code_bytes(),
                                   records->state_cache_bytes(), 0};
                const std::string name = ivf_lvc::mode_name(mode);
                report_compression(name, chain_max, used_forest, fp, compression_seconds);
                for (size_t nprobe : probes) {
                    index->nprobe = nprobe;
                    index->set_pq_filter(false);
                    const auto exact = search(*index, queries, dim, truth, rounds);
                    report_search(name, chain_max, false, 1, nprobe, exact, nullptr,
                                  fp.total(coarse_bytes));
                    if (!use_pq_filter) continue;
                    for (double alpha : alphas) {
                        index->set_pq_filter(true, alpha);
                        const auto filtered = search(*index, queries, dim, truth, rounds);
                        if (alpha == 1 && (exact.ids != filtered.ids || exact.distances != filtered.distances))
                            throw std::runtime_error("pq_alpha=1 differs from exact IVF-LVC search");
                        report_search(name, chain_max, true, alpha, nprobe, filtered,
                                      &exact, fp.total(coarse_bytes));
                    }
                }
            }
        }
        std::cout << "Results: " << output_dir << "/compression.csv and search.csv\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL IVF-LVC PQ: " << error.what() << '\n';
        return 1;
    }
}
