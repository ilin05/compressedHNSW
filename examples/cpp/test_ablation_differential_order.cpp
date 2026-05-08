#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <omp.h>

#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/compressed_hnsw_framework.h"
#include "../../hnswlib/compressed_codecs.h"
#include "../utils/memory_stream_writer.h"

using namespace std;
using namespace hnswlib;

// ---------------------------------------------------------------------------
// Data loading (reused from test_build_compressed_hnsw_ablation.cpp)
// ---------------------------------------------------------------------------
double* load_fvecs_as_double(const string& filename, size_t& num_vectors, size_t& dim) {
    ifstream input(filename, ios::binary);
    if (!input) {
        cerr << "Cannot open " << filename << endl;
        return nullptr;
    }
    int32_t d = 0;
    input.read((char*)&d, 4);
    dim = static_cast<size_t>(d);

    input.seekg(0, ios::end);
    size_t file_size = static_cast<size_t>(input.tellg());
    num_vectors = file_size / (4 + dim * 4);

    double* data = new double[num_vectors * dim];
    float* tmp = new float[dim];
    input.seekg(0, ios::beg);

    for (size_t i = 0; i < num_vectors; ++i) {
        input.read((char*)&d, 4);
        input.read((char*)tmp, dim * 4);
        for (size_t j = 0; j < dim; ++j) {
            data[i * dim + j] = static_cast<double>(tmp[j]);
        }
    }
    delete[] tmp;
    return data;
}

// ---------------------------------------------------------------------------
// Strategy 1: Intra-Vector  (same-vector cross-dimension)
//   For each vector u: encode x_u[0], x_u[1], ..., x_u[d-1]
//   State resets per vector.
// ---------------------------------------------------------------------------
template<typename CodecPolicy>
double encode_intra_vector(double* data, size_t N, size_t d) {
    vector<char> buffer;
    utils::MemoryStreamWriter writer(&buffer);
    long long total_bits = 0;

    for (size_t u = 0; u < N; ++u) {
        typename CodecPolicy::StateType state;
        for (size_t k = 0; k < d; ++k) {
            CodecPolicy::encode(data[u * d + k], state, writer);
            total_bits += writer.track_bits();
        }
        writer.align();
        total_bits += writer.track_bits();
    }
    // writer.align();
    // total_bits += writer.track_bits();

    size_t compressed_bytes = buffer.size();
    size_t original_bytes = N * d * sizeof(double);
    if (compressed_bytes == 0) return 0.0;
    return static_cast<double>(original_bytes) / static_cast<double>(compressed_bytes);
}

// ---------------------------------------------------------------------------
// Strategy 2: ID-Order Cross-Vector Same-Dimension
//   For each dimension k: encode x_0[k], x_1[k], ..., x_{N-1}[k]
//   Per-dimension independent states.
// ---------------------------------------------------------------------------
template<typename CodecPolicy>
double encode_cross_vector_id_order(double* data, size_t N, size_t d) {
    vector<char> buffer;
    utils::MemoryStreamWriter writer(&buffer);
    long long total_bits = 0;

    std::vector<typename CodecPolicy::StateType> states(d); // one state per dimension
    for (size_t u = 0; u < N; ++u) {
        for (size_t k = 0; k < d; ++k) {
            CodecPolicy::encode(data[u * d + k], states[k], writer);
            total_bits += writer.track_bits();
        }
        writer.align();
        total_bits += writer.track_bits();
    }
    // for (size_t k = 0; k < d; ++k) {
    //     typename CodecPolicy::StateType state;
    //     for (size_t u = 0; u < N; ++u) {
    //         CodecPolicy::encode(data[u * d + k], state, writer);
    //         total_bits += writer.track_bits();
    //     }
    //     writer.align();
    //     total_bits += writer.track_bits();
    // }
    // writer.align();
    // total_bits += writer.track_bits();

    size_t compressed_bytes = buffer.size();
    size_t original_bytes = N * d * sizeof(double);
    if (compressed_bytes == 0) return 0.0;
    return static_cast<double>(original_bytes) / static_cast<double>(compressed_bytes);
}

static std::string chain_tag(int chain_max_length) {
    if (chain_max_length < 0) return "chunlim";
    return "ch" + std::to_string(chain_max_length);
}

// ---------------------------------------------------------------------------
// Strategy 3: Graph-Guided  (cross-vector same-dimension, BFS order)
//   Use HierarchicalNSWCABFRAMEWORK to build HNSW + compress.
//   Returns {compression_ratio, prenode_array_for_residuals}.
// ---------------------------------------------------------------------------
template<typename CodecPolicy>
pair<double, vector<tableint>> encode_graph_guided(
    double* data, size_t N, size_t d,
    int M, int ef_construction, int chain_max_length, const std::string& dataset, const std::string& algo)
{
    const std::string index_file = dataset + "_" + algo + "_" + chain_tag(chain_max_length) + "_pq.bin";
    L2SpaceDouble l2space(d);
    size_t cache_size = N / 100;
    
    auto* index = new HierarchicalNSWCABFRAMEWORK<double, CodecPolicy>(
        &l2space, index_file, true);
    
    // auto* index = new HierarchicalNSWCABFRAMEWORK<double, CodecPolicy>(
    //     &l2space, N, M, ef_construction, true, cache_size, 100, false);

    // // Build HNSW
    // #pragma omp parallel for
    // for (long long i = 0; i < static_cast<long long>(N); ++i) {
    //     index->addPoint(data + i * d, static_cast<labeltype>(i));
    // }

    // Compress
    // index->compress_dataset(chain_max_length);

    // Compression ratio
    size_t original_data_size = N * d * sizeof(double);
    size_t compressed_data_size = index->getCompressedDataSize();
    double ratio = 0.0;
    if (compressed_data_size > 0) {
        ratio = static_cast<double>(original_data_size) /
                static_cast<double>(compressed_data_size);
    }

    // Extract prenode assignments for residual computation
    vector<tableint> prenodes(N);
    for (size_t i = 0; i < N; ++i) {
        prenodes[i] = index->getPrenodeId(static_cast<tableint>(i));
    }

    delete index;
    return {ratio, prenodes};
}

// ---------------------------------------------------------------------------
// Residual computation
// ---------------------------------------------------------------------------

/// Intra-Vector residuals: x_u[k] - x_u[k-1]  for k >= 1
vector<double> compute_residuals_intra(double* data, size_t N, size_t d) {
    vector<double> res;
    res.reserve(N * (d - 1));
    for (size_t u = 0; u < N; ++u) {
        for (size_t k = 1; k < d; ++k) {
            res.push_back(data[u * d + k] - data[u * d + (k - 1)]);
        }
    }
    return res;
}

/// ID-Order residuals: x_u[k] - x_{u-1}[k]  for u >= 1
vector<double> compute_residuals_id_order(double* data, size_t N, size_t d) {
    vector<double> res;
    res.reserve((N - 1) * d);
    for (size_t k = 0; k < d; ++k) {
        for (size_t u = 1; u < N; ++u) {
            res.push_back(data[u * d + k] - data[(u - 1) * d + k]);
        }
    }
    return res;
}

/// Graph-Guided residuals: x_u[k] - x_{prenode[u]}[k] for prenode[u] != -1
vector<double> compute_residuals_graph(double* data, size_t N, size_t d,
                                       const vector<tableint>& prenodes) {
    vector<double> res;
    for (size_t u = 0; u < N; ++u) {
        tableint p = prenodes[u];
        if (p == static_cast<tableint>(-1)) continue;
        for (size_t k = 0; k < d; ++k) {
            res.push_back(data[u * d + k] - data[p * d + k]);
        }
    }
    return res;
}

// ---------------------------------------------------------------------------
// Histogram binning
// ---------------------------------------------------------------------------
struct Histogram {
    vector<double> bin_centers;
    vector<size_t> counts;
};

Histogram build_histogram(const vector<double>& values, size_t num_bins = 100) {
    Histogram h;
    if (values.empty()) return h;

    double vmin = *min_element(values.begin(), values.end());
    double vmax = *max_element(values.begin(), values.end());

    // Expand range slightly for edge values
    double margin = (vmax - vmin) * 0.01;
    if (margin < 1e-10) margin = 0.1;
    vmin -= margin;
    vmax += margin;

    double bin_width = (vmax - vmin) / num_bins;
    h.counts.resize(num_bins, 0);
    h.bin_centers.resize(num_bins);

    for (size_t i = 0; i < num_bins; ++i) {
        h.bin_centers[i] = vmin + (i + 0.5) * bin_width;
    }

    for (double v : values) {
        int idx = static_cast<int>((v - vmin) / bin_width);
        if (idx < 0) idx = 0;
        if (idx >= static_cast<int>(num_bins)) idx = static_cast<int>(num_bins) - 1;
        h.counts[idx]++;
    }

    return h;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    string base_dir = "../datasets/hdf5files/";
    string out_ratio_csv = "differential_order_compression.csv";
    string out_residual_csv = "differential_order_residuals.csv";
    int threads = 32;
    int M = 16;
    int ef_construction = 200;
    int chain_max = -1;
    size_t hist_bins = 100;

    vector<string> datasets = {
        "fashion-mnist-784-euclidean",
        "mnist-784-euclidean",
        "sift-128-euclidean",
        "gist-960-euclidean",
        "deep-image-96-angular"
    };

    vector<string> algorithms = {"DeXOR", "Gorilla", "Elf", "Camel"};

    // --- CLI parsing (reuse pattern from ablation build script) ---
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--base_dir" && i + 1 < argc) {
            base_dir = argv[++i];
        } else if (arg == "--output_ratio" && i + 1 < argc) {
            out_ratio_csv = argv[++i];
        } else if (arg == "--output_residual" && i + 1 < argc) {
            out_residual_csv = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            threads = stoi(argv[++i]);
        } else if (arg == "--M" && i + 1 < argc) {
            M = stoi(argv[++i]);
        } else if (arg == "--ef_construction" && i + 1 < argc) {
            ef_construction = stoi(argv[++i]);
        } else if (arg == "--chain_max" && i + 1 < argc) {
            chain_max = stoi(argv[++i]);
        } else if (arg == "--hist_bins" && i + 1 < argc) {
            hist_bins = static_cast<size_t>(stoi(argv[++i]));
        } else if (arg == "--dataset" && i + 1 < argc) {
            datasets.clear();
            while (i + 1 < argc && string(argv[i + 1]).rfind("--", 0) != 0) {
                datasets.push_back(argv[++i]);
            }
        } else if (arg == "--algorithm" && i + 1 < argc) {
            algorithms.clear();
            while (i + 1 < argc && string(argv[i + 1]).rfind("--", 0) != 0) {
                algorithms.push_back(argv[++i]);
            }
        } else {
            cerr << "Unknown argument: " << arg << endl;
            return -1;
        }
    }

    if (!base_dir.empty() && base_dir.back() != '/' && base_dir.back() != '\\') {
        base_dir += '/';
    }

    omp_set_num_threads(threads);

    // Open output CSVs
    ofstream csv_ratio(out_ratio_csv);
    ofstream csv_residual(out_residual_csv);
    if (!csv_ratio.is_open() || !csv_residual.is_open()) {
        cerr << "Failed to open output CSV files." << endl;
        return -1;
    }

    csv_ratio << "Dataset,Algorithm,Strategy,CompressionRatio\n";
    csv_ratio << fixed << setprecision(6);
    csv_residual << "Dataset,Strategy,BinCenter,BinCount\n";
    csv_residual << fixed << setprecision(6);

    for (const auto& ds : datasets) {
        string train_file = base_dir + ds + "_train.fvecs";
        size_t N = 0, d = 0;
        double* data = load_fvecs_as_double(train_file, N, d);
        if (!data) {
            cerr << "Skipping " << ds << " (failed to load)" << endl;
            continue;
        }
        cout << "\n=== Dataset: " << ds << "  N=" << N << "  d=" << d << " ===" << endl;

        // -----------------------------------------------------------------
        // 1) Intra-Vector residuals (independent of codec algorithm)
        // -----------------------------------------------------------------
        {
            vector<double> residuals = compute_residuals_intra(data, N, d);
            Histogram hist = build_histogram(residuals, hist_bins);
            for (size_t b = 0; b < hist.bin_centers.size(); ++b) {
                csv_residual << ds << ",IntraVector,"
                             << hist.bin_centers[b] << ","
                             << hist.counts[b] << "\n";
            }
            cout << "  IntraVector residuals: " << residuals.size()
                 << " values, range [" << hist.bin_centers.front() - (hist.bin_centers[1]-hist.bin_centers[0])/2
                 << ", " << hist.bin_centers.back() + (hist.bin_centers[1]-hist.bin_centers[0])/2
                 << "]" << endl;
        }

        // 2) ID-Order residuals (independent of codec algorithm)
        {
            vector<double> residuals = compute_residuals_id_order(data, N, d);
            Histogram hist = build_histogram(residuals, hist_bins);
            for (size_t b = 0; b < hist.bin_centers.size(); ++b) {
                csv_residual << ds << ",IDOrder,"
                             << hist.bin_centers[b] << ","
                             << hist.counts[b] << "\n";
            }
            cout << "  IDOrder residuals: " << residuals.size() << " values" << endl;
        }

        // -----------------------------------------------------------------
        // Compression ratios for Intra-Vector and ID-Order
        // -----------------------------------------------------------------
        for (const auto& algo : algorithms) {
            if (algo == "DeXOR") {
                double r = encode_intra_vector<codecs::DeXORCodecPolicy>(data, N, d);
                csv_ratio << ds << ",DeXOR,IntraVector," << r << "\n";
                cout << "  DeXOR IntraVector ratio=" << r << endl;

                r = encode_cross_vector_id_order<codecs::DeXORCodecPolicy>(data, N, d);
                csv_ratio << ds << ",DeXOR,IDOrder," << r << "\n";
                cout << "  DeXOR IDOrder ratio=" << r << endl;
            } else if (algo == "Gorilla") {
                double r = encode_intra_vector<codecs::GorillaCodecPolicy>(data, N, d);
                csv_ratio << ds << ",Gorilla,IntraVector," << r << "\n";
                cout << "  Gorilla IntraVector ratio=" << r << endl;

                r = encode_cross_vector_id_order<codecs::GorillaCodecPolicy>(data, N, d);
                csv_ratio << ds << ",Gorilla,IDOrder," << r << "\n";
                cout << "  Gorilla IDOrder ratio=" << r << endl;
            } else if (algo == "Elf") {
                double r = encode_intra_vector<codecs::ElfCodecPolicy>(data, N, d);
                csv_ratio << ds << ",Elf,IntraVector," << r << "\n";
                cout << "  Elf IntraVector ratio=" << r << endl;

                r = encode_cross_vector_id_order<codecs::ElfCodecPolicy>(data, N, d);
                csv_ratio << ds << ",Elf,IDOrder," << r << "\n";
                cout << "  Elf IDOrder ratio=" << r << endl;
            } else if (algo == "Camel") {
                double r = encode_intra_vector<codecs::CamelCodecPolicy>(data, N, d);
                csv_ratio << ds << ",Camel,IntraVector," << r << "\n";
                cout << "  Camel IntraVector ratio=" << r << endl;

                r = encode_cross_vector_id_order<codecs::CamelCodecPolicy>(data, N, d);
                csv_ratio << ds << ",Camel,IDOrder," << r << "\n";
                cout << "  Camel IDOrder ratio=" << r << endl;
            }
        }

        // -----------------------------------------------------------------
        // 3) Graph-Guided compression ratio + residuals
        // -----------------------------------------------------------------
        vector<tableint> graph_prenodes;
        for (const auto& algo : algorithms) {
            double ratio = 0.0;
            vector<tableint> pn;

            if (algo == "DeXOR") {
                tie(ratio, pn) = encode_graph_guided<codecs::DeXORCodecPolicy>(
                    data, N, d, M, ef_construction, chain_max, ds, algo);
            } else if (algo == "Gorilla") {
                tie(ratio, pn) = encode_graph_guided<codecs::GorillaCodecPolicy>(
                    data, N, d, M, ef_construction, chain_max, ds, algo);
            } else if (algo == "Elf") {
                tie(ratio, pn) = encode_graph_guided<codecs::ElfCodecPolicy>(
                    data, N, d, M, ef_construction, chain_max, ds, algo);
            } else if (algo == "Camel") {
                tie(ratio, pn) = encode_graph_guided<codecs::CamelCodecPolicy>(
                    data, N, d, M, ef_construction, chain_max, ds, algo);
            }

            csv_ratio << ds << "," << algo << ",GraphGuided," << ratio << "\n";
            cout << "  " << algo << " GraphGuided ratio=" << ratio << endl;

            // Save prenodes from the first algorithm (they differ slightly per codec
            // due to random tie-breaking; use DeXOR's for residual analysis)
            if (algo == "DeXOR" && !pn.empty()) {
                graph_prenodes = move(pn);
            }
        }

        // Graph-Guided residuals (use prenodes from DeXOR build as representative)
        if (!graph_prenodes.empty()) {
            vector<double> residuals = compute_residuals_graph(data, N, d, graph_prenodes);
            Histogram hist = build_histogram(residuals, hist_bins);
            for (size_t b = 0; b < hist.bin_centers.size(); ++b) {
                csv_residual << ds << ",GraphGuided,"
                             << hist.bin_centers[b] << ","
                             << hist.counts[b] << "\n";
            }
            cout << "  GraphGuided residuals: " << residuals.size() << " values" << endl;
        }

        delete[] data;
    }

    csv_ratio.close();
    csv_residual.close();

    cout << "\n=== Done ===" << endl;
    cout << "Compression ratios  -> " << out_ratio_csv << endl;
    cout << "Residual histograms -> " << out_residual_csv << endl;
    return 0;
}
