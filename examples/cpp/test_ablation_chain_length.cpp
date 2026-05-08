#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <omp.h>

#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/compressed_hnsw_framework.h"

using namespace std;
using namespace hnswlib;

// Read dimension from .fvecs header (first 4 bytes)
size_t read_fvecs_dim(const string& filename) {
    ifstream in(filename, ios::binary);
    if (!in) return 0;
    int32_t d = 0;
    in.read((char*)&d, 4);
    return static_cast<size_t>(d);
}

static string chain_tag(int chain_max_length) {
    if (chain_max_length < 0) return "chunlim";
    return "ch" + to_string(chain_max_length);
}

// ---------------------------------------------------------------------------
// Core measurement for one (dataset, algorithm, chain_max) combination
// ---------------------------------------------------------------------------
template <typename CodecPolicy>
void measure_one(const string& dataset,
                 const string& base_dir,
                 const string& algo,
                 int chain_max_length,
                 size_t num_samples,
                 ofstream& csv) {
    const string index_file = dataset + "_" + algo + "_" + chain_tag(chain_max_length) + "_pq.bin";

    ifstream test(index_file, ios::binary);
    if (!test) {
        cerr << "  [SKIP] Index not found: " << index_file << endl;
        return;
    }
    test.close();

    const string train_file = base_dir + dataset + "_train.fvecs";
    size_t dim = read_fvecs_dim(train_file);
    if (dim == 0) {
        cerr << "  [SKIP] Cannot read dimension from: " << train_file << endl;
        return;
    }

    L2SpaceDouble l2space(dim);
    auto* index = new HierarchicalNSWCABFRAMEWORK<double, CodecPolicy>(
        &l2space, index_file, true, 0);

    size_t N = index->getCurrentElementCount();
    size_t original_bytes = N * dim * sizeof(double);
    size_t compressed_bytes = index->getCompressedDataSize();
    double ratio = (compressed_bytes > 0)
        ? static_cast<double>(original_bytes) / static_cast<double>(compressed_bytes)
        : 0.0;

    size_t actual_samples = min(num_samples, N);
    vector<tableint> sample_ids(actual_samples);

    // Systematic sampling for coverage across the graph
    if (N <= actual_samples) {
        for (size_t i = 0; i < N; ++i) sample_ids[i] = static_cast<tableint>(i);
    } else {
        size_t step = N / actual_samples;
        for (size_t i = 0; i < actual_samples; ++i) {
            sample_ids[i] = static_cast<tableint>(i * step);
        }
        // Shuffle to avoid correlated timing effects
        mt19937 rng(42);
        shuffle(sample_ids.begin(), sample_ids.end(), rng);
    }

    vector<double> latencies;
    latencies.reserve(actual_samples);

    // Warmup: first call may include cold-cache overhead
    index->getOriginalDataByInternalId(0);

    for (size_t i = 0; i < actual_samples; ++i) {
        auto t0 = chrono::steady_clock::now();
        auto vec = index->getOriginalDataByInternalId(sample_ids[i]);
        (void)vec; // suppress unused warning
        auto t1 = chrono::steady_clock::now();
        double us = static_cast<double>(
            chrono::duration_cast<chrono::microseconds>(t1 - t0).count());
        latencies.push_back(us);
    }

    sort(latencies.begin(), latencies.end());

    auto percentile = [&](double p) -> double {
        size_t idx = static_cast<size_t>(latencies.size() * p);
        if (idx >= latencies.size()) idx = latencies.size() - 1;
        return latencies[idx];
    };

    double avg_us = accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
    double p50_us = percentile(0.50);
    double p95_us = percentile(0.95);
    double p99_us = percentile(0.99);

    csv << dataset << ","
        << algo << ","
        << chain_max_length << ","
        << fixed << setprecision(6) << ratio << ","
        << setprecision(3) << avg_us << ","
        << p50_us << ","
        << p95_us << ","
        << p99_us << ","
        << actual_samples << "\n";

    cout << "  " << algo << " chain_max=";
    if (chain_max_length < 0) cout << "unlim";
    else cout << chain_max_length;
    cout << " ratio=" << setprecision(4) << ratio
         << " avg=" << avg_us << "us"
         << " p50=" << p50_us << "us"
         << " p95=" << p95_us << "us"
         << " p99=" << p99_us << "us"
         << " n=" << actual_samples << endl;

    delete index;
}

// ---------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------
static void dispatch(const string& dataset,
                     const string& base_dir,
                     const string& algo,
                     int chain_max_length,
                     size_t num_samples,
                     ofstream& csv) {
    if (algo == "DeXOR")
        measure_one<codecs::DeXORCodecPolicy>(dataset, base_dir, algo, chain_max_length, num_samples, csv);
    else if (algo == "Gorilla")
        measure_one<codecs::GorillaCodecPolicy>(dataset, base_dir, algo, chain_max_length, num_samples, csv);
    else if (algo == "Elf")
        measure_one<codecs::ElfCodecPolicy>(dataset, base_dir, algo, chain_max_length, num_samples, csv);
    else if (algo == "Camel")
        measure_one<codecs::CamelCodecPolicy>(dataset, base_dir, algo, chain_max_length, num_samples, csv);
    else
        cerr << "Unknown algorithm: " << algo << endl;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    string base_dir = "../datasets/hdf5files/";
    string out_csv = "chain_length_ablation.csv";
    int threads = 32;
    size_t num_samples = 5000;

    vector<string> datasets = {
        "fashion-mnist-784-euclidean",
        "mnist-784-euclidean",
        "sift-128-euclidean",
        "gist-960-euclidean",
        "deep-image-96-angular"
    };

    vector<string> algorithms = {"DeXOR", "Gorilla", "Elf", "Camel"};
    vector<int> chain_max_list = {2, 5, -1};

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--base_dir" && i + 1 < argc) {
            base_dir = argv[++i];
        } else if (arg == "--output_csv" && i + 1 < argc) {
            out_csv = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            threads = stoi(argv[++i]);
        } else if (arg == "--num_samples" && i + 1 < argc) {
            num_samples = static_cast<size_t>(stoul(argv[++i]));
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
        } else if (arg == "--chain_max" && i + 1 < argc) {
            chain_max_list.clear();
            while (i + 1 < argc && string(argv[i + 1]).rfind("--", 0) != 0) {
                chain_max_list.push_back(stoi(argv[++i]));
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

    ofstream csv(out_csv);
    if (!csv.is_open()) {
        cerr << "Failed to open output CSV: " << out_csv << endl;
        return -1;
    }

    csv << "Dataset,Algorithm,ChainMaxLength,CompressionRatio,"
        << "AvgDecompressUs,P50DecompressUs,P95DecompressUs,P99DecompressUs,"
        << "NumSamples\n";

    for (const auto& ds : datasets) {
        cout << "\n=== Dataset: " << ds << " ===" << endl;
        for (const auto& algo : algorithms) {
            for (int chain_max : chain_max_list) {
                dispatch(ds, base_dir, algo, chain_max, num_samples, csv);
            }
        }
    }

    csv.close();
    cout << "\nResults saved to: " << out_csv << endl;
    return 0;
}
