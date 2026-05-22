#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <queue>
#include <unordered_set>
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

double* load_fvecs_as_double(const string& filename, size_t& num_vectors, size_t& dim) {
    ifstream input(filename, ios::binary);
    if (!input) {
        cerr << "Cannot open " << filename << endl;
        return nullptr;
    }

    int32_t d;
    input.read((char*)&d, 4);
    dim = static_cast<size_t>(d);

    input.seekg(0, ios::end);
    size_t file_size = input.tellg();
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

unsigned int* load_ivecs(const string& filename, size_t& num_vectors, size_t& dim) {
    ifstream input(filename, ios::binary);
    if (!input) {
        cerr << "Cannot open " << filename << endl;
        return nullptr;
    }

    int32_t d;
    input.read((char*)&d, 4);
    dim = static_cast<size_t>(d);

    input.seekg(0, ios::end);
    size_t file_size = input.tellg();
    num_vectors = file_size / (4 + dim * 4);

    unsigned int* data = new unsigned int[num_vectors * dim];
    input.seekg(0, ios::beg);

    for (size_t i = 0; i < num_vectors; ++i) {
        input.read((char*)&d, 4);
        input.read((char*)(data + i * dim), dim * 4);
    }

    return data;
}

struct SearchMetrics {
    double qps = 0.0;
    double mean_latency_us = 0.0;
    double p99_latency_us = 0.0;
    double recall_at_1 = 0.0;
    double avg_replay_depth = 0.0;
};

template <typename CodecPolicy>
SearchMetrics measure_search(
    HierarchicalNSWCABFRAMEWORK<double, CodecPolicy>* index,
    const string& query_file,
    const string& gt_file,
    size_t dim,
    size_t ef,
    int search_threads) {
    SearchMetrics metrics;
    omp_set_num_threads(search_threads);

    size_t qsize = 0, qdim = 0;
    double* queries = load_fvecs_as_double(query_file, qsize, qdim);
    if (!queries) return metrics;

    size_t gt_num = 0, gt_dim = 0;
    unsigned int* groundtruth = load_ivecs(gt_file, gt_num, gt_dim);
    if (!groundtruth) {
        delete[] queries;
        return metrics;
    }

    if (qdim != dim || qsize != gt_num || gt_dim == 0) {
        cerr << "  [SKIP] Query/ground-truth metadata mismatch." << endl;
        delete[] queries;
        delete[] groundtruth;
        return metrics;
    }

    index->setEf(ef);
    index->setProfilingMetrics(true);
    index->resetProfilingMetrics();

    size_t correct = 0;
    vector<double> latencies;
    latencies.reserve(qsize);

    auto total_start = chrono::steady_clock::now();
    for (size_t i = 0; i < qsize; ++i) {
        auto q_start = chrono::steady_clock::now();
        auto result = index->searchKnn(queries + i * qdim, 1);
        auto q_end = chrono::steady_clock::now();
        latencies.push_back(static_cast<double>(
            chrono::duration_cast<chrono::microseconds>(q_end - q_start).count()));

        if (!result.empty() && result.top().second == static_cast<labeltype>(groundtruth[i * gt_dim])) {
            ++correct;
        }
    }
    auto total_end = chrono::steady_clock::now();

    double total_us = static_cast<double>(
        chrono::duration_cast<chrono::microseconds>(total_end - total_start).count());
    metrics.qps = static_cast<double>(qsize) / (total_us / 1e6);
    metrics.recall_at_1 = static_cast<double>(correct) / static_cast<double>(qsize);
    metrics.mean_latency_us = accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();

    sort(latencies.begin(), latencies.end());
    size_t p99_idx = min(latencies.size() - 1, static_cast<size_t>(ceil(latencies.size() * 0.99)) - 1);
    metrics.p99_latency_us = latencies[p99_idx];

    long calls = index->getOriginalDataCallCount();
    metrics.avg_replay_depth = calls > 0
        ? static_cast<double>(index->getOriginalDataBacktrackHops()) / static_cast<double>(calls)
        : 0.0;

    delete[] queries;
    delete[] groundtruth;
    return metrics;
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
                 size_t search_ef,
                 int search_threads,
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

    SearchMetrics search = measure_search(
        index,
        base_dir + dataset + "_test.fvecs",
        base_dir + dataset + "_neighbors.ivecs",
        dim,
        search_ef,
        search_threads);

    csv << dataset << ","
        << algo << ","
        << chain_max_length << ","
        << fixed << setprecision(6) << ratio << ","
        << setprecision(3) << avg_us << ","
        << p50_us << ","
        << p95_us << ","
        << p99_us << ","
        << actual_samples << ","
        << search_ef << ","
        << fixed << setprecision(2) << search.qps << ","
        << search.mean_latency_us << ","
        << search.p99_latency_us << ","
        << fixed << setprecision(6) << search.recall_at_1 << ","
        << search.avg_replay_depth << "\n";

    cout << "  " << algo << " chain_max=";
    if (chain_max_length < 0) cout << "unlim";
    else cout << chain_max_length;
    cout << " ratio=" << setprecision(4) << ratio
         << " avg=" << avg_us << "us"
         << " p50=" << p50_us << "us"
         << " p95=" << p95_us << "us"
         << " p99=" << p99_us << "us"
         << " qps=" << fixed << setprecision(2) << search.qps
         << " mean_search=" << search.mean_latency_us << "us"
         << " p99_search=" << search.p99_latency_us << "us"
         << " recall@1=" << setprecision(4) << search.recall_at_1
         << " replay_depth=" << search.avg_replay_depth
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
                     size_t search_ef,
                     int search_threads,
                     ofstream& csv) {
    if (algo == "DeXOR")
        measure_one<codecs::DeXORCodecPolicy>(dataset, base_dir, algo, chain_max_length, num_samples, search_ef, search_threads, csv);
    else if (algo == "Gorilla")
        measure_one<codecs::GorillaCodecPolicy>(dataset, base_dir, algo, chain_max_length, num_samples, search_ef, search_threads, csv);
    else if (algo == "Elf")
        measure_one<codecs::ElfCodecPolicy>(dataset, base_dir, algo, chain_max_length, num_samples, search_ef, search_threads, csv);
    else if (algo == "Camel")
        measure_one<codecs::CamelCodecPolicy>(dataset, base_dir, algo, chain_max_length, num_samples, search_ef, search_threads, csv);
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
    int search_threads = 1;
    size_t search_ef = 200;
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
        } else if (arg == "--search_threads" && i + 1 < argc) {
            search_threads = stoi(argv[++i]);
        } else if (arg == "--search_ef" && i + 1 < argc) {
            search_ef = static_cast<size_t>(stoul(argv[++i]));
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
        << "NumSamples,SearchEf,QPS,MeanLatencyUs,P99LatencyUs,Recall@1,AverageReplayDepth\n";

    for (const auto& ds : datasets) {
        cout << "\n=== Dataset: " << ds << " ===" << endl;
        for (const auto& algo : algorithms) {
            for (int chain_max : chain_max_list) {
                dispatch(ds, base_dir, algo, chain_max, num_samples, search_ef, search_threads, csv);
            }
        }
    }

    csv.close();
    cout << "\nResults saved to: " << out_csv << endl;
    return 0;
}
