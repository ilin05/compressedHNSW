#include <chrono>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <omp.h>

#include <faiss/IndexHNSW.h>
#include <faiss/IndexScalarQuantizer.h>

using namespace std;

class StopW {
    std::chrono::steady_clock::time_point time_begin;

 public:
    StopW() { reset(); }
    void reset() { time_begin = std::chrono::steady_clock::now(); }
    double getElapsedTimeMicro() const {
        auto time_end = std::chrono::steady_clock::now();
        return static_cast<double>(
                std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
    }
};

struct TestConfig {
    string method;
    int compression_ratio;
    int pq_m = 0;
    int code_bits = 0;
    faiss::ScalarQuantizer::QuantizerType sq_type = faiss::ScalarQuantizer::QT_8bit;
    bool supported = true;
    string message;
};

static float* load_fvecs(const string& filename, size_t& num_vectors, size_t& dim) {
    ifstream input(filename, ios::binary);
    if (!input) return nullptr;

    int32_t d;
    input.read((char*)&d, 4);
    dim = static_cast<size_t>(d);
    input.seekg(0, ios::end);
    size_t file_size = static_cast<size_t>(input.tellg());
    num_vectors = file_size / (4 + dim * 4);

    float* data = new float[num_vectors * dim];
    input.seekg(0, ios::beg);
    for (size_t i = 0; i < num_vectors; ++i) {
        input.read((char*)&d, 4);
        input.read((char*)(data + i * dim), dim * 4);
    }
    return data;
}

static unsigned int* load_ivecs(const string& filename, size_t& num_vectors, size_t& dim) {
    ifstream input(filename, ios::binary);
    if (!input) return nullptr;

    int32_t d;
    input.read((char*)&d, 4);
    dim = static_cast<size_t>(d);
    input.seekg(0, ios::end);
    size_t file_size = static_cast<size_t>(input.tellg());
    num_vectors = file_size / (4 + dim * 4);

    unsigned int* data = new unsigned int[num_vectors * dim];
    input.seekg(0, ios::beg);
    for (size_t i = 0; i < num_vectors; ++i) {
        input.read((char*)&d, 4);
        input.read((char*)(data + i * dim), dim * 4);
    }
    return data;
}

static string clean_csv_field(string s) {
    for (char& c : s) {
        if (c == ',' || c == '\n' || c == '\r') c = ' ';
    }
    return s;
}

static double compute_recall(
        size_t qn,
        size_t gt_k,
        const unsigned int* gt,
        const vector<faiss::idx_t>& ids,
        size_t k) {
    const size_t effective_k = std::min(k, gt_k);
    size_t correct = 0;
    for (size_t i = 0; i < qn; ++i) {
        unordered_set<faiss::idx_t> gt_set;
        for (size_t j = 0; j < effective_k; ++j) {
            gt_set.insert(static_cast<faiss::idx_t>(gt[i * gt_k + j]));
        }
        for (size_t j = 0; j < k; ++j) {
            if (gt_set.find(ids[i * k + j]) != gt_set.end()) {
                ++correct;
            }
        }
    }
    return static_cast<double>(correct) / static_cast<double>(qn * k);
}

static unique_ptr<faiss::Index> build_index_for_config(
        const TestConfig& cfg,
        int dim,
        int hnsw_m,
        int ef_construction) {
    unique_ptr<faiss::Index> index;
    if (cfg.method == "HNSW") {
        index = make_unique<faiss::IndexHNSWFlat>(dim, hnsw_m, faiss::METRIC_L2);
    } else if (cfg.method == "SQ") {
        index = make_unique<faiss::IndexHNSWSQ>(dim, cfg.sq_type, hnsw_m, faiss::METRIC_L2);
    } else if (cfg.method == "PQ") {
        index = make_unique<faiss::IndexHNSWPQ>(dim, cfg.pq_m, hnsw_m, cfg.code_bits, faiss::METRIC_L2);
    }

    auto* hnsw = dynamic_cast<faiss::IndexHNSW*>(index.get());
    if (hnsw) {
        hnsw->hnsw.efConstruction = ef_construction;
    }
    return index;
}

static vector<TestConfig> make_configs_for_dim(
        size_t dim,
        const vector<string>& methods,
        const vector<int>& ratios) {
    vector<TestConfig> configs;
    for (const string& method : methods) {
        for (int ratio : ratios) {
            if (ratio == 1) {
                configs.push_back({"HNSW", 1, 0, 32});
                continue;
            }
            if (method == "HNSW") continue;

            if (method == "SQ") {
                if (ratio == 2) {
                    configs.push_back({"SQ", ratio, 0, 16, faiss::ScalarQuantizer::QT_fp16});
                } else if (ratio == 4) {
                    configs.push_back({"SQ", ratio, 0, 8, faiss::ScalarQuantizer::QT_8bit});
                } else if (ratio == 8) {
                    configs.push_back({"SQ", ratio, 0, 4, faiss::ScalarQuantizer::QT_4bit});
                } else {
                    configs.push_back({"SQ", ratio, 0, 0, faiss::ScalarQuantizer::QT_8bit, false,
                                       "FAISS scalar quantizer has no 2-bit SQ type for 16x compression"});
                }
            } else if (method == "PQ") {
                int code_bits = 0;
                if (ratio == 2) code_bits = 16;
                else if (ratio == 4) code_bits = 8;
                else if (ratio == 8) code_bits = 4;
                else if (ratio == 16) code_bits = 2;

                if (code_bits == 0) {
                    configs.push_back({"PQ", ratio, 0, 0, faiss::ScalarQuantizer::QT_8bit, false,
                                       "unsupported PQ compression ratio"});
                } else {
                    configs.push_back({"PQ", ratio, static_cast<int>(dim), code_bits});
                }
            }
        }
    }

    vector<TestConfig> dedup;
    unordered_set<string> seen;
    for (const auto& cfg : configs) {
        string key = cfg.method + ":" + to_string(cfg.compression_ratio);
        if (seen.insert(key).second) dedup.push_back(cfg);
    }
    return dedup;
}

static void write_failure_row(
        ofstream& csv,
        const string& dataset,
        const TestConfig& cfg,
        size_t k,
        const string& message) {
    csv << dataset << ',' << cfg.method << ',' << cfg.compression_ratio << ','
        << cfg.pq_m << ',' << cfg.code_bits << ',' << k << ",,"
        << ",,,,,,,failed," << clean_csv_field(message) << "\n";
}

int main(int argc, char** argv) {
    string base_dir = "../datasets/hdf5files/";
    vector<string> datasets = {
            "fashion-mnist-784-euclidean",
            "gist-960-euclidean",
            "mnist-784-euclidean",
            "sift-128-euclidean"};
    vector<string> methods = {"SQ", "PQ"};
    vector<int> ratios = {1, 2, 4, 8, 16};
    vector<size_t> efs = {10, 20, 40, 80, 120, 200, 400, 600, 800, 1000};
    size_t k = 1;
    int num_rounds = 1;
    int build_threads = 32;
    int search_threads = 1;
    int hnsw_m = 16;
    int ef_construction = 200;
    string output_csv = "quantization_recall_compression_ratio_results.csv";

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--base_dir" && i + 1 < argc) {
            base_dir = argv[++i];
            if (!base_dir.empty() && base_dir.back() != '/' && base_dir.back() != '\\') base_dir += "/";
        } else if (arg == "--dataset" && i + 1 < argc) {
            datasets.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') datasets.push_back(argv[++i]);
        } else if (arg == "--methods" && i + 1 < argc) {
            methods.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') methods.push_back(argv[++i]);
        } else if (arg == "--compression_ratios" && i + 1 < argc) {
            ratios.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') ratios.push_back(stoi(argv[++i]));
        } else if (arg == "--efs" && i + 1 < argc) {
            efs.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') efs.push_back(static_cast<size_t>(stoull(argv[++i])));
        } else if (arg == "--k" && i + 1 < argc) {
            k = static_cast<size_t>(stoull(argv[++i]));
        } else if (arg == "--num_rounds" && i + 1 < argc) {
            num_rounds = stoi(argv[++i]);
        } else if (arg == "--build_threads" && i + 1 < argc) {
            build_threads = stoi(argv[++i]);
        } else if (arg == "--search_threads" && i + 1 < argc) {
            search_threads = stoi(argv[++i]);
        } else if (arg == "--M" && i + 1 < argc) {
            hnsw_m = stoi(argv[++i]);
        } else if (arg == "--ef_construction" && i + 1 < argc) {
            ef_construction = stoi(argv[++i]);
        } else if (arg == "--output_csv" && i + 1 < argc) {
            output_csv = argv[++i];
        }
    }

    ofstream csv(output_csv);
    csv << "Dataset,Method,CompressionRatio,PQ_M,CodeBits,K,ef,Recall,TimePerQuery(ms),Latency(us),QPS,TrainTime(s),AddTime(s),BuildTime(s),Status,Message\n";

    for (const string& ds : datasets) {
        string train_file = base_dir + ds + "_train.fvecs";
        string query_file = base_dir + ds + "_test.fvecs";
        string gt_file = base_dir + ds + "_neighbors.ivecs";

        size_t n = 0, dim = 0, qn = 0, qdim = 0, gt_n = 0, gt_k = 0;
        unique_ptr<float[]> train(load_fvecs(train_file, n, dim));
        unique_ptr<float[]> queries(load_fvecs(query_file, qn, qdim));
        unique_ptr<unsigned int[]> gt(load_ivecs(gt_file, gt_n, gt_k));

        if (!train || !queries || !gt || dim != qdim || qn != gt_n) {
            cerr << "Skip dataset due to load failure or dimension mismatch: " << ds << endl;
            continue;
        }

        cout << "\n=== Dataset: " << ds << " n=" << n << " dim=" << dim << " q=" << qn << " ===" << endl;
        vector<TestConfig> configs = make_configs_for_dim(dim, methods, ratios);

        for (const TestConfig& cfg : configs) {
            if (!cfg.supported) {
                cout << "Skip " << cfg.method << " " << cfg.compression_ratio << "x: " << cfg.message << endl;
                write_failure_row(csv, ds, cfg, k, cfg.message);
                continue;
            }

            cout << "Build " << cfg.method << " ratio=" << cfg.compression_ratio
                 << " pq_m=" << cfg.pq_m << " bits=" << cfg.code_bits << endl;

            unique_ptr<faiss::Index> index;
            double train_s = 0.0;
            double add_s = 0.0;
            try {
                index = build_index_for_config(cfg, static_cast<int>(dim), hnsw_m, ef_construction);

                omp_set_num_threads(build_threads);
                StopW timer;
                index->train(static_cast<faiss::idx_t>(n), train.get());
                train_s = timer.getElapsedTimeMicro() / 1e6;

                timer.reset();
                index->add(static_cast<faiss::idx_t>(n), train.get());
                add_s = timer.getElapsedTimeMicro() / 1e6;
            } catch (const exception& e) {
                cerr << "Failed to build " << cfg.method << " " << cfg.compression_ratio
                     << "x on " << ds << ": " << e.what() << endl;
                write_failure_row(csv, ds, cfg, k, e.what());
                continue;
            }

            vector<faiss::idx_t> ids(qn * k);
            vector<float> distances(qn * k);
            omp_set_num_threads(search_threads);

            for (size_t ef : efs) {
                if (auto* hnsw = dynamic_cast<faiss::IndexHNSW*>(index.get())) {
                    hnsw->hnsw.efSearch = static_cast<int>(ef);
                }

                double total_us = 0.0;
                double recall_sum = 0.0;
                for (int round = 0; round < num_rounds; ++round) {
                    StopW timer;
                    index->search(static_cast<faiss::idx_t>(qn), queries.get(), static_cast<faiss::idx_t>(k),
                                  distances.data(), ids.data());
                    double elapsed_us = timer.getElapsedTimeMicro();
                    total_us += elapsed_us;
                    recall_sum += compute_recall(qn, gt_k, gt.get(), ids, k);
                }

                double avg_us = total_us / static_cast<double>(num_rounds);
                double recall = recall_sum / static_cast<double>(num_rounds);
                double latency_us = avg_us / static_cast<double>(qn);
                double latency_ms = latency_us / 1000.0;
                double qps = static_cast<double>(qn) * 1e6 / avg_us;
                double build_s = train_s + add_s;

                cout << cfg.method << " " << cfg.compression_ratio << "x ef=" << ef
                     << " recall=" << fixed << setprecision(6) << recall
                     << " latency_us=" << latency_us << " qps=" << qps << endl;

                csv << ds << ',' << cfg.method << ',' << cfg.compression_ratio << ','
                    << cfg.pq_m << ',' << cfg.code_bits << ',' << k << ',' << ef << ','
                    << fixed << setprecision(6) << recall << ',' << latency_ms << ','
                    << latency_us << ',' << qps << ',' << train_s << ',' << add_s << ','
                    << build_s << ",ok,\n";
            }
        }
    }

    csv.close();
    cout << "\nSaved CSV: " << output_csv << endl;
    return 0;
}
