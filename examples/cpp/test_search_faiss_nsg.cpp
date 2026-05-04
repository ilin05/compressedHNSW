#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>
#include <omp.h>

#include <faiss/IndexNSG.h>
#include <faiss/index_io.h>

#include "../../examples/data_processor/data_loader.h"
#include "bigann_test_utils.h"

using namespace std;

class StopW {
    std::chrono::steady_clock::time_point time_begin;
 public:
    StopW() { time_begin = std::chrono::steady_clock::now(); }
    double getElapsedTimeMicro() const {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return static_cast<double>(
                std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin)
                        .count());
    }
};

static float* load_fvecs(const string& filename, size_t& num_vectors, size_t& dim) {
    ifstream input(filename, ios::binary);
    if (!input) {
        throw runtime_error("Cannot open file " + filename);
    }
    int32_t d = 0;
    input.read(reinterpret_cast<char*>(&d), 4);
    dim = static_cast<size_t>(d);
    input.seekg(0, ios::end);
    size_t file_size = static_cast<size_t>(input.tellg());
    num_vectors = file_size / (4 + dim * 4);
    float* data = new float[num_vectors * dim];
    input.seekg(0, ios::beg);
    for (size_t i = 0; i < num_vectors; ++i) {
        input.read(reinterpret_cast<char*>(&d), 4);
        input.read(reinterpret_cast<char*>(data + i * dim), dim * 4);
    }
    return data;
}

static unsigned int* load_ivecs(const string& filename, size_t& num_vectors, size_t& dim) {
    ifstream input(filename, ios::binary);
    if (!input) return nullptr;
    int32_t d = 0;
    input.read(reinterpret_cast<char*>(&d), 4);
    dim = static_cast<size_t>(d);
    input.seekg(0, ios::end);
    size_t file_size = static_cast<size_t>(input.tellg());
    num_vectors = file_size / (4 + dim * 4);
    unsigned int* data = new unsigned int[num_vectors * dim];
    input.seekg(0, ios::beg);
    for (size_t i = 0; i < num_vectors; ++i) {
        input.read(reinterpret_cast<char*>(&d), 4);
        input.read(reinterpret_cast<char*>(data + i * dim), dim * 4);
    }
    return data;
}

static vector<float> to_float_buffer(const double* src, size_t n) {
    vector<float> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(src[i]);
    return out;
}

static bool is_bigann_dataset_token(const string& token, size_t& subset_million) {
    return bigann_test_utils::parse_subset_token(token, subset_million);
}

int main(int argc, char** argv) {
    omp_set_num_threads(1);

    string normal_base_dir = "../datasets/hdf5files/";
    string bigann_query_file = "../bigann/bigann_query.bvecs";
    string bigann_gt_dir = "../bigann/gnd";

    vector<string> datasets = {
            "fashion-mnist-784-euclidean",
            "gist-960-euclidean",
            "mnist-784-euclidean",
            "sift-128-euclidean"};

    size_t qsize = 10000;
    // evaluate Recall@1 and Recall@10; ensure GT is loaded with top-10
    size_t gt_k_to_load = 10;
    int R = 32;
    int k = 1; // default to recall@1; can be set to 10 for recall@10

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--dataset" && i + 1 < argc) {
            datasets.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') datasets.push_back(argv[++i]);
        } else if (arg == "--qsize" && i + 1 < argc) {
            qsize = static_cast<size_t>(stoull(argv[++i]));
        } else if (arg == "--k" && i + 1 < argc) {
            k = static_cast<size_t>(stoull(argv[++i]));
        } else if (arg == "--normal_base_dir" && i + 1 < argc) {
            normal_base_dir = argv[++i]; if (normal_base_dir.back() != '/' && normal_base_dir.back() != '\\') normal_base_dir += "/";
        } else if (arg == "--bigann_query_file" && i + 1 < argc) {
            bigann_query_file = argv[++i];
        } else if (arg == "--bigann_gt_dir" && i + 1 < argc) {
            bigann_gt_dir = argv[++i];
        } else if (arg == "--R" && i + 1 < argc) {
            R = stoi(argv[++i]);
        }
    }

    vector<float> bigann_queries;
    int qdim_bigann = 0;
    try {
        double* qraw = data_loader::loadBvecsChunk(bigann_query_file, 0, qsize, qdim_bigann);
        bigann_queries = to_float_buffer(qraw, qsize * static_cast<size_t>(qdim_bigann));
        delete[] qraw;
    } catch (const exception& e) {
        cerr << "Warning: cannot preload bigann queries: " << e.what() << endl;
    }

    ofstream csv("faiss_nsg_search_results.csv");
    csv << "Dataset,Type,SearchL,K,Recall,TimePerQuery(ms),IndexSizeKB\n";

    for (const auto& ds : datasets) {
        size_t subset_m = 0;
        const bool is_bigann = is_bigann_dataset_token(ds, subset_m);

        float* queries = nullptr;
        size_t qn = 0;
        size_t qdim = 0;
        unsigned int* gt_dense = nullptr;
        size_t gt_dim_normal = 0;
        vector<vector<unsigned int>> gt_rows;

        if (is_bigann) {
            if (bigann_queries.empty()) { cerr << "Skip " << ds << " because bigann queries are unavailable." << endl; continue; }
            queries = bigann_queries.data();
            qn = qsize;
            qdim = static_cast<size_t>(qdim_bigann);
            const string gt_file = bigann_gt_dir + "/idx_" + to_string(subset_m) + "M.ivecs";
            if (!bigann_test_utils::load_gt_ivecs_topk(gt_file, qn, gt_k_to_load, gt_rows)) { cerr << "Skip " << ds << " due to GT load failure." << endl; continue; }
        } else {
            const string query_file = normal_base_dir + ds + "_test.fvecs";
            const string gt_file = normal_base_dir + ds + "_neighbors.ivecs";
            try {
                queries = load_fvecs(query_file, qn, qdim);
                size_t gt_n = 0, gt_d = 0;
                gt_dense = load_ivecs(gt_file, gt_n, gt_d);
                gt_dim_normal = gt_d;
                if (!gt_dense) { cerr << "Skip " << ds << " due to GT file load failure." << endl; delete[] queries; continue; }
            } catch (const exception& e) {
                cerr << "Skip " << ds << " due to load error: " << e.what() << endl;
                if (queries) delete[] queries;
                continue;
            }
        }

        string index_path = ds + "_NSG_R" + to_string(R) + ".bin";
        faiss::Index* nsg = nullptr;
        try {
            nsg = faiss::read_index(index_path.c_str());
        } catch (const exception& e) {
            cerr << "Cannot load NSG index " << index_path << ": " << e.what() << endl;
            if (!is_bigann) { delete[] queries; if (gt_dense) delete[] gt_dense; }
            continue;
        }

        size_t nvecs = nsg->ntotal;
        double index_kb = 0.0;
        try {
            std::ifstream in(index_path, ios::binary | ios::ate);
            if (in) index_kb = static_cast<double>(in.tellg()) / 1024.0;
        } catch (...) {}

        for (size_t kk : {1u, 10u}) {
            for (int search_L = static_cast<int>(kk); search_L <= 20; search_L += 1) {
                faiss::IndexNSG* nsg_ptr = dynamic_cast<faiss::IndexNSG*>(nsg);
                if (nsg_ptr) nsg_ptr->setSearchL(search_L);

                vector<faiss::idx_t> I(qn * kk);
                vector<float> D(qn * kk);

                StopW timer;
                nsg->search(static_cast<faiss::idx_t>(qn), queries, static_cast<faiss::idx_t>(kk), D.data(), I.data());
                double time_ms = timer.getElapsedTimeMicro() / 1000.0 / static_cast<double>(qn);
                double qps = 0.0;
                if (time_ms > 0.0) qps = 1000.0 / time_ms;

                // compute recall
                size_t correct = 0;
                for (size_t i = 0; i < qn; ++i) {
                    unordered_set<unsigned int> gset;
                    if (is_bigann) {
                        gset = unordered_set<unsigned int>(gt_rows[i].begin(), gt_rows[i].end());
                    } else {
                        const size_t topk = min(kk, gt_dim_normal);
                        gset = unordered_set<unsigned int>(gt_dense + i * gt_dim_normal, gt_dense + i * gt_dim_normal + topk);
                    }
                    for (size_t j = 0; j < kk; ++j) {
                        faiss::idx_t id = I[i * kk + j];
                        if (id >= 0 && gset.find(static_cast<unsigned int>(id)) != gset.end()) ++correct;
                    }
                }
                double recall = static_cast<double>(correct) / static_cast<double>(qn * kk);

                cout << "Dataset=" << ds << " NSG search_L=" << search_L << " k=" << kk << " recall=" << fixed << setprecision(6) << recall
                     << " time(ms)=" << setprecision(3) << time_ms << " QPS=" << qps << " IndexKB=" << index_kb << endl;

                csv << ds << ',' << (is_bigann ? "bigann" : "normal") << ',' << search_L << ',' << kk << ',' << fixed << setprecision(6)
                    << recall << ',' << time_ms << ',' << index_kb << "\n";
            }
        }

        delete nsg;
        if (!is_bigann) { delete[] queries; if (gt_dense) delete[] gt_dense; }
    }

    csv.close();
    cout << "\nSaved CSV: faiss_nsg_search_results.csv" << endl;
    return 0;
}
