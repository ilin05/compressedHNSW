#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <unordered_set>
#include <iomanip>
#include <omp.h>

#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/hnswalg_simplified_alp_PQ.h"

#include <faiss/index_io.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexNSG.h>

using namespace std;
using namespace hnswlib;

class StopW {
    std::chrono::steady_clock::time_point time_begin;
 public:
    StopW() { time_begin = std::chrono::steady_clock::now(); }
    double getElapsedTimeMicro() const {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return static_cast<double>(
                std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
    }
    void reset() { time_begin = std::chrono::steady_clock::now(); }
};

static float* load_fvecs(const std::string& filename, size_t& num_vectors, size_t& dim) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) return nullptr;
    int32_t d;
    input.read((char*)&d, 4);
    dim = d;
    input.seekg(0, std::ios::end);
    size_t file_size = input.tellg();
    num_vectors = file_size / (4 + dim * 4);
    float* data = new float[num_vectors * dim];
    float* tmp = new float[dim];
    input.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num_vectors; ++i) {
        input.read((char*)&d, 4);
        input.read((char*)tmp, dim * 4);
        for (size_t j = 0; j < dim; ++j) data[i * dim + j] = tmp[j];
    }
    delete[] tmp;
    return data;
}

static unsigned int* load_ivecs(const std::string& filename, size_t& num_vectors, size_t& dim) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) return nullptr;
    int32_t d;
    input.read((char*)&d, 4);
    dim = d;
    input.seekg(0, std::ios::end);
    size_t file_size = input.tellg();
    num_vectors = file_size / (4 + dim * 4);
    unsigned int* data = new unsigned int[num_vectors * dim];
    input.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num_vectors; ++i) {
        input.read((char*)&d, 4);
        input.read((char*)(data + i * dim), dim * 4);
    }
    return data;
}

static size_t file_size_bytes(const std::string& filename) {
    std::ifstream input(filename, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("Cannot open file " + filename);
    }
    return static_cast<size_t>(input.tellg());
}

/**
 * 修改后的标准 Recall@K 计算函数
 * @param qn 查询数量
 * @param gt_k Ground Truth 文件中每一行包含的邻居总数 (通常为 100)
 * @param gt_rows Ground Truth 数据指针
 * @param I 检索出的结果 ID 向量
 * @param k 当前测试的 Recall@K 中的 K 值 (如 1 或 10)
 */
static double compute_recall_from_gt(size_t qn, size_t gt_k, unsigned int* gt_rows, const vector<faiss::idx_t>& I, size_t k) {
    // 基础安全检查：检索数不能超过 GT 文件提供的总数
    if (k > gt_k) {
        static bool warned = false;
        if (!warned) {
            cerr << "Warning: Requesting Recall@" << k << " but Ground Truth only has " << gt_k << " neighbors." << endl;
            warned = true;
        }
        k = gt_k;
    }

    size_t correct = 0;
    for (size_t qi = 0; qi < qn; ++qi) {
        // --- 关键修改：只将 GT 中的前 K 个邻居放入判定集 ---
        unordered_set<faiss::idx_t> g;
        for (size_t j = 0; j < k; ++j) {
            g.insert(static_cast<faiss::idx_t>(gt_rows[qi * gt_k + j]));
        }

        // 检查检索出的前 K 个结果
        for (size_t j = 0; j < k; ++j) {
            faiss::idx_t pred = I[qi * k + j];
            if (g.find(pred) != g.end()) {
                correct++;
            }
        }
    }
    // 返回平均召回率：总命中数 / (查询数 * k)
    return static_cast<double>(correct) / static_cast<double>(qn * k);
}

int main(int argc, char** argv) {
    omp_set_num_threads(1);

    string base_dir = "../datasets/hdf5files/";
    vector<string> datasets = {"fashion-mnist-784-euclidean","gist-960-euclidean","mnist-784-euclidean","sift-128-euclidean"};

    // parameter sweeps
    vector<int> hnsw_efs = {10,20,40,80,160,320};
    vector<int> ivf_nprobes = {1,2,4,8,16,32};
    vector<int> pq_ms = {1,2};
    vector<int> sq_nbits = {4,8};
    vector<float> tls_ratios = {0.1f, 0.2f, 0.3f};

    // Algorithm flags
    bool test_hnswalp = false;
    bool test_hnsw_hnswlib = false;
    bool test_hnsw_faiss = false;
    bool test_ivf = false;
    bool test_nsg = false;
    bool test_hnswpq = false;
    bool test_hnswsq = false;
    vector<string> algorithms;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--dataset" && i + 1 < argc) {
            datasets.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') datasets.push_back(argv[++i]);
        } else if (arg == "--base_dir" && i + 1 < argc) {
            base_dir = argv[++i]; if (base_dir.back() != '/' && base_dir.back() != '\\') base_dir += "/";
        } else if (arg == "--algorithm" && i + 1 < argc) {
            algorithms.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') algorithms.push_back(argv[++i]);
        }
    }

    // Parse algorithm flags
    if (algorithms.empty()) {
        // Default: test all algorithms
        test_hnswalp = test_hnsw_hnswlib = test_hnsw_faiss = test_ivf = test_nsg = test_hnswpq = test_hnswsq = true;
    } else {
        for (const auto& algo : algorithms) {
            if (algo == "HNSWALP") {test_hnswalp = true; tls_ratios.clear(); tls_ratios.push_back(0.0f);} // default to no TLS for "HNSWALP"
            else if (algo == "HNSWALP_0.1") { test_hnswalp = true; tls_ratios.push_back(0.1f); }
            else if (algo == "HNSWALP_0.2") { test_hnswalp = true; tls_ratios.push_back(0.2f); }
            else if (algo == "HNSWALP_0.3") { test_hnswalp = true; tls_ratios.push_back(0.3f); }
            else if (algo == "HNSWALP_0.5") { test_hnswalp = true; tls_ratios.push_back(0.5f); }
            else if (algo == "HNSW(hnswlib)") test_hnsw_hnswlib = true;
            else if (algo == "HNSW(faiss)") test_hnsw_faiss = true;
            else if (algo == "IVF(faiss)") test_ivf = true;
            else if (algo == "NSG(faiss)") test_nsg = true;
            else if (algo == "HNSWPQ1") { test_hnswpq = true; pq_ms.clear(); pq_ms.push_back(1); }
            else if (algo == "HNSWPQ2") { test_hnswpq = true; pq_ms.clear(); pq_ms.push_back(2); }
            else if (algo == "HNSWSQ4") { test_hnswsq = true; sq_nbits.clear(); sq_nbits.push_back(4); }
            else if (algo == "HNSWSQ8") { test_hnswsq = true; sq_nbits.clear(); sq_nbits.push_back(8); }
        }
    }

    ofstream csv("vq_recall_results.csv");
    csv << "Dataset,IndexType,Param,K,Recall,QPS,Latency(us),IndexSizeKB,VectorsPerKB,VQ\n";

    for (const auto& ds : datasets) {
        cout << "\n--- Dataset: " << ds << " ---" << endl;
        string query_file = base_dir + ds + "_test.fvecs";
        string gt_file = base_dir + ds + "_neighbors.ivecs";

        size_t qn = 0, qdim = 0;
        float* queries_f = load_fvecs(query_file, qn, qdim);
        if (!queries_f) { cerr << "Cannot load queries for " << ds << endl; continue; }

        size_t gt_n = 0, gt_k = 0;
        unsigned int* gt_rows = load_ivecs(gt_file, gt_n, gt_k);
        if (!gt_rows) { cerr << "Cannot load GT for " << ds << endl; delete[] queries_f; continue; }
        if (qn != gt_n) { cerr << "Query/GT size mismatch for " << ds << endl; delete[] queries_f; delete[] gt_rows; continue; }

        // --- Compressed HNSW (HNSWALP) ---
        if (test_hnswalp) {
            string idx_path = ds + "_train.fvecs_hnswalp_simplified_pq.bin";
            L2Space l2space(static_cast<int>(qdim));
            HierarchicalNSWALPSIMPLIFIEDPQ<float>* cidx = nullptr;
            try {
                cidx = new HierarchicalNSWALPSIMPLIFIEDPQ<float>(&l2space, idx_path, false);
            } catch (exception& e) { cerr << "Load compressed HNSW failed: " << e.what() << endl; }
            if (cidx) {
                for(float tls_ratio : tls_ratios) {
                    cidx->use_tls_ = (tls_ratio > 0.0f);
                    cidx->tls_ratio_ = tls_ratio;

                    size_t nvecs = cidx->getCurrentElementCount();
                    size_t index_size_bytes = cidx->getCompressedIndexSize();
                    double index_kb = static_cast<double>(index_size_bytes) / 1024.0;
                    double v_per_kb = static_cast<double>(nvecs) / index_kb;

                    // test k=1 and k=10 using the HNSW ef sweep (use ef values for search)
                    for (int k : {1,10}) {
                        for (int ef : hnsw_efs) {
                            cidx->setEf(ef);
                            StopW t0;
                            vector<faiss::idx_t> I(qn * k);
                            for (size_t qi = 0; qi < qn; ++qi) {
                                auto pq = cidx->searchKnn(queries_f + qi * qdim, k);
                                for (int j = k-1; j >= 0; --j) {
                                    if (pq.empty()) { I[qi*k + (k-1-j)] = -1; continue; }
                                    I[qi*k + (k-1-j)] = pq.top().second; pq.pop();
                                }
                            }
                            double elapsed_us = t0.getElapsedTimeMicro();
                            double qps = qn * 1e6 / elapsed_us;
                            double latency = elapsed_us / qn;
                            double recall = compute_recall_from_gt(qn, gt_k, gt_rows, I, k);
                            double vq = v_per_kb * qps;
                            csv << ds << ",HNSWALP" << tls_ratio <<",ef=" << ef << "," << k << "," << fixed << setprecision(6) << recall << "," << qps << "," << latency << "," << index_kb << "," << v_per_kb << "," << vq << "\n";
                            cout << "HNSWALP" << tls_ratio << " ef=" << ef << " k=" << k << " recall=" << recall << " QPS=" << qps << " Latency=" << latency << "us VQ=" << vq << endl;
                        }
                    }
                }
                delete cidx;
            }
        }

        // --- Original HNSW (hnswlib) ---
        if (test_hnsw_hnswlib) {
            string idx_path = ds + "_train.fvecs_hnsw_float.bin";
            L2Space l2space(static_cast<int>(qdim));
            HierarchicalNSW<float>* idx = nullptr;
            try { idx = new HierarchicalNSW<float>(&l2space, idx_path, false); } catch (exception& e) { cerr << "Load HNSW failed: " << e.what() << endl; }
            if (idx) {
                size_t nvecs = idx->getCurrentElementCount();
                size_t index_size_bytes = idx->indexFileSize();
                double index_kb = static_cast<double>(index_size_bytes) / 1024.0;
                double v_per_kb = static_cast<double>(nvecs) / index_kb;

                for (int k : {1,10}) {
                    for (int ef : hnsw_efs) {
                        idx->setEf(ef);
                        StopW t0;
                        vector<faiss::idx_t> I(qn * k);
                        for (size_t qi = 0; qi < qn; ++qi) {
                            auto pq = idx->searchKnn(queries_f + qi * qdim, k);
                            for (int j = k-1; j >= 0; --j) {
                                if (pq.empty()) { I[qi*k + (k-1-j)] = -1; continue; }
                                I[qi*k + (k-1-j)] = pq.top().second; pq.pop();
                            }
                        }
                        double elapsed_us = t0.getElapsedTimeMicro();
                        double qps = qn * 1e6 / elapsed_us;
                        double latency = elapsed_us / qn;
                        double recall = compute_recall_from_gt(qn, gt_k, gt_rows, I, k);
                        double vq = v_per_kb * qps;
                        csv << ds << ",HNSW,ef=" << ef << "," << k << "," << fixed << setprecision(6) << recall << "," << qps << "," << latency << "," << index_kb << "," << v_per_kb << "," << vq << "\n";
                        cout << "HNSW ef=" << ef << " k=" << k << " recall=" << recall << " QPS=" << qps << " Latency=" << latency << "us VQ=" << vq << endl;
                    }
                }

                delete idx;
            }
        }

        // --- Faiss HNSW baseline ---
        if (test_hnsw_faiss) {
            string idx_path = ds + "_train.fvecs_faiss_hnsw_M16_efConstruction200.bin";
            try {
                faiss::Index* base = faiss::read_index(idx_path.c_str());
                auto* hnsw = dynamic_cast<faiss::IndexHNSW*>(base);
                if (!hnsw) {
                    cerr << "Loaded index is not Faiss HNSW: " << idx_path << endl;
                    delete base;
                } else {
                    size_t nvecs = base->ntotal;
                    double index_kb = static_cast<double>(file_size_bytes(idx_path)) / 1024.0;
                    double v_per_kb = static_cast<double>(nvecs) / index_kb;

                    for (int k : {1,10}) {
                        for (int ef : hnsw_efs) {
                            hnsw->hnsw.efSearch = ef;
                            vector<faiss::idx_t> I(qn * k);
                            vector<float> D(qn * k);
                            StopW t0;
                            base->search(static_cast<faiss::idx_t>(qn), queries_f, k, D.data(), I.data());
                            double elapsed_us = t0.getElapsedTimeMicro();
                            double qps = qn * 1e6 / elapsed_us;
                            double latency = elapsed_us / qn;
                            double recall = compute_recall_from_gt(qn, gt_k, gt_rows, I, k);
                            double vq = v_per_kb * qps;
                            csv << ds << ",FaissHNSW,ef=" << ef << "," << k << "," << fixed << setprecision(6)
                                << recall << "," << qps << "," << latency << "," << index_kb << "," << v_per_kb << ","
                                << vq << "\n";
                            cout << "FaissHNSW ef=" << ef << " k=" << k << " recall=" << recall
                                 << " QPS=" << qps << " Latency=" << latency << "us VQ=" << vq << endl;
                        }
                    }

                    delete base;
                }
            } catch (const exception& e) {
                cerr << "Load Faiss HNSW failed: " << e.what() << endl;
            }
        }

        // --- Faiss IVF ---
        if (test_ivf) {
            string ivf_index_path = ds + "_IVFFlat_nlist1024.bin"; // common naming from build script; user can adjust
            try {
                faiss::Index* ivf = faiss::read_index(ivf_index_path.c_str());
                size_t nvecs = ivf->ntotal;
                double index_kb = static_cast<double>(file_size_bytes(ivf_index_path)) / 1024.0;
                double v_per_kb = static_cast<double>(nvecs) / index_kb;

                for (int k : {1,10}) {
                    for (int nprobe : ivf_nprobes) {
                        faiss::IndexIVF* iivf = dynamic_cast<faiss::IndexIVF*>(ivf);
                        if (iivf) iivf->nprobe = nprobe;
                        // prepare buffers
                        vector<faiss::idx_t> I(qn * k);
                        vector<float> D(qn * k);
                        StopW t0;
                        ivf->search(static_cast<faiss::idx_t>(qn), queries_f, k, D.data(), I.data());
                        double elapsed_us = t0.getElapsedTimeMicro();
                        double qps = qn * 1e6 / elapsed_us;
                        double latency = elapsed_us / qn;
                        double recall = compute_recall_from_gt(qn, gt_k, gt_rows, I, k);
                        double vq = v_per_kb * qps;
                        csv << ds << ",IVF,nprobe=" << nprobe << "," << k << "," << fixed << setprecision(6) << recall << "," << qps << "," << latency << "," << index_kb << "," << v_per_kb << "," << vq << "\n";
                        cout << "IVF nprobe=" << nprobe << " k=" << k << " recall=" << recall << " QPS=" << qps << " Latency=" << latency << "us VQ=" << vq << endl;
                    }
                }

                delete ivf;
            } catch (exception& e) {
                cerr << "Load IVF index failed: " << e.what() << endl;
            }
        }

        // --- Faiss NSG ---
        if (test_nsg) {
            string nsg_index_path = ds + "_NSG_R32.bin"; // adapt name as produced by build
            try {
                faiss::Index* nsg = faiss::read_index(nsg_index_path.c_str());
                size_t nvecs = nsg->ntotal;
                double index_kb = static_cast<double>(file_size_bytes(nsg_index_path)) / 1024.0;
                double v_per_kb = static_cast<double>(nvecs) / index_kb;

                // Faiss NSG typically uses default search; sweeping nprobe not applicable. We'll run single config for k=1,10
                for (int k : {1,10}) {
                    for(int search_L = k; search_L <= 20; search_L += 1){
                        faiss::IndexNSG* nsg_ptr = dynamic_cast<faiss::IndexNSG*>(nsg);
                        if (nsg_ptr) nsg_ptr->setSearchL(search_L);
                        // prepare buffers
                        vector<faiss::idx_t> I(qn * k);
                        vector<float> D(qn * k);
                        StopW t0;
                        nsg->search(static_cast<faiss::idx_t>(qn), queries_f, k, D.data(), I.data());
                        double elapsed_us = t0.getElapsedTimeMicro();
                        double qps = qn * 1e6 / elapsed_us;
                        double latency = elapsed_us / qn;
                        double recall = compute_recall_from_gt(qn, gt_k, gt_rows, I, k);
                        double vq = v_per_kb * qps;
                        csv << ds << ",NSG," << search_L << "," << k << "," << fixed << setprecision(6) << recall << "," << qps << "," << latency << "," << index_kb << "," << v_per_kb << "," << vq << "\n";
                        cout << "NSG search_L=" << search_L << " k=" << k << " recall=" << recall << " QPS=" << qps << " Latency=" << latency << "us VQ=" << vq << endl;
                    }
                }

                delete nsg;
            } catch (exception& e) {
                cerr << "Load NSG index failed: " << e.what() << endl;
            }
        }

        // --- Faiss HNSWPQ ---
        if (test_hnswpq) {
            for (int pq_m : pq_ms) {
                string current_algo_name = "HNSWPQ_" + std::to_string(pq_m);
                string idx_path = ds + "_train.fvecs_" + current_algo_name + ".bin";
                try {
                    faiss::Index* index = faiss::read_index(idx_path.c_str());
                    auto* real_index = dynamic_cast<faiss::IndexHNSW*>(index);
                    if (!index) {
                        cerr << "Cannot load HNSWPQ index " << idx_path << endl;
                        continue;
                    }

                    size_t nvecs = index->ntotal;
                    double index_kb = static_cast<double>(file_size_bytes(idx_path)) / 1024.0;
                    double v_per_kb = static_cast<double>(nvecs) / index_kb;

                    for (int k : {1, 10}) {
                        for (int ef : hnsw_efs) {
                            if (real_index) {
                                real_index->hnsw.efSearch = ef;
                            }
                            vector<faiss::idx_t> I(qn * k);
                            vector<float> D(qn * k);
                            StopW t0;
                            index->search(static_cast<faiss::idx_t>(qn), queries_f, k, D.data(), I.data());
                            double elapsed_us = t0.getElapsedTimeMicro();
                            double qps = qn * 1e6 / elapsed_us;
                            double latency = elapsed_us / qn;
                            double recall = compute_recall_from_gt(qn, gt_k, gt_rows, I, k);
                            double vq = v_per_kb * qps;
                            csv << ds << "," << current_algo_name << ",ef=" << ef << "," << k << "," << fixed << setprecision(6)
                                << recall << "," << qps << "," << latency << "," << index_kb << "," << v_per_kb << ","
                                << vq << "\n";
                            cout << current_algo_name << " ef=" << ef << " k=" << k << " recall=" << recall
                                 << " QPS=" << qps << " Latency=" << latency << "us VQ=" << vq << endl;
                        }
                    }

                    delete index;
                } catch (const exception& e) {
                    cerr << "Load HNSWPQ index failed: " << e.what() << endl;
                }
            }
        }

        // --- Faiss HNSWSQ ---
        if (test_hnswsq) {
            for (int nbits : sq_nbits) {
                string current_algo_name = "HNSWSQ_" + std::to_string(nbits);
                string idx_path = ds + "_train.fvecs_" + current_algo_name + ".bin";
                try {
                    faiss::Index* index = faiss::read_index(idx_path.c_str());
                    auto* real_index = dynamic_cast<faiss::IndexHNSW*>(index);
                    if (!index) {
                        cerr << "Cannot load HNSWSQ index " << idx_path << endl;
                        continue;
                    }

                    size_t nvecs = index->ntotal;
                    double index_kb = static_cast<double>(file_size_bytes(idx_path)) / 1024.0;
                    double v_per_kb = static_cast<double>(nvecs) / index_kb;

                    for (int k : {1, 10}) {
                        for (int ef : hnsw_efs) {
                            if (real_index) {
                                real_index->hnsw.efSearch = ef;
                            }
                            vector<faiss::idx_t> I(qn * k);
                            vector<float> D(qn * k);
                            StopW t0;
                            index->search(static_cast<faiss::idx_t>(qn), queries_f, k, D.data(), I.data());
                            double elapsed_us = t0.getElapsedTimeMicro();
                            double qps = qn * 1e6 / elapsed_us;
                            double latency = elapsed_us / qn;
                            double recall = compute_recall_from_gt(qn, gt_k, gt_rows, I, k);
                            double vq = v_per_kb * qps;
                            csv << ds << "," << current_algo_name << ",ef=" << ef << "," << k << "," << fixed << setprecision(6)
                                << recall << "," << qps << "," << latency << "," << index_kb << "," << v_per_kb << ","
                                << vq << "\n";
                            cout << current_algo_name << " ef=" << ef << " k=" << k << " recall=" << recall
                                 << " QPS=" << qps << " Latency=" << latency << "us VQ=" << vq << endl;
                        }
                    }

                    delete index;
                } catch (const exception& e) {
                    cerr << "Load HNSWSQ index failed: " << e.what() << endl;
                }
            }
        }

        delete[] queries_f;
        delete[] gt_rows;
    }

    csv.close();
    cout << "\nSaved CSV: vq_recall_results.csv" << endl;
    return 0;
}
