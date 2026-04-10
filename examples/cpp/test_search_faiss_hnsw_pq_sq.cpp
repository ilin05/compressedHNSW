#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <omp.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/IndexPQ.h>
#include <faiss/index_factory.h>
#include <faiss/index_io.h>

using namespace std;

class StopW {
    std::chrono::steady_clock::time_point time_begin;
 public:
    StopW() {
        time_begin = std::chrono::steady_clock::now();
    }
    float getElapsedTimeMicro() {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
    }
    void reset() {
        time_begin = std::chrono::steady_clock::now();
    }
};

float* load_fvecs(const std::string& filename, size_t& num_vectors, size_t& dim) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot open file " + filename);
    }
    int32_t d;
    input.read((char*)&d, 4);
    dim = d;
    input.seekg(0, std::ios::end);
    size_t file_size = input.tellg();
    num_vectors = file_size / (4 + dim * 4);
    float* data = new float[num_vectors * dim];
    input.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num_vectors; ++i) {
        input.read((char*)&d, 4);
        input.read((char*)(data + i * dim), dim * 4);
    }
    return data;
}

unsigned int* load_ivecs(const std::string& filename, size_t& num_vectors, size_t& dim) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        return nullptr;
    }
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

int main() {
    omp_set_num_threads(32);
    std::string base_dir = "../datasets/hdf5files/";
    vector<string> datasets = {
        "fashion-mnist-784-euclidean",
        "gist-960-euclidean",
        "mnist-784-euclidean",
        "sift-128-euclidean"
    };

    vector<string> algos = {"HNSWSQ", "HNSWPQ"};

    for (const auto& ds_base : datasets) {
        std::string query_file = base_dir + ds_base + "_test.fvecs";
        std::string gt_file = base_dir + ds_base + "_neighbors.ivecs";

        size_t qsize, qdim;
        float* queries = nullptr;
        unsigned int* gt = nullptr;
        try {
            queries = load_fvecs(query_file, qsize, qdim);
            size_t gt_num, gt_dim;
            gt = load_ivecs(gt_file, gt_num, gt_dim);
        } catch (...) { continue; }

        for (const auto& algo : algos) {
            std::string index_path = ds_base + "_train.fvecs_" + algo + ".bin";
            cout << "Loading " << index_path << endl;

            faiss::Index* index = nullptr;
            try {
                index = faiss::read_index(index_path.c_str());
            } catch (...) {
                cout << "Index not found\n";
                continue;
            }

            faiss::IndexHNSW* real_index = nullptr;
            if (algo == "HNSWPQ" || algo == "HNSWSQ") {
               // HNSW is top level
               // wait, index factory returns IndexPreTransform or IndexHNSW.
               // It doesn't matter, we can search it directly, but to set efSearch:
               auto raw_ptr = dynamic_cast<faiss::IndexHNSW*>(index);
               if(raw_ptr) real_index = raw_ptr;
            }
            
            vector<size_t> efs = {10, 20, 40, 80, 120, 200, 400, 600, 800, 1000};
            int k = 10;
            vector<faiss::idx_t> I(qsize * k);
            vector<float> D(qsize * k);

            for (size_t ef : efs) {
                // If it is faiss::IndexHNSW, we can set efSearch.
                if(real_index) {
                    real_index->hnsw.efSearch = ef;
                } else {
                    faiss::SearchParametersHNSW params;
                    params.efSearch = ef;
                    index->search(qsize, queries, k, D.data(), I.data(), &params);
                    // wait, maybe the factory adds pretransforms. We can just set globally or cast to right index if needed.
                }

                StopW sw;
                index->search(qsize, queries, k, D.data(), I.data());
                double search_time = sw.getElapsedTimeMicro() / 1000.0 / qsize;

                size_t correct = 0;
                for (size_t i = 0; i < qsize; i++) {
                    std::unordered_set<unsigned int> g(gt + i * 100, gt + i * 100 + k);
                    for (int j = 0; j < k; j++) {
                        if (g.count(I[i * k + j])) {
                            correct++;
                        }
                    }
                }
                double recall = (double)correct / (qsize * k);
                cout << "Algo: " << algo << " ef: " << ef << " Recall: " << recall << " Time: " << search_time << " ms/q\n";
            }
            delete index;
        }
        delete[] queries;
        if(gt) delete[] gt;
    }
    return 0;
}
