#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <queue>
#include <string>
#include <unordered_set>
#include <vector>
#include <omp.h>

#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/compressed_hnsw_framework.h"

using namespace std;
using namespace hnswlib;

class StopW {
    std::chrono::steady_clock::time_point begin_;
public:
    StopW() : begin_(std::chrono::steady_clock::now()) {}
    double elapsed_us() const {
        return static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - begin_).count());
    }
};

double* load_fvecs_as_double(const std::string& filename, size_t& num_vectors, size_t& dim) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        std::cerr << "Cannot open " << filename << std::endl;
        return nullptr;
    }

    int32_t d = 0;
    input.read((char*)&d, 4);
    dim = static_cast<size_t>(d);

    input.seekg(0, std::ios::end);
    size_t file_size = static_cast<size_t>(input.tellg());
    num_vectors = file_size / (4 + dim * 4);

    double* data = new double[num_vectors * dim];
    float* tmp = new float[dim];
    input.seekg(0, std::ios::beg);

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

unsigned int* load_ivecs(const std::string& filename, size_t& num_vectors, size_t& dim) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        std::cerr << "Cannot open " << filename << std::endl;
        return nullptr;
    }

    int32_t d = 0;
    input.read((char*)&d, 4);
    dim = static_cast<size_t>(d);

    input.seekg(0, std::ios::end);
    size_t file_size = static_cast<size_t>(input.tellg());
    num_vectors = file_size / (4 + dim * 4);

    unsigned int* data = new unsigned int[num_vectors * dim];
    input.seekg(0, std::ios::beg);

    for (size_t i = 0; i < num_vectors; ++i) {
        input.read((char*)&d, 4);
        input.read((char*)(data + i * dim), dim * 4);
    }

    return data;
}

size_t count_fvecs(const std::string& filename) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) return 0;

    int32_t d = 0;
    input.read((char*)&d, 4);
    if (d <= 0) return 0;

    input.seekg(0, std::ios::end);
    size_t file_size = static_cast<size_t>(input.tellg());
    return file_size / (4 + static_cast<size_t>(d) * 4);
}

static std::string chain_tag(int chain_max_length) {
    if (chain_max_length < 0) return "chunlim";
    return "ch" + std::to_string(chain_max_length);
}

template <typename CodecPolicy>
void search_one_config(const std::string& dataset,
                       const std::string& base_dir,
                       const std::string& algo,
                       int chain_max_length,
                       bool use_cache,
                       bool use_tls,
                       double tls_ratio,
                       size_t k,
                       const std::vector<size_t>& efs,
                       std::ofstream& csv) {
    const std::string query_file = base_dir + dataset + "_test.fvecs";
    const std::string gt_file = base_dir + dataset + "_neighbors.ivecs";
    const std::string train_file = base_dir + dataset + "_train.fvecs";

    size_t qsize = 0, qdim = 0;
    double* queries = load_fvecs_as_double(query_file, qsize, qdim);
    if (!queries) return;

    size_t gt_num = 0, gt_dim = 0;
    unsigned int* gt = load_ivecs(gt_file, gt_num, gt_dim);
    if (!gt) {
        delete[] queries;
        return;
    }

    if (qsize != gt_num) {
        std::cerr << "Query size and GT size mismatch on dataset: " << dataset << std::endl;
        delete[] queries;
        delete[] gt;
        return;
    }

    size_t train_num = count_fvecs(train_file);
    size_t cache_size = use_cache ? std::max<size_t>(1, train_num / 100) : 0;

    const std::string index_file = dataset + "_" + algo + "_" + chain_tag(chain_max_length) + "_pq.bin";

    L2SpaceDouble l2space(qdim);
    auto* index = new HierarchicalNSWCABFRAMEWORK<double, CodecPolicy>(
        &l2space, index_file, true, cache_size);
    index->setUseTls(use_tls);
    index->setTlsRatio(tls_ratio);

    size_t topk = std::min(k, gt_dim);

    std::cout << "\n=== Search dataset=" << dataset
              << " algo=" << algo
              << " chain_max=" << chain_max_length
              << " cache=" << (use_cache ? 1 : 0)
              << " tls=" << (use_tls ? 1 : 0)
              << " tls_ratio=" << tls_ratio << " ===" << std::endl;

    for (size_t ef : efs) {
        index->setEf(ef);

        size_t correct = 0;
        StopW timer;

        for (long long i = 0; i < static_cast<long long>(qsize); ++i) {
            auto result = index->searchKnn(queries + i * qdim, topk);
            std::unordered_set<labeltype> gt_set;
            for (size_t j = 0; j < topk; ++j) {
                gt_set.insert(gt[i * gt_dim + j]);
            }

            while (!result.empty()) {
                if (gt_set.find(result.top().second) != gt_set.end()) {
                    ++correct;
                }
                result.pop();
            }
        }

        double recall = static_cast<double>(correct) / static_cast<double>(qsize * topk);
        double time_us = timer.elapsed_us() / static_cast<double>(qsize);

        std::cout << "ef=" << ef
                  << " recall=" << std::fixed << std::setprecision(4) << recall
                  << " time=" << std::setprecision(2) << time_us << " us/query" << std::endl;

        csv << dataset << ','
            << algo << ','
            << chain_max_length << ','
            << (use_cache ? 1 : 0) << ','
            << (use_tls ? 1 : 0) << ','
            << std::fixed << std::setprecision(4) << tls_ratio << ','
            << topk << ','
            << ef << ','
            << std::setprecision(6) << recall << ','
            << std::setprecision(6) << time_us << '\n';

        if (recall >= 0.99) {
            break;
        }
    }

    delete index;
    delete[] queries;
    delete[] gt;
}

void dispatch_search(const std::string& dataset,
                     const std::string& base_dir,
                     const std::string& algo,
                     int chain_max_length,
                     bool use_cache,
                     bool use_tls,
                     double tls_ratio,
                     size_t k,
                     const std::vector<size_t>& efs,
                     std::ofstream& csv) {
    if (algo == "DeXOR") return search_one_config<codecs::DeXORCodecPolicy>(dataset, base_dir, algo, chain_max_length, use_cache, use_tls, tls_ratio, k, efs, csv);
    if (algo == "Gorilla") return search_one_config<codecs::GorillaCodecPolicy>(dataset, base_dir, algo, chain_max_length, use_cache, use_tls, tls_ratio, k, efs, csv);
    if (algo == "Elf") return search_one_config<codecs::ElfCodecPolicy>(dataset, base_dir, algo, chain_max_length, use_cache, use_tls, tls_ratio, k, efs, csv);
    if (algo == "Camel") return search_one_config<codecs::CamelCodecPolicy>(dataset, base_dir, algo, chain_max_length, use_cache, use_tls, tls_ratio, k, efs, csv);
    if (algo == "DeXORPlus") return search_one_config<codecs::DeXORPlusCodecPolicy>(dataset, base_dir, algo, chain_max_length, use_cache, use_tls, tls_ratio, k, efs, csv);
    throw std::runtime_error("Unknown algorithm: " + algo);
}

int main(int argc, char** argv) {
    std::string base_dir = "../datasets/hdf5files/";
    std::string out_csv = "compressed_hnsw_ablation_search_results.csv";
    int threads = 32;
    size_t k = 1;

    std::vector<std::string> datasets = {
        "fashion-mnist-784-euclidean",
        "gist-960-euclidean",
        "mnist-784-euclidean",
        "sift-128-euclidean"
    };

    std::vector<std::string> algorithms = {
        "DeXOR",
        "Gorilla",
        "Elf",
        "Camel",
        "DeXORPlus"
    };

    std::vector<int> chain_max_list = {2, 5, -1};
    std::vector<int> cache_flags = {0, 1};
    std::vector<int> tls_flags = {0, 1};
    std::vector<double> tls_ratios = {0.2};
    std::vector<size_t> efs = {10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 300, 400};

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--base_dir" && i + 1 < argc) {
            base_dir = argv[++i];
        } else if (arg == "--output_csv" && i + 1 < argc) {
            out_csv = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            threads = std::stoi(argv[++i]);
        } else if (arg == "--k" && i + 1 < argc) {
            k = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (arg == "--dataset" && i + 1 < argc) {
            datasets.clear();
            while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                datasets.push_back(argv[++i]);
            }
        } else if (arg == "--algorithm" && i + 1 < argc) {
            algorithms.clear();
            while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                algorithms.push_back(argv[++i]);
            }
        } else if (arg == "--chain_max" && i + 1 < argc) {
            chain_max_list.clear();
            while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                chain_max_list.push_back(std::stoi(argv[++i]));
            }
        } else if (arg == "--use_cache" && i + 1 < argc) {
            cache_flags.clear();
            while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                cache_flags.push_back(std::stoi(argv[++i]));
            }
        } else if (arg == "--use_tls" && i + 1 < argc) {
            tls_flags.clear();
            while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                tls_flags.push_back(std::stoi(argv[++i]));
            }
        } else if (arg == "--tls_ratio" && i + 1 < argc) {
            tls_ratios.clear();
            while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                tls_ratios.push_back(std::stod(argv[++i]));
            }
        } else if (arg == "--ef" && i + 1 < argc) {
            efs.clear();
            while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                efs.push_back(static_cast<size_t>(std::stoul(argv[++i])));
            }
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << std::endl;
            return -1;
        }
    }

    if (!base_dir.empty() && base_dir.back() != '/' && base_dir.back() != '\\') {
        base_dir += '/';
    }

    omp_set_num_threads(threads);

    std::ofstream csv(out_csv);
    if (!csv.is_open()) {
        std::cerr << "Failed to open csv: " << out_csv << std::endl;
        return -1;
    }

    csv << "Dataset,Algorithm,ChainMaxLength,UseCacheTop1PctByLevel,UseTwoLevelSearch,TlsRatio,K,ef,Recall,TimePerQuery(us)\n";

    for (const auto& ds : datasets) {
        for (const auto& algo : algorithms) {
            for (int chain_max : chain_max_list) {
                for (int cache_flag : cache_flags) {
                    for (int tls_flag : tls_flags) {
                        if (tls_flag == 0) {
                            dispatch_search(ds, base_dir, algo, chain_max, cache_flag != 0, false, 0.0, k, efs, csv);
                        } else {
                            for (double ratio : tls_ratios) {
                                dispatch_search(ds, base_dir, algo, chain_max, cache_flag != 0, true, ratio, k, efs, csv);
                            }
                        }
                    }
                }
            }
        }
    }

    csv.close();
    std::cout << "\nSaved CSV: " << out_csv << std::endl;
    return 0;
}
