#include <faiss/IndexFlatCodes.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQFastScan.h>
#include <faiss/IndexNSG.h>
#include <faiss/index_io.h>
#include <faiss/invlists/BlockInvertedLists.h>
#include <faiss/invlists/InvertedLists.h>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

int main(int argc, char** argv) {
    try {
        std::string path, algorithm;
        for (int i = 1; i < argc; ++i) {
            if (i + 1 >= argc) throw std::invalid_argument("missing argument value");
            const std::string option = argv[i];
            const std::string value = argv[++i];
            if (option == "--index") path = value;
            else if (option == "--algorithm") algorithm = value;
            else throw std::invalid_argument("unknown option " + option);
        }
        if (path.empty()) throw std::invalid_argument("--index is required");
        std::unique_ptr<faiss::Index> index(faiss::read_index(path.c_str()));
        std::vector<std::pair<std::string, uint64_t>> parts;
        auto add = [&](const std::string& name, uint64_t bytes) {
            parts.emplace_back(name, bytes);
        };

        if (algorithm == "ivf_flat" || algorithm == "ivfpq_fastscan") {
            auto* ivf = dynamic_cast<faiss::IndexIVF*>(index.get());
            if (!ivf || !ivf->invlists) throw std::runtime_error("expected an IVF index");
            auto* coarse = dynamic_cast<faiss::IndexFlatCodes*>(ivf->quantizer);
            if (!coarse) throw std::runtime_error("expected a flat coarse quantizer");
            uint64_t codes = 0, ids = 0;
            if (algorithm == "ivf_flat") {
                if (!dynamic_cast<faiss::IndexIVFFlat*>(ivf))
                    throw std::runtime_error("expected IndexIVFFlat");
                auto* lists = dynamic_cast<faiss::ArrayInvertedLists*>(ivf->invlists);
                if (!lists) throw std::runtime_error("expected ArrayInvertedLists");
                for (size_t i = 0; i < lists->nlist; ++i) {
                    codes += lists->codes[i].size();
                    ids += lists->ids[i].size() * sizeof(faiss::idx_t);
                }
                add("Vector Payloads", codes);
            } else {
                auto* fast = dynamic_cast<faiss::IndexIVFPQFastScan*>(ivf);
                auto* lists = dynamic_cast<faiss::BlockInvertedLists*>(ivf->invlists);
                if (!fast || !lists) throw std::runtime_error("expected blocked IVFPQFastScan");
                for (size_t i = 0; i < lists->nlist; ++i) {
                    codes += lists->codes[i].size(); // includes SIMD block padding
                    ids += lists->ids[i].size() * sizeof(faiss::idx_t);
                }
                add("Quantized Vector Codes", codes);
                add("Quantizer Codebook", fast->pq.centroids.size() * sizeof(float));
            }
            add("Vector IDs", ids);
            add("Coarse Centroids", coarse->codes.size());
        } else if (algorithm == "nsg") {
            auto* nsg = dynamic_cast<faiss::IndexNSGFlat*>(index.get());
            if (!nsg || !nsg->nsg.final_graph)
                throw std::runtime_error("expected a built IndexNSGFlat");
            auto* storage = dynamic_cast<faiss::IndexFlatCodes*>(nsg->storage);
            if (!storage) throw std::runtime_error("expected flat NSG vector storage");
            const auto& graph = *nsg->nsg.final_graph;
            uint64_t edges_and_terminators = 0;
            for (int i = 0; i < graph.N; ++i) {
                for (int j = 0; j < graph.K; ++j) {
                    if (graph.at(i, j) < 0) break;
                    ++edges_and_terminators;
                }
                ++edges_and_terminators; // Faiss writes one -1 after every adjacency list
            }
            add("Vector Payloads", storage->codes.size());
            add("Graph Structure", edges_and_terminators * sizeof(int32_t));
        } else {
            throw std::invalid_argument("algorithm must be ivf_flat, ivfpq_fastscan, or nsg");
        }

        const uint64_t total = std::filesystem::file_size(path);
        uint64_t explained = 0;
        for (const auto& part : parts) explained += part.second;
        if (explained > total) throw std::runtime_error("components exceed serialized size");
        add("Original Metadata", total - explained);
        std::cout << "component,bytes,nodes,dim,file_size_bytes\n";
        for (const auto& part : parts)
            std::cout << part.first << ',' << part.second << ','
                      << index->ntotal << ',' << index->d << ',' << total << '\n';
    } catch (const std::exception& error) {
        std::cerr << "storage breakdown failed: " << error.what() << '\n';
        return 1;
    }
}
