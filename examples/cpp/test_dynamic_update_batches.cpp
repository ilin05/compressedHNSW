#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>
#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/compressed_hnsw_framework.h"
#include "../../hnswlib/hnswalg_simplified_alp_PQ.h"
using namespace hnswlib;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
static double us(Clock::time_point start) { return std::chrono::duration<double, std::micro>(Clock::now()-start).count(); }

struct Fvecs {
    size_t dim, count;
    std::vector<float> data;
    std::vector<uint32_t> source_ids;
    Fvecs(const std::string& path, size_t limit, uint64_t seed=0) {
        std::ifstream in(path, std::ios::binary);
        int32_t d = 0;
        if (!in.read((char*)&d, 4) || d <= 0) throw std::runtime_error("invalid fvecs: " + path);
        dim = d;
        const auto bytes = fs::file_size(path);
        if (bytes % (4 + dim*4)) throw std::runtime_error("invalid fvecs size: " + path);
        size_t total=bytes/(4+dim*4);
        count = std::min(limit, total);
        data.resize(count*dim);
        source_ids.resize(total);
        std::iota(source_ids.begin(),source_ids.end(),0);
        if (seed) { std::mt19937_64 rng(seed); std::shuffle(source_ids.begin(),source_ids.end(),rng); }
        source_ids.resize(count);
        for (size_t i=0; i<count; ++i) {
            in.seekg(uint64_t(source_ids[i])*(4+dim*4));
            if (!in.read((char*)&d, 4) || size_t(d)!=dim || !in.read((char*)(data.data()+i*dim), dim*4))
                throw std::runtime_error("invalid fvecs row: " + std::to_string(i));
        }
    }
    const float* row(size_t i) const { return data.data()+i*dim; }
};

template<class Scalar> static std::vector<Scalar> convert(const float* row, size_t dim) {
    return std::vector<Scalar>(row, row+dim);
}

static std::vector<std::vector<uint32_t>> ground_truth(const Fvecs& train, const Fvecs& queries,
        const std::vector<uint32_t>& active) {
    constexpr size_t k=1;
    if (active.size()<k) throw std::runtime_error("no active nodes");
    std::vector<std::vector<uint32_t>> truth(queries.count);
    for (size_t qi=0; qi<queries.count; ++qi) {
        std::vector<std::pair<double,uint32_t>> dist;
        dist.reserve(active.size());
        for (uint32_t id:active) {
            double sum=0;
            for (size_t j=0; j<train.dim; ++j) {
                double delta=double(queries.row(qi)[j])-train.row(id)[j];
                sum+=delta*delta;
            }
            dist.emplace_back(sum,id);
        }
        std::partial_sort(dist.begin(),dist.begin()+k,dist.end());
        for (size_t j=0;j<k;++j) truth[qi].push_back(dist[j].second);
    }
    return truth;
}

template<class Index,class Scalar> static std::pair<double,double> search(Index& index,
        const Fvecs& train,const Fvecs& queries,const std::vector<uint32_t>& active) {
    const auto truth=ground_truth(train,queries,active);
    std::unordered_set<uint32_t> valid(active.begin(),active.end());
    size_t hits=0;
    auto started=Clock::now();
    for (size_t qi=0;qi<queries.count;++qi) {
        auto query=convert<Scalar>(queries.row(qi),queries.dim);
        auto result=index.searchKnn(query.data(),1);
        if (result.size()!=1) throw std::runtime_error("search returned no result");
        while (!result.empty()) {
            auto id=uint32_t(result.top().second); result.pop();
            if (!valid.count(id)) throw std::runtime_error("deleted/unknown node returned: "+std::to_string(id));
            hits+=std::find(truth[qi].begin(),truth[qi].end(),id)!=truth[qi].end();
        }
    }
    const double elapsed=us(started);
    return {double(hits)/queries.count,elapsed/queries.count};
}

template<class Index,class Scalar,class Prepare,class SearchConfig,class Meter>
static void run(Index& index,const std::string& codec,const Fvecs& train,const Fvecs& queries,
        size_t base,size_t batch_size,size_t batches,uint64_t seed,const fs::path& out,
        bool profile,Prepare prepare,SearchConfig search_config,Meter meter) {
    std::mt19937_64 rng(seed);
    std::vector<uint32_t> ids(train.count);
    std::iota(ids.begin(),ids.end(),0);
    std::shuffle(ids.begin(),ids.end(),rng);
    std::vector<uint32_t> active(ids.begin(),ids.begin()+base);
    for (uint32_t id:active) {
        auto row=convert<Scalar>(train.row(id),train.dim);
        index.addPoint(row.data(),id);
    }
    prepare(); search_config();
    fs::create_directories(out);
    if (profile) {
        // The wrapper attaches perf after READY; only addPoint runs before DONE.
        std::ofstream(out/"READY").put('\n');
        while (!fs::exists(out/"GO")) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        for (size_t j=0;j<batch_size;++j) {
            uint32_t id=ids[base+j];
            auto row=convert<Scalar>(train.row(id),train.dim);
            index.addPoint(row.data(),id);
        }
        std::ofstream(out/"DONE").put('\n');
        while (!fs::exists(out/"EXIT")) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        return;
    }
    std::ofstream csv(out/"metrics.csv");
    csv << "codec,stage,batch,active,deleted,index_bytes,recall_at_1,mean_query_us,qps,insert_mean_us,delete_mean_us,graph_mean_us,encoding_mean_us,max_insert_abs_error\n";
    auto write_stage=[&](const char* stage,size_t batch,double insert_us,double delete_us,double graph_us,double encoding_us,double max_error) {
        index.saveIndex((out/"index.bin").string());
        const auto bytes=fs::file_size(out/"index.bin");
        auto [recall,mean_us]=search<Index,Scalar>(index,train,queries,active);
        csv << codec << ',' << stage << ',' << batch << ',' << active.size() << ','
            << index.getDeletedCount() << ',' << bytes << ',' << recall << ',' << mean_us << ','
            << 1e6/mean_us << ',' << insert_us << ',' << delete_us << ',' << graph_us << ','
            << encoding_us << ',' << max_error << '\n';
        csv.flush();
        std::cout << codec << ' ' << stage << " batch=" << batch << " bytes=" << bytes
                  << " recall@1=" << recall << " mean_us=" << mean_us << std::endl;
    };
    write_stage("before",0,0,0,0,0,0);
    for (size_t batch=0;batch<batches;++batch) {
        std::shuffle(active.begin(),active.end(),rng);
        auto started=Clock::now();
        for (size_t j=0;j<batch_size;++j) index.markDelete(active[j]);
        const double delete_us=us(started)/batch_size;
        active.erase(active.begin(),active.begin()+batch_size);
        const auto [graph_before,encoding_before]=meter();
        double max_error=0;
        double insertion_total_us=0;
        for (size_t j=0;j<batch_size;++j) {
            uint32_t id=ids[base+batch*batch_size+j];
            auto row=convert<Scalar>(train.row(id),train.dim);
            started=Clock::now();
            index.addPoint(row.data(),id);
            insertion_total_us+=us(started);
            auto decoded=index.template getDataByLabel<Scalar>(id);
            for (size_t d=0;d<train.dim;++d) {
                if (codec!="alp") {
                    float value=float(decoded[d]);
                    if (std::memcmp(&value,train.row(id)+d,4)) {
                        uint32_t expected_bits,actual_bits;
                        std::memcpy(&expected_bits,train.row(id)+d,4);
                        std::memcpy(&actual_bits,&value,4);
                        throw std::runtime_error("lossless insert mismatch at id="+std::to_string(id)+
                            " coordinate="+std::to_string(d)+" expected_bits="+std::to_string(expected_bits)+
                            " actual_bits="+std::to_string(actual_bits));
                    }
                } else max_error=std::max(max_error,std::abs(double(decoded[d])-train.row(id)[d]));
            }
            active.push_back(id);
        }
        const double insert_us=insertion_total_us/batch_size;
        const auto [graph_after,encoding_after]=meter();
        write_stage("after",batch+1,insert_us,delete_us,
                    (graph_after-graph_before)/batch_size,
                    (encoding_after-encoding_before)/batch_size,max_error);
    }
}

int main(int argc,char** argv) {
    try {
        if (argc!=11) throw std::runtime_error("usage: benchmark TRAIN TEST CODEC BASE BATCH BATCHES QUERIES SEED OUT MODE(normal|profile)");
        const std::string codec=argv[3],mode=argv[10];
        size_t base=std::stoull(argv[4]),batch=std::stoull(argv[5]),batches=std::stoull(argv[6]);
        size_t qcount=std::stoull(argv[7]); uint64_t seed=std::stoull(argv[8]);
        if (base<256 || batch==0 || batches==0 || qcount==0 || batch>=base || (mode!="normal" && mode!="profile"))
            throw std::runtime_error("invalid benchmark parameters");
        Fvecs train(argv[1],base+batch*batches,seed),queries(argv[2],qcount);
        if (train.count<base+batch*batches || queries.count<qcount || train.dim!=queries.dim)
            throw std::runtime_error("dataset too small or dimension mismatch");
        fs::path out(argv[9]);
        fs::create_directories(out);
        std::ofstream sampled(out/"sampled_train_ids.csv");
        sampled << "local_id,source_row\n";
        for (size_t i=0;i<train.count;++i) sampled << i << ',' << train.source_ids[i] << '\n';
        if (codec=="alp") {
            L2Space space(train.dim);
            HierarchicalNSWALPSIMPLIFIEDPQ<float> index(&space,base+batch*batches,16,200,
                std::max<size_t>(1,base/100),false,1e-4f,true);
            run<decltype(index),float>(index,codec,train,queries,base,batch,batches,seed,out,mode=="profile",
                [&]{index.compress_dataset();},[&]{index.setEf(50);index.setUseTLS(true);index.setTLSRatio(0.2f);},
                [&]{return std::make_pair(index.dynamic_graph_us_,index.dynamic_encoding_us_);});
        } else if (codec=="raw_f32") {
            L2Space space(train.dim);
            HierarchicalNSW<float> index(&space,base+batch*batches,16,200);
            run<decltype(index),float>(index,codec,train,queries,base,batch,batches,seed,out,mode=="profile",
                []{},[&]{index.setEf(50);},[]{return std::make_pair(0.0,0.0);});
        } else if (codec=="raw_f64") {
            L2SpaceDouble space(train.dim);
            HierarchicalNSW<double> index(&space,base+batch*batches,16,200);
            run<decltype(index),double>(index,codec,train,queries,base,batch,batches,seed,out,mode=="profile",
                []{},[&]{index.setEf(50);},[]{return std::make_pair(0.0,0.0);});
        } else {
            L2SpaceDouble space(train.dim);
            auto execute=[&](auto policy) {
                using Policy=decltype(policy); using Index=HierarchicalNSWCABFRAMEWORK<double,Policy>;
                Index index(&space,base+batch*batches,16,200,true,0,
                    std::max<size_t>(1,base/100),false,true);
                run<Index,double>(index,codec,train,queries,base,batch,batches,seed,out,mode=="profile",
                    [&]{index.compress_dataset(2);},[&]{index.setEf(50);index.setUseTls(true);index.setTlsRatio(0.2);},
                    [&]{return std::make_pair(index.dynamic_graph_us_,index.dynamic_encoding_us_);});
            };
            if (codec=="dexor") execute(codecs::DeXORCodecPolicy{});
            else if (codec=="gorilla") execute(codecs::GorillaCodecPolicy{});
            else if (codec=="elf") execute(codecs::ElfCodecPolicy{});
            else throw std::runtime_error("unknown codec: "+codec);
        }
        return 0;
    } catch(const std::exception& e) {std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
