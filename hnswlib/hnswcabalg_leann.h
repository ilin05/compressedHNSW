#pragma once

#include "visited_list_pool.h"
#include "hnswlib.h"
#include <atomic>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <list>
#include <memory>
#include <functional>
#include <algorithm>
#include <numeric>
#include "../examples/utils/memory_block_stream_reader.h"
#include "../examples/utils/memory_stream_writer.h"
#include "../encoding_algorithms/dexor/dexor_tools.h"

namespace hnswlib {
typedef unsigned int tableint;
typedef unsigned int linklistsizeint;

template<typename dist_t>
class HierarchicalNSWCABLEANN : public AlgorithmInterface<dist_t> {
 public:
    static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;
    static const unsigned char DELETE_MARK = 0x01;

    // DeXOR State Structure
    struct DeXORState {
        double previous_value = 0.0;
        int previous_q = 0;
        int previous_delta = 0;
        long long previous_exp = 1023; // Bias for double (1023)
        int EL = 1;
        int contract_step = 0;
        int rho = 1; // Default rho
        
        void reset() {
            previous_value = 0.0;
            previous_q = 0;
            previous_delta = 0;
            previous_exp = 1023;
            EL = 1;
            contract_step = 0;
        }
    };

    size_t max_elements_{0};
    mutable std::atomic<size_t> cur_element_count{0};  // current number of elements
    size_t size_data_per_element_{0};
    size_t size_links_per_element_{0};
    mutable std::atomic<size_t> num_deleted_{0};  // number of deleted elements
    size_t M_{0};
    size_t maxM_{0};
    size_t maxM0_{0};
    size_t ef_construction_{0};
    size_t ef_{ 0 };

    double mult_{0.0}, revSize_{0.0};
    int maxlevel_{0};

    std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

    // Locks operations with element by label value
    mutable std::vector<std::mutex> label_op_locks_;

    std::mutex global;
    std::vector<std::mutex> link_list_locks_;

    tableint enterpoint_node_{0};

    size_t size_links_level0_{0};
    size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{ 0 }, prenode_offset_{ 0 };

    // 记录level0每个element数据的开始位置
    std::vector<size_t> level0_element_start_positions_;

    std::vector<char> data_level0_memory_;
    char **linkLists_{nullptr};
    std::vector<int> element_levels_;  // keeps level of each element

    size_t data_size_{0};

    DISTFUNC<dist_t> fstdistfunc_;
    void *dist_func_param_{nullptr};

    mutable std::mutex label_lookup_lock;  // lock for label_lookup_
    std::unordered_map<labeltype, tableint> label_lookup_;

    std::default_random_engine level_generator_;
    std::default_random_engine update_probability_generator_;

    mutable std::atomic<long> metric_distance_computations{0};
    mutable std::atomic<long> metric_hops{0};

    bool allow_replace_deleted_ = false;  // flag to replace deleted elements (marked as deleted) during insertions

    std::mutex deleted_elements_lock;  // lock for deleted_elements
    std::unordered_set<tableint> deleted_elements;  // contains internal ids of deleted elements

    std::string encoding_algorithm_name_ = "DeXOR";

    // 记录getOriginalData消耗的时间
    mutable std::atomic<long> getOriginalData_time{0};

    // 记录decoding消耗的时间
    mutable std::atomic<long> decoding_time{0};

    // 记录decoding的次数
    mutable std::atomic<long> decoding_count{0};

    // data cache for getOriginalDataByInternalId
    size_t cache_max_size_ = 0;
    mutable std::list<tableint> lru_history_;
    // mutable std::unordered_map<tableint, std::pair<std::vector<double>, std::list<tableint>::iterator>> getOriginalData_cache_;
    mutable std::unordered_map<tableint, std::vector<DeXORState>> root_state_cache_;
    mutable std::mutex cache_lock_;
    mutable std::atomic<int> cache_pop_count_{0};
    mutable std::atomic<int> getOriginalData_call_count_{0};
    
    // Flag to indicate if the index has been compacted
    bool is_compacted_ = false;

    bool use_encoding_algorithm_ = true;

    // Inline DeXOR Encoding Logic
    void dexor_encode(double value, DeXORState& state, utils::MemoryStreamWriter& writer) const {
        using namespace encoding_algorithm::dexor;
        
        // Decimal_XOR Logic
        int q = DeXORTools::getEnd(value, state.previous_q);

        int delta = 0;
        double alpha = 0;
        while (delta < 16) {
            double pow = DeXORTools::getP10(q + delta);
            long long a = DeXORTools::truncate(value / pow);
            long long b = DeXORTools::truncate(state.previous_value / pow);
            if (a == b) {
                alpha = a * pow;
                break;
            }
            delta++;
        }

        double pow = DeXORTools::getP10(q);
        double residual = value - alpha;
        long long beta = std::llround(residual / pow);

        if (delta >= 16 || DeXORTools::comp(alpha + beta * pow, value, pow) != 0) {
            writer.write(true);
            writer.write(true);
            dexor_exception_handle(value, state, writer);
            return;
        }

        beta = std::llabs(beta);
        bool flag = q == state.previous_q;
        if (flag && delta == state.previous_delta) {
            writer.write(true);
            writer.write(false);
        } else {
            writer.write(false);
            writer.write(flag);
            if (!flag) {
                writer.write(q + 20, 5);
                state.previous_q = q;
            }
            writer.write(delta, 4);
            state.previous_delta = delta;
        }

        if (DeXORTools::comp(alpha, 0) == 0) {
            writer.write(value > 0); // sign bit
        }

        writer.write(beta, DeXORTools::decimalBits(delta));
        state.previous_value = value;
    }

    void dexor_exception_handle(double value, DeXORState& state, utils::MemoryStreamWriter& writer) const {
        using namespace encoding_algorithm::dexor;
        union { double d; long long l; } u;
        u.d = value;
        long long exp = DeXORTools::segment(u.l, 2, 12);
        long long delta = exp - state.previous_exp;
        int bias = DeXORTools::getP2(state.EL - 1) - 1;

        if (delta >= -bias && delta <= bias) {
            writer.write(delta + bias, state.EL);
            writer.write(u.l < 0); // sign bit
            writer.write(u.l, 52); // mantissa

            if (state.EL > 1) {
                int su_bias = DeXORTools::getP2(state.EL - 2) - 1;
                if(delta >= -su_bias && delta <= su_bias) {
                    state.contract_step++;
                }else{
                    state.contract_step = 0;
                }
                if (state.contract_step == state.rho) {
                    state.EL--;
                    state.contract_step = 0;
                }
            }
        } else {
            writer.write(DeXORTools::getP2(state.EL) - 1, state.EL);
            writer.write(u.l, 64);
            state.contract_step = 0;

            if (state.EL < 10) {
                state.EL++;
            }
        }
        state.previous_exp = exp;
    }

    // Inline DeXOR Decoding Logic
    double dexor_decode(DeXORState& state, utils::MemoryBlockStreamReader& reader) const {
        using namespace encoding_algorithm::dexor;
        
        int con = reader.readInt(2);
        if (con == 3) {
            return dexor_exception_decode(state, reader);
        }

        if (con == 0 || con == 1) {
            if (con == 0) {
                state.previous_q = reader.readInt(5) - 20;
            }
            state.previous_delta = reader.readInt(4);
        }
        
        // Reconstruct alpha
        const double pow = DeXORTools::getP10(state.previous_q + state.previous_delta);
        const double truncated = static_cast<double>(DeXORTools::truncate(state.previous_value / pow));
        double previous_alpha = truncated * pow;

        long long sign = previous_alpha > 0 ? 1LL : -1LL;
        if (DeXORTools::comp(previous_alpha, 0) == 0) {
            sign = reader.readBoolean() ? 1LL : -1LL;
        }

        const int bits = DeXORTools::decimalBits(state.previous_delta);
        long long magnitude = bits > 0 ? reader.readLong(bits) : 0;
        const double beta = static_cast<double>(sign * magnitude) * DeXORTools::getP10(state.previous_q);

        state.previous_value = previous_alpha + beta;
        return state.previous_value;
    }

    double dexor_exception_decode(DeXORState& state, utils::MemoryBlockStreamReader& reader) const {
        using namespace encoding_algorithm::dexor;
        const int bias = DeXORTools::getP2(state.EL - 1) - 1;
        const long long delta = reader.readLong(state.EL) - bias;
        uint64_t lv = 0;

        if (delta >= -bias && delta <= bias) {
            state.previous_exp += delta;
            const uint64_t sign = static_cast<uint64_t>(reader.readLong(1));
            const uint64_t mantissa = static_cast<uint64_t>(reader.readLong(52));
            lv = (sign << 63) | (static_cast<uint64_t>(state.previous_exp) << 52) | mantissa;

            if (state.EL > 1) {
                const int su_bias = DeXORTools::getP2(state.EL - 2) - 1;
                if (delta >= -su_bias && delta <= su_bias) {
                    state.contract_step++;
                } else {
                    state.contract_step = 0;
                }
                if (state.contract_step == state.rho) {
                    state.EL--;
                    state.contract_step = 0;
                }
            }
        } else {
            lv = static_cast<uint64_t>(reader.readLong(64));
            state.previous_exp = DeXORTools::segment(static_cast<long long>(lv), 2, 12);

            if (state.EL < 10) {
                state.EL++;
                state.contract_step = 0;
            }
        }

        union {
            uint64_t bits;
            double value;
        } converter{};
        converter.bits = lv;
        return converter.value;
    }

    HierarchicalNSWCABLEANN(SpaceInterface<dist_t> *s) {
    }


    HierarchicalNSWCABLEANN(
        SpaceInterface<dist_t> *s,
        const std::string &location,
        bool use_encoding_algorithm = true,
        size_t cache_max_size = 0,
        bool nmslib = false,
        size_t max_elements = 0,
        bool allow_replace_deleted = false)
        : allow_replace_deleted_(allow_replace_deleted),
          use_encoding_algorithm_(use_encoding_algorithm),
          cache_max_size_(cache_max_size){
        loadIndex(location, s, max_elements);
    }


    HierarchicalNSWCABLEANN(
        SpaceInterface<dist_t> *s,
        size_t max_elements,
        const std::string &encoding_algorithm_name = "DeXOR",
        size_t M = 16,
        size_t ef_construction = 200,
        bool use_encoding_algorithm = true,
        size_t cache_max_size = 0,
        size_t random_seed = 100,
        bool allow_replace_deleted = false)
        : label_op_locks_(MAX_LABEL_OPERATION_LOCKS),
            link_list_locks_(max_elements),
            level0_element_start_positions_(max_elements),
            element_levels_(max_elements),
            allow_replace_deleted_(allow_replace_deleted),
            encoding_algorithm_name_(encoding_algorithm_name),
            use_encoding_algorithm_(use_encoding_algorithm),
            cache_max_size_(cache_max_size) {
        max_elements_ = max_elements;
        num_deleted_ = 0;
        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        if ( M <= 10000 ) {
            M_ = M;
        } else {
            HNSWERR << "warning: M parameter exceeds 10000 which may lead to adverse effects." << std::endl;
            HNSWERR << "         Cap to 10000 will be applied for the rest of the processing." << std::endl;
            M_ = 10000;
        }
        maxM_ = M_;
        maxM0_ = M_ * 2;
        ef_construction_ = std::max(ef_construction, M_);
        ef_ = 10;

        level_generator_.seed(random_seed);
        update_probability_generator_.seed(random_seed + 1);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype) + sizeof(tableint);
        
        offsetLevel0_ = 0;
        prenode_offset_ = size_links_level0_;
        label_offset_ = prenode_offset_ + sizeof(tableint);
        offsetData_ = label_offset_ + sizeof(labeltype);

        data_level0_memory_.reserve(max_elements_ * size_data_per_element_);

        cur_element_count = 0;

        visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements));

        // initializations for special treatment of the first node
        enterpoint_node_ = -1;
        maxlevel_ = -1;

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: HierarchicalNSWCABLEANN failed to allocate linklists");
        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
        mult_ = 1 / log(1.0 * M_);
        revSize_ = 1.0 / mult_;

        // // 为getOriginalData缓存预留空间
        // if(cache_max_size_ > 0){
        //     getOriginalData_cache_.reserve(cache_max_size_);
        //     loadCache();
        // }
    }


    ~HierarchicalNSWCABLEANN() {
        clear();
    }

    void clear() {
        data_level0_memory_.clear();
        data_level0_memory_.shrink_to_fit();
        root_state_cache_.clear();
        lru_history_.clear();
        for (tableint i = 0; i < cur_element_count; i++) {
            if (element_levels_[i] > 0)
                free(linkLists_[i]);
        }
        free(linkLists_);
        linkLists_ = nullptr;
        cur_element_count = 0;
        visited_list_pool_.reset(nullptr);
    }


    struct CompareByFirst {
        constexpr bool operator()(std::pair<dist_t, tableint> const& a,
            std::pair<dist_t, tableint> const& b) const noexcept {
            return a.first < b.first;
        }
    };


    void setEf(size_t ef) {
        ef_ = ef;
    }


    inline std::mutex& getLabelOpMutex(labeltype label) const {
        // calculate hash
        size_t lock_id = label & (MAX_LABEL_OPERATION_LOCKS - 1);
        return label_op_locks_[lock_id];
    }


    inline labeltype getExternalLabel(tableint internal_id) const {
        labeltype return_label;
        size_t offset = is_compacted_ ? sizeof(tableint) : label_offset_;
        memcpy(&return_label, (data_level0_memory_.data() + level0_element_start_positions_[internal_id] + offset), sizeof(labeltype));
        return return_label;
    }


    inline void setExternalLabel(tableint internal_id, labeltype label) const {
        size_t offset = is_compacted_ ? sizeof(tableint) : label_offset_;
        memcpy((char*)(data_level0_memory_.data() + level0_element_start_positions_[internal_id] + offset), &label, sizeof(labeltype));
    }


    inline labeltype *getExternalLabeLp(tableint internal_id) const {
        size_t offset = is_compacted_ ? sizeof(tableint) : label_offset_;
        return (labeltype *) (data_level0_memory_.data() + level0_element_start_positions_[internal_id] + offset);
    }

    inline tableint getPrenodeId(tableint internal_id) const {
        tableint prenode;
        size_t offset = is_compacted_ ? 0 : prenode_offset_;
        memcpy(&prenode, (data_level0_memory_.data() + level0_element_start_positions_[internal_id] + offset), sizeof(tableint));
        return prenode;
    }

    inline void setPrenodeId(tableint internal_id, tableint prenode) {
        size_t offset = is_compacted_ ? 0 : prenode_offset_;
        memcpy((char*)(data_level0_memory_.data() + level0_element_start_positions_[internal_id] + offset), &prenode, sizeof(tableint));
    }

    // 数据存储在第0层，每个element的大小都是固定的，size_data_per_element。可以通过HNSW内部的id随机读取数据
    // 如果要使用差分编码压缩data，将无法通过简单的随机读取获取数据。或许需要页表之类的结构进行索引；同时，data需要解压缩。可以在每个element的开头记录上一个data的internal_id，接着回溯到第0个data，然后依次解压缩
    inline char *getDataByInternalId(tableint internal_id) const {
        size_t offset;
        if (is_compacted_) {
            // Compacted layout: [Prenode (4)] [Label (4/8)] [LinkListSize (2)] [Neighbors (size*4)] [Data...]
            // We need to read LinkListSize to determine Data offset
            // Prenode (4) + Label (sizeof(labeltype))
            size_t linklist_size_offset = sizeof(tableint) + sizeof(labeltype);
            unsigned short int size = *((unsigned short int*)(data_level0_memory_.data() + level0_element_start_positions_[internal_id] + linklist_size_offset));
            // Data starts after neighbors
            offset = linklist_size_offset + sizeof(linklistsizeint) + size * sizeof(tableint);
        } else {
            offset = offsetData_;
        }
        return (char*)(data_level0_memory_.data() + level0_element_start_positions_[internal_id] + offset);
    }

    inline char *getDataByInternalId(tableint internal_id, char* data_level0_memory) const {
        size_t offset;
        if (is_compacted_) {
            size_t linklist_size_offset = sizeof(tableint) + sizeof(labeltype);
            unsigned short int size = *((unsigned short int*)(data_level0_memory + level0_element_start_positions_[internal_id] + linklist_size_offset));
            offset = linklist_size_offset + sizeof(linklistsizeint) + size * sizeof(tableint);
        } else {
            offset = offsetData_;
        }
        return (char*)(data_level0_memory + level0_element_start_positions_[internal_id] + offset);
    }

    std::vector<double> getOriginalDataByInternalId(tableint internal_id) const {
        // auto start_time = std::chrono::high_resolution_clock::now();

        if(!use_encoding_algorithm_ || !is_compacted_) {
            size_t dim = data_size_ / sizeof(double);
            std::vector<double> result(dim);
            char* data_ptr = getDataByInternalId(internal_id);
            memcpy(result.data(), data_ptr, dim * sizeof(double));
            return result;
        }
        
        size_t dim = *((size_t *) dist_func_param_);
        std::vector<double> result(dim);
        
        tableint prenode = getPrenodeId(internal_id);
        
        // Case 1: Node is a Root (prenode == -1)
        if (prenode == (tableint)-1) {
            // Check cache for state
            if (cache_max_size_ > 0) {
                auto it = root_state_cache_.find(internal_id);
                if (it != root_state_cache_.end()) {
                    // Reconstruct data from state
                    const std::vector<DeXORState>& states = it->second;
                    for(size_t i=0; i<dim; ++i) {
                        result[i] = states[i].previous_value;
                    }
                    return result;
                }
            }
            
            // Not in cache, decode normally (against default state)
            utils::MemoryBlockStreamReader reader((const unsigned char*)data_level0_memory_.data());
            
            // Calculate start/end for this node
            size_t start, end;
            // ... (Same offset calculation logic as before) ...
            size_t linklist_size_offset = sizeof(tableint) + sizeof(labeltype);
            unsigned short int size = *((unsigned short int*)(data_level0_memory_.data() + level0_element_start_positions_[internal_id] + linklist_size_offset));
            size_t data_offset = linklist_size_offset + sizeof(linklistsizeint) + size * sizeof(tableint);
            start = level0_element_start_positions_[internal_id] + data_offset;
            
            if (internal_id + 1 < cur_element_count && level0_element_start_positions_[internal_id+1] > 0) {
                end = level0_element_start_positions_[internal_id+1];
            } else {
                end = data_level0_memory_.size();
            }
            
            reader.resetBuffer((const unsigned char*)(data_level0_memory_.data() + start), end - start);
            
            std::vector<DeXORState> states(dim); // Default states
            auto decode_start = std::chrono::high_resolution_clock::now();
            for(size_t i=0; i<dim; ++i) {
                result[i] = dexor_decode(states[i], reader);
            }
            decoding_count ++;
            auto decode_end = std::chrono::high_resolution_clock::now();
            decoding_time += std::chrono::duration_cast<std::chrono::microseconds>(decode_end - decode_start).count();
            
            // Optional: Cache this root state if we have space? 
            // For now, we rely on loadCache to populate cache.
            return result;
        }
        
        // Case 2: Node is a Child (prenode != -1)
        // We need the state AFTER decoding the Root (prenode)
        
        std::vector<DeXORState> states(dim);
        bool cache_hit = false;
        
        if (cache_max_size_ > 0) {
            auto it = root_state_cache_.find(prenode);
            if (it != root_state_cache_.end()) {
                states = it->second; // Copy state
                cache_hit = true;
            }
        }
        
        if (!cache_hit) {
            // Decode Root to get state
            utils::MemoryBlockStreamReader reader((const unsigned char*)data_level0_memory_.data());
            
            // Calculate start/end for Root
            size_t start, end;
            size_t linklist_size_offset = sizeof(tableint) + sizeof(labeltype);
            unsigned short int size = *((unsigned short int*)(data_level0_memory_.data() + level0_element_start_positions_[prenode] + linklist_size_offset));
            size_t data_offset = linklist_size_offset + sizeof(linklistsizeint) + size * sizeof(tableint);
            start = level0_element_start_positions_[prenode] + data_offset;
            
            if (prenode + 1 < cur_element_count && level0_element_start_positions_[prenode+1] > 0) {
                end = level0_element_start_positions_[prenode+1];
            } else {
                end = data_level0_memory_.size();
            }
            
            reader.resetBuffer((const unsigned char*)(data_level0_memory_.data() + start), end - start);
            
            auto decode_start = std::chrono::high_resolution_clock::now();
            for(size_t i=0; i<dim; ++i) {
                dexor_decode(states[i], reader);
            }
            decoding_count ++;
            auto decode_end = std::chrono::high_resolution_clock::now();
            decoding_time += std::chrono::duration_cast<std::chrono::microseconds>(decode_end - decode_start).count();
        }
        
        // Now decode Child using the state
        utils::MemoryBlockStreamReader reader((const unsigned char*)data_level0_memory_.data());
        
        // Calculate start/end for Child
        size_t start, end;
        size_t linklist_size_offset = sizeof(tableint) + sizeof(labeltype);
        unsigned short int size = *((unsigned short int*)(data_level0_memory_.data() + level0_element_start_positions_[internal_id] + linklist_size_offset));
        size_t data_offset = linklist_size_offset + sizeof(linklistsizeint) + size * sizeof(tableint);
        start = level0_element_start_positions_[internal_id] + data_offset;
        
        if (internal_id + 1 < cur_element_count && level0_element_start_positions_[internal_id+1] > 0) {
            end = level0_element_start_positions_[internal_id+1];
        } else {
            end = data_level0_memory_.size();
        }
        
        reader.resetBuffer((const unsigned char*)(data_level0_memory_.data() + start), end - start);
        
        auto decode_start = std::chrono::high_resolution_clock::now();
        for(size_t i=0; i<dim; ++i) {
            result[i] = dexor_decode(states[i], reader);
        }
        decoding_count ++;
        auto decode_end = std::chrono::high_resolution_clock::now();
        decoding_time += std::chrono::duration_cast<std::chrono::microseconds>(decode_end - decode_start).count();

        return result;
    }


    int getRandomLevel(double reverse_size) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator_)) * reverse_size;
        return (int) r;
    }

    size_t getMaxElements() {
        return max_elements_;
    }

    size_t getCurrentElementCount() {
        return cur_element_count;
    }

    size_t getDeletedCount() {
        return num_deleted_;
    }

    // 这里是论文中的alg.2---找出layer中距离data point(也就是论文中的q)最近的前ef个元素，返回动态列表top_candidates
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayer(tableint ep_id, const void *data_point, int layer) {
        // visitedList 包括数组和数，visited_array 是数组，tag 是数
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        // visited_array 存储已访问的元素
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        // top_candidates存储每一层距离datapoint最近的ef个邻居，对应于论文中的动态列表W
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        // candidateSet存储候选元素，对应动态列表中的C
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

        dist_t lowerBound;
        if (!isMarkedDeleted(ep_id)) {
            // 计算data_point(query) 到enterpoint的距离，结果保存在dist中。
            std::vector<double> vec_ep = getOriginalDataByInternalId(ep_id);
            dist_t dist = fstdistfunc_(data_point, vec_ep.data(), dist_func_param_);
            // enterpoint加入到最近邻列表
            top_candidates.emplace(dist, ep_id);
            // lowerBound存储当前到datapoint的最近距离
            lowerBound = dist;
            // enterpoint加入到候选列表
            candidateSet.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidateSet.emplace(-lowerBound, ep_id);
        }
        // enterpoint加入已访问列表
        visited_array[ep_id] = visited_array_tag;

        // std::cout << "Searching base layer " << layer << " from entry point " << ep_id << std::endl;

        // 当候选列表不为空时 while |C|>0
        while (!candidateSet.empty()) {
            // 从C中取出距离q最近的元素
            std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
            // 如果C中最近元素与q的距离 > W中与q最远元素的距离，说明W中每一个元素都已评估过，退出循环
            if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) {
                break;
            }
            // 弹出候选者列表队头
            candidateSet.pop();

            // 获取当前元素的label
            tableint curNodeNum = curr_el_pair.second;

            std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]);

            // 获取当前元素的邻居
            int *data;  // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
            // 如果在第0层
            if (layer == 0) {
                // 计算当前元素邻居的内存
                data = (int*)get_linklist0(curNodeNum);
            } else {
                // 计算当前元素邻居的内存
                data = (int*)get_linklist(curNodeNum, layer);
//                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
            }
            // 获取当前元素的邻居数量
            size_t size = getListCount((linklistsizeint*)data);
            // datal表示当前元素第一个邻居的label
            tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
            if (size > 0) {
                _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
            }
            if (size > 1) {
                _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
            }
#endif

            // std::cout << "Evaluating " << size << " neighbors." << std::endl;
            // 对于layer层中的当前元素的每一个邻居candidate
            for (size_t j = 0; j < size; j++) {
                tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                if (j + 1 < size) {
                    // std::cout << "Prefetching data for neighbor " << j + 1 << std::endl;
                    _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                    // std::cout << "_mm_prefetch visited_array done." << std::endl;
                    _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
                    // std::cout << "_mm_prefetch data done." << std::endl;
                }
#endif
                // 如果candidate已经访问过（对应论文中，如果e属于v，无操作，循环次数加1）
                if (visited_array[candidate_id] == visited_array_tag) continue;
                // 没有访问过，将已访问列表中并入candidate
                visited_array[candidate_id] = visited_array_tag;
                // 根据candidate的id号获取这个candidate元素，也就是currObj1
                // std::cout << "Visiting candidate " << candidate_id << std::endl;
                // char *currObj1 = (getDataByInternalId(candidate_id));
                std::vector<double> currObjVec = getOriginalDataByInternalId(candidate_id);
                const void* currObj1 = currObjVec.data();

                // std::cout << "Data for candidate retrieved." << std::endl;
                // 计算currObj1到data point之间的距离，对应于论文中的distance(e, q)
                dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                // 取出top_candidates中距离data point最远的元素并比较大小
                // 对应于论文中，如果distance(e, q) < distance(f, 1) || |W| < ef
                if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                    // 将currObj1的id——candidate_id加到候选列表中
                    candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                    if (!isMarkedDeleted(candidate_id)){
                        // 将currObj1的id——candidate_id加到动态列表中
                        top_candidates.emplace(dist1, candidate_id);
                    }
                    // 如果动态列表的长度大于ef，那么减掉“最弱的”元素，对应于论文中|W| > ef
                    if (top_candidates.size() > ef_construction_){
                        // 取出W中距离q最远的元素
                        top_candidates.pop();
                    }
                    if (!top_candidates.empty()){
                        // 更新distance（f,q）
                        lowerBound = top_candidates.top().first;
                    }
                }
            }

            // std::cout << "Top candidates size: " << top_candidates.size() << ", lower bound: " << lowerBound << std::endl;
        }
        visited_list_pool_->releaseVisitedList(vl);

        // 返回动态列表，也就是返回layer层中距离q最近的ef个邻居
        return top_candidates;
    }


    // bare_bone_search means there is no check for deletions and stop condition is ignored in return of extra performance
    template <bool bare_bone_search = true, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerST(
        tableint ep_id,
        const void *data_point,
        size_t ef,
        BaseFilterFunctor* isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

        dist_t lowerBound;
        if (bare_bone_search || 
            (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
            std::vector<double> ep_vec = getOriginalDataByInternalId(ep_id);
            const void* ep_data = ep_vec.data();
            dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            if (!bare_bone_search && stop_condition) {
                stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
            }
            candidate_set.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;

        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            bool flag_stop_search;
            if (bare_bone_search) {
                flag_stop_search = candidate_dist > lowerBound;
            } else {
                if (stop_condition) {
                    flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
                } else {
                    flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                }
            }
            if (flag_stop_search) {
                break;
            }
            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;
            int *data = (int *) get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint*)data);
//                bool cur_node_deleted = isMarkedDeleted(current_node_id);
            if (collect_metrics) {
                metric_hops++;
                metric_distance_computations+=size;
            }

#ifdef USE_SSE
            if (size > 0) {
                _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
            }
#endif

            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                if (j < size) {
                    _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(getDataByInternalId(*(data + j + 1)), _MM_HINT_T0);
                }
#endif
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;

                    // char *currObj1 = (getDataByInternalId(candidate_id));
                    std::vector<double> currObjVec = getOriginalDataByInternalId(candidate_id);
                    const void* currObj1 = currObjVec.data();

                    dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                    bool flag_consider_candidate;
                    if (!bare_bone_search && stop_condition) {
                        flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                    } else {
                        flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                    }

                    if (flag_consider_candidate) {
                        candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                        if (candidate_set.size() > 0) {
                            size_t offset = is_compacted_ ? (sizeof(tableint) + sizeof(labeltype)) : offsetLevel0_;
                            _mm_prefetch(data_level0_memory_.data() + level0_element_start_positions_[candidate_set.top().second] +
                                            offset,  ///////////
                                            _MM_HINT_T0);  ////////////////////////
                        }
#endif

                        if (bare_bone_search || 
                            (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
                            top_candidates.emplace(dist, candidate_id);
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                            }
                        }

                        bool flag_remove_extra = false;
                        if (!bare_bone_search && stop_condition) {
                            flag_remove_extra = stop_condition->should_remove_extra();
                        } else {
                            flag_remove_extra = top_candidates.size() > ef;
                        }
                        while (flag_remove_extra) {
                            tableint id = top_candidates.top().second;
                            top_candidates.pop();
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                                flag_remove_extra = stop_condition->should_remove_extra();
                            } else {
                                flag_remove_extra = top_candidates.size() > ef;
                            }
                        }

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }


    // 论文中的alg.4--启发式方法选择邻居，从top_candidates中选择距离q最近的M个元素
    void getNeighborsByHeuristic2(
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        const size_t M) {
        // 如果top_candidates里元素个数小于M,那还选个啥，直接return
        if (top_candidates.size() < M) {
            return;
        }
        // queue_closest是working queue for the candidates，论文中是W ，存放候选者
        std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
        // return_list存放最终的M个结果,论文中是R，初始为空集 
        std::vector<std::pair<dist_t, tableint>> return_list;
        // 将queue_closest初始化为top_candidates,论文中为W<--C
        while (top_candidates.size() > 0) {
            queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
            top_candidates.pop();
        }
        // 当queue_closest内的元素个数大于0
        while (queue_closest.size()) {
            // 如果return_list内元素个数已经大于M,那么启发式查找过程结束
            if (return_list.size() >= M)
                break;
            // curent_pair是queue_closest（W）的元素
            std::pair<dist_t, tableint> curent_pair = queue_closest.top();
            // dist_to_query是curent_pair与query的距离
            dist_t dist_to_query = -curent_pair.first;
            // queue_cloest元素减一
            queue_closest.pop();
            bool good = true;

            std::vector<double> vec2 = getOriginalDataByInternalId(curent_pair.second);

            // 对于return_list(R)中的每一个元素
            for (std::pair<dist_t, tableint> second_pair : return_list) {
                std::vector<double> vec1 = getOriginalDataByInternalId(second_pair.second);
                dist_t curdist =
                        fstdistfunc_(vec1.data(),
                                        vec2.data(),
                                        dist_func_param_);
                // 如果curent_pair 与已经与q连接元素的距离 < curent_pair与query的距离
                if (curdist < dist_to_query) {
                    // curent_pair将不会作为q的邻居返回
                    good = false;
                    break;
                }
            }
            // 如果curent_pair 与已经与q连接元素的距离 >= curent_pair与query的距离，见论文中Fig.2
            // 那么将curent_pair并入return_list(也就是论文里的R)
            if (good) {
                return_list.push_back(curent_pair);
            }
        }

        for (std::pair<dist_t, tableint> curent_pair : return_list) {
            top_candidates.emplace(-curent_pair.first, curent_pair.second);
        }
    }


    linklistsizeint *get_linklist0(tableint internal_id) const {
        if (is_compacted_) {
            return (linklistsizeint *) (data_level0_memory_.data() + level0_element_start_positions_[internal_id] + sizeof(tableint) + sizeof(labeltype));
        }
        return (linklistsizeint *) (data_level0_memory_.data() + level0_element_start_positions_[internal_id] + offsetLevel0_);
    }


    linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) const {
        if (is_compacted_) {
            return (linklistsizeint *) (data_level0_memory_ + level0_element_start_positions_[internal_id] + sizeof(tableint) + sizeof(labeltype));
        }
        return (linklistsizeint *) (data_level0_memory_ + level0_element_start_positions_[internal_id] + offsetLevel0_);
    }


    linklistsizeint *get_linklist(tableint internal_id, int level) const {
        return (linklistsizeint *) (linkLists_[internal_id] + (level - 1) * size_links_per_element_);
    }


    linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const {
        return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
    }


    tableint mutuallyConnectNewElement(
        const void *data_point,
        tableint cur_c,
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        int level,
        bool isUpdate) {
        size_t Mcurmax = level ? maxM_ : maxM0_;
        getNeighborsByHeuristic2(top_candidates, M_);
        if (top_candidates.size() > M_)
            throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

        std::vector<tableint> selectedNeighbors;
        selectedNeighbors.reserve(M_);
        while (top_candidates.size() > 0) {
            selectedNeighbors.push_back(top_candidates.top().second);
            top_candidates.pop();
        }

        tableint next_closest_entry_point = selectedNeighbors.back();

        {
            // lock only during the update
            // because during the addition the lock for cur_c is already acquired
            std::unique_lock <std::mutex> lock(link_list_locks_[cur_c], std::defer_lock);
            if (isUpdate) {
                lock.lock();
            }
            linklistsizeint *ll_cur;
            if (level == 0)
                ll_cur = get_linklist0(cur_c);
            else
                ll_cur = get_linklist(cur_c, level);

            if (*ll_cur && !isUpdate) {
                throw std::runtime_error("The newly inserted element should have blank link list");
            }
            setListCount(ll_cur, selectedNeighbors.size());
            tableint *data = (tableint *) (ll_cur + 1);
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                if (data[idx] && !isUpdate)
                    throw std::runtime_error("Possible memory corruption");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                data[idx] = selectedNeighbors[idx];
            }
        }

        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
            std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

            linklistsizeint *ll_other;
            if (level == 0)
                ll_other = get_linklist0(selectedNeighbors[idx]);
            else
                ll_other = get_linklist(selectedNeighbors[idx], level);

            size_t sz_link_list_other = getListCount(ll_other);

            if (sz_link_list_other > Mcurmax)
                throw std::runtime_error("Bad value of sz_link_list_other");
            if (selectedNeighbors[idx] == cur_c)
                throw std::runtime_error("Trying to connect an element to itself");
            if (level > element_levels_[selectedNeighbors[idx]])
                throw std::runtime_error("Trying to make a link on a non-existent level");

            tableint *data = (tableint *) (ll_other + 1);

            bool is_cur_c_present = false;
            if (isUpdate) {
                for (size_t j = 0; j < sz_link_list_other; j++) {
                    if (data[j] == cur_c) {
                        is_cur_c_present = true;
                        break;
                    }
                }
            }

            // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
            if (!is_cur_c_present) {
                if (sz_link_list_other < Mcurmax) {
                    data[sz_link_list_other] = cur_c;
                    setListCount(ll_other, sz_link_list_other + 1);
                } else {
                    std::vector<double> vec_cur = getOriginalDataByInternalId(cur_c);
                    std::vector<double> vec_neigh = getOriginalDataByInternalId(selectedNeighbors[idx]);

                    // finding the "weakest" element to replace it with the new one
                    dist_t d_max = fstdistfunc_(vec_cur.data(), vec_neigh.data(),
                                                dist_func_param_);
                    // Heuristic:
                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    candidates.emplace(d_max, cur_c);

                    // 邻居的linkList已经满了，需要重新选出最好的Mcurmax个邻居。逐个计算距离，放入candidates中
                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        std::vector<double> vec_cand = getOriginalDataByInternalId(data[j]);
                        candidates.emplace(
                                fstdistfunc_(vec_cand.data(), vec_neigh.data(),
                                                dist_func_param_), data[j]);
                    }

                    getNeighborsByHeuristic2(candidates, Mcurmax);

                    int indx = 0;
                    while (candidates.size() > 0) {
                        data[indx] = candidates.top().second;
                        candidates.pop();
                        indx++;
                    }

                    setListCount(ll_other, indx);
                    // Nearest K:
                    /*int indx = -1;
                    for (int j = 0; j < sz_link_list_other; j++) {
                        dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                        if (d > d_max) {
                            indx = j;
                            d_max = d;
                        }
                    }
                    if (indx >= 0) {
                        data[indx] = cur_c;
                    } */
                }
            }
        }

        return next_closest_entry_point;
    }


    void compactLevel0() {
        if (is_compacted_) return;

        std::vector<char> new_memory;
        new_memory.reserve(data_level0_memory_.size());

        std::vector<size_t> new_start_positions(cur_element_count);

        for (size_t i = 0; i < cur_element_count; ++i) {
            size_t old_start = level0_element_start_positions_[i];
            size_t new_start = new_memory.size();
            new_start_positions[i] = new_start;

            // 1. Copy Prenode
            tableint prenode = getPrenodeId(i);
            size_t prenode_size = sizeof(tableint);
            new_memory.resize(new_memory.size() + prenode_size);
            *((tableint*)(new_memory.data() + new_start)) = prenode;

            // 2. Copy Label
            labeltype label = getExternalLabel(i);
            size_t label_size = sizeof(labeltype);
            new_memory.resize(new_memory.size() + label_size);
            *((labeltype*)(new_memory.data() + new_start + prenode_size)) = label;

            // 3. Copy LinkList
            // Note: get_linklist0 uses offsetLevel0_ which is 0 in original layout.
            // We are currently NOT compacted, so get_linklist0 works as expected (using offsetLevel0_).
            
            unsigned char* old_ll_ptr = (unsigned char*)get_linklist0(i);
            linklistsizeint size = *((linklistsizeint*)old_ll_ptr);
            
            size_t ll_header_size = sizeof(linklistsizeint);
            size_t neighbors_size = size * sizeof(tableint);
            
            new_memory.resize(new_memory.size() + ll_header_size + neighbors_size);
            
            // Copy size
            *((linklistsizeint*)(new_memory.data() + new_start + prenode_size + label_size)) = size;
            
            // Copy neighbors
            memcpy(new_memory.data() + new_start + prenode_size + label_size + ll_header_size, 
                   old_ll_ptr + ll_header_size, 
                   neighbors_size);

            // 4. Copy Data
            size_t old_data_start = old_start + offsetData_;
            size_t old_data_end;
            if (i + 1 < cur_element_count && level0_element_start_positions_[i+1] > 0) {
                old_data_end = level0_element_start_positions_[i+1];
            } else {
                old_data_end = data_level0_memory_.size();
            }
            size_t data_len = old_data_end - old_data_start;
            
            new_memory.resize(new_memory.size() + data_len);
            memcpy(new_memory.data() + new_start + prenode_size + label_size + ll_header_size + neighbors_size,
                   data_level0_memory_.data() + old_data_start,
                   data_len);
        }

        data_level0_memory_ = std::move(new_memory);
        level0_element_start_positions_ = std::move(new_start_positions);
        is_compacted_ = true;
        
        // Shrink to fit to release unused memory
        data_level0_memory_.shrink_to_fit();
    }

    void compress_dataset() {
        if (is_compacted_) return;
        if (!use_encoding_algorithm_) return;

        size_t dim = data_size_ / sizeof(double);
        size_t target_root_count = std::max((size_t)1, cur_element_count / 100);
        
        std::vector<tableint> assigned_root(cur_element_count, -1);
        std::vector<dist_t> min_dist(cur_element_count, std::numeric_limits<dist_t>::max());
        std::vector<bool> is_root(cur_element_count, false);

        // 1. Select Roots (Top 1% by level)
        std::vector<tableint> indices(cur_element_count);
        std::iota(indices.begin(), indices.end(), 0);
        
        std::sort(indices.begin(), indices.end(), [&](tableint a, tableint b) {
            return element_levels_[a] > element_levels_[b];
        });

        size_t roots_found = 0;
        for(size_t i=0; i<cur_element_count && roots_found < target_root_count; ++i) {
            tableint idx = indices[i];
            if (isMarkedDeleted(idx)) continue;
            
            is_root[idx] = true;
            assigned_root[idx] = idx;
            min_dist[idx] = 0;
            roots_found++;
        }
        
        // Fallback if no roots found
        if(roots_found == 0 && cur_element_count > 0) {
             for(size_t i=0; i<cur_element_count; ++i) {
                 if(!isMarkedDeleted(i)) {
                     is_root[i] = true;
                     assigned_root[i] = i;
                     min_dist[i] = 0;
                     roots_found++;
                     break;
                 }
             }
        }

        std::cout << "Compression: Selected " << roots_found << " roots." << std::endl;

        // 2. Assign Roots (Label Propagation)
        int max_iters = 100; 
        bool changed = true;
        
        // Reusable buffers for distance calculation
        std::vector<double> vec_i(dim);
        std::vector<double> vec_root(dim);

        for(int iter=0; iter<max_iters && changed; ++iter) {
            changed = false;
            for(size_t i=0; i<cur_element_count; ++i) {
                if (isMarkedDeleted(i)) continue;
                if (is_root[i]) continue; // Roots stay roots

                // Read vec_i once
                memcpy(vec_i.data(), getDataByInternalId(i), dim * sizeof(double));
                
                // Check neighbors
                unsigned int* data = get_linklist_at_level(i, 0);
                int size = getListCount(data);
                tableint* neighbors = (tableint*)(data + 1);
                
                for(int j=0; j<size; ++j) {
                    tableint neighbor = neighbors[j];
                    tableint neighbor_root = assigned_root[neighbor];
                    
                    if(neighbor_root != -1) {
                        if(neighbor_root == assigned_root[i]) continue;

                        // Calculate distance to neighbor's root
                        memcpy(vec_root.data(), getDataByInternalId(neighbor_root), dim * sizeof(double));
                        dist_t d = fstdistfunc_(vec_i.data(), vec_root.data(), dist_func_param_);
                        
                        if(d < min_dist[i]) {
                            min_dist[i] = d;
                            assigned_root[i] = neighbor_root;
                            changed = true;
                        }
                    }
                }
            }
        }
        
        // Handle unassigned nodes (assign to nearest root found via brute force)
        std::vector<tableint> root_indices;
        for(size_t i=0; i<cur_element_count; ++i) {
            if(is_root[i]) root_indices.push_back(i);
        }

        for(size_t i=0; i<cur_element_count; ++i) {
             if(!isMarkedDeleted(i) && assigned_root[i] == -1) {
                 dist_t best_d = std::numeric_limits<dist_t>::max();
                 tableint best_r = -1;
                 memcpy(vec_i.data(), getDataByInternalId(i), dim * sizeof(double));
                 
                 for(tableint r : root_indices) {
                     memcpy(vec_root.data(), getDataByInternalId(r), dim * sizeof(double));
                     dist_t d = fstdistfunc_(vec_i.data(), vec_root.data(), dist_func_param_);
                     if(d < best_d) {
                         best_d = d;
                         best_r = r;
                     }
                 }
                 assigned_root[i] = best_r;
             }
        }

        // 3. Compress and Rebuild Memory
        std::vector<char> new_memory;
        new_memory.reserve(data_level0_memory_.size() / 2); 

        std::vector<size_t> new_start_positions(cur_element_count);
        
        {
             std::lock_guard<std::mutex> lock(cache_lock_);
             root_state_cache_.clear();
        }
        
        for (size_t i = 0; i < cur_element_count; ++i) {
            if (isMarkedDeleted(i)) {
                assigned_root[i] = i; 
            }
            
            size_t new_start = new_memory.size();
            new_start_positions[i] = new_start;
            
            tableint root = assigned_root[i];
            tableint prenode = (i == root) ? -1 : root;
            
            // --- Copy Header ---
            size_t prenode_size = sizeof(tableint);
            new_memory.resize(new_memory.size() + prenode_size);
            *((tableint*)(new_memory.data() + new_start)) = prenode;

            labeltype label = getExternalLabel(i);
            size_t label_size = sizeof(labeltype);
            new_memory.resize(new_memory.size() + label_size);
            *((labeltype*)(new_memory.data() + new_start + prenode_size)) = label;

            unsigned char* old_ll_ptr = (unsigned char*)get_linklist0(i);
            linklistsizeint size = *((linklistsizeint*)old_ll_ptr);
            
            size_t ll_header_size = sizeof(linklistsizeint);
            size_t neighbors_size = size * sizeof(tableint);
            
            new_memory.resize(new_memory.size() + ll_header_size + neighbors_size);
            
            *((linklistsizeint*)(new_memory.data() + new_start + prenode_size + label_size)) = size;
            
            memcpy(new_memory.data() + new_start + prenode_size + label_size + ll_header_size, 
                   old_ll_ptr + ll_header_size, 
                   neighbors_size);

            // --- Compress Data ---
            const double* my_data_ptr = (const double*)getDataByInternalId(i);
            
            // if (i == root) {
            //     std::vector<double> my_data_vec(my_data_ptr, my_data_ptr + dim);
            //     std::lock_guard<std::mutex> lock(cache_lock_);
            //     getOriginalData_cache_[i] = my_data_vec;
            // }
            
            std::vector<char> compressed_buffer;
            utils::MemoryStreamWriter writer(&compressed_buffer);
            std::vector<DeXORState> states(dim); 
            
            if (prenode != -1) {
                const double* root_data_ptr = (const double*)getDataByInternalId(root);
                std::vector<char> temp_buffer;
                utils::MemoryStreamWriter temp_writer(&temp_buffer);
                for(size_t k=0; k<dim; ++k) {
                    dexor_encode(root_data_ptr[k], states[k], temp_writer);
                }
            }
            
            for(size_t k=0; k<dim; ++k) {
                dexor_encode(my_data_ptr[k], states[k], writer);
            }
            writer.align();
            
            size_t data_len = compressed_buffer.size();
            new_memory.resize(new_memory.size() + data_len);
            memcpy(new_memory.data() + new_start + prenode_size + label_size + ll_header_size + neighbors_size,
                   compressed_buffer.data(),
                   data_len);
        }

        data_level0_memory_ = std::move(new_memory);
        level0_element_start_positions_ = std::move(new_start_positions);
        is_compacted_ = true;
        
        data_level0_memory_.shrink_to_fit();
    }

    void resizeIndex(size_t new_max_elements) {
        if (new_max_elements < cur_element_count)
            throw std::runtime_error("Cannot resize, max element is less than the current number of elements");

        visited_list_pool_.reset(new VisitedListPool(1, new_max_elements));

        element_levels_.resize(new_max_elements);
        level0_element_start_positions_.resize(new_max_elements);

        std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

        // Reallocate base layer
        // data_level0_memory_ is std::vector, no need to realloc manually.
        // We can reserve if we want.
        // data_level0_memory_.reserve(new_max_elements * size_data_per_element_);

        // Reallocate all other layers
        char ** linkLists_new = (char **) realloc(linkLists_, sizeof(void *) * new_max_elements);
        if (linkLists_new == nullptr)
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
        linkLists_ = linkLists_new;

        max_elements_ = new_max_elements;
    }

    size_t indexFileSize() const {
        size_t size = 0;
        size += sizeof(offsetLevel0_);
        size += sizeof(max_elements_);
        size += sizeof(cur_element_count);
        size += sizeof(size_data_per_element_);
        size += sizeof(label_offset_);
        size += sizeof(offsetData_);
        size += sizeof(maxlevel_);
        size += sizeof(enterpoint_node_);
        size += sizeof(maxM_);

        size += sizeof(maxM0_);
        size += sizeof(M_);
        size += sizeof(mult_);
        size += sizeof(ef_construction_);

        size += cur_element_count * sizeof(size_t);

        size += sizeof(size_t);
        size += data_level0_memory_.size();

        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            size += sizeof(linkListSize);
            size += linkListSize;
        }
        return size;
    }

    void saveIndex(const std::string &location) {
        // if (is_compacted_)
        //     throw std::runtime_error("Cannot save compacted index");
        std::ofstream output(location, std::ios::binary);
        std::streampos position;

        writeBinaryPOD(output, is_compacted_);
        writeBinaryPOD(output, offsetLevel0_);
        writeBinaryPOD(output, max_elements_);
        writeBinaryPOD(output, cur_element_count);
        writeBinaryPOD(output, size_data_per_element_);
        writeBinaryPOD(output, prenode_offset_);
        writeBinaryPOD(output, label_offset_);
        writeBinaryPOD(output, offsetData_);
        writeBinaryPOD(output, maxlevel_);
        writeBinaryPOD(output, enterpoint_node_);
        writeBinaryPOD(output, maxM_);

        writeBinaryPOD(output, maxM0_);
        writeBinaryPOD(output, M_);
        writeBinaryPOD(output, mult_);
        writeBinaryPOD(output, ef_construction_);

        output.write(reinterpret_cast<const char*>(level0_element_start_positions_.data()), cur_element_count * sizeof(size_t));

        size_t level0_memory_size = data_level0_memory_.size();
        writeBinaryPOD(output, level0_memory_size);
        output.write(data_level0_memory_.data(), level0_memory_size);

        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            writeBinaryPOD(output, linkListSize);
            if (linkListSize)
                output.write(linkLists_[i], linkListSize);
        }
        output.close();
    }


    void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i = 0) {
        std::ifstream input(location, std::ios::binary);

        if (!input.is_open())
            throw std::runtime_error("Cannot open file");

        clear();
        // get file size:
        input.seekg(0, input.end);
        std::streampos total_filesize = input.tellg();
        input.seekg(0, input.beg);

        readBinaryPOD(input, is_compacted_);
        readBinaryPOD(input, offsetLevel0_);
        readBinaryPOD(input, max_elements_);
        readBinaryPOD(input, cur_element_count);

        size_t max_elements = max_elements_i;
        if (max_elements < cur_element_count)
            max_elements = max_elements_;
        max_elements_ = max_elements;
        readBinaryPOD(input, size_data_per_element_);
        readBinaryPOD(input, prenode_offset_);
        readBinaryPOD(input, label_offset_);
        readBinaryPOD(input, offsetData_);
        readBinaryPOD(input, maxlevel_);
        readBinaryPOD(input, enterpoint_node_);

        readBinaryPOD(input, maxM_);
        readBinaryPOD(input, maxM0_);
        readBinaryPOD(input, M_);
        readBinaryPOD(input, mult_);
        readBinaryPOD(input, ef_construction_);

        level0_element_start_positions_.resize(max_elements_);
        input.read(reinterpret_cast<char*>(level0_element_start_positions_.data()), cur_element_count * sizeof(size_t));

        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();

        auto pos = input.tellg();

        /// Optional - check if index is ok:
        /*
        input.seekg(cur_element_count * size_data_per_element_, input.cur);
        for (size_t i = 0; i < cur_element_count; i++) {
            if (input.tellg() < 0 || input.tellg() >= total_filesize) {
                throw std::runtime_error("Index seems to be corrupted or unsupported");
            }

            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize != 0) {
                input.seekg(linkListSize, input.cur);
            }
        }

        // throw exception if it either corrupted or old index
        if (input.tellg() != total_filesize)
            throw std::runtime_error("Index seems to be corrupted or unsupported");

        input.clear();
        */
        /// Optional check end

        input.seekg(pos, input.beg);

        size_t level0_memory_size;
        readBinaryPOD(input, level0_memory_size);
        data_level0_memory_.resize(level0_memory_size);
        input.read(data_level0_memory_.data(), level0_memory_size);

        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        std::vector<std::mutex>(max_elements).swap(link_list_locks_);
        std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS).swap(label_op_locks_);

        visited_list_pool_.reset(new VisitedListPool(1, max_elements));

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
        element_levels_ = std::vector<int>(max_elements);
        revSize_ = 1.0 / mult_;
        ef_ = 10;
        for (size_t i = 0; i < cur_element_count; i++) {
            label_lookup_[getExternalLabel(i)] = i;
            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize == 0) {
                element_levels_[i] = 0;
                linkLists_[i] = nullptr;
            } else {
                element_levels_[i] = linkListSize / size_links_per_element_;
                linkLists_[i] = (char *) malloc(linkListSize);
                if (linkLists_[i] == nullptr)
                    throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                input.read(linkLists_[i], linkListSize);
            }
        }

        for (size_t i = 0; i < cur_element_count; i++) {
            if (isMarkedDeleted(i)) {
                num_deleted_ += 1;
                if (allow_replace_deleted_) deleted_elements.insert(i);
            }
        }

        input.close();

        // 加载cache
        if( cache_max_size_ > 0){
            root_state_cache_.reserve(cache_max_size_);
            loadCache();
        }

        return;
    }


    template<typename data_t>
    std::vector<data_t> getDataByLabel(labeltype label) const {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));
        
        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        // char* data_ptrv = getDataByInternalId(internalId);
        std::vector<double> vec = getOriginalDataByInternalId(internalId);
        
        std::vector<data_t> data;
        for(double v : vec) {
            data.push_back((data_t)v);
        }
        return data;
    }


    /*
    * Marks an element with the given label deleted, does NOT really change the current graph.
    */
    void markDelete(labeltype label) {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        markDeletedInternal(internalId);
    }


    /*
    * Uses the last 16 bits of the memory for the linked list size to store the mark,
    * whereas maxM0_ has to be limited to the lower 16 bits, however, still large enough in almost all cases.
    */
    void markDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count);
        if (!isMarkedDeleted(internalId)) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId))+2;
            *ll_cur |= DELETE_MARK;
            num_deleted_ += 1;
            if (allow_replace_deleted_) {
                std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.insert(internalId);
            }
        } else {
            throw std::runtime_error("The requested to delete element is already deleted");
        }
    }


    /*
    * Removes the deleted mark of the node, does NOT really change the current graph.
    * 
    * Note: the method is not safe to use when replacement of deleted elements is enabled,
    *  because elements marked as deleted can be completely removed by addPoint
    */
    void unmarkDelete(labeltype label) {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        unmarkDeletedInternal(internalId);
    }



    /*
    * Remove the deleted mark of the node.
    */
    void unmarkDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count);
        if (isMarkedDeleted(internalId)) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
            *ll_cur &= ~DELETE_MARK;
            num_deleted_ -= 1;
            if (allow_replace_deleted_) {
                std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.erase(internalId);
            }
        } else {
            throw std::runtime_error("The requested to undelete element is not deleted");
        }
    }


    /*
    * Checks the first 16 bits of the memory to see if the element is marked deleted.
    */
    bool isMarkedDeleted(tableint internalId) const {
        unsigned char *ll_cur = ((unsigned char*)get_linklist0(internalId)) + 2;
        return *ll_cur & DELETE_MARK;
    }


    unsigned short int getListCount(linklistsizeint * ptr) const {
        return *((unsigned short int *)ptr);
    }


    void setListCount(linklistsizeint * ptr, unsigned short int size) const {
        *((unsigned short int*)(ptr))=*((unsigned short int *)&size);
    }


    /*
    * Adds point. Updates the point if it is already in the index.
    * If replacement of deleted elements is enabled: replaces previously deleted point if any, updating it with new point
    */
    void addPoint(const void *data_point, labeltype label, bool replace_deleted = false) {
        if ((allow_replace_deleted_ == false) && (replace_deleted == true)) {
            throw std::runtime_error("Replacement of deleted elements is disabled in constructor");
        }

        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));
        if (!replace_deleted) {
            addPoint(data_point, label, -1);
            return;
        }
        // check if there is vacant place
        tableint internal_id_replaced;
        std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
        bool is_vacant_place = !deleted_elements.empty();
        if (is_vacant_place) {
            internal_id_replaced = *deleted_elements.begin();
            deleted_elements.erase(internal_id_replaced);
        }
        lock_deleted_elements.unlock();

        // if there is no vacant place then add or update point
        // else add point to vacant place
        if (!is_vacant_place) {
            addPoint(data_point, label, -1);
        } else {
            // we assume that there are no concurrent operations on deleted element
            labeltype label_replaced = getExternalLabel(internal_id_replaced);
            setExternalLabel(internal_id_replaced, label);

            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            label_lookup_.erase(label_replaced);
            label_lookup_[label] = internal_id_replaced;
            lock_table.unlock();

            unmarkDeletedInternal(internal_id_replaced);
            updatePoint(data_point, internal_id_replaced, 1.0);
        }
    }


    void updatePoint(const void *dataPoint, tableint internalId, float updateNeighborProbability) {
        if (is_compacted_)
            throw std::runtime_error("Cannot update point in compacted index");

        // update the feature vector associated with existing point with new vector
        // memcpy(getDataByInternalId(internalId), dataPoint, data_size_);

        size_t dim = data_size_ / sizeof(double);
        tableint prenode = getPrenodeId(internalId);
        
        std::vector<double> prenode_data(dim, 0.0);
        if (prenode != (tableint)-1) {
            prenode_data = getOriginalDataByInternalId(prenode);
        }
        
        std::vector<char> compressed_buffer;
        utils::MemoryStreamWriter writer(&compressed_buffer);
        
        std::vector<DeXORState> states(dim);
        
        if (prenode != (tableint)-1) {
            std::vector<char> dummy_buffer;
            utils::MemoryStreamWriter dummy_writer(&dummy_buffer);
            for(size_t i=0; i<dim; ++i) {
                dexor_encode(prenode_data[i], states[i], dummy_writer);
            }
            dummy_writer.align();
        }
        
        const double* data_arr = (const double*)dataPoint;
        for(size_t i=0; i<dim; ++i) {
            dexor_encode(data_arr[i], states[i], writer);
        }
        writer.align();
        
        size_t old_pos = level0_element_start_positions_[internalId];
        size_t new_pos = data_level0_memory_.size();
        
        size_t total_size = offsetData_ + compressed_buffer.size();
        data_level0_memory_.resize(new_pos + total_size);
        
        memcpy(data_level0_memory_.data() + new_pos, data_level0_memory_.data() + old_pos, offsetData_);
        memcpy(data_level0_memory_.data() + new_pos + offsetData_, compressed_buffer.data(), compressed_buffer.size());
        
        level0_element_start_positions_[internalId] = new_pos;

        int maxLevelCopy = maxlevel_;
        tableint entryPointCopy = enterpoint_node_;
        // If point to be updated is entry point and graph just contains single element then just return.
        if (entryPointCopy == internalId && cur_element_count == 1)
            return;

        int elemLevel = element_levels_[internalId];
        std::uniform_real_distribution<float> distribution(0.0, 1.0);
        for (int layer = 0; layer <= elemLevel; layer++) {
            std::unordered_set<tableint> sCand;
            std::unordered_set<tableint> sNeigh;
            std::vector<tableint> listOneHop = getConnectionsWithLock(internalId, layer);
            if (listOneHop.size() == 0)
                continue;

            sCand.insert(internalId);

            for (auto&& elOneHop : listOneHop) {
                sCand.insert(elOneHop);

                if (distribution(update_probability_generator_) > updateNeighborProbability)
                    continue;

                sNeigh.insert(elOneHop);

                std::vector<tableint> listTwoHop = getConnectionsWithLock(elOneHop, layer);
                for (auto&& elTwoHop : listTwoHop) {
                    sCand.insert(elTwoHop);
                }
            }

            for (auto&& neigh : sNeigh) {
                // if (neigh == internalId)
                //     continue;

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                size_t size = sCand.find(neigh) == sCand.end() ? sCand.size() : sCand.size() - 1;  // sCand guaranteed to have size >= 1
                size_t elementsToKeep = std::min(ef_construction_, size);
                for (auto&& cand : sCand) {
                    if (cand == neigh)
                        continue;

                    std::vector<double> vec_neigh = getOriginalDataByInternalId(neigh);
                    std::vector<double> vec_cand = getOriginalDataByInternalId(cand);
                    dist_t distance = fstdistfunc_(vec_neigh.data(), vec_cand.data(), dist_func_param_);
                    if (candidates.size() < elementsToKeep) {
                        candidates.emplace(distance, cand);
                    } else {
                        if (distance < candidates.top().first) {
                            candidates.pop();
                            candidates.emplace(distance, cand);
                        }
                    }
                }

                // Retrieve neighbours using heuristic and set connections.
                getNeighborsByHeuristic2(candidates, layer == 0 ? maxM0_ : maxM_);

                {
                    std::unique_lock <std::mutex> lock(link_list_locks_[neigh]);
                    linklistsizeint *ll_cur;
                    ll_cur = get_linklist_at_level(neigh, layer);
                    size_t candSize = candidates.size();
                    setListCount(ll_cur, candSize);
                    tableint *data = (tableint *) (ll_cur + 1);
                    for (size_t idx = 0; idx < candSize; idx++) {
                        data[idx] = candidates.top().second;
                        candidates.pop();
                    }
                }
            }
        }

        repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy);
    }


    void repairConnectionsForUpdate(
        const void *dataPoint,
        tableint entryPointInternalId,
        tableint dataPointInternalId,
        int dataPointLevel,
        int maxLevel) {
        tableint currObj = entryPointInternalId;
        if (dataPointLevel < maxLevel) {
            std::vector<double> vec_curr = getOriginalDataByInternalId(currObj);
            dist_t curdist = fstdistfunc_(dataPoint, vec_curr.data(), dist_func_param_);
            for (int level = maxLevel; level > dataPointLevel; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int *data;
                    std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                    data = get_linklist_at_level(currObj, level);
                    int size = getListCount(data);
                    tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
#endif
                    for (int i = 0; i < size; i++) {
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);
#endif
                        tableint cand = datal[i];
                        std::vector<double> vec_cand = getOriginalDataByInternalId(cand);
                        dist_t d = fstdistfunc_(dataPoint, vec_cand.data(), dist_func_param_);
                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }
        }

        if (dataPointLevel > maxLevel)
            throw std::runtime_error("Level of item to be updated cannot be bigger than max level");

        for (int level = dataPointLevel; level >= 0; level--) {
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> topCandidates = searchBaseLayer(
                    currObj, dataPoint, level);

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> filteredTopCandidates;
            while (topCandidates.size() > 0) {
                if (topCandidates.top().second != dataPointInternalId)
                    filteredTopCandidates.push(topCandidates.top());

                topCandidates.pop();
            }

            // Since element_levels_ is being used to get `dataPointLevel`, there could be cases where `topCandidates` could just contains entry point itself.
            // To prevent self loops, the `topCandidates` is filtered and thus can be empty.
            if (filteredTopCandidates.size() > 0) {
                bool epDeleted = isMarkedDeleted(entryPointInternalId);
                if (epDeleted) {
                    std::vector<double> vec_ep = getOriginalDataByInternalId(entryPointInternalId);
                    filteredTopCandidates.emplace(fstdistfunc_(dataPoint, vec_ep.data(), dist_func_param_), entryPointInternalId);
                    if (filteredTopCandidates.size() > ef_construction_)
                        filteredTopCandidates.pop();
                }

                currObj = mutuallyConnectNewElement(dataPoint, dataPointInternalId, filteredTopCandidates, level, true);
            }
        }
    }


    std::vector<tableint> getConnectionsWithLock(tableint internalId, int level) {
        std::unique_lock <std::mutex> lock(link_list_locks_[internalId]);
        unsigned int *data = get_linklist_at_level(internalId, level);
        int size = getListCount(data);
        std::vector<tableint> result(size);
        tableint *ll = (tableint *) (data + 1);
        memcpy(result.data(), ll, size * sizeof(tableint));
        return result;
    }


    // label是外部的id。存储在label_lookup_中的是外部id到内部id的映射
    tableint addPoint(const void *data_point, labeltype label, int level) {
        if (is_compacted_)
            throw std::runtime_error("Cannot add point in compacted index");
        tableint cur_c = 0;
        {
            // Checking if the element with the same label already exists
            // if so, updating it *instead* of creating a new element.
            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search != label_lookup_.end()) {
                tableint existingInternalId = search->second;
                if (allow_replace_deleted_) {
                    if (isMarkedDeleted(existingInternalId)) {
                        throw std::runtime_error("Can't use addPoint to update deleted elements if replacement of deleted elements is enabled.");
                    }
                }
                lock_table.unlock();

                if (isMarkedDeleted(existingInternalId)) {
                    unmarkDeletedInternal(existingInternalId);
                }
                // 更新原本的数据
                updatePoint(data_point, existingInternalId, 1.0);

                return existingInternalId;
            }

            // 如果当前元素数量已经达到上限，抛出异常
            if (cur_element_count >= max_elements_) {
                throw std::runtime_error("The number of elements exceeds the specified limit");
            }

            cur_c = cur_element_count;
            cur_element_count++;
            label_lookup_[label] = cur_c;
        }

        std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);
        // 确定新元素所在的层数，curLevel按照几何分布确定
        int curlevel = getRandomLevel(mult_);
        if (level > 0)
            curlevel = level;

        element_levels_[cur_c] = curlevel;

        std::unique_lock <std::mutex> templock(global);
        int maxlevelcopy = maxlevel_;
        if (curlevel <= maxlevelcopy)
            templock.unlock();

        // currObj: 当前距离待插节点最近的节点
        tableint currObj = enterpoint_node_;
        tableint enterpoint_copy = enterpoint_node_;

        if ((signed)currObj != -1) {
            if (curlevel < maxlevelcopy) {
                std::vector<double> vec_curr = getOriginalDataByInternalId(currObj);
                dist_t curdist = fstdistfunc_(data_point, vec_curr.data(), dist_func_param_);
                for (int level = maxlevelcopy; level > curlevel; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        unsigned int *data;
                        std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                        // 获取currObj的所有邻居neighbors
                        data = get_linklist(currObj, level);
                        int size = getListCount(data);

                        tableint *datal = (tableint *) (data + 1);
                        for (int i = 0; i < size; i++) {
                            tableint cand = datal[i];
                            if (cand < 0 || cand > max_elements_)
                                throw std::runtime_error("cand error");
                            std::vector<double> vec_cand = getOriginalDataByInternalId(cand);
                            dist_t d = fstdistfunc_(data_point, vec_cand.data(), dist_func_param_);
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }
        }

        // ============================================================
        // New Logic: Search -> Compress -> Connect
        // ============================================================
        
        std::vector<std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>> candidates_cache;
        int start_level = std::min(curlevel, maxlevelcopy);
        
        if ((signed)currObj != -1) {
            candidates_cache.resize(start_level + 1);
            bool epDeleted = isMarkedDeleted(enterpoint_copy);
            
            // Phase 1: Search and Cache
            for (int level = start_level; level >= 0; level--) {
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                        currObj, data_point, level);
                
                if (epDeleted) {
                    std::vector<double> vec_ep = getOriginalDataByInternalId(enterpoint_copy);
                    top_candidates.emplace(fstdistfunc_(data_point, vec_ep.data(), dist_func_param_), enterpoint_copy);
                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();
                }
                
                candidates_cache[level] = top_candidates;
                
                // Update currObj to the closest candidate for the next iteration
                if (!top_candidates.empty()) {
                    // top_candidates is max-heap (furthest on top). 
                    // We need to iterate to find closest.
                    // Since we need to keep top_candidates for Phase 3, we should copy it?
                    // Or just traverse it non-destructively? 
                    // Priority queue doesn't support iteration.
                    // We have to copy.
                    
                    auto temp_queue = top_candidates;
                    dist_t min_dist = std::numeric_limits<dist_t>::max();
                    tableint closest = -1;
                    while(!temp_queue.empty()){
                        auto p = temp_queue.top();
                        temp_queue.pop();
                        if(p.first < min_dist){
                            min_dist = p.first;
                            closest = p.second;
                        }
                    }
                    if(closest != -1) currObj = closest;
                }
            }
        }

        // No compression during build phase
        std::vector<char> compressed_buffer;
        tableint prenode = -1;
        
        size_t dim = data_size_ / sizeof(double);
        compressed_buffer.resize(dim * sizeof(double));
        memcpy(compressed_buffer.data(), data_point, dim * sizeof(double));
        
        size_t start_pos = data_level0_memory_.size();
        level0_element_start_positions_[cur_c] = start_pos;
        
        size_t total_size = offsetData_ + compressed_buffer.size();
        data_level0_memory_.resize(start_pos + total_size);
        
        memset(data_level0_memory_.data() + start_pos, 0, size_links_level0_);
        setPrenodeId(cur_c, prenode);
        setExternalLabel(cur_c, label);
        memcpy(data_level0_memory_.data() + start_pos + offsetData_, compressed_buffer.data(), compressed_buffer.size());

        // 输出当前添加的点的信息
        // std::cout << "Adding point " << cur_c << " at level " << curlevel << std::endl;
        // std::cout << "Data point: ";
        // std::vector<double> cur_data_vec(dim, 0.0);
        // cur_data_vec = getOriginalDataByInternalId(cur_c);
        // for(double v : cur_data_vec) {
        //     std::cout << v << " ";
        // }
        // std::cout << std::endl;

        if (curlevel) {
            // 为非level0的层分配内存
            linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);
            if (linkLists_[cur_c] == nullptr)
                throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
            
            // 初始化link list为空
            memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
        }

        if ((signed)enterpoint_copy != -1) {
            // std::cout << "Entering level <= maxlevel loop for point " << cur_c << " at level " << curlevel << std::endl;

            for (int level = start_level; level >= 0; level--) {
                if (level > maxlevelcopy || level < 0)  // possible? 这不可能吧？
                    throw std::runtime_error("Level error");

                // Retrieve candidates from cache
                auto& top_candidates = candidates_cache[level];

                // 在level层建立data_point与top_candidates中每一个元素的连接
                mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
            }

            // std::cout << "Point " << cur_c << " connected up to level " << curlevel << std::endl;
        } else {
            // Do nothing for the first element
            enterpoint_node_ = 0;
            maxlevel_ = curlevel;
        }

        // Releasing lock for the maximum level
        if (curlevel > maxlevelcopy) {
            enterpoint_node_ = cur_c;
            maxlevel_ = curlevel;
        }
        return cur_c;
    }


    //knn搜索 ---论文中的alg.5
    std::priority_queue<std::pair<dist_t, labeltype >>
    searchKnn(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const {
        std::priority_queue<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;

        // currObj和curdist分别记录距离data point最近的点和距离
        tableint currObj = enterpoint_node_;
        std::vector<double> vec_ep = getOriginalDataByInternalId(enterpoint_node_);
        dist_t curdist = fstdistfunc_(query_data, vec_ep.data(), dist_func_param_);
        // 在层L...1之间
        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                // 首先没有变化，表示在同一层中搜索
                changed = false;
                unsigned int *data;
                // 获得currObj的连接数，也就是邻居
                data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations+=size;

                tableint *datal = (tableint *) (data + 1);
                // 对于currObj的每一个邻居，计算它与data point的距离，并及时更新currObj和currdist
                std::vector<dist_t> dists(size);
                #pragma omp parallel for
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < max_elements_) {
                        std::vector<double> vec_cand = getOriginalDataByInternalId(cand);
                        dists[i] = fstdistfunc_(query_data, vec_cand.data(), dist_func_param_);
                    }
                }

                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    
                    if (dists[i] < curdist) {
                        curdist = dists[i];
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        // 目前已获得第一层与query最近的元素currObj
        // 在第零层获取currObj邻居中距离query最近的max(k, ef)个近邻，也就是动态列表top_candidates-----论文中是W
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        bool bare_bone_search = !num_deleted_ && !isIdAllowed;
        if (bare_bone_search) {
            top_candidates = searchBaseLayerST<true>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed);
        } else {
            top_candidates = searchBaseLayerST<false>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed);
        }
        // top_candidates修建为k个
        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        // 将结果放入result中返回
        while (top_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
    }


    std::vector<std::pair<dist_t, labeltype >>
    searchStopConditionClosest(
        const void *query_data,
        BaseSearchStopCondition<dist_t>& stop_condition,
        BaseFilterFunctor* isIdAllowed = nullptr) const {
        std::vector<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations+=size;

                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        top_candidates = searchBaseLayerST<false>(currObj, query_data, 0, isIdAllowed, &stop_condition);

        size_t sz = top_candidates.size();
        result.resize(sz);
        while (!top_candidates.empty()) {
            result[--sz] = top_candidates.top();
            top_candidates.pop();
        }

        stop_condition.filter_results(result);

        return result;
    }


    void checkIntegrity() {
        int connections_checked = 0;
        std::vector <int > inbound_connections_num(cur_element_count, 0);
        for (int i = 0; i < cur_element_count; i++) {
            for (int l = 0; l <= element_levels_[i]; l++) {
                linklistsizeint *ll_cur = get_linklist_at_level(i, l);
                int size = getListCount(ll_cur);
                tableint *data = (tableint *) (ll_cur + 1);
                std::unordered_set<tableint> s;
                for (int j = 0; j < size; j++) {
                    assert(data[j] < cur_element_count);
                    assert(data[j] != i);
                    inbound_connections_num[data[j]]++;
                    s.insert(data[j]);
                    connections_checked++;
                }
                assert(s.size() == size);
            }
        }
        if (cur_element_count > 1) {
            int min1 = inbound_connections_num[0], max1 = inbound_connections_num[0];
            for (int i=0; i < cur_element_count; i++) {
                assert(inbound_connections_num[i] > 0);
                min1 = std::min(inbound_connections_num[i], min1);
                max1 = std::max(inbound_connections_num[i], max1);
            }
            std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
        }
        std::cout << "integrity ok, checked " << connections_checked << " connections\n";
    }

    void printCompressionTree(const std::string& filename = "storage/encoding_order.txt") {
        std::ofstream out(filename);
        if (!out.is_open()) {
            std::cerr << "Failed to open file for writing: " << filename << std::endl;
            return;
        }

        std::unordered_map<tableint, std::vector<tableint>> adj;
        std::vector<tableint> roots;

        for (tableint i = 0; i < cur_element_count; i++) {
            tableint pre = getPrenodeId(i);
            if (pre == (tableint)-1) {
                roots.push_back(i);
            } else {
                adj[pre].push_back(i);
            }
        }

        out << "\n=== Compression Tree Structure (Prenode -> Children) ===\n";
        out << "Total elements: " << cur_element_count << "\n";
        out << "Roots count: " << roots.size() << "\n";

        // Use a stack for iterative DFS to avoid recursion depth issues, 
        // or just simple recursion if depth is not expected to be huge.
        // Given it's a compression chain, it might be deep.
        // But for visualization, recursion is simpler. 
        
        std::function<void(tableint, int)> printNode = 
            [&](tableint node, int depth) {
            for (int i = 0; i < depth; ++i) out << "  ";
            out << "|- " << node << " (Label: " << getExternalLabel(node) << ")";
            
            // Optional: Print data size or other info
            // size_t start = level0_element_start_positions_[node];
            // size_t end = (node + 1 < cur_element_count) ? level0_element_start_positions_[node+1] : data_level0_memory_.size();
            // out << " [Size: " << (end - start) << "]";
            
            out << "\n";

            if (adj.count(node)) {
                for (tableint child : adj[node]) {
                    printNode(child, depth + 1);
                }
            }
        };

        for (tableint root : roots) {
            printNode(root, 0);
        }
        out << "====================================================\n";
        out.close();
        std::cout << "Compression tree structure written to " << filename << std::endl;
    }

    int getCompressedDataSize() const {
        // 返回压缩的所有element的data部分的总大小
        size_t total_size = 0;
        for (tableint i = 0; i < cur_element_count; i++) {
            size_t start;
            if (is_compacted_) {
                size_t linklist_size_offset = sizeof(tableint) + sizeof(labeltype);
                unsigned short int size = *((unsigned short int*)(data_level0_memory_.data() + level0_element_start_positions_[i] + linklist_size_offset));
                size_t data_offset = linklist_size_offset + sizeof(linklistsizeint) + size * sizeof(tableint);
                start = level0_element_start_positions_[i] + data_offset;
            } else {
                start = level0_element_start_positions_[i] + offsetData_;
            }

            size_t end;
            if (i + 1 < cur_element_count && level0_element_start_positions_[i+1] > 0) {
                end = level0_element_start_positions_[i+1];
            } else {
                end = data_level0_memory_.size();
            }
            
            if (end > start) {
                total_size += (end - start);
            }
        }
        return (int)total_size;
    }

    size_t getIndexSize() const {
        size_t total_size = 0;
        // total_size += sizeof(*this);
        total_size += data_level0_memory_.size();
        total_size += level0_element_start_positions_.size() * sizeof(size_t);
        total_size += element_levels_.size() * sizeof(int);
        
        // linkLists_ 指针数组的大小
        total_size += max_elements_ * sizeof(void*);

        // linkLists_ 实际指向的内存大小 (仅统计 level > 0 的元素)
        for (size_t i = 0; i < cur_element_count; i++) {
            int level = element_levels_[i];
            if (level > 0) {
                total_size += size_links_per_element_ * level;
            }
        }
        
        return total_size;
    }

    // 获取getOriginalData的时间
    long getTotalTimeGetOriginalData() const {
        return getOriginalData_time;
    }

    void resetTotalTimeGetOriginalData() {
        getOriginalData_time = 0;
    }

    // 根据internal id获取该节点在level0的linkLists中元素的数量
    int getLevel0LinkListSize(tableint internalId) {
        std::unique_lock <std::mutex> lock(link_list_locks_[internalId]);
        linklistsizeint *ll_cur = get_linklist_at_level(internalId, 0);
        int size = getListCount(ll_cur);
        return size;
    }

    // 获取internal id的encoding chain长度
    int getEncodingChainLength(tableint internalId) {
        int length = 0;
        tableint curr = getPrenodeId(internalId);
        while (curr != (tableint)-1) {
            length++;
            curr = getPrenodeId(curr);
            if (length > 1000) break; 
        }
        return length;
    }

    // 获取internal id的encoding chain
    std::vector<tableint> getEncodingChain(tableint internalId) {
        std::vector<tableint> chain;
        tableint curr = internalId;
        while (curr != (tableint)-1) {
            chain.push_back(curr);
            curr = getPrenodeId(curr);
            if (chain.size() > 1000) break; 
        }
        return chain;
    }

    void checkPreNodeInNeighbors() {
        for (tableint i = 0; i < cur_element_count; i++) {
            tableint prenode = getPrenodeId(i);
            if (prenode == (tableint)-1) continue;
            bool found = false;
            linklistsizeint *ll_cur = get_linklist_at_level(i, 0);
            int size = getListCount(ll_cur);
            tableint *data = (tableint *) (ll_cur + 1);
            for (int j = 0; j < size; j++) {
                if (data[j] == prenode) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                std::cout << "Prenode " << prenode << " of node " << i << " is: " << prenode << ", not found in its neighbors!\n";
                std::cout << "Neighbors are: ";
                for (int j = 0; j < size; j++) {
                    std::cout << data[j] << " ";
                }
                std::cout << std::endl;
            }
        }
    }

    int getCachePopCount() const {
        return cache_pop_count_;
    }

    int getGetOriginalDataCallCount() const {
        return getOriginalData_call_count_;
    }

    std::vector<size_t> getLevel0ElementStartPositions() const {
        return level0_element_start_positions_;
    }

    std::vector<char> getDataLevel0Memory() const {
        return data_level0_memory_;
    }

    // 获取decoding消耗的时间
    long getTotalTimeDecoding() const {
        return decoding_time;
    }

    // 获取decoding调用次数
    int getDecodingCallCount() const {
        return decoding_count;
    }

    // 加载cache，根据max_cache_size设置cache大小，cache中固定存储热门数据，不使用lru
    void loadCache() {
        if (cache_max_size_ > 0) {
            // 这里实现加载热门数据到cache的逻辑
            // 加載作为root的点为热门点
            std::vector<tableint> root_nodes;
            for (tableint i = 0; i < cur_element_count; i++) {
                tableint prenode = getPrenodeId(i);
                if (prenode == (tableint)-1) {
                    root_nodes.push_back(i);
                }
            }
            size_t to_load = std::min((size_t)cache_max_size_, root_nodes.size());
            
            size_t dim = *((size_t *) dist_func_param_);
            utils::MemoryBlockStreamReader reader((const unsigned char*)data_level0_memory_.data());

            for (size_t i = 0; i < to_load; i++) {
                tableint id = root_nodes[i];
                
                // Decode Root to get state
                std::vector<DeXORState> states(dim);
                
                size_t start, end;
                size_t linklist_size_offset = sizeof(tableint) + sizeof(labeltype);
                unsigned short int size = *((unsigned short int*)(data_level0_memory_.data() + level0_element_start_positions_[id] + linklist_size_offset));
                size_t data_offset = linklist_size_offset + sizeof(linklistsizeint) + size * sizeof(tableint);
                start = level0_element_start_positions_[id] + data_offset;
                
                if (id + 1 < cur_element_count && level0_element_start_positions_[id+1] > 0) {
                    end = level0_element_start_positions_[id+1];
                } else {
                    end = data_level0_memory_.size();
                }
                
                reader.resetBuffer((const unsigned char*)(data_level0_memory_.data() + start), end - start);
                
                for(size_t k=0; k<dim; ++k) {
                    dexor_decode(states[k], reader);
                }
                
                {
                    // std::lock_guard<std::mutex> lock(cache_lock_);
                    root_state_cache_[id] = states;
                }
            }

            std::cout << "Cache loaded with " << to_load << " root states." << std::endl;
        }
    }
};
}  // namespace hnswlib
