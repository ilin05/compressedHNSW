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
#include "../examples/utils/memory_block_stream_reader.h"
#include "../examples/utils/memory_stream_writer.h"
#include "../encoding_algorithms/algorithms_manager.h"

namespace hnswlib {
typedef unsigned int tableint;
typedef unsigned int linklistsizeint;

template<typename dist_t>
class HierarchicalNSWCW : public AlgorithmInterface<dist_t> {
 public:
    static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;
    static const unsigned char DELETE_MARK = 0x01;

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


    HierarchicalNSWCW(SpaceInterface<dist_t> *s) {
    }


    HierarchicalNSWCW(
        SpaceInterface<dist_t> *s,
        const std::string &location,
        bool nmslib = false,
        size_t max_elements = 0,
        bool allow_replace_deleted = false)
        : allow_replace_deleted_(allow_replace_deleted) {
        loadIndex(location, s, max_elements);
    }


    HierarchicalNSWCW(
        SpaceInterface<dist_t> *s,
        size_t max_elements,
        const std::string &encoding_algorithm_name = "DeXOR",
        size_t M = 16,
        size_t ef_construction = 200,
        size_t random_seed = 100,
        bool allow_replace_deleted = false)
        : label_op_locks_(MAX_LABEL_OPERATION_LOCKS),
            link_list_locks_(max_elements),
            level0_element_start_positions_(max_elements),
            element_levels_(max_elements),
            allow_replace_deleted_(allow_replace_deleted),
            encoding_algorithm_name_(encoding_algorithm_name) {
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
            throw std::runtime_error("Not enough memory: HierarchicalNSWCW failed to allocate linklists");
        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
        mult_ = 1 / log(1.0 * M_);
        revSize_ = 1.0 / mult_;
    }


    ~HierarchicalNSWCW() {
        clear();
    }

    void clear() {
        data_level0_memory_.clear();
        data_level0_memory_.shrink_to_fit();
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
        memcpy(&return_label, (data_level0_memory_.data() + level0_element_start_positions_[internal_id] + label_offset_), sizeof(labeltype));
        return return_label;
    }


    inline void setExternalLabel(tableint internal_id, labeltype label) const {
        memcpy((char*)(data_level0_memory_.data() + level0_element_start_positions_[internal_id] + label_offset_), &label, sizeof(labeltype));
    }


    inline labeltype *getExternalLabeLp(tableint internal_id) const {
        return (labeltype *) (data_level0_memory_.data() + level0_element_start_positions_[internal_id] + label_offset_);
    }

    inline tableint getPrenodeId(tableint internal_id) const {
        tableint prenode;
        memcpy(&prenode, (data_level0_memory_.data() + level0_element_start_positions_[internal_id] + prenode_offset_), sizeof(tableint));
        return prenode;
    }

    inline void setPrenodeId(tableint internal_id, tableint prenode) {
        memcpy((char*)(data_level0_memory_.data() + level0_element_start_positions_[internal_id] + prenode_offset_), &prenode, sizeof(tableint));
    }

    // 数据存储在第0层，每个element的大小都是固定的，size_data_per_element。可以通过HNSW内部的id随机读取数据
    // 如果要使用差分编码压缩data，将无法通过简单的随机读取获取数据。或许需要页表之类的结构进行索引；同时，data需要解压缩。可以在每个element的开头记录上一个data的internal_id，接着回溯到第0个data，然后依次解压缩
    inline char *getDataByInternalId(tableint internal_id) const {
        return (char*)(data_level0_memory_.data() + level0_element_start_positions_[internal_id] + offsetData_);
    }

    inline char *getDataByInternalId(tableint internal_id, char* data_level0_memory) const {
        return (char*)(data_level0_memory + level0_element_start_positions_[internal_id] + offsetData_);
    }

    std::vector<double> getOriginalDataByInternalId(tableint internal_id) const {
        auto start_time = std::chrono::high_resolution_clock::now();
        size_t dim = *((size_t *) dist_func_param_);
        std::vector<double> result(dim);
        
        std::vector<tableint> chain;
        tableint curr = internal_id;
        while (curr != (tableint)-1) {
            chain.push_back(curr);
            curr = getPrenodeId(curr);
            if (chain.size() > 1000) break; 
        }

        // std::cout << "Decompression chain length: " << chain.size() << std::endl;
        
        std::reverse(chain.begin(), chain.end());
        
        auto reader = std::make_shared<utils::MemoryBlockStreamReader>((const unsigned char*)data_level0_memory_.data());
        
        std::vector<std::unique_ptr<encoding_algorithm::Decoder>> decoders;
        for(size_t i=0; i<dim; ++i) {
            decoders.push_back(encoding_algorithm::AlgorithmsManager::getDecoder("Double", encoding_algorithm_name_, reader));
        }
        
        for (tableint id : chain) {
            // std::cout << "Decoding id: " << id << "cur_element_count" << cur_element_count << std::endl;
            size_t start = level0_element_start_positions_[id] + offsetData_;
            size_t end;
            if (id + 1 < cur_element_count && level0_element_start_positions_[id+1] > 0) {
                end = level0_element_start_positions_[id+1];
            } else {
                end = data_level0_memory_.size();
            }

            // std::cout << "Data range in memory: " << start << " to " << end << std::endl;
            
            reader->resetBuffer((const unsigned char*)(data_level0_memory_.data() + start), end - start);
            
            for(size_t i=0; i<dim; ++i) {
                // std::cout << "Decoding dimension: " << i << std::endl;
                result[i] = decoders[i]->decodeDouble();
            }
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        getOriginalData_time += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
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
            char* ep_data = getDataByInternalId(ep_id);
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
                _mm_prefetch(data_level0_memory_.data() + level0_element_start_positions_[*(data + 1)] + offsetData_, _MM_HINT_T0);
                _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
            }
#endif

            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                if (j < size) {
                    _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(data_level0_memory_.data() + level0_element_start_positions_[*(data + j + 1)] + offsetData_,
                                    _MM_HINT_T0);  ////////////
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
                            _mm_prefetch(data_level0_memory_.data() + level0_element_start_positions_[candidate_set.top().second] +
                                            offsetLevel0_,  ///////////
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
        return (linklistsizeint *) (data_level0_memory_.data() + level0_element_start_positions_[internal_id] + offsetLevel0_);
    }


    linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) const {
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

        size += cur_element_count * size_data_per_element_;

        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            size += sizeof(linkListSize);
            size += linkListSize;
        }
        return size;
    }

    void saveIndex(const std::string &location) {
        std::ofstream output(location, std::ios::binary);
        std::streampos position;

        writeBinaryPOD(output, offsetLevel0_);
        writeBinaryPOD(output, max_elements_);
        writeBinaryPOD(output, cur_element_count);
        writeBinaryPOD(output, size_data_per_element_);
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

        readBinaryPOD(input, offsetLevel0_);
        readBinaryPOD(input, max_elements_);
        readBinaryPOD(input, cur_element_count);

        size_t max_elements = max_elements_i;
        if (max_elements < cur_element_count)
            max_elements = max_elements_;
        max_elements_ = max_elements;
        readBinaryPOD(input, size_data_per_element_);
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
        level0_element_start_positions_ = std::vector<size_t>(max_elements);
        revSize_ = 1.0 / mult_;
        ef_ = 10;
        for (size_t i = 0; i < cur_element_count; i++) {
            // level0_element_start_positions_[i] = i * size_data_per_element_;
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
        // update the feature vector associated with existing point with new vector
        // memcpy(getDataByInternalId(internalId), dataPoint, data_size_);

        size_t dim = data_size_ / sizeof(double);
        tableint prenode = getPrenodeId(internalId);
        
        std::vector<double> prenode_data(dim, 0.0);
        if (prenode != (tableint)-1) {
            prenode_data = getOriginalDataByInternalId(prenode);
        }
        
        std::vector<char> compressed_buffer;
        auto writer = std::make_shared<utils::MemoryStreamWriter>(&compressed_buffer);
        
        std::vector<std::unique_ptr<encoding_algorithm::Encoder>> encoders;
        for(size_t i=0; i<dim; ++i) {
            encoders.push_back(encoding_algorithm::AlgorithmsManager::getEncoder("Double", encoding_algorithm_name_, writer));
        }
        
        if (prenode != (tableint)-1) {
            std::vector<char> dummy_buffer;
            writer->setBuffer(&dummy_buffer);
            for(size_t i=0; i<dim; ++i) {
                encoders[i]->encode(prenode_data[i]);
            }
            writer->align();
            writer->setBuffer(&compressed_buffer);
        }
        
        const double* data_arr = (const double*)dataPoint;
        for(size_t i=0; i<dim; ++i) {
            encoders[i]->encode(data_arr[i]);
        }
        writer->align();
        
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

        // Compression Logic
        size_t dim = data_size_ / sizeof(double);
        tableint prenode = -1;
        if ((signed)currObj != -1) {
            prenode = currObj; 
        }
        
        // Build chain from prenode to root
        std::vector<tableint> chain;
        tableint curr = prenode;
        while (curr != (tableint)-1) {
            chain.push_back(curr);
            curr = getPrenodeId(curr);
            if (chain.size() > 1000) break; 
        }
        std::reverse(chain.begin(), chain.end());

        // Prepare Encoders
        std::vector<char> compressed_buffer;
        std::vector<char> dummy_buffer; 
        auto writer = std::make_shared<utils::MemoryStreamWriter>(&dummy_buffer);
        
        std::vector<std::unique_ptr<encoding_algorithm::Encoder>> encoders;
        for(size_t i=0; i<dim; ++i) {
            encoders.push_back(encoding_algorithm::AlgorithmsManager::getEncoder("Double", encoding_algorithm_name_, writer));
        }

        // Prepare Decoders for reading chain
        auto reader = std::make_shared<utils::MemoryBlockStreamReader>((const unsigned char*)data_level0_memory_.data());
        std::vector<std::unique_ptr<encoding_algorithm::Decoder>> decoders;
        for(size_t i=0; i<dim; ++i) {
            decoders.push_back(encoding_algorithm::AlgorithmsManager::getDecoder("Double", encoding_algorithm_name_, reader));
        }

        // Process chain: Decode -> Encode (to update state)
        for (tableint id : chain) {
             size_t start = level0_element_start_positions_[id] + offsetData_;
             size_t end;
             if (id + 1 < cur_element_count && level0_element_start_positions_[id+1] > 0) {
                end = level0_element_start_positions_[id+1];
             } else {
                end = data_level0_memory_.size();
             }
             
             reader->resetBuffer((const unsigned char*)(data_level0_memory_.data() + start), end - start);
             
             for(size_t i=0; i<dim; ++i) {
                 double val = decoders[i]->decodeDouble();
                 encoders[i]->encode(val);
             }
             writer->align(); 
             // Clear dummy buffer to avoid growing indefinitely
             dummy_buffer.clear();
             writer->setBuffer(&dummy_buffer);
        }

        // Switch to real buffer for the new point
        writer->setBuffer(&compressed_buffer);
        
        const double* data_arr = (const double*)data_point;
        for(size_t i=0; i<dim; ++i) {
            encoders[i]->encode(data_arr[i]);
        }
        writer->align();

        // std::cout << "compressed buffer size for point " << cur_c << " is " << compressed_buffer.size() << std::endl;
        
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

        if ((signed)currObj != -1) {
            // std::cout << "Entering level <= maxlevel loop for point " << cur_c << " at level " << curlevel << std::endl;

            bool epDeleted = isMarkedDeleted(enterpoint_copy);
            for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                if (level > maxlevelcopy || level < 0)  // possible? 这不可能吧？
                    throw std::runtime_error("Level error");

                // 在level层找到与data_point距离最近的ef个节点，存储在列表中
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                        currObj, data_point, level);

                // std::cout << "Point " << cur_c << " found " << top_candidates.size() << " candidates at level " << level << std::endl;
                if (epDeleted) {
                    std::vector<double> vec_ep = getOriginalDataByInternalId(enterpoint_copy);
                    top_candidates.emplace(fstdistfunc_(data_point, vec_ep.data(), dist_func_param_), enterpoint_copy);
                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();
                }
                // 在level层建立data_point与top_candidates中每一个元素的连接
                currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
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
                for (int i = 0; i < size; i++) {
                    // 获取邻居的id
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    // 根据id获取邻居并计算其到query的距离
                    std::vector<double> vec_cand = getOriginalDataByInternalId(cand);
                    dist_t d = fstdistfunc_(query_data, vec_cand.data(), dist_func_param_);
                    // 如果这个邻居与query的距离比curdist还小，更新curdist为这个邻居，changed改为true
                    if (d < curdist) {
                        curdist = d;
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
        int total_size = 0;
        for (tableint i = 0; i < cur_element_count; i++) {
            size_t start = level0_element_start_positions_[i] + offsetData_;
            size_t end;
            if (i + 1 < cur_element_count && level0_element_start_positions_[i+1] > 0) {
                end = level0_element_start_positions_[i+1];
            } else {
                end = data_level0_memory_.size();
            }
            total_size += (end - start);
        }
        return total_size;
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
};
}  // namespace hnswlib
