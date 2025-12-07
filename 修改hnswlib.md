<center><h1>compressed_hnsw方案设计

现在我希望压缩hnsw索引，主要是压缩其中的data部分。

我已经成功试验了对double类型vector集合的逐个压缩和解压缩。压缩算法是基于差分的思想设计的，每次压缩/解压缩需要利用encoder压缩完前一个元素data部分的状态，来压缩下一个元素的data。

针对当前的hnsw系统，需要进行以下修改：

## 1. 修改存储结构

我们只需要修改level 0的存储结构即可。

在以前，每个element的大小是一定的，`size_data_per_element_`。可以通过这个值直接计算得到data的位置。

现在，我们使用`std::vector<size_t> level0_element_start_positions_`来记录level0每个element数据的开始位置。

每个element内部通过以下结构存储：

* links_level0：该element在第0层的邻居列表。通过`size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);`表示它的大小。
  * 该部分是element在内存中的第一项，通过`level0_element_start_positions_`可以用internal_id直接获取
* label：是外部的id（？存疑），大小是`sizeof(labelType)`
  * 该部分是element在内存中的第二项，位于links_level0的后面。因此有一个偏移量`label_offset_ = size_links_level0_`
* data：存储该点的实际数据。目前的大小是`data_size_ = dim * sizeof(double);`
  * 该部分是element在内存中的第三项，位于label后面，因此有一个偏移量`offsetData_ = sizeof(labeltype) + size_links_level0_`

现在，我们需要在links_level0后面添加一个`prenode`，表示该element进行差分压缩编码的一个前导节点。大小是sizeof(tableint)，和links_level0中一个元素的大小是相同的。全1表示该点没有压缩，存储的是原始数据。可以当作压缩/解压缩的起点。

* 相应地，offsetData和label_offset需要修改，同时添加一个prenode_offset_，设置为`prenode_offset_ = size_links_level0_`，并添加一个获取prenode_offset的函数

  * ```c++
    tableint getPrenodeId(tableint internal_id) const {
        tableint prenode_id;
        memcpy(&prenode_id, data_level0_memory_ + level0_element_start_positions_[internal_id] + prenode_offset_), sizeof(labeltype));
        return prenode_id;
    }
    ```



## 2. 修改获取data的函数

当前该函数是这么写的：

```c++
inline char *getDataByInternalId(tableint internal_id) const {
    return (data_level0_memory_ + level0_element_start_positions_[internal_id] + offsetData_);
}
```

这里的函数倒是不用大改，只要改了offsetData，只不过获取的这个data是压缩后的值。要想获取其原数据的话，我们需要额外添加一个解压缩的函数，下面是所打的一个草稿：

```c++
stack<tableint> decodeOrder = new stack<tableint>();
decodeOrder.push(cur_node);
tableint cur_node_copy = cur_node;
while(getPrenodeId(cur_node_copy) != 0xffffffff){
    tableint prenode = getPrenodeId(cur_node_copy));
    decodeOrder.push(prenode);
    cur_node_copy = prenode;
}
// decode
std::shared_ptr<utils::MemoryBlockStreamReader> reader = std::make_shared<utils::MemoryBlockStreamReader>(sharedOutputPath, static_cast<size_t>(byteNum));	// 这里的sharedOutputPath指的是存储data的文件路径。在当前系统中，压缩后的hnsw不一定会存储在文件中，可能在内存中的某个位置，因此之后还需要修改BlockStreamReader相关的代码，将其改为MemoryBlockStreamReader。
// 每列需要一个decoder
std::vector<std::unique_ptr<encoding_algorithm::Decoder>> decoders;
for (int col = 0; col < cols; ++col) {
    	decoders.emplace_back(encoding_algorithm::AlgorithmsManager::getDecoder(kDataType, algorithm, reader));
}
while(cur_node_copy != cur_node){
    cur_node_copy = decodeOrder.pop();
    char* begin_byte = getDataByInternalId(cur_node_copy);
	// BlockStreamReader读入cur_node_data到下一个cur_node_copy + 1这个element的起始位置这块数据
    char* end_byte = level0_element_start_positions_[cur_node_copy + 1];
    reader.cacheBytes(begin_byte, end_byte);
    for(int i = 0; i < dim; i++){
        decoders[i]->decodeDouble();
    }
}
// decode当前node的值
char* begin_byte = getDataByInternalId(cur_node);
char* end_byte = level0_element_start_positions_[cur_node + 1];
reader.cacheBytes(begin_byte, end_byte);
double* cur_node_data = (double*)malloc(sizeof(double) * dim);
for(int i = 0; i < dim; i++){
    cur_node_data[i] = decoders[i]->decodeDouble();
}

return cur_node_data;
```

简单来说，就是要反向溯源，找到开始encoding的起始点，然后逐个解压缩，一直到cur_node。之前都是通过`getDataByInternalId`函数直接获取data的，我们要用上述的代码替换所有用到这个函数的地方。



## 3. 修改计算距离的函数

目前的压缩算法只针对double类型的数据。hnswlib中定义了double的距离空间及相关计算距离的函数：

```c++
class L2SpaceDouble : public SpaceInterface<double> {
    DISTFUNC<double> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

 public:
    explicit L2SpaceDouble(size_t dim) {
        fstdistfunc_ = L2SqrDouble;
        dim_ = dim;
        data_size_ = dim * sizeof(double);
    }

    size_t get_data_size() {
        return data_size_;
    }

    DISTFUNC<double> get_dist_func() {
        return fstdistfunc_;
    }

    void *get_dist_func_param() {
        return &dim_;
    }

    ~L2SpaceDouble() {}
};

static double
L2SqrDouble(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    const double *pVect1 = static_cast<const double *>(pVect1v);
    const double *pVect2 = static_cast<const double *>(pVect2v);
    size_t qty = *static_cast<const size_t *>(qty_ptr);

    double res = 0.0;
    for (size_t i = 0; i < qty; i++) {
        double diff = pVect1[i] - pVect2[i];
        res += diff * diff;
    }
    return res;
}
```

其用法一般如下：

```c++
dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
```

data_point是const void\*类型的，直接指向内存地址；getDataByInternalId(ep_id)返回的则是char\*，指向ep_id这个element的data所在的内存地址。在计算距离时，直接强制类型转换为const double*，然后逐个数字解析，计算两个点之间的距离。

当我们适配了压缩算法后，不能直接向L2SqrDouble函数传递getDataByInternalId(ep_id)，而是需要通过第2项内容的修改计算距离的函数，传递解压缩后的data。



## 4. hnsw索引的存储和读取

hnsw索引的save和load需要修改。

以往的saveIndex函数，先后写入了offsetLevel0_, max_elements_, cur_element_count等metadata，以及data_level0_memory_, linkLists_这些data。其中，size_data_per_element_代表的是每个第0层里每个element的大小，对于压缩后的hnsw，我们就不再需要这一项了；相应地，写入第0层的`data_level0_memory_`这部分代码也需要修改。原本是`output.write(data_level0_memory_, cur_element_count * size_data_per_element_);`，我们可以利用`level0_element_start_positions_`计算`data_level0_memory_`的大小。

loadIndex函数也要做出相应修改。

此外，以前我们使用StreamWriter和BlockStreamReader来存储和读取压缩后的vector。它们有一个成员变量fileName或者path，表示读取和存储的文件位置。也就是说，它们是直接存储到磁盘文件、直接从磁盘文件读取的。现在，只有在saveIndex或者loadIndex时，我们才能写入指定的hnsw索引文件，一般是叫hnsw.bin。而在构建hnsw时和loadIndex之后，我们都是写入和读取一部分的内存位置的。因此，需要添加一个MemoryBlockStreamReader和MemoryStreamWriter，用于进行内存的存储和读取。

* 此外，以前的BlockStreamReader是根据指定的beginBit和endBit来读取文件中的一部分内容的，现在需要根据beginByte和endByte来读取内存中指定的一部分内容，单位从bit变成了byte；相应地，每次encoding一个vector，可能还需要对编码后的data补齐byte。



## 5. addPoint

添加新的element非常重要。我们主要需要修改为element分配level0的内存这部分的代码。

在以往的逻辑中，这部分是这样写的：

```c++
// 每个节点的数据大小是一定的，size_data_per_element_。在这里为level0分配内存
level0_element_start_positions_[cur_c] = cur_c * size_data_per_element_;
memset(data_level0_memory_ + level0_element_start_positions_[cur_c] + offsetLevel0_, 0, size_data_per_element_);

// Initialisation of the data and label
memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
memcpy(getDataByInternalId(cur_c), data_point, data_size_);
```

其中的data_point是const void*类型的，代表要存储的element的data部分。

1. 首先，每个element的size不是固定的了，不能用cur_c * size_data_per_element_来确定新的element的位置了。我们或许需要一个全局的变量来记录level0已经使用的总空间

2. 不能直接把data_point通过memset来写入指定的内存位置。我们需要先把这一步放着，先进行下一步：设置每一层的邻居，记录在linklists中。

3. 第二步包含了逐层的searchBaseLayer过程。在这里，我们会找到各层的top_candidates，然后通过mutuallyConnectNewElement将data_point和邻居建立连接。我们可以在寻找新插入的点在第0层的邻居时找到最近的一个邻居，用它作为新插入点的prenode，并进行压缩。

4. 压缩的过程需要一轮解压缩和一轮压缩。需要先顺着prenode的记录找到压缩的起点，然后设置一组encoder和一组decoder：decoder在前，先把下一个点的data解压缩出来，encoder在后，按照顺序逐个压缩，一直到我们要新插入的点。将其压缩后，得到了压缩后的data，写入内存中，这时再进行memset和memcpy。该过程写成如下代码草稿：

   ```c++
   int cur_node = getNearestNeighbor(top_candidates, data_point);	// cur_node是距离data_point最近的邻居，用它作为encoding的prenode
   stack<tableint> decodeOrder = new stack<tableint>();
   decodeOrder.push(cur_c);	// 向栈中压入新插入element的internal_id
   decodeOrder.push(cur_node);
   tableint cur_node_copy = cur_node;
   while(getPrenodeId(cur_node_copy) != 0xffffffff){
       tableint prenode = getPrenodeId(cur_node_copy));
       decodeOrder.push(prenode);
       cur_node_copy = prenode;
   }
   // decode and encode
   std::shared_ptr<utils::MemoryBlockStreamReader> reader = std::make_shared<utils::MemoryBlockStreamReader>(sharedOutputPath, static_cast<size_t>(byteNum));	// 这里的sharedOutputPath指的是存储data的文件路径。在当前系统中，压缩后的hnsw不一定会存储在文件中，可能在内存中的某个位置，因此之后还需要修改BlockStreamReader相关的代码。
   char* data_position = (level0_element_start_positions_[cur_c] + offsetData_);	// data_position是通过element内存位置映射表获取到的新的element的data应该存储的位置
   std::make_shared<utils::MemoryStreamWriter> writer = std::make_shared<utils::MemoryStreamWriter>(data_position);
   // 每列需要一个decoder和encoder
   std::vector<std::unique_ptr<encoding_algorithm::Decoder>> decoders;
   std::vector<std::unique_ptr<encoding_algorithm::Encoder>> encoders;
   
   for (int col = 0; col < cols; ++col) {
       	decoders.emplace_back(encoding_algorithm::AlgorithmsManager::getDecoder(kDataType, algorithm, reader));
       encoders.emplace_back(encoding_algorithm::AlgorithmsManager::getEecoder(kDataType, algorithm, writer));
   }
   while(cur_node_copy != cur_c){
       cur_node_copy = decodeOrder.pop();
       char* begin_byte = getDataByInternalId(cur_node_copy);
   	// BlockStreamReader读入cur_node_data到下一个cur_node_copy + 1这个element的起始位置这块数据
       char* end_byte = level0_element_start_positions_[cur_node_copy + 1];
       reader.cacheBytes(begin_byte, end_byte);
       std::vector<double> cur_vector = new vector<double>[dim];
       for(int i = 0; i < dim; i++){
           cur_vector[i] = decoders[i]->decodeDouble();
       	encoders[i]->encode(cur_vector[i], false);	//第二个bool参数表示是否写入内存。在编码到新插入的点之前，我们都不写入内存。
       }
   }
   // encode新插入element的值
   for(int i = 0; i < dim; i++){
   	encoders[i] -> encode(data_point[i], true);
   }
   ```

5. 在写入了新的element的data之后，也要更新level0_element_start_positions_的内容





## 6. 其他要修改的内容

getDataByLabel、searchKnn、searchStopConditionClosest等函数都用到了计算距离的函数，比如：`dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);`这里的getDataByInternalId只要替换成我们新版的获取data的函数就好了，前文已经提及，其他的就没有什么要修改的内容。



