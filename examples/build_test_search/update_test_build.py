import sys, re

filename = r'd:\ZJU\SuDIS\hnswlib_cpp_py\examples\cpp\test_build_compressed_hnsw.cpp'
with open(filename, 'r', encoding='utf-8') as f:
    text = f.read()

helper_function = '''
template<typename dist_t, class Codec>
TestResult run_build_test_for_codec(
    SpaceInterface<dist_t>* l2space,
    size_t num_vectors,
    const std::string& algo_name,
    int M,
    int efConstruction,
    size_t cache_max_size,
    const std::string& dataset_name,
    double* data,
    size_t dim) 
{
    TestResult res;
    res.dataset_name = dataset_name;
    res.algo_name = algo_name;
    res.build_time = 0.0;
    res.compress_time = 0.0;
    res.data_compression_ratio = 0.0;
    res.index_compression_ratio = 0.0;

    cout << "Allocating memory for index..." << endl;
    HierarchicalNSWCABFRAMEWORK<dist_t, Codec>* appr_alg = 
        new HierarchicalNSWCABFRAMEWORK<dist_t, Codec>(l2space, num_vectors, algo_name, M, efConstruction, true, cache_max_size);

    StopW stopw;
    
    cout << "Inserting elements into HNSW... " << endl;
    // 多线程并发插入
    #pragma omp parallel for
    for (long i = 0; i < (long)num_vectors; ++i) {
        appr_alg->addPoint(data + i * dim, i);
    }
    
    res.build_time = 1e-6 * stopw.getElapsedTimeMicro();
    cout << "Graph Construction Time: " << res.build_time << " seconds" << endl;
    
    stopw.reset();
    cout << "Compressing dataset using " << algo_name << "..." << endl;
    
    appr_alg->compress_dataset();
    
    res.compress_time = 1e-6 * stopw.getElapsedTimeMicro();
    cout << "Compression Time: " << res.compress_time << " seconds" << endl;
    
    size_t original_data_size = num_vectors * dim * sizeof(double);
    size_t compressed_data_size = appr_alg->getCompressedDataSize();
    cout << "Original data size: " << original_data_size << " bytes." << endl;
    cout << "Total compressed data size: " << compressed_data_size << " bytes." << endl;
    res.data_compression_ratio = static_cast<double>(original_data_size) / static_cast<double>(compressed_data_size);
    cout << "Data compression ratio: " << res.data_compression_ratio << endl;

    size_t original_index_size = appr_alg->getIndexSize();
    size_t compressed_index_size = appr_alg->getCompressedIndexSize();
    cout << "Original index size: " << original_index_size << " bytes." << endl;
    cout << "Total compressed index size: " << compressed_index_size << " bytes." << endl;
    res.index_compression_ratio = static_cast<double>(original_index_size) / static_cast<double>(compressed_index_size);
    cout << "Index compression ratio: " << res.index_compression_ratio << endl;
    
    std::string index_path = dataset_name + "_compressed_hnsw_framework.bin";
    appr_alg->saveIndex(index_path);

    delete appr_alg;
    return res;
}
'''

# The function 	est_build_index currently creates the instance natively.
# We will use regex to wrap the middle part with a dispatch condition.

# Actually, it's easier to just rewrite test_build_index entirely manually in python.
# We know what the current test_build_index does (we have it).

regex_replace = r'''
    cout \<\< "Allocating memory for index..." \<\< endl;.*?delete appr_alg;
'''

replacement = '''
    if (algo_name == "DeXOR") {
        res = run_build_test_for_codec<double, codecs::DeXORCodec>(&l2space, num_vectors, algo_name, M, efConstruction, cache_max_size, dataset_name, data, dim);
    } else if (algo_name == "Gorilla") {
        res = run_build_test_for_codec<double, codecs::GorillaCodec>(&l2space, num_vectors, algo_name, M, efConstruction, cache_max_size, dataset_name, data, dim);
    } else if (algo_name == "Elf") {
        res = run_build_test_for_codec<double, codecs::ElfCodec>(&l2space, num_vectors, algo_name, M, efConstruction, cache_max_size, dataset_name, data, dim);
    } else if (algo_name == "Camel") {
        res = run_build_test_for_codec<double, codecs::CamelCodec>(&l2space, num_vectors, algo_name, M, efConstruction, cache_max_size, dataset_name, data, dim);
    } else {
        res = run_build_test_for_codec<double, codecs::DeXORCodec>(&l2space, num_vectors, algo_name, M, efConstruction, cache_max_size, dataset_name, data, dim);
    }
'''

text = re.sub(regex_replace, replacement, text, flags=re.DOTALL)
text = text.replace('TestResult test_build_index', helper_function + '\n\nTestResult test_build_index')

with open(filename, 'w', encoding='utf-8') as f:
    f.write(text)
