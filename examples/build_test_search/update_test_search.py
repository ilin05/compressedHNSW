import sys, re

filename = r'd:\ZJU\SuDIS\hnswlib_cpp_py\examples\cpp\test_search_compressed_hnsw.cpp'
with open(filename, 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('HierarchicalNSWCABFRAMEWORK<double>* appr_alg = nullptr;', 'AlgorithmInterface<double>* appr_alg = nullptr;')

init_logic = '''
        if (algo_name == "DeXOR") {
            appr_alg = new HierarchicalNSWCABFRAMEWORK<double, codecs::DeXORCodec>(&l2space, index_path, true, algo_name, cache_sz);
        } else if (algo_name == "Gorilla") {
            appr_alg = new HierarchicalNSWCABFRAMEWORK<double, codecs::GorillaCodec>(&l2space, index_path, true, algo_name, cache_sz);
        } else if (algo_name == "Elf") {
            appr_alg = new HierarchicalNSWCABFRAMEWORK<double, codecs::ElfCodec>(&l2space, index_path, true, algo_name, cache_sz);
        } else if (algo_name == "Camel") {
            appr_alg = new HierarchicalNSWCABFRAMEWORK<double, codecs::CamelCodec>(&l2space, index_path, true, algo_name, cache_sz);
        } else {
            appr_alg = new HierarchicalNSWCABFRAMEWORK<double, codecs::DeXORCodec>(&l2space, index_path, true, algo_name, cache_sz);
        }
'''

text = re.sub(
    r'appr_alg = new HierarchicalNSWCABFRAMEWORK<double>\(&l2space, index_path, true, algo_name, cache_sz\);',
    init_logic,
    text
)

ef_logic = '''
        if (auto p = dynamic_cast<HierarchicalNSWCABFRAMEWORK<double, codecs::DeXORCodec>*>(appr_alg)) p->setEf(ef);
        else if (auto p = dynamic_cast<HierarchicalNSWCABFRAMEWORK<double, codecs::GorillaCodec>*>(appr_alg)) p->setEf(ef);
        else if (auto p = dynamic_cast<HierarchicalNSWCABFRAMEWORK<double, codecs::ElfCodec>*>(appr_alg)) p->setEf(ef);
        else if (auto p = dynamic_cast<HierarchicalNSWCABFRAMEWORK<double, codecs::CamelCodec>*>(appr_alg)) p->setEf(ef);
'''

text = text.replace('appr_alg->setEf(ef);', ef_logic)

with open(filename, 'w', encoding='utf-8') as f:
    f.write(text)
