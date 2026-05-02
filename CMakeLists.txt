cmake_minimum_required(VERSION 3.0...3.26)

project(hnswlib
    LANGUAGES CXX)

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)

add_library(hnswlib INTERFACE)
add_library(hnswlib::hnswlib ALIAS hnswlib)

target_include_directories(hnswlib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# Install
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/hnswlib
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS hnswlib
    EXPORT hnswlibTargets)

install(EXPORT hnswlibTargets
    FILE hnswlibConfig.cmake
    NAMESPACE hnswlib::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hnswlib)

# Examples and tests
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    option(HNSWLIB_EXAMPLES "Build examples and tests." ON)
else()
    option(HNSWLIB_EXAMPLES "Build examples and tests." OFF)
endif()
if(HNSWLIB_EXAMPLES)
    set(CMAKE_CXX_STANDARD 17)

    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      SET( CMAKE_CXX_FLAGS  "-Ofast -std=c++17 -DHAVE_CXX0X -openmp -fpic -ftree-vectorize" )
      check_cxx_compiler_flag("-march=native" COMPILER_SUPPORT_NATIVE_FLAG)
      if(COMPILER_SUPPORT_NATIVE_FLAG)
        SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native" )
        message("set -march=native flag")
      else()
        check_cxx_compiler_flag("-mcpu=apple-m1" COMPILER_SUPPORT_M1_FLAG)
        if(COMPILER_SUPPORT_M1_FLAG)
          SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=apple-m1" )
          message("set -mcpu=apple-m1 flag")
        endif()
      endif()
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      SET( CMAKE_CXX_FLAGS  "-Ofast -g -lrt -std=c++17 -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
      SET( CMAKE_CXX_FLAGS  "/O2 -DHAVE_CXX0X /W1 /openmp /EHsc" )
    endif()

    # examples
    file(GLOB_RECURSE ENCODING_ALGO_SRC
      "encoding_algorithms/dexor/*.cpp"
      "encoding_algorithms/gorilla/*.cpp"
      "encoding_algorithms/elf/*.cpp"
      "encoding_algorithms/camel/*.cpp")

    # ----- faiss -----
    find_package(OpenMP REQUIRED)
    set(FAISS_ENABLE_PYTHON OFF CACHE BOOL "" FORCE)
    set(FAISS_ENABLE_GPU OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    add_subdirectory(third_party/faiss-1.14.1 EXCLUDE_FROM_ALL)
    include_directories(third_party/faiss-1.14.1)

    add_executable(test_build_faiss_hnsw_pq_sq examples/cpp/test_build_faiss_hnsw_pq_sq.cpp)
    target_link_libraries(test_build_faiss_hnsw_pq_sq faiss OpenMP::OpenMP_CXX)

    add_executable(test_search_faiss_hnsw_pq_sq examples/cpp/test_search_faiss_hnsw_pq_sq.cpp)
    target_link_libraries(test_search_faiss_hnsw_pq_sq faiss OpenMP::OpenMP_CXX)

    add_executable(test_build_faiss_ivf examples/cpp/test_build_faiss_ivf.cpp)
    target_link_libraries(test_build_faiss_ivf faiss OpenMP::OpenMP_CXX)

    add_executable(test_search_faiss_ivf examples/cpp/test_search_faiss_ivf.cpp)
    target_link_libraries(test_search_faiss_ivf faiss OpenMP::OpenMP_CXX)

    add_executable(test_build_faiss_nsg examples/cpp/test_build_faiss_nsg.cpp)
    target_link_libraries(test_build_faiss_nsg faiss OpenMP::OpenMP_CXX)

    add_executable(test_vq_recall examples/cpp/test_vq_recall.cpp)
    target_link_libraries(test_vq_recall faiss hnswlib OpenMP::OpenMP_CXX)

    add_executable(test_build_bigann_faiss_hnsw_pq_sq examples/cpp/test_build_bigann_faiss_hnsw_pq_sq.cpp)
    target_link_libraries(test_build_bigann_faiss_hnsw_pq_sq faiss OpenMP::OpenMP_CXX)

    add_executable(test_search_bigann_faiss_hnsw_pq_sq examples/cpp/test_search_bigann_faiss_hnsw_pq_sq.cpp)
    target_link_libraries(test_search_bigann_faiss_hnsw_pq_sq faiss OpenMP::OpenMP_CXX)
    # -----------------

    add_executable(example_search examples/cpp/example_search.cpp)
    target_link_libraries(example_search hnswlib)

    add_executable(example_epsilon_search examples/cpp/example_epsilon_search.cpp)
    target_link_libraries(example_epsilon_search hnswlib)

    add_executable(example_multivector_search examples/cpp/example_multivector_search.cpp)
    target_link_libraries(example_multivector_search hnswlib)

    add_executable(example_filter examples/cpp/example_filter.cpp)
    target_link_libraries(example_filter hnswlib)

    add_executable(example_replace_deleted examples/cpp/example_replace_deleted.cpp)
    target_link_libraries(example_replace_deleted hnswlib)

    add_executable(example_mt_search examples/cpp/example_mt_search.cpp)
    target_link_libraries(example_mt_search hnswlib)

    add_executable(example_mt_filter examples/cpp/example_mt_filter.cpp)
    target_link_libraries(example_mt_filter hnswlib)

    add_executable(example_mt_replace_deleted examples/cpp/example_mt_replace_deleted.cpp)
    target_link_libraries(example_mt_replace_deleted hnswlib)

    add_executable(example_double_vector examples/cpp/example_double_vector.cpp ${ENCODING_ALGO_SRC})
    target_link_libraries(example_double_vector hnswlib)

    add_executable(vector_compress examples/cpp/vector_compress.cpp ${ENCODING_ALGO_SRC})
    target_link_libraries(vector_compress hnswlib)

    add_executable(memory_vector_compress examples/cpp/memory_vector_compress.cpp ${ENCODING_ALGO_SRC})
    target_link_libraries(memory_vector_compress hnswlib)

    add_executable(single_vector_decompress examples/cpp/single_vector_decompress.cpp ${ENCODING_ALGO_SRC})
    target_link_libraries(single_vector_decompress hnswlib)

    add_executable(test_hnswcw examples/cpp/test_hnswcw.cpp ${ENCODING_ALGO_SRC})
    target_link_libraries(test_hnswcw hnswlib)

    add_executable(test_hnswcw_cached examples/cpp/test_hnswcw_cached.cpp ${ENCODING_ALGO_SRC})
    target_link_libraries(test_hnswcw_cached hnswlib)

    add_executable(test_flame examples/cpp/test_flame.cpp ${ENCODING_ALGO_SRC})
    target_link_libraries(test_flame hnswlib)

    add_executable(test_save_and_load_hnswcw examples/cpp/test_save_and_load_hnswcw.cpp ${ENCODING_ALGO_SRC})
    target_link_libraries(test_save_and_load_hnswcw hnswlib)

    add_executable(test_save_hnswcw_and_hnsw examples/cpp/test_save_hnswcw_and_hnsw.cpp ${ENCODING_ALGO_SRC})
    target_link_libraries(test_save_hnswcw_and_hnsw hnswlib)

    add_executable(test_load_hnswcw_and_hnsw examples/cpp/test_load_hnswcw_and_hnsw.cpp ${ENCODING_ALGO_SRC})
    target_link_libraries(test_load_hnswcw_and_hnsw hnswlib)

    add_executable(test_save_hnswcab_and_hnsw examples/cpp/test_save_hnswcab_and_hnsw.cpp ${ENCODING_ALGO_SRC})
    target_link_libraries(test_save_hnswcab_and_hnsw hnswlib)

    add_executable(test_save_hnswcab_leann_and_hnsw examples/cpp/test_save_hnswcab_leann_and_hnsw.cpp ${ENCODING_ALGO_SRC})
    target_link_libraries(test_save_hnswcab_leann_and_hnsw hnswlib)

    add_executable(test_save_hnswalp_and_hnsw examples/cpp/test_save_hnswalp_and_hnsw.cpp ${ENCODING_ALGO_SRC})
    target_link_libraries(test_save_hnswalp_and_hnsw hnswlib)

    add_executable(test_save_hnswalp_leann_and_hnsw examples/cpp/test_save_hnswalp_leann_and_hnsw.cpp ${ENCODING_ALGO_SRC})
    target_link_libraries(test_save_hnswalp_leann_and_hnsw hnswlib)

    add_executable(test_bigann_hnswalp_and_hnsw examples/cpp/test_bigann_hnswalp_and_hnsw.cpp ${ENCODING_ALGO_SRC})
    target_link_libraries(test_bigann_hnswalp_and_hnsw hnswlib)

    add_executable(test_save_hnswalp_origin_and_hnsw examples/cpp/test_save_hnswalp_origin_and_hnsw.cpp ${ENCODING_ALGO_SRC})
    target_link_libraries(test_save_hnswalp_origin_and_hnsw hnswlib)

    add_executable(test_save_hnswalp_simplified_and_hnsw examples/cpp/test_save_hnswalp_simplified_and_hnsw.cpp ${ENCODING_ALGO_SRC})
    target_link_libraries(test_save_hnswalp_simplified_and_hnsw hnswlib)

    add_executable(test_save_hnswalp_simdized_and_hnsw examples/cpp/test_save_hnswalp_simdized_and_hnsw.cpp ${ENCODING_ALGO_SRC})
    target_link_libraries(test_save_hnswalp_simdized_and_hnsw hnswlib)

    add_executable(test_build_hnswalp_pq examples/cpp/test_build_hnswalp_pq.cpp ${ENCODING_ALGO_SRC})
    target_link_libraries(test_build_hnswalp_pq hnswlib)

    add_executable(test_build_hnswdexor_pq examples/cpp/test_build_hnswdexor_pq.cpp ${ENCODING_ALGO_SRC})
    target_link_libraries(test_build_hnswdexor_pq hnswlib)

    add_executable(test_search_hnswdexor_pq examples/cpp/test_search_hnswdexor_pq.cpp ${ENCODING_ALGO_SRC})
    target_link_libraries(test_search_hnswdexor_pq hnswlib)

    add_executable(test_search_hnswalp_pq examples/cpp/test_search_hnswalp_pq.cpp ${ENCODING_ALGO_SRC})
    target_link_libraries(test_search_hnswalp_pq hnswlib)

    add_executable(test_search_compressed_hnsw examples/cpp/test_search_compressed_hnsw.cpp ${ENCODING_ALGO_SRC})
    target_link_libraries(test_search_compressed_hnsw hnswlib)

    add_executable(test_build_compressed_hnsw examples/cpp/test_build_compressed_hnsw.cpp ${ENCODING_ALGO_SRC})
    target_link_libraries(test_build_compressed_hnsw hnswlib)

    add_executable(test_build_compressed_hnsw_ablation examples/cpp/test_build_compressed_hnsw_ablation.cpp ${ENCODING_ALGO_SRC})
    target_link_libraries(test_build_compressed_hnsw_ablation hnswlib)

    add_executable(test_search_compressed_hnsw_ablation examples/cpp/test_search_compressed_hnsw_ablation.cpp ${ENCODING_ALGO_SRC})
    target_link_libraries(test_search_compressed_hnsw_ablation hnswlib)

    add_executable(test_profile_compressed_hnsw_search examples/cpp/test_profile_compressed_hnsw_search.cpp ${ENCODING_ALGO_SRC})
    target_link_libraries(test_profile_compressed_hnsw_search hnswlib)

    add_executable(test_build_hnsw examples/cpp/test_build_hnsw.cpp ${ENCODING_ALGO_SRC})
    target_link_libraries(test_build_hnsw hnswlib)

    add_executable(test_search_hnsw examples/cpp/test_search_hnsw.cpp ${ENCODING_ALGO_SRC})
    target_link_libraries(test_search_hnsw hnswlib)

    add_executable(test_profile_hnsw_node_stats examples/cpp/test_profile_hnsw_node_stats.cpp ${ENCODING_ALGO_SRC})
    target_link_libraries(test_profile_hnsw_node_stats hnswlib)
    target_compile_definitions(test_profile_hnsw_node_stats PRIVATE HNSWLIB_ENABLE_ACCESS_COUNT)

    add_executable(test_profile_hnsw_search examples/cpp/test_profile_hnsw_search.cpp ${ENCODING_ALGO_SRC})
    target_link_libraries(test_profile_hnsw_search hnswlib)

    add_executable(test_build_bigann_hnsw examples/cpp/test_build_bigann_hnsw.cpp ${ENCODING_ALGO_SRC})
    target_link_libraries(test_build_bigann_hnsw hnswlib)

    add_executable(test_search_bigann_hnsw examples/cpp/test_search_bigann_hnsw.cpp ${ENCODING_ALGO_SRC})
    target_link_libraries(test_search_bigann_hnsw hnswlib)

    add_executable(test_build_bigann_hnswalp_pq examples/cpp/test_build_bigann_hnswalp_pq.cpp ${ENCODING_ALGO_SRC})
    target_link_libraries(test_build_bigann_hnswalp_pq hnswlib)

    add_executable(test_search_bigann_hnswalp_pq examples/cpp/test_search_bigann_hnswalp_pq.cpp ${ENCODING_ALGO_SRC})
    target_link_libraries(test_search_bigann_hnswalp_pq hnswlib)

    add_executable(test_build_bigann_hnswdexor_pq examples/cpp/test_build_bigann_hnswdexor_pq.cpp ${ENCODING_ALGO_SRC})
    target_link_libraries(test_build_bigann_hnswdexor_pq hnswlib)

    add_executable(test_search_bigann_hnswdexor_pq examples/cpp/test_search_bigann_hnswdexor_pq.cpp ${ENCODING_ALGO_SRC})
    target_link_libraries(test_search_bigann_hnswdexor_pq hnswlib)

    add_executable(test_build_bigann_compressed_hnsw examples/cpp/test_build_bigann_compressed_hnsw.cpp ${ENCODING_ALGO_SRC})
    target_link_libraries(test_build_bigann_compressed_hnsw hnswlib)

    add_executable(test_search_bigann_compressed_hnsw examples/cpp/test_search_bigann_compressed_hnsw.cpp ${ENCODING_ALGO_SRC})
    target_link_libraries(test_search_bigann_compressed_hnsw hnswlib)

    add_executable(test_refresh_compressed_hnsw_ratios examples/cpp/test_refresh_compressed_hnsw_ratios.cpp ${ENCODING_ALGO_SRC})
    target_link_libraries(test_refresh_compressed_hnsw_ratios hnswlib)

    add_executable(test_build_index_size_comparison examples/cpp/test_build_index_size_comparison.cpp ${ENCODING_ALGO_SRC})
    target_link_libraries(test_build_index_size_comparison hnswlib faiss OpenMP::OpenMP_CXX)

    # tests
    add_executable(multivector_search_test tests/cpp/multivector_search_test.cpp)
    target_link_libraries(multivector_search_test hnswlib)

    add_executable(epsilon_search_test tests/cpp/epsilon_search_test.cpp)
    target_link_libraries(epsilon_search_test hnswlib)

    add_executable(test_updates tests/cpp/updates_test.cpp)
    target_link_libraries(test_updates hnswlib)

    add_executable(searchKnnCloserFirst_test tests/cpp/searchKnnCloserFirst_test.cpp)
    target_link_libraries(searchKnnCloserFirst_test hnswlib)

    add_executable(searchKnnWithFilter_test tests/cpp/searchKnnWithFilter_test.cpp)
    target_link_libraries(searchKnnWithFilter_test hnswlib)

    add_executable(multiThreadLoad_test tests/cpp/multiThreadLoad_test.cpp)
    target_link_libraries(multiThreadLoad_test hnswlib)

    add_executable(multiThread_replace_test tests/cpp/multiThread_replace_test.cpp)
    target_link_libraries(multiThread_replace_test hnswlib)

    add_executable(main tests/cpp/main.cpp tests/cpp/sift_1b.cpp tests/cpp/sift_1b_double_hnsw.cpp tests/cpp/sift_1b_double_hnswalp.cpp tests/cpp/sift_1b_double_hnswalp_pq.cpp tests/cpp/sift_1b_double_hnswcab.cpp tests/cpp/sift_1b_double_hnswcab_pq.cpp)
    target_link_libraries(main hnswlib)
    
    # Enable search/cache macros for compressed-HNSW related executables.
    # target_compile_definitions only accepts one target per call, so apply in a loop.
    set(COMPRESSED_HNSW_TARGETS
      main
      test_build_hnswalp_pq
      test_search_hnswalp_pq
      test_build_hnswdexor_pq
      test_search_hnswdexor_pq
      test_build_compressed_hnsw
      test_search_compressed_hnsw
      test_build_compressed_hnsw_ablation
      test_search_compressed_hnsw_ablation
      test_build_bigann_hnswalp_pq
      test_search_bigann_hnswalp_pq
      test_build_bigann_hnswdexor_pq
      test_search_bigann_hnswdexor_pq
      test_build_bigann_compressed_hnsw
      test_search_bigann_compressed_hnsw
      test_refresh_compressed_hnsw_ratios
      test_build_index_size_comparison)

    foreach(target_name IN LISTS COMPRESSED_HNSW_TARGETS)
      target_compile_definitions(${target_name} PRIVATE TWO_LEVEL_SEARCH SELECT_HUBS_FOR_CACHE)
    endforeach()
endif()
