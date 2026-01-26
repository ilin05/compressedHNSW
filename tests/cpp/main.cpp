

#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

void sift_test1B();
void sift_test1B_double_hnsw();
void sift_test1B_double_hnswalp();
void sift_test1B_double_hnswalp_pq();
int main() {
#ifdef _OPENMP
    omp_set_num_threads(20);
    std::cout << "OpenMP Threads set to: 20" << std::endl;
#endif

    // sift_test1B();
    sift_test1B_double_hnsw();
    sift_test1B_double_hnswalp();
    sift_test1B_double_hnswalp_pq();

    return 0;
}
