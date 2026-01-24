#include <iostream>
#include <fstream>
#include <queue>
#include <chrono>
#include "../../hnswlib/hnswlib.h"
#include "../../examples/data_processor/data_loader.h"


#include <unordered_set>

using namespace std;
using namespace hnswlib;

class StopW {
    std::chrono::steady_clock::time_point time_begin;
 public:
    StopW() {
        time_begin = std::chrono::steady_clock::now();
    }

    float getElapsedTimeMicro() {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
    }

    void reset() {
        time_begin = std::chrono::steady_clock::now();
    }
};



/*
* Author:  David Robert Nadeau
* Site:    http://NadeauSoftware.com/
* License: Creative Commons Attribution 3.0 Unported License
*          http://creativecommons.org/licenses/by/3.0/deed.en_US
*/

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))

#include <unistd.h>
#include <sys/resource.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
#include <fcntl.h>
#include <procfs.h>

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)

#endif

#else
#error "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS."
#endif


/**
* Returns the peak (maximum so far) resident set size (physical
* memory use) measured in bytes, or zero if the value cannot be
* determined on this OS.
*/
static size_t getPeakRSS() {
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.PeakWorkingSetSize;

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
    /* AIX and Solaris ------------------------------------------ */
    struct psinfo psinfo;
    int fd = -1;
    if ((fd = open("/proc/self/psinfo", O_RDONLY)) == -1)
        return (size_t)0L;      /* Can't open? */
    if (read(fd, &psinfo, sizeof(psinfo)) != sizeof(psinfo)) {
        close(fd);
        return (size_t)0L;      /* Can't read? */
    }
    close(fd);
    return (size_t)(psinfo.pr_rssize * 1024L);

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
    /* BSD, Linux, and OSX -------------------------------------- */
    struct rusage rusage;
    getrusage(RUSAGE_SELF, &rusage);
#if defined(__APPLE__) && defined(__MACH__)
    return (size_t)rusage.ru_maxrss;
#else
    return (size_t) (rusage.ru_maxrss * 1024L);
#endif

#else
    /* Unknown OS ----------------------------------------------- */
    return (size_t)0L;          /* Unsupported. */
#endif
}


/**
* Returns the current resident set size (physical memory use) measured
* in bytes, or zero if the value cannot be determined on this OS.
*/
static size_t getCurrentRSS() {
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.WorkingSetSize;

#elif defined(__APPLE__) && defined(__MACH__)
    /* OSX ------------------------------------------------------ */
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
        (task_info_t)&info, &infoCount) != KERN_SUCCESS)
        return (size_t)0L;      /* Can't access? */
    return (size_t)info.resident_size;

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
    /* Linux ---------------------------------------------------- */
    long rss = 0L;
    FILE *fp = NULL;
    if ((fp = fopen("/proc/self/statm", "r")) == NULL)
        return (size_t) 0L;      /* Can't open? */
    if (fscanf(fp, "%*s%ld", &rss) != 1) {
        fclose(fp);
        return (size_t) 0L;      /* Can't read? */
    }
    fclose(fp);
    return (size_t) rss * (size_t) sysconf(_SC_PAGESIZE);

#else
    /* AIX, BSD, Solaris, and Unknown OS ------------------------ */
    return (size_t)0L;          /* Unsupported. */
#endif
}


static void
get_gt(
    unsigned int *massQA,
    double *massQ,
    double *mass,
    size_t vecsize,
    size_t qsize,
    L2SpaceDouble &l2space,
    size_t vecdim,
    vector<std::priority_queue<std::pair<double, labeltype>>> &answers,
    size_t k) {
    (vector<std::priority_queue<std::pair<double, labeltype >>>(qsize)).swap(answers);
    DISTFUNC<double> fstdistfunc_ = l2space.get_dist_func();
    cout << qsize << "" << endl;
    for (int i = 0; i < qsize; i++) {
        for (int j = 0; j < k; j++) {
            answers[i].emplace(0.0f, massQA[1000 * i + j]);
        }
    }
}

static float
test_approx(
    double *massQ,
    size_t vecsize,
    size_t qsize,
    HierarchicalNSWALPSIMPLIFIED<double> &appr_alg,
    size_t vecdim,
    vector<std::priority_queue<std::pair<double, labeltype>>> &answers,
    size_t k) {
    size_t correct = 0;
    size_t total = 0;
    // uncomment to test in parallel mode:
    //#pragma omp parallel for
    for (int i = 0; i < qsize; i++) {
        std::priority_queue<std::pair<double, labeltype >> result = appr_alg.searchKnn(massQ + vecdim * i, k);
        std::priority_queue<std::pair<double, labeltype >> gt(answers[i]);
        unordered_set<labeltype> g;
        total += gt.size();

        while (gt.size()) {
            g.insert(gt.top().second);
            gt.pop();
        }

        while (result.size()) {
            if (g.find(result.top().second) != g.end()) {
                correct++;
            } else {
            }
            result.pop();
        }
    }
    return 1.0f * correct / total;
}

static void
test_vs_recall(
    double *massQ,
    size_t vecsize,
    size_t qsize,
    HierarchicalNSWALPSIMPLIFIED<double> &appr_alg,
    size_t vecdim,
    vector<std::priority_queue<std::pair<double, labeltype>>> &answers,
    size_t k) {

    std::string csv_file_path = "sift1b_hnswalp_recall_results.csv";
    std::ofstream csv_file(csv_file_path);
    csv_file << "ef,recall,time_us_per_query\n";

    vector<size_t> efs;  // = { 10,10,10,10,10 };
    for (int i = k; i < 30; i++) {
        efs.push_back(i);
    }
    for (int i = 30; i < 100; i += 10) {
        efs.push_back(i);
    }
    for (int i = 100; i < 500; i += 40) {
        efs.push_back(i);
    }
    for (size_t ef : efs) {
        appr_alg.setEf(ef);
        StopW stopw = StopW();

        float recall = test_approx(massQ, vecsize, qsize, appr_alg, vecdim, answers, k);
        float time_us_per_query = stopw.getElapsedTimeMicro() / qsize;

        cout << ef << "\t" << recall << "\t" << time_us_per_query << " us" << endl;
        csv_file << ef << "," << recall << "," << time_us_per_query << "\n";
        if (recall > 1.0) {
            cout << recall << "\t" << time_us_per_query << " us" << endl;
            break;
        }
    }
    csv_file.close();
}

inline bool exists_test(const std::string &name) {
    ifstream f(name.c_str());
    return f.good();
}


void sift_test1B_double_hnswalp() {
    int subset_size_milllions = 10;
    int efConstruction = 300;
    int M = 32;

    size_t vecsize = subset_size_milllions * 1000000;

    size_t qsize = 10000;
    size_t vecdim = 128; // This will be detected from file
    char path_index[1024];
    char path_gt[1024];
    const std::string path_q = "../bigann/bigann_query.bvecs";
    const std::string path_data = "../bigann/bigann_base.bvecs";
    snprintf(path_index, sizeof(path_index), "sift1b_hnswalp_%dm_ef_%d_M_%d.bin", subset_size_milllions, efConstruction, M);

    snprintf(path_gt, sizeof(path_gt), "../bigann/gnd/idx_%dM.ivecs", subset_size_milllions);

    cout << "Loading GT:" << endl;
    ifstream inputGT(path_gt, ios::binary);
    unsigned int *massQA = new unsigned int[qsize * 1000];
    for (int i = 0; i < qsize; i++) {
        int t;
        inputGT.read((char *) &t, 4);
        inputGT.read((char *) (massQA + 1000 * i), t * 4);
        if (t != 1000) {
            cout << "err";
            return;
        }
    }
    inputGT.close();

    cout << "Loading queries:" << endl;
    int dim_q;
    double* massQ = data_loader::loadBvecsChunk(path_q, 0, qsize, dim_q);
    if(dim_q != vecdim) {
        // Warning if default assumption wrong, but update is better.
        vecdim = dim_q;
    }

    L2SpaceDouble l2space(vecdim);
    HierarchicalNSWALPSIMPLIFIED<double> *appr_alg;
    
    if (exists_test(path_index)) {
        cout << "Loading index from " << path_index << ":" << endl;
        appr_alg = new HierarchicalNSWALPSIMPLIFIED<double>(&l2space, path_index, false);
        cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb " << endl;
    } else {
        cout << "Building index:" << endl;
        appr_alg = new HierarchicalNSWALPSIMPLIFIED<double>(&l2space, vecsize, M, efConstruction);

        size_t CHUNK_SIZE = 500000; // 500k chunks
        size_t loaded_count = 0;
        
        StopW stopw_full = StopW();
        
        while (loaded_count < vecsize) {
            size_t this_batch = std::min(CHUNK_SIZE, vecsize - loaded_count);
            // cout << "Loading chunk: offset " << loaded_count << ", size " << this_batch << endl;
            
            int loaded_dim;
            double* chunk_data = data_loader::loadBvecsChunk(path_data, loaded_count, this_batch, loaded_dim);
            
            if (loaded_dim != vecdim) {
                cout << "Error: Dimension mismatch in chunk at offset " << loaded_count << endl;
                delete[] chunk_data;
                break;
            }

            #pragma omp parallel for
            for (long i = 0; i < static_cast<long>(this_batch); ++i) {
                 appr_alg->addPoint(chunk_data + i * vecdim, loaded_count + i);
            }
            
            delete[] chunk_data;
            loaded_count += this_batch;
            
            cout << "Added " << loaded_count << " points. Mem: " 
                 << getCurrentRSS() / 1000000 << " Mb " << endl;
        }
        appr_alg->compress_dataset();
        cout << "Build time:" << 1e-6 * stopw_full.getElapsedTimeMicro() << "  seconds" << endl;
        appr_alg->saveIndex(path_index);
    }


    vector<std::priority_queue<std::pair<double, labeltype >>> answers;
    size_t k = 1;
    cout << "Parsing gt:" << endl; // Should rename to just preparing GT struct or similar since GT is already loaded, but keeping string from user code
    // Note: get_gt signature changed but checks massQ and mass which are now doubles (or NULL)
    get_gt(massQA, massQ, NULL, vecsize, qsize, l2space, vecdim, answers, k);
    cout << "Loaded gt" << endl;
    for (int i = 0; i < 1; i++)
        test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);
    cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb " << endl;
    
    delete[] massQA;
    delete[] massQ;
    if(appr_alg) delete appr_alg;
    
    return;
}
