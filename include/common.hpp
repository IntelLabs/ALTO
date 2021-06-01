#ifndef COMMON_HPP_
#define COMMON_HPP_

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <inttypes.h>
#include <float.h>
#include <sys/mman.h>
#ifdef USE_MKL_
#include <mkl.h>
#endif
#ifdef USE_ESSL_
#include <essl.h>
#include <cblas.h>
#include <lapacke.h>
#endif
#ifdef USE_OPENBLAS_
#include <cblas.h>
#include <lapacke.h>
#endif
#include <omp.h>

#define MAP_HUGE_SHIFT 26
#define MAP_HUGE_2MB    (21 << MAP_HUGE_SHIFT)
#define MAP_HUGE_1GB    (30 << MAP_HUGE_SHIFT)
// set to 1 to use 1G hugepages and to 0 if you want to use 2M hugepages
#define USE_1G 0
// set the PRE_ALLOC[_ALTO] definitions in main.c and alto.c accordingly for the general
// usage of hugepages

typedef unsigned long long IType;
#if 1
typedef double FType;
#define POTRF  dpotrf_
#define POTRS  dpotrs_
#define GELSY  dgelsy_
#define SYRK   cblas_dsyrk
#else
typedef float FType;
#define POTRF  spotrf
#define POTRS  spotrs
#define GELSY  sgelsy
#define SYRK   cblas_ssyrk
#endif

#define MAX(a,b) (((a)<(b))?(b):(a))
#define MIN(a,b) (((a)>(b))?(b):(a))
#define CACHELINE      64

#define ROW 1

inline uint64_t ReadTSC(void)
{
#if defined(__i386__)

    uint64_t x;
    __asm__ __volatile__(".byte 0x0f, 0x31":"=A"(x));
    return x;

#elif defined(__x86_64__)

    uint32_t hi, lo;
    __asm__ __volatile__("rdtsc":"=a"(lo), "=d"(hi));
    return ((uint64_t) lo) | (((uint64_t) hi) << 32);

#elif defined(__powerpc__)

    uint64_t result = 0;
    uint64_t upper, lower, tmp;
    __asm__ __volatile__("0:                  \n"
                         "\tmftbu   %0           \n"
                         "\tmftb    %1           \n"
                         "\tmftbu   %2           \n"
                         "\tcmpw    %2,%0        \n"
                         "\tbne     0b         \n":"=r"(upper), "=r"(lower),
                         "=r"(tmp)
        );
    result = upper;
    result = result << 32;
    result = result | lower;
    return result;
#else
    return 0ULL;
#endif // defined(__i386__)
}

inline void *AlignedMalloc(size_t size)
{
    void *addr = NULL;
    if (posix_memalign(&addr, CACHELINE, size) != 0) {
        addr = NULL;
    }
    return addr;
}


inline void AlignedFree(void *addr)
{
    if (addr != NULL) {
        free(addr);
    }
    addr = NULL;
}

inline void* HPMalloc(size_t nbytes) {
    uint64_t page_size;
    if (USE_1G) {
        page_size = 1073741824;
    } else {
        page_size = 2097152;
    }
    void* ret_ptr = NULL;
    size_t num_large_pages = nbytes / page_size;
    if (nbytes > num_large_pages * page_size) {
        num_large_pages++;
    }
    nbytes = (size_t) num_large_pages * page_size;
    //printf("trying to allocate %ld pages\n", num_large_pages);
    if (USE_1G) {
        ret_ptr = mmap(NULL, nbytes,
            PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE | MAP_HUGETLB | MAP_HUGE_1GB,
            -1, 0);
    } else {
        ret_ptr = mmap(NULL, nbytes,
            PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE | MAP_HUGETLB,
            -1, 0);
    }
    if ((ret_ptr == (void *)(-1))) {
        fprintf(stderr,"mmap call failed\n");
        exit(1);
    }
    return ret_ptr;
}

inline void HPFree(void * ptr, size_t nbytes) {
    munmap(ptr, nbytes);
}

static inline
void ParMemcpy(void * const dst, void * const src, size_t const bytes)
{
  #pragma omp parallel proc_bind(close)
  {
    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();

    size_t n_per_thread = (bytes + nthreads - 1) / nthreads;
    size_t n_begin = MIN(n_per_thread * tid, bytes);
    size_t n_end = MIN(n_begin + n_per_thread, bytes);

    memcpy((char *)dst + n_begin, (char *)src + n_begin, n_end - n_begin);
  }
}

static inline
void ParMemset(void * dst, int val, size_t bytes)
{
  #pragma omp parallel proc_bind(close)
  {
    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();

    size_t n_per_thread = (bytes + nthreads - 1) / nthreads;
    size_t n_begin = MIN(n_per_thread * tid, bytes);
    size_t n_end = MIN(n_begin + n_per_thread, bytes);

    memset((char *)dst + n_begin, val, n_end - n_begin);
  }
}

#ifndef TIME
#define TIME 0
#endif
static inline void BEGIN_TIMER(
  uint64_t* ticks
)
{
  #if TIME
  *ticks = ReadTSC();
  #endif
}

static inline void END_TIMER(
  uint64_t* ticks
)
{
  #if TIME
  *ticks = ReadTSC();
  #endif
}

void ELAPSED_TIME(
  uint64_t start,
  uint64_t end,
  double* t_elapsed
);

void PRINT_TIMER(
  const char* message,
  double t
);

void ASMTrace(char *str);

void InitTSC(void);

double ElapsedTime(uint64_t ticks);

void PrintFPMatrix(char *name, FType * a, size_t m, size_t n);

void PrintIntMatrix(char *name, size_t * a, size_t m, size_t n);
#endif // COMMON_HPP_
