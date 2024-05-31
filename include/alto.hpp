#ifndef ALTO_HPP_
#define ALTO_HPP_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <cmath>
#include <algorithm>

#include <omp.h>

#include "common.hpp"
#include "bitops.hpp"
#include "sptensor.hpp"

#ifdef OPT_ALTO
typedef unsigned long  LPType;     // Linearized partition id type
#endif

typedef struct Interval_ {
    IType start;
    IType stop;
} Interval;

template <typename LIT>
struct AltoTensor {
    int nmode;                  // number of modes
    int nprtn;                  // number of partitions
    IType nnz;                  // number of nonzeros
    IType *dims;                // dimensions
    LIT alto_mask;
    LIT *mode_masks;            // gather/scatter masks (nmode)
#ifdef ALT_PEXT
    int *mode_pos;              // starting point for each mode mask (nmode)
#endif
    LIT *idx;                   // ALTO index
    FType *vals;                // nonzeros

    IType *prtn_ptr;            // pointer to children nnzs
    Interval *prtn_intervals;   // ALTO partition/subspace intervals (nprtn*nmode)
    LIT alto_cr_mask;
    LIT *cr_masks;              // optional conflict resolution masks (nmode)

#ifdef OPT_ALTO
    LPType *prtn_id;           // ALTO partition/subspace id
    LPType *prtn_mask;
    LPType *prtn_mode_masks;   // ALTO partition/subspace gather/scatter masks (nprtn*nmode)
#endif
};

//Adaptive Linearized Tensor Order (ALTO) APIs
template <typename LIT>
static inline void create_alto(SparseTensor *spt, AltoTensor<LIT> **at, int nprtn);

template <typename LIT>
static inline void destroy_alto(AltoTensor<LIT> *at);

template <typename LIT>
static inline void create_da_mem(int target_mode, IType rank, AltoTensor<LIT> *at, FType ***ofibs);

template <typename LIT>
static inline void destroy_da_mem(AltoTensor<LIT> *at, FType **ofibs, IType rank, IType target_mode);

template <typename LIT>
static inline void unmap_da_mem(AltoTensor<LIT> *at, FType **ofibs, IType rank, IType target_mode);

template <typename LIT>
static inline void evaluate_delinearization(AltoTensor<LIT> *at);

template <typename LIT> 
void mttkrp_alto(int target_mode, FType **factors, IType rank, AltoTensor<LIT> *at) __attribute__((noinline));

template <typename LIT>
static inline void mttkrp_alto_par(int target_mode, FType **factors, IType rank, AltoTensor<LIT> *at, omp_lock_t* wlocks, FType **ofibs);

//#define TEST_ALTO
//#define ALTO_MEM_TRACE
//#define ALTO_DEBUG
#ifdef ALTO_MEM_TRACE
typedef unsigned long long AddType;
#endif

// worst case da_mem cost = 4 mem ops
#define MIN_FIBER_REUSE 4
// cr_bits can improve the performance when there is limited fiber reuse.
#define GENERATE_CR_BITS

//ALTO currently supports multiple options:
// 1) packing (LSB first or MSB first)
// 2) mode order within a group of bits (natural, shortest first, longest first)
typedef enum PackOrder_ { LSB_FIRST, MSB_FIRST } PackOrder;
typedef enum ModeOrder_ { SHORT_FIRST, LONG_FIRST, NATURAL } ModeOrder;

template <typename LIT, typename ModeT, typename RankT>
static inline void
mttkrp_alto_atomic_cr(int const target_mode, FType** factors, const AltoTensor<LIT>* const at, ModeT nmode = ModeT(), RankT rank = RankT());

template <typename LIT, typename ModeT, typename RankT>
static inline void
mttkrp_alto_da_mem_pull(int const target_mode, FType** factors, const AltoTensor<LIT>* const at, FType** ofibs, ModeT nmode = ModeT(), RankT rank = RankT());

template <typename LIT>
static inline void
create_alto(SparseTensor* spt, AltoTensor<LIT>** at, int nprtn)
{
    //uint64_t ticks;
    double wtime_s, wtime;

    assert(spt->nmodes <= MAX_NUM_MODES);
    int nmode = spt->nmodes;
    IType nnz = spt->nnz;

    AltoTensor<LIT>* _at = (AltoTensor<LIT>*)AlignedMalloc(sizeof(AltoTensor<LIT>));
    assert(_at);

    _at->nmode = nmode;
    _at->nprtn = nprtn;
    _at->nnz = nnz;

    _at->dims = (IType*)AlignedMalloc(nmode * sizeof(IType));
    assert(_at->dims);
    memcpy(_at->dims, spt->dims, nmode * sizeof(IType));

    _at->mode_masks = (LIT*)AlignedMalloc(nmode * sizeof(LIT));
    assert(_at->mode_masks);
    memset(_at->mode_masks, 0, nmode * sizeof(LIT));

#ifdef ALT_PEXT
    _at->mode_pos = (int*)AlignedMalloc(nmode * sizeof(int));
    assert(_at->mode_pos);
#endif
    
    _at->idx = (LIT*)AlignedMalloc(nnz * sizeof(LIT));
    assert(_at->idx);

    _at->vals = (FType*)AlignedMalloc(nnz * sizeof(FType));
    assert(_at->vals);

    _at->prtn_ptr = (IType*)AlignedMalloc((nprtn + 1) * sizeof(IType));
    assert(_at->prtn_ptr);

    _at->prtn_intervals = (Interval*)AlignedMalloc(static_cast<Interval>(nprtn) * static_cast<Interval>(nmode) * sizeof(Interval));
    assert(_at->prtn_intervals);

    _at->cr_masks = (LIT*)AlignedMalloc(nmode * sizeof(LIT));
    assert(_at->cr_masks);
    memset(_at->cr_masks, 0, nmode * sizeof(LIT));

#ifdef OPT_ALTO
    _at->prtn_id = (LPType*)AlignedMalloc(nprtn * sizeof(LPType));
    assert(_at->prtn_id);
    memset(_at->prtn_id, 0, nprtn * sizeof(LPType));

    _at->prtn_mask = (LPType*)AlignedMalloc(nprtn * sizeof(LPType));
    assert(_at->prtn_mask);
    memset(_at->prtn_mask, 0, nprtn * sizeof(LPType));

    _at->prtn_mode_masks = (LPType*)AlignedMalloc(nprtn * nmode * sizeof(LPType));
    assert(_at->prtn_mode_masks);
    memset(_at->prtn_mode_masks, 0, nprtn * nmode * sizeof(LPType));
#endif

    //Setup the linearization scheme.
    //ticks = ReadTSC();
    wtime_s = omp_get_wtime();
    setup_packed_alto(_at, LSB_FIRST, SHORT_FIRST);
    //wtime = ElapsedTime (ReadTSC() - ticks);
    wtime = omp_get_wtime() - wtime_s;
    printf("ALTO: setup time = %f (s)\n", wtime);

    //local buffer
    LIT ALTO_MASKS[MAX_NUM_MODES];
    for (int n = 0; n < nmode; ++n) {
        ALTO_MASKS[n] = _at->mode_masks[n];
    }

    //Linearization
    wtime_s = omp_get_wtime();
    #pragma omp parallel for
    for (IType i = 0; i < nnz; i++) {
        LIT alto = 0;

        _at->vals[i] = spt->vals[i];
        for (int j = 0; j < nmode; j++) {
            alto |= pdep(spt->cidx[j][i], ALTO_MASKS[j]);
        }
        _at->idx[i] = alto;
#ifdef TEST_ALTO
        for (int j = 0; j < nmode; j++) {
            IType mode_idx = 0;
            mode_idx = pext(alto, ALTO_MASKS[j]);
            assert(mode_idx == spt->cidx[j][i]);
        }
#endif
    }
    wtime = omp_get_wtime() - wtime_s;
    printf("ALTO: Linearization time = %f (s)\n", wtime);

    //Sort the nonzeros based on their line position.  
    wtime_s = omp_get_wtime();
    sort_alto(_at);
    wtime = omp_get_wtime() - wtime_s;
    printf("ALTO: sort time = %f (s)\n", wtime);

#ifdef ALT_PEXT
    //Re-encode the ALTO index.

    //local buffer
    int ALTO_POS[MAX_NUM_MODES];
    for (int n = 0; n < nmode; ++n) {
        ALTO_POS[n] = _at->mode_pos[n];
    }
    
    wtime_s = omp_get_wtime();
    #pragma omp parallel for schedule(static)
    for (IType i = 0; i < nnz; i++) {
        LIT index = _at->idx[i];
        LIT new_index = 0;
        for (int n = 0; n < nmode; ++n) {
            LIT mode_idx = pext(index, ALTO_MASKS[n]);
            new_index |= (mode_idx << ALTO_POS[n]);
        }
        _at->idx[i] = new_index;
    }
    //Update the mode masks to match num_bits.
    for (int n = 0; n < nmode; ++n) {
        int num_bits = (sizeof(IType) * 8) - clz(_at->dims[n] - 1);
        _at->mode_masks[n] = ((1 << num_bits) - 1);
    }
    wtime = omp_get_wtime() - wtime_s;
    printf("ALTO: reorder time = %f (s)\n", wtime);
#ifdef ALTO_DEBUG
    for (int n = 0; n < nmode; n++) {
        printf("ALTO_MASKS[%d] = 0x%llx, pos=%d\n", n, _at->mode_masks[n], _at->mode_pos[n]);
    }
#endif
#endif

    //Workload partitioning
    wtime_s = omp_get_wtime();
    prtn_alto(_at, nprtn);
    wtime = omp_get_wtime() - wtime_s;
    printf("ALTO: prtn time = %f (s)\n", wtime);

    *at = _at;
}

template <typename LIT>
static inline void
destroy_alto(AltoTensor<LIT>* at)
{
    AlignedFree(at->dims);
    AlignedFree(at->mode_masks);
#ifdef ALT_PEXT
    AlignedFree(at->mode_pos);
#endif
    AlignedFree(at->idx);
    AlignedFree(at->vals);
    AlignedFree(at->prtn_ptr);
    AlignedFree(at->prtn_intervals);
    AlignedFree(at->cr_masks);
#ifdef OPT_ALTO
    AlignedFree(at->prtn_id);
    AlignedFree(at->prtn_mask);
    AlignedFree(at->prtn_mode_masks);
#endif
    AlignedFree(at);
}

template <typename LIT>
static inline void
create_da_mem(int target_mode, IType rank, AltoTensor<LIT>* at, FType*** ofibs)
{
    int const nprtn = at->nprtn;
    int const nmode = at->nmode;

#ifndef ALTO_PRE_ALLOC
    FType** _ofibs = (FType**)AlignedMalloc(nprtn * sizeof(FType*));
#else
    FType** _ofibs = (FType**)HPMalloc(nprtn * sizeof(FType*));
#endif
    assert(_ofibs);

    {
        double total_storage = 0.0;
        #pragma omp parallel for reduction(+: total_storage) proc_bind(close)
        for (int p = 0; p < nprtn; p++) {
            IType num_fibs = 0;
            if (target_mode == -1) {
                //allocate enough da_mem for all modes with reuse > threshold
                for (int n = 0; n < nmode; ++n) {
                    IType fib_reuse = at->nnz / at->dims[n];
                    if (fib_reuse > MIN_FIBER_REUSE) {
                        Interval const intvl = at->prtn_intervals[p * nmode + n];
                        IType const mode_fibs = intvl.stop - intvl.start + 1;
                        num_fibs = std::max(num_fibs, mode_fibs);
                    }
                }
            }
            else {
                //allocate enough da_mem if reuse > threshold
                IType fib_reuse = at->nnz / at->dims[target_mode];
                if (fib_reuse > MIN_FIBER_REUSE) {
                    Interval const intvl = at->prtn_intervals[p * nmode + target_mode];
                    num_fibs = intvl.stop - intvl.start + 1;
                }
            }

            if (num_fibs) {
#ifndef ALTO_PRE_ALLOC
                _ofibs[p] = (FType*)AlignedMalloc(num_fibs * rank * sizeof(FType));
#else
                _ofibs[p] = (FType*)HPMalloc(num_fibs * rank * sizeof(FType));
#endif
                assert(_ofibs[p]);
            }
            else
                _ofibs[p] = NULL;
            //printf("p%d: storage=%f MB\n", p, ((double) num_fibs * rank * sizeof(FType)) / (1024.0*1024.0));
            total_storage += ((double) num_fibs * rank * sizeof(FType)) / (1024.0*1024.0);
        } // nprtn
        printf("ofibs storage/prtn: %f MB\n", total_storage/(double)nprtn);
    } // omp parallel
    *ofibs = _ofibs;
}

template <typename LIT>
static inline void
destroy_da_mem(AltoTensor<LIT>* at, FType** ofibs, IType rank, int target_mode)
{
#ifndef ALTO_PRE_ALLOC
    int const nprtn = at->nprtn;

    #pragma omp parallel for
    for (int p = 0; p < nprtn; p++) {
        AlignedFree(ofibs[p]);
    }
#else
    unmap_da_mem(at, ofibs, rank, target_mode);
#endif
    AlignedFree(ofibs);
}

template <typename LIT>
static inline void
unmap_da_mem(AltoTensor<LIT>* at, FType** ofibs, IType rank, int target_mode)
{
    int const nprtn = at->nprtn;
    int const nmode = at->nmode;

    #pragma omp parallel for
    for (int p = 0; p < nprtn; p++) {
        IType num_fibs = 0;
        if (target_mode == -1) {
            for (int n = 0; n < nmode; ++n) {
                IType fib_reuse = at->nnz / at->dims[n];
                if (fib_reuse > MIN_FIBER_REUSE) {
                    Interval const intvl = at->prtn_intervals[p * nmode + n];
                    IType const mode_fibs = intvl.stop - intvl.start + 1;
                    num_fibs = std::max(num_fibs, mode_fibs);
                }
            }
        } else {
            IType fib_reuse = at->nnz / at->dims[target_mode];
            if (fib_reuse > MIN_FIBER_REUSE) {
                Interval const intvl = at->prtn_intervals[p * nmode + target_mode];
                num_fibs = intvl.stop - intvl.start + 1;
            }
        }
        if (num_fibs)
            HPFree((void*)ofibs[p], num_fibs * rank * sizeof(FType));
    }
}

template <typename LIT>
static inline void
evaluate_delinearization(AltoTensor<LIT>* at)
{
    int nmode = at->nmode;
    IType nnz = at->nnz;

    //local buffer
    LIT ALTO_MASKS[MAX_NUM_MODES];
    for (int n = 0; n < nmode; ++n) {
        ALTO_MASKS[n] = at->mode_masks[n];
    }

    #pragma omp parallel for
    for (IType i = 0; i < nnz; i++) {
        LIT alto_idx = at->idx[i];
        for (int j = 0; j < nmode; j++) {
#ifndef ALT_PEXT
            volatile IType mode_idx = pext(alto_idx, ALTO_MASKS[j]);
#else
            volatile IType mode_idx = pext(alto_idx, ALTO_MASKS[j], at->mode_pos[j]);
#endif
        }
    }
}

template <typename LIT>
void mttkrp_alto(int target_mode, FType** factors, IType rank, AltoTensor<LIT>* at)
{
    int const nmode = at->nmode;
    IType const nnz = at->nnz;

    LIT* idx = at->idx;
    FType* vals = at->vals;

#ifdef ALTO_MEM_TRACE
    FILE* trace_file = fopen("mem_trace.txt", "w");
    assert(trace_file);
#endif
    
    //local buffer
    LIT ALTO_MASKS[MAX_NUM_MODES];
    #pragma omp simd
    for (int n = 0; n < nmode; ++n) {
        ALTO_MASKS[n] = at->mode_masks[n];
    }

    //FType *row = (FType*)AlignedMalloc(rank * sizeof(FType));
    //assert(row);
    FType row[rank]; //Allocate an auto array of variable size.

    for (IType i = 0; i < nnz; ++i) {
        LIT const alto_idx = idx[i];
#ifdef ALTO_MEM_TRACE
        {
            AddType mem_add = (AddType) & (idx[i]);
            fprintf(trace_file, "%llu\n", mem_add);
        }
#endif
        FType const val = vals[i];
#ifdef ALTO_MEM_TRACE
        {
            AddType mem_add = (AddType) & (vals[i]);
            fprintf(trace_file, "%llu\n", mem_add);
        }
#endif

        #pragma omp simd
        for (IType r = 0; r < rank; ++r) {
            row[r] = val;
        }

        for (int m = 0; m < nmode; ++m) {
            if (m != target_mode) { //input fibers
#ifndef ALT_PEXT
                IType const row_id = pext(alto_idx, ALTO_MASKS[m]);
#else
                IType const row_id = pext(alto_idx, ALTO_MASKS[m], at->mode_pos[m]);
#endif
                #pragma omp simd
                for (IType r = 0; r < rank; ++r) {
                    row[r] *= factors[m][row_id * rank + r];
#ifdef ALTO_MEM_TRACE
                    {
                        AddType mem_add = (AddType) & (factors[m][row_id * rank + r]);
                        fprintf(trace_file, "%llu\n", mem_add);
                    }
#endif
                }
            }
        }

        //Output fibers
#ifndef ALT_PEXT
        IType row_id = pext(alto_idx, ALTO_MASKS[target_mode]);
#else
        IType row_id = pext(alto_idx, ALTO_MASKS[target_mode], at->mode_pos[target_mode]);
#endif
        #pragma omp simd
        for (IType r = 0; r < rank; ++r) {
            factors[target_mode][row_id * rank + r] += row[r];
#ifdef ALTO_MEM_TRACE
            {
                AddType mem_add = (AddType) & (factors[target_mode][row_id * rank + r]);
                fprintf(trace_file, "%llu\n", mem_add);
            }
#endif
        }
    } // for(IType i = 0; i < nnz; i++)
#ifdef ALTO_MEM_TRACE
    fclose(trace_file);
#endif
}

template <typename LIT, typename MT, IType... Ranks>
struct mttkrp_alto_rank_specializer;

template <typename LIT, typename MT>
struct mttkrp_alto_rank_specializer<LIT, MT, 0>
{
    void
    operator()(int target_mode, IType fib_reuse, FType** factors, IType rank, AltoTensor<LIT>* at, omp_lock_t* wlocks, FType** ofibs, MT nmodes)
    {
        if (fib_reuse <= MIN_FIBER_REUSE) {
            if (at->cr_masks[target_mode])
                return mttkrp_alto_atomic_cr(target_mode, factors, at, nmodes, rank);
            return mttkrp_alto_atomic(target_mode, factors, at, nmodes, rank);
        }
        else
            return mttkrp_alto_da_mem_pull(target_mode, factors, at, ofibs, nmodes, rank);
    }
};

template <typename LIT, typename MT, IType Head, IType... Tail>
struct mttkrp_alto_rank_specializer<LIT, MT, 0, Head, Tail...>
{
    void
    operator()(int target_mode, IType fib_reuse, FType** factors, IType rank, AltoTensor<LIT>* at, omp_lock_t* wlocks, FType** ofibs, MT nmodes)
    {
        if (rank == Head) {
            using const_rank = std::integral_constant<IType, Head>;
            if (fib_reuse <= MIN_FIBER_REUSE) {
                if (at->cr_masks[target_mode])
                    return mttkrp_alto_atomic_cr(target_mode, factors, at, nmodes, const_rank());
                return mttkrp_alto_atomic(target_mode, factors, at, nmodes, const_rank());
            }
            else
                return mttkrp_alto_da_mem_pull(target_mode, factors, at, ofibs, nmodes, const_rank());
        }
        else
            return mttkrp_alto_rank_specializer<LIT, MT, 0, Tail...>()(target_mode, fib_reuse, factors, rank, at, wlocks, ofibs, nmodes);
    }
};

template <typename LIT, int... Modes>
struct mttkrp_alto_mode_specializer;

template <typename LIT>
struct mttkrp_alto_mode_specializer<LIT, 0>
{
    void
    operator()(int target_mode, IType fib_reuse, FType** factors, IType rank, AltoTensor<LIT>* at, omp_lock_t* wlocks, FType** ofibs)
    { mttkrp_alto_rank_specializer<LIT, int, ALTO_RANKS_SPECIALIZED>()(target_mode, fib_reuse, factors, rank, at, wlocks, ofibs, at->nmode); }
};

template <typename LIT, int Head, int... Tail>
struct mttkrp_alto_mode_specializer<LIT, 0, Head, Tail...>
{
    void
    operator()(int target_mode, IType fib_reuse, FType** factors, IType rank, AltoTensor<LIT>* at, omp_lock_t* wlocks, FType** ofibs)
    {
        if (at->nmode == Head) {
            using const_mode = std::integral_constant<int, Head>;
            mttkrp_alto_rank_specializer<LIT, const_mode, ALTO_RANKS_SPECIALIZED>()(target_mode, fib_reuse, factors, rank, at, wlocks, ofibs, const_mode());
        }
        else
          mttkrp_alto_mode_specializer<LIT, 0, Tail...>()(target_mode, fib_reuse, factors, rank, at, wlocks, ofibs);
    }
};

template <typename LIT>
static inline void
mttkrp_alto_par(int target_mode, FType** factors, IType rank, AltoTensor<LIT>* at, omp_lock_t* wlocks, FType** ofibs)
{
    IType fib_reuse = at->nnz / at->dims[target_mode];
    return mttkrp_alto_mode_specializer<LIT, ALTO_MODES_SPECIALIZED>()(target_mode, fib_reuse, factors, rank, at, wlocks, ofibs);
}

template <typename LIT>
static inline void
setup_alto(AltoTensor<LIT>* at)
{
    LIT ALTO_MASKS[MAX_NUM_MODES] = {}; //Initialized to zeros by default.

    int nmode = at->nmode;
    int alto_bits_min = 0, alto_bits_max = 0;
    LIT alto_mask = 0;

    for (int n = 0; n < nmode; ++n) {
        int num_bits = (sizeof(IType) * 8) - clz(at->dims[n] - 1);
        alto_bits_min += num_bits;
        alto_bits_max = std::max(alto_bits_max, num_bits);
        printf("num_bits for mode-%d=%d\n", n + 1, num_bits);

        for (int i = 0; i < num_bits - 1; ++i) {
            ALTO_MASKS[n] |= 0x1;
            ALTO_MASKS[n] <<= nmode; //Dilate
        }
        ALTO_MASKS[n] |= 0x1;
        ALTO_MASKS[n] <<= n; //Shift
        alto_mask |= ALTO_MASKS[n];
        at->mode_masks[n] = ALTO_MASKS[n];
#ifdef ALTO_DEBUG
        printf("ALTO_MASKS[%d] = 0x%llx\n", n, ALTO_MASKS[n]);
#endif
    }
    at->alto_mask = alto_mask;

    alto_bits_max *= nmode;
    printf("alto_bits_min=%d, alto_bits_max=%d\n", alto_bits_min, alto_bits_max);
#ifdef ALTO_DEBUG
    printf("alto_mask = 0x%llx\n", alto_mask);
#endif    

    assert(alto_bits_max <= ((int)sizeof(LIT) * 8));
}

struct MPair {
    int mode;
    int bits;
};

// Achieving alto_bits_min requires packing/compression.
template <typename LIT>
static inline void
setup_packed_alto(AltoTensor<LIT>* at, PackOrder po, ModeOrder mo)
{
    LIT ALTO_MASKS[MAX_NUM_MODES] = {}; //initialized to zeros by default

    int nmode = at->nmode;
    int alto_bits_min = 0, alto_bits_max = 0;
    LIT alto_mask = 0;
    int max_num_bits = 0, min_num_bits = sizeof(IType) * 8;
    
    MPair* mode_bits = (MPair*)AlignedMalloc(nmode * sizeof(MPair));
    assert(mode_bits);

    //Initial mode values.
    for (int n = 0; n < nmode; ++n) {
        int mbits = (sizeof(IType) * 8) - clz(at->dims[n] - 1);
        mode_bits[n].mode = n;
        mode_bits[n].bits = mbits;
        alto_bits_min += mbits;
        max_num_bits = std::max(max_num_bits, mbits);
        min_num_bits = std::min(min_num_bits, mbits);
        printf("num_bits for mode-%d=%d\n", n + 1, mbits);
    }
    
#ifdef ALT_PEXT
    //Simple prefix sum
    at->mode_pos[0] = 0;
    for (int n = 1; n < nmode; ++n) {
        at->mode_pos[n] = at->mode_pos[n-1] + mode_bits[n-1].bits;
    }
#endif
    
    alto_bits_max = max_num_bits * nmode;
    //printf("range of mode bits=[%d %d]\n", min_num_bits, max_num_bits);
    printf("alto_bits_min=%d, alto_bits_max=%d\n", alto_bits_min, alto_bits_max);

    assert(alto_bits_min <= ((int)sizeof(LIT) * 8));

    //Assuming we use a power-2 data type for ALTO_idx with a minimum size of a byte
    //int alto_bits = pow(2, (sizeof(int) * 8) - __builtin_clz(alto_bits_min));
    int alto_bits = (int)0x1 << std::max<int>(3, (sizeof(int) * 8) - __builtin_clz(alto_bits_min));
    printf("alto_bits=%d\n", alto_bits);

    double alto_storage = 0;
    alto_storage = at->nnz * (sizeof(FType) + sizeof(LIT));
    printf("Alto format storage:    %g Bytes\n", alto_storage);
    
    alto_storage = at->nnz * (sizeof(FType) + (alto_bits >> 3));
    printf("Alto-power-2 format storage:    %g Bytes\n", alto_storage);

    alto_storage = at->nnz * (sizeof(FType) + (alto_bits_min >> 3));
    printf("Alto-opt format storage:    %g Bytes\n", alto_storage);
    
    {//Dilation & shifting.
        int level = 0, shift = 0, inc = 1;

        //Sort modes, if needed.
        if (mo == SHORT_FIRST)
            std::sort(mode_bits, mode_bits + nmode, [](auto& a, auto& b) { return a.bits < b.bits; });
        else if(mo == LONG_FIRST)
            std::sort(mode_bits, mode_bits + nmode, [](auto& a, auto& b) { return a.bits > b.bits; });
          
        if (po == MSB_FIRST) {
            shift = alto_bits_min - 1;
            inc = -1;
        }
        
        bool done;
        do {
            done = true;

            for (int n = 0; n < nmode; ++n) {
                if (level < mode_bits[n].bits) {
                    ALTO_MASKS[mode_bits[n].mode] |= (LIT)0x1 << shift;
                    shift += inc;
                    done = false;
                }
            }
            ++level;
        } while (!done);
        
        assert(level == (max_num_bits+1));
        assert(po == MSB_FIRST ? (shift == -1) : (shift == alto_bits_min));
    }

    for (int n = 0; n < nmode; ++n) {
        at->mode_masks[n] = ALTO_MASKS[n];
        alto_mask |= ALTO_MASKS[n];
#ifdef ALTO_DEBUG        
        printf("ALTO_MASKS[%d] = 0x%llx\n", n, ALTO_MASKS[n]);
#endif
    }
    at->alto_mask = alto_mask;
#ifdef ALTO_DEBUG
    printf("alto_mask = 0x%llx\n", alto_mask);
#endif
    free(mode_bits);
}

//Use STL for now. MSB-radix sort could be more efficient.
template <typename LIT>
struct APair {
    LIT idx;
    FType val;
};

template <typename LIT>
static inline void
sort_alto(AltoTensor<LIT>* at)
{
    IType nnz = at->nnz;
    APair<LIT>* at_pair = (APair<LIT>*)AlignedMalloc(nnz * sizeof(APair<LIT>));
    assert(at_pair);


    #pragma omp parallel for
    for (IType i = 0; i < nnz; ++i) {
        at_pair[i].idx = at->idx[i];
        at_pair[i].val = at->vals[i];
    }

    std::sort(at_pair, at_pair + nnz, [](auto& a, auto& b) { return a.idx <b.idx; });

    #pragma omp parallel for
    for (IType i = 0; i < nnz; ++i) {
        at->idx[i] = at_pair[i].idx;
        at->vals[i] = at_pair[i].val;
    }

    AlignedFree(at_pair);
}

template <typename LIT>
static inline void
prtn_alto(AltoTensor<LIT>* at, int nprtn)
{
    int nmode = at->nmode;
    IType nnz = at->nnz;
    IType nnz_ptrn = (nnz + nprtn - 1) / nprtn;
    printf("num_ptrn=%d, nnz_ptrn=%llu\n", nprtn, nnz_ptrn);

    //local buffer
    LIT ALTO_MASKS[MAX_NUM_MODES];
    for (int n = 0; n < nmode; ++n) {
        ALTO_MASKS[n] = at->mode_masks[n];
    }

    int alto_bits = popcount(at->alto_mask);

    at->prtn_ptr[0] = 0;
    #pragma omp parallel for schedule(static,1) proc_bind(close)
    for (int p = 0; p < nprtn; ++p) {
        IType start_i = p * nnz_ptrn;
        IType end_i = (p + 1) * nnz_ptrn;

        if (end_i > nnz)
            end_i = nnz;

        if (start_i > end_i)
            start_i = end_i;

        //at->prtn_ptr[p] = start_i;
        at->prtn_ptr[p + 1] = end_i;
        
        // Find the subspace of a given partition.
#ifdef OPT_ALTO
        if (start_i != end_i) {
            LIT alto_idx_s = at->idx[start_i];
            LIT alto_idx_e = at->idx[end_i - 1];
#ifdef ALTO_DEBUG
            //printf("p%d: start alto_idx[%llu]=0x%llx\n", p, start_i, alto_idx_s);
            //printf("p%d: end  alto_idx[%llu]=0x%llx\n", p, end_i-1, alto_idx_e);
#endif 
            LIT mask = at->idx[start_i] ^ at->idx[end_i - 1];
            int prefix_bits = clz(mask) - ((sizeof(LIT) * 8) - alto_bits);

            for (int i = 0; i < prefix_bits; ++i) {
                at->prtn_mask[p] |= (LPType)0x1 << i;
            }
            at->prtn_id[p] = (LPType)((at->idx[start_i] >> (alto_bits - prefix_bits)) & at->prtn_mask[p]);

#ifdef ALTO_DEBUG            
            //printf("p%d: mask=0x%lx, id=0x%lx\n", p, at->prtn_mask[p], at->prtn_id[p]);
#endif
            for (int n = 0; n < nmode; ++n) {
                mask = ((ALTO_MASKS[n] >> (alto_bits - prefix_bits)) & at->prtn_mask[p]);
                at->prtn_mode_masks[p * nmode + n] = mask;
#ifdef ALTO_DEBUG                
                //printf("p%d: mode%d: prefix_mask=0x%llx\n", p, n+1, mask);
#endif                
            }
        }
        else { //empty partition
            printf("p%d: is empty\n", p);
        }
#endif
    }// omp parallel
    
    // O(storage requirements) for conflict resolution, using dense/direct-access storage, 
    // can be computed in constant time from the subspace id. The code below finds tighter bounds 
    // using interval analysis in linear time (where nnz>> nptrn>> nmode).
    #pragma omp parallel for schedule(static,1) proc_bind(close)
    for (int p = 0; p < nprtn; ++p) {
        Interval fib[MAX_NUM_MODES];
        for (int n = 0; n < nmode; ++n) {
            fib[n].start = at->dims[n];
            fib[n].stop = 0;
        }

        for (IType i = at->prtn_ptr[p]; i < at->prtn_ptr[p + 1]; ++i) {
            LIT alto_idx = at->idx[i];
            for (int n = 0; n < nmode; ++n) {
#ifndef ALT_PEXT
                IType mode_idx = pext(alto_idx, ALTO_MASKS[n]);
#else
                IType mode_idx = pext(alto_idx, ALTO_MASKS[n], at->mode_pos[n]);
#endif
                fib[n].start = std::min(fib[n].start, mode_idx);
                fib[n].stop = std::max(fib[n].stop, mode_idx);
            }
        }

        for (int n = 0; n < nmode; ++n) {
            at->prtn_intervals[p * nmode + n].start = fib[n].start;
            at->prtn_intervals[p * nmode + n].stop = fib[n].stop;
        }
    }

#ifdef TEST_ALTO
    // Checking conflicts using interval intersections.
    for (int p = 0; p < nprtn; ++p) {
        for (int n = 0; n < nmode; ++n) {
            printf("p%d: mode%d: conflicts ", p, n+1);
            Interval myintvl = at->prtn_intervals[p*nmode+n];
            for (int i = 0; i < nprtn; ++i) {
                if( i != p) {
                    Interval intvl = at->prtn_intervals[i*nmode+n];
                    if( (intvl.start > myintvl.stop) || (intvl.stop < myintvl.start) ) {
                        //no conflict
                    } else {
                        //conflict
                        printf("%d[%llu %llu] ", i, std::max(intvl.start, myintvl.start), std::min(intvl.stop, myintvl.stop));
                    }
                }
            }
            printf("\n");
        }
    }
#endif

// CR bits can be computed from the interval intersections.
// The code below finds the exact conflicts for each fiber in linear time (where nnz>> nptrn>> nmode).
#ifdef GENERATE_CR_BITS
    //local buffer
    LIT ALTO_CR_MASKS[MAX_NUM_MODES];
    LIT alto_cr_mask = 0;
    
    int free_bits = (sizeof(LIT) * 8) - alto_bits;
    printf("ALTO: free_bits = %d\n", free_bits);
    //short int is large enough to support an appropriate number of partitions. 
    short int** out_fibs = (short int**)AlignedMalloc(nmode * sizeof(short int*));
    assert(out_fibs);
    
    //Setup CR bit masks.
    for (int n = 0; n < nmode; ++n) {
        IType num_fibs = at->dims[n];
        IType fib_reuse = at->nnz / num_fibs;
        printf("ALTO: fib_reuse[%d]=%llu\n", n, fib_reuse);
        if ((fib_reuse <= MIN_FIBER_REUSE) && free_bits) {
            //if(free_bits){
            at->cr_masks[n] = (LIT)0x1 << ((sizeof(LIT) * 8) - free_bits);
            alto_cr_mask |= at->cr_masks[n];
#ifdef ALTO_DEBUG            
            printf("ALTO: cr_masks[%d]=0x%llx\n", n, at->cr_masks[n]);
#endif            
            out_fibs[n] = (short int*)AlignedMalloc(num_fibs * sizeof(short int));
        #pragma omp parallel for
            for (IType i = 0; i < num_fibs; ++i) {
                out_fibs[n][i] = 0; //Not owned by any prtn.
            }
            free_bits--;
        }
        ALTO_CR_MASKS[n] = at->cr_masks[n];
    }//modes
    at->alto_cr_mask = alto_cr_mask;
#ifdef ALTO_DEBUG    
    printf("ALTO: alto_cr_mask=0x%llx\n", alto_cr_mask);
#endif
    
    if (alto_cr_mask) {
        //Find conflicting fibers.
        #pragma omp parallel for schedule(static,1) proc_bind(close)
        for (int p = 0; p < nprtn; ++p) {
            for (IType i = at->prtn_ptr[p]; i < at->prtn_ptr[p + 1]; ++i) {
                LIT alto_idx = at->idx[i];
                for (int n = 0; n < nmode; ++n) {
                    if (ALTO_CR_MASKS[n]) {
#ifndef ALT_PEXT
                        IType mode_idx = pext(alto_idx, ALTO_MASKS[n]);
#else
                        IType mode_idx = pext(alto_idx, ALTO_MASKS[n], at->mode_pos[n]);
#endif
                        bool done;
                        short int old = 0, desired = p + 1; //internal
                        do {
                            done = __atomic_compare_exchange(&out_fibs[n][mode_idx], &old, &desired,
                                                             false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
                            if ((old == p+1) || (old == -1))
                                done = true;
                            else if (old > 0)
                                desired = -1; // external                           
                        } while (!done);   
                    }
                } //modes
            } //nnz
        } //prtns
        
        //Set cr_bits.
        #pragma omp parallel for schedule(static,1) proc_bind(close)
        for (int p = 0; p < nprtn; ++p) {
            for (IType i = at->prtn_ptr[p]; i < at->prtn_ptr[p + 1]; ++i) {
                LIT alto_idx = at->idx[i];
                LIT cr_mask = 0;
                for (int n = 0; n < nmode; ++n) {
                    if (ALTO_CR_MASKS[n]) {
#ifndef ALT_PEXT
                        IType mode_idx = pext(alto_idx, ALTO_MASKS[n]);
#else
                        IType mode_idx = pext(alto_idx, ALTO_MASKS[n], at->mode_pos[n]);
#endif

                        if (out_fibs[n][mode_idx] == -1)
                            cr_mask |= ALTO_CR_MASKS[n];
                    }
                }//modes
                if (cr_mask)
                    at->idx[i] = alto_idx | cr_mask;
            }//nnz
        }//prtns

        for (int n = 0; n < nmode; ++n) {
            if (ALTO_CR_MASKS[n])
                AlignedFree(out_fibs[n]);
        }//modes
    }
    AlignedFree(out_fibs);
#endif
}

template <typename LIT, typename ModeT, typename RankT>
static inline void
mttkrp_alto_atomic(int const target_mode, FType** factors, const AltoTensor<LIT>* const at, ModeT nmode, RankT rank)
{
    int const nprtn = at->nprtn;
    assert(at->nmode == nmode);
    #pragma omp parallel for schedule(static,1) proc_bind(close)
    for (int p = 0; p < nprtn; ++p) {
        //local buffer
        LIT ALTO_MASKS[MAX_NUM_MODES];
        for (int n = 0; n < nmode; ++n) {
            ALTO_MASKS[n] = at->mode_masks[n];
        }
        
        //FType *row = (FType*)AlignedMalloc(rank * sizeof(FType));
        //assert(row);
        FType row[rank]; //Allocate an auto array of variable size.

        LIT* const idx = at->idx;
        FType* const vals = at->vals;
        IType const nnz_s = at->prtn_ptr[p];
        IType const nnz_e = at->prtn_ptr[p + 1];

        for (IType i = nnz_s; i < nnz_e; ++i) {
            FType const val = vals[i];
            LIT const alto_idx = idx[i];

            #pragma omp simd
            for (IType r = 0; r < rank; ++r) {
                row[r] = val;
            }

            for (int m = 0; m < nmode; ++m) {
                if (m != target_mode) { //input fibers
#ifndef ALT_PEXT
                    IType const row_id = pext(alto_idx, ALTO_MASKS[m]);
#else
                    IType const row_id = pext(alto_idx, ALTO_MASKS[m], at->mode_pos[m]);
#endif
                    #pragma omp simd
                    for (IType r = 0; r < rank; r++) {
                        row[r] *= factors[m][row_id * rank + r];
                    }
                }
            }

            //Output fibers
#ifndef ALT_PEXT
            IType const row_id = pext(alto_idx, ALTO_MASKS[target_mode]);
#else
            IType const row_id = pext(alto_idx, ALTO_MASKS[target_mode], at->mode_pos[target_mode]);
#endif
            for (IType r = 0; r < rank; ++r) {
                #pragma omp atomic update
                factors[target_mode][row_id * rank + r] += row[r];
            }
        } //nnzs
    } //prtns
}

template <typename LIT, typename ModeT, typename RankT>
static inline void
mttkrp_alto_atomic_cr(int const target_mode, FType** factors, const AltoTensor<LIT>* const at, ModeT nmode, RankT rank)
{
    int const nprtn = at->nprtn;
    assert(at->nmode == nmode);
    LIT const cr_mask = at->cr_masks[target_mode];

    #pragma omp parallel for schedule(static,1) proc_bind(close)
    for (int p = 0; p < nprtn; ++p) {
        //local buffer
        LIT ALTO_MASKS[MAX_NUM_MODES];
        for (int n = 0; n < nmode; ++n) {
            ALTO_MASKS[n] = at->mode_masks[n];
        }

        //FType *row = (FType*)AlignedMalloc(rank * sizeof(FType));
        //assert(row);
        FType row[rank]; //Allocate an auto array of variable size.

        LIT* const idx = at->idx;
        FType* const vals = at->vals;
        IType const nnz_s = at->prtn_ptr[p];
        IType const nnz_e = at->prtn_ptr[p + 1];

        for (IType i = nnz_s; i < nnz_e; ++i) {
            FType const val = vals[i];
            LIT const alto_idx = idx[i];

            #pragma omp simd
            for (IType r = 0; r < rank; ++r) {
                row[r] = val;
            }

            for (int m = 0; m < nmode; ++m) {
                if (m != target_mode) { //input fibers
#ifndef ALT_PEXT
                    IType const row_id = pext(alto_idx, ALTO_MASKS[m]);
#else
                    IType const row_id = pext(alto_idx, ALTO_MASKS[m], at->mode_pos[m]);
#endif
                    #pragma omp simd
                    for (IType r = 0; r < rank; r++) {
                        row[r] *= factors[m][row_id * rank + r];
                    }
                }
            }

            //Output fibers
#ifndef ALT_PEXT
            IType const row_id = pext(alto_idx, ALTO_MASKS[target_mode]);
#else
            IType const row_id = pext(alto_idx, ALTO_MASKS[target_mode], at->mode_pos[target_mode]);
#endif
            if (alto_idx & cr_mask) {
                for (IType r = 0; r < rank; ++r) {
                    #pragma omp atomic update
                    factors[target_mode][row_id * rank + r] += row[r];
                }
            }
            else {
                #pragma omp simd
                for (IType r = 0; r < rank; ++r) {
                    factors[target_mode][row_id * rank + r] += row[r];
                }
            }
        } //nnzs
    } //prtns
}

template <typename LIT>
static inline void
mttkrp_alto_lock_cr(int const target_mode, FType** factors, IType const rank, const AltoTensor<LIT>* const at, omp_lock_t* wlocks)
{
    int const nprtn = at->nprtn;
    int const nmode = at->nmode;
    LIT const cr_mask = at->cr_masks[target_mode];

    #pragma omp parallel for schedule(static,1) proc_bind(close)
    for (int p = 0; p < nprtn; ++p) {
        //local buffer
        LIT ALTO_MASKS[MAX_NUM_MODES];
        for (int n = 0; n < nmode; ++n) {
            ALTO_MASKS[n] = at->mode_masks[n];
        }

        //FType *row = (FType*)AlignedMalloc(rank * sizeof(FType));
        //assert(row);
        FType row[rank]; //Allocate an auto array of variable size.

        LIT* const idx = at->idx;
        FType* const vals = at->vals;
        IType const nnz_s = at->prtn_ptr[p];
        IType const nnz_e = at->prtn_ptr[p + 1];

        for (IType i = nnz_s; i < nnz_e; ++i) {
            FType const val = vals[i];
            LIT const alto_idx = idx[i];

            #pragma omp simd
            for (IType r = 0; r < rank; ++r) {
                row[r] = val;
            }

            for (int m = 0; m < nmode; ++m) {
                if (m != target_mode) { //input fibers
#ifndef ALT_PEXT
                    IType const row_id = pext(alto_idx, ALTO_MASKS[m]);
#else
                    IType const row_id = pext(alto_idx, ALTO_MASKS[m], at->mode_pos[m]);
#endif
                    #pragma omp simd
                    for (IType r = 0; r < rank; r++) {
                        row[r] *= factors[m][row_id * rank + r];
                    }
                }
            }

            //Output fibers
#ifndef ALT_PEXT
            IType const row_id = pext(alto_idx, ALTO_MASKS[target_mode]);
#else
            IType const row_id = pext(alto_idx, ALTO_MASKS[target_mode], at->mode_pos[target_mode]);
#endif

            if (alto_idx & cr_mask)
                omp_set_lock(&(wlocks[row_id]));
            #pragma omp simd
            for (IType r = 0; r < rank; ++r) {
                factors[target_mode][row_id * rank + r] += row[r];
            }
            if (alto_idx & cr_mask)
                omp_unset_lock(&(wlocks[row_id]));
        } //nnzs
    } //prtns
}

template <typename LIT, typename ModeT, typename RankT>
static inline void
mttkrp_alto_da_mem_pull(int const target_mode, FType** factors, const AltoTensor<LIT>* const at, FType** ofibs, ModeT nmode, RankT rank)
{
    int const nprtn = at->nprtn;
    assert(nmode == at->nmode);
    IType const num_fibs = at->dims[target_mode];

    #pragma omp parallel proc_bind(close)
    {
        #pragma omp for schedule(static, 1)
        for (int p = 0; p < nprtn; ++p) {
            //local buffer
            LIT ALTO_MASKS[MAX_NUM_MODES];

            #pragma omp simd
            for (int n = 0; n < nmode; ++n) {
                ALTO_MASKS[n] = at->mode_masks[n];
            }
            //FType *row = (FType*)AlignedMalloc(rank * sizeof(FType));
            //assert(row);
            FType row[rank]; //Allocate an auto array of variable size.

            FType* const out = ofibs[p];
            Interval const intvl = at->prtn_intervals[p * nmode + target_mode];
            IType const offset = intvl.start;
            IType const stop = intvl.stop;
            memset(out, 0, (stop - offset + 1) * rank * sizeof(FType));

            LIT* const idx = at->idx;
            FType* const vals = at->vals;
            IType const nnz_s = at->prtn_ptr[p];
            IType const nnz_e = at->prtn_ptr[p + 1];

            for (IType i = nnz_s; i < nnz_e; ++i) {
                FType const val = vals[i];
                LIT const alto_idx = idx[i];

                #pragma omp simd
                for (IType r = 0; r < rank; ++r) {
                    row[r] = val;
                }

                for (int m = 0; m < nmode; ++m) {
                    if (m != target_mode) { //input fibers
#ifndef ALT_PEXT
                        IType const row_id = pext(alto_idx, ALTO_MASKS[m]) * rank;
#else
                        IType const row_id = pext(alto_idx, ALTO_MASKS[m], at->mode_pos[m]) * rank;
#endif
                        #pragma omp simd 
                        for (IType r = 0; r < rank; ++r) {
                            row[r] *= factors[m][row_id + r];
                        }
                    }
                }

                //Output fibers
#ifndef ALT_PEXT
                IType row_id = pext(alto_idx, ALTO_MASKS[target_mode]) - offset;
#else
                IType row_id = pext(alto_idx, ALTO_MASKS[target_mode], at->mode_pos[target_mode]) - offset;
#endif
                row_id *= rank;
                #pragma omp simd
                for (IType r = 0; r < rank; ++r) {
                    out[row_id + r] += row[r];
                }
            } //nnzs
        } //prtns

        //pull-based accumulation
        #pragma omp for schedule(static)
        for (IType i = 0; i < num_fibs; ++i) {
            for (int p = 0; p < nprtn; p++)
            {
                Interval const intvl = at->prtn_intervals[p * nmode + target_mode];
                IType const offset = intvl.start;
                IType const stop = intvl.stop;

                if ((i >= offset) && (i <= stop)) {
                    FType* const out = ofibs[p];
                    IType const j = i - offset;
                    #pragma omp simd
                    for (IType r = 0; r < rank; r++) {
                        factors[target_mode][i * rank + r] += out[j * rank + r];
                    }
                }
            } //prtns
        } //ofibs
    } // omp parallel
}

#endif
