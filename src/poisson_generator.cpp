#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>

#include "poisson_generator.hpp"
#include "rng_stream.hpp"


#define BSIZE 100000
#define SEED  1234


static void **rng_streams = NULL;


static void UniformRandom(IType n, FType low, FType high, FType *a)
{
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();        
        void *stream = rng_streams[tid];
        IType chunk = (n + nthreads - 1)/nthreads; 
        IType start = tid * chunk;
        IType end = start + chunk;
        end = (end > n ? n : end);
        if (start < n) {
            RNGStreamUniform(stream, low, high, end - start, &(a[start]));
        }
    }
}


static IType Binning(FType rv, IType nbins, FType *bins)
{
    IType start = 0;
    IType end = nbins - 1;
    while (start <= end) {
        IType mid = (end + start)/2;
        if (rv > bins[mid]) {
            start = mid + 1;
        } else if (rv == bins[mid]) {
            if (mid == nbins - 1) {
                mid = mid - 1;
            }
            return mid;
        } else {
            end = mid - 1;
        }
    }
    return end;
}


static void ProbSample(IType nprob, FType *prob,
                       IType nsamples, IType *samples, IType lds,
                       FType *work, IType lwork)
{
    for (IType n = 0; n < nsamples; n+=lwork) {
        IType len = (n + lwork < nsamples ? lwork : nsamples - n);
        UniformRandom(len, 0.0, 1.0, work);
        #pragma omp parallel for
        for (IType i = 0; i < len; i++) {
            samples[(n + i) * lds] = Binning(work[i], nprob + 1, prob);
        }
    }    
}


static void ProbCount(IType nprob, FType *prob, IType nsamples, IType *count,
                      FType *work, IType lwork)
{
    for (IType n = 0; n < nsamples; n+=lwork) {
        IType len = (n + lwork < nsamples ? lwork : nsamples - n);
        UniformRandom(len, 0.0, 1.0, work);
        for (IType i = 0; i < len; i++) {
            IType idx = Binning(work[i], nprob + 1, prob);
            count[idx]++;
        }
    }    
}


static void RandomPerm(IType n, IType nbig, IType *perm)
{
    for (IType i = 0; i < n; i++) {
        perm[i] = i;
    }
    for (IType i = 0; i < nbig; i++) {
        IType idx = (rand()%(n - i)) + i;
        IType tmp = perm[i];
        perm[i] = perm[idx];
        perm[idx] = tmp;
    }
}


void CreatePoissonGenerator(int mode, IType *dims, PoissonGenerator **pg_)
{
    PoissonGenerator *pg =
        (PoissonGenerator *)AlignedMalloc(sizeof(PoissonGenerator));
    assert(pg != NULL);
    
    assert(mode > 0);
    pg->mode = mode;
    IType max_dims = 0.0;
    for (int n = 0; n < mode; n++) {
        assert(dims[n] >= 1);
        max_dims = (max_dims > dims[n] ? max_dims : dims[n]);
    }   
    pg->dims = (IType *)AlignedMalloc(sizeof(IType) * mode);
    assert(pg->dims != NULL);
    memcpy(pg->dims, dims, sizeof(IType) * mode);

    // create random streams
    int nthreads = omp_get_max_threads();
    rng_streams = (void **)AlignedMalloc(sizeof(void *) * nthreads);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        CreateRNGStream(SEED + tid, &(rng_streams[tid]));
    }
    
    *pg_ = pg;
}


void DestroyPoissonGenerator(PoissonGenerator *pg)
{
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        DestroyRNGStream(rng_streams[tid]);
    }
    AlignedFree(rng_streams);
    AlignedFree(pg->dims);
    AlignedFree(pg);
}


static void GenerateRandomModel(PoissonGenerator *pg, KruskalModel *M)
{
    int mode = pg->mode;    
    IType rank = M->rank;
    IType *dims = pg->dims;
    IType max_dims = 0;
    for (int n = 0; n < mode; n++) {
        max_dims = (max_dims > dims[n] ? max_dims : dims[n]);
    }
    IType *perm = (IType *)AlignedMalloc(sizeof(IType) * (max_dims + 1));
    assert(perm != NULL);
    FType *vals = (FType *)AlignedMalloc(sizeof(FType) * (max_dims + 1));
    assert(vals != NULL);
    
    for (int n = 0; n < mode; n++) {
        FType *U_n = M->U[n];
        IType dim_n = dims[n];
        // initially set all entires to a fixed 0.1
        for (IType i = 0; i < dim_n * rank; i++) {
            U_n[i] = 0.1;
        }
        // take 20% of the entries in that column randomly
        IType nbig = (IType)ceil(0.2 * dim_n);
        // do column by column
        for (IType r = 0; r < rank; r++) {
            RandomPerm(dim_n, nbig, perm);
            UniformRandom(nbig, 1.0, 10.0 * rank, vals);
            // for those 20% of entires, set it to 1 + 10Rx, where x is a
            // random value from [0 1]
            for (IType i = 0; i < nbig; i++) {
                U_n[perm[i] * rank + r] = vals[i];
            }
        }
    }
    
    // lambda is random number from [0 1]
    UniformRandom(rank, 0.0, 1.0, M->lambda);
    
    // normalize
    KruskalModelNormalize(M);      
    FType tmp = 0.0;
    for (IType i = 0; i < rank; i++) {
        tmp += M->lambda[i];
    }
    assert(tmp);
    tmp = 1.0/tmp;
    for (IType i = 0; i < rank; i++) {
        M->lambda[i] *= tmp;
    }

    AlignedFree(perm);
    AlignedFree(vals);    
}


static void GenerateEdges(PoissonGenerator *pg, KruskalModel *M,
                          IType num_edges, IType **eidx_)
{
    int mode = pg->mode;
    IType *dims = pg->dims;
    IType rank = M->rank;
    FType *lambda = M->lambda;
    IType max_dims = 0;
    for (int n = 0; n < mode; n++) {
        max_dims = (max_dims > dims[n] ? max_dims : dims[n]);
    }

    FType *bins = (FType *)AlignedMalloc(sizeof(FType) * (max_dims + 1));
    assert(bins != NULL);
    // determine how many samples per component
    bins[0] = 0.0; // (rank + 1) * 1
    for (IType i = 0; i < rank; i++) {
        bins[i + 1] = lambda[i] + bins[i];
        /*
        if((bins[i+1] > 1.0)) {
            fprintf (stderr, "%d %d %g\n", i, rank, bins[i+1]);
        }
         */
        // assert(bins[i + 1] <= 1.0);
        bins[i + 1] = (bins[i + 1] < 1.0 ? bins[i + 1] : 1.0);
    }
    IType *r_count = (IType *)AlignedMalloc(sizeof(rank) * (max_dims + 1));
    assert(r_count != NULL);
    FType *work = (FType *)AlignedMalloc(sizeof(FType) * BSIZE);
    assert(work != NULL);
    memset(r_count, 0, sizeof(IType) * rank);
    ProbCount(rank, bins, num_edges, r_count, work, BSIZE);
    
    IType *eidx = (IType *)AlignedMalloc(sizeof(IType) * mode * num_edges);
    assert(eidx != NULL);
    // for each component
    IType count = 0;
    for (IType r = 0; r < rank; r++) {
        IType nc = r_count[r];
        // for each subscript find a value
        for (int n = 0; n < mode; n++) {
            IType dim_n = dims[n];
            FType *U = M->U[n];
            bins[0] = 0.0;  // (dim_n + 1) * 1
            for (IType i = 0; i < dim_n; i++) {
                bins[i + 1] = U[i * rank + r] + bins[i];
                bins[i + 1] = (bins[i + 1] < 1.0 ? bins[i + 1] : 1.0);
            }
            ProbSample(dim_n, bins, nc, &(eidx[count * mode + n]),
                       mode, work, BSIZE);
        }
        count += nc;
    }

    // rescale lambda so that it is proportional to the number of nonzeros
    for (IType r = 0; r < rank; r++) {
        M->lambda[r] *= (FType)num_edges;
    }

    AlignedFree(r_count);
    AlignedFree(bins);
    AlignedFree(work);
    *eidx_ = eidx;
}


static void ConstructCIdx(int mode, IType *dims, IType num_edges, IType *eidx,
                          IType *nnz_, FType **vals_)
{
    IType *cidx_tmp = (IType *)AlignedMalloc(sizeof(IType) * num_edges * mode);
    assert(cidx_tmp != NULL);
    IType *cidx_old = eidx;
    IType *cidx_new = cidx_tmp;
    FType *vals = (FType *)AlignedMalloc(sizeof(FType) * num_edges);
    assert(vals != NULL);
    IType max_dims = 0;
    for (int n = 0; n < mode; n++) {
        max_dims = (max_dims > dims[n] ? max_dims : dims[n]);
    }

    // radix sort
    IType *offset = (IType *)AlignedMalloc(sizeof(IType) * (max_dims + 1));
    assert(offset != NULL);
    for (int n = 0; n < mode; n++) {
        memset(offset, 0, sizeof(IType) * dims[n]);
        for (IType i = 0; i < num_edges; i++) {
            IType idx = cidx_old[i * mode + n];
            offset[idx + 1] += 1;
        }
        for (IType i = 0; i < dims[n]; i++) {
            offset[i + 1] += offset[i];
        }
        for (IType i = 0; i < num_edges; i++) {
            IType idx = cidx_old[i * mode + n];
            IType pos = offset[idx];
            memcpy(&(cidx_new[pos * mode]), &(cidx_old[i * mode]),
                   sizeof(IType) * mode);
            offset[idx] += 1;            
        }
        IType *tmp = cidx_new;
        cidx_new = cidx_old;
        cidx_old = tmp;
    }
    if (cidx_old != cidx_tmp) {
        memcpy(cidx_tmp, cidx_old, sizeof(IType) * mode * num_edges);
    }

    // find the same points
    memcpy(eidx, cidx_tmp, sizeof(IType) * mode);
    vals[0] = 1.0;    
    IType nnz = 1;
    for (IType i = 1; i < num_edges; i++) {
        bool dup = true;
        for (int n = 0; n < mode; n++) {
            if (cidx_tmp[i * mode + n] != cidx_tmp[(i - 1) * mode + n]) {
                dup = false;
                break;
            }
        }
        if (dup == true) {
            vals[nnz - 1] += 1.0;
        } else {
            memcpy(&(eidx[nnz * mode]), &(cidx_tmp[i * mode]),
                   sizeof(IType) * mode);
            vals[nnz] = 1.0;
            nnz++;
        }
    }

    AlignedFree(offset);
    AlignedFree(cidx_tmp);
    *nnz_ = nnz;
    *vals_ = vals;
}


void PoissonGeneratorRun(PoissonGenerator *pg, IType num_edges, IType rank,
                         KruskalModel **M_, SparseTensor **X_)
{
    int mode = pg->mode;
    IType *dims = pg->dims;   
    for (int n = 0; n < mode; n++) {
        assert(dims[n] >= rank);
    }
    assert(num_edges > 0);
    printf("Generating a sparse tensor: ");
    for (int n = 0; n < mode - 1; n++) {
        printf("%llux", (unsigned long long)dims[n]);
    }
    printf("%llu, rank = %llu ...\n",
        (unsigned long long)dims[mode - 1], (unsigned long long)rank);
    
    // create a random Kruskal model
    KruskalModel *M;
    CreateKruskalModel(mode, dims, rank, &M);
    GenerateRandomModel(pg, M);

    // generate edges
    IType *eidx;
    GenerateEdges(pg, M, num_edges, &eidx);

    // compute cidx and vals
    IType nnz = 0;
    FType *vals;
    ConstructCIdx(mode, dims, num_edges, eidx, &nnz, &vals);

    SparseTensor *X;
    CreateSparseTensor((IType) mode, dims, nnz, eidx, vals, &X);
    
    printf("    Nonzeros = %llu\n", (unsigned long long)nnz);
    AlignedFree(eidx);
    AlignedFree(vals);

    // crate a sparse tensor    
    *M_ = M;
    *X_ = X;
}
