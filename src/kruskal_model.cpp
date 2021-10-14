#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <omp.h>

#include "kruskal_model.hpp"

// #define DEBUG 1


void ExportKruskalModel(KruskalModel *M, char *file_path)
{
    // factor matrices
    for (int n = 0; n < M->mode; n++) {
        char str[1000];
        sprintf(str, "%s.%d.out", file_path, n);
        FILE *fp = fopen(str, "w");
        assert(fp);
        fprintf(fp, "matrix\n");
        fprintf(fp, "2\n");
        fprintf(fp, "%llu %llu\n", M->dims[n], M->rank);
        for (IType i = 0; i < M->rank; i++) {
            for (IType j = 0; j < M->dims[n]; j++) {
                fprintf(fp, "%.20lf\n", M->U[n][j * M->rank + i]);
            }
        }
        fclose(fp);
    }

    // lambda
    char str[1000];
    sprintf(str, "%s.lambda.out", file_path);
    FILE* fp = fopen(str, "w");
    assert(fp);
    fprintf(fp, "vector\n");
    fprintf(fp, "1\n");
    fprintf(fp, "%llu\n", M->rank);
    for(IType i = 0; i < M->rank; i++) {
        fprintf(fp, "%.20lf\n", M->lambda[i]);
    }
    fclose(fp);
}


void PrintKruskalModel(KruskalModel *M)
{
    for (int n = 0; n < M->mode; n++) {
        printf("mode %d:\n", n);
        for (IType j = 0; j < M->dims[n]; j++) {
            for (IType i = 0; i < M->rank; i++) {
                printf("%g ", M->U[n][j * M->rank + i]);
            }
            printf("\n");
        }
    }
    printf("lambda:");
    for (IType r = 0; r < M->rank; r++) {
        printf("%g ", M->lambda[r]);
    }
    printf("\n");
}


void CreateKruskalModel(int mode, IType *dims, IType rank, KruskalModel **M_)
{
    assert(mode >= 1);
    assert(rank >= 1);
    for (int n = 0; n < mode; n++) {
        assert(dims[n] >= 1);
        //assert(rank <= dims[n]);
    }
    
    KruskalModel *M = (KruskalModel *)AlignedMalloc(sizeof(KruskalModel));
    assert(M != NULL);
    M->mode = mode;
    M->rank = rank;
    M->dims = (IType *)AlignedMalloc(mode * sizeof(IType));
    assert(M->dims != NULL);
    memcpy(M->dims, dims, sizeof(IType) * mode);  
    M->U = (FType **)AlignedMalloc(mode * sizeof(FType *));
    assert(M->U != NULL);
    for (int n = 0; n < mode; n++) {
        M->U[n] = (FType *)AlignedMalloc(dims[n] * rank * sizeof(FType));
        assert(M->U[n] != NULL);
    }
    M->lambda = (FType *)AlignedMalloc(rank * sizeof(FType));
    assert(M->lambda != NULL);

    *M_ = M;
}

// Expand the size of the Kruskal Model given new dimensions
// This version is the one that maintains the full temporal factor matrices
/*
void GrowKruskalModel(IType *dims, KruskalModel **M_, int streaming_mode)
{
    IType mode = (*M_)->mode;
    IType rank = (*M_)->rank;
    IType * old_dims = (*M_)->dims; // previous dimension sizes
    FType ** U = (*M_)->U;
    FType * lambda = (*M_)->lambda;

    assert(rank >= 1);
    assert(mode >= 1);

    KruskalModel *M = (KruskalModel *)AlignedMalloc(sizeof(KruskalModel));
    assert(M != NULL);

    M->mode = mode;
    M->rank = rank;
    M->dims = (IType *) AlignedMalloc(mode * sizeof(IType));
    assert(M->dims != NULL);
    memcpy(M->dims, dims, sizeof(IType) * mode); // copy new dims to new kruskal

    M->U = (FType **)AlignedMalloc(mode * sizeof(FType *));
    assert(M->U != NULL);
    for (int n = 0; n < mode; ++n) {
        if (n != streaming_mode) {
            M->U[n] = (FType *)AlignedMalloc(dims[n] * rank * sizeof(FType));
            assert(M->U[n] != NULL);

            // Copy values from previous factor matrix to expanded factor matrix
            memcpy(M->U[n], U[n], old_dims[n] * rank * sizeof(FType));

            // Fill exceeding values with zeros
            int added_nrows = dims[n] - old_dims[n];
            if (added_nrows > 0) { // If we need to add more rows
                for (int r = 0; r < added_nrows * rank; ++r) {
                    // Set all added entries to 0.0
                    M->U[n][old_dims[n] * rank + r] = 0.0;
                }
            }
        } else { 
            // Grow the streaming mode by 1
            int current_dim = (*M_)->dims[n];
            M->U[n] = (FType *)AlignedMalloc((current_dim + 1) * rank * sizeof(FType));
            assert(M->U[n] != NULL);

            // Copy values from previous factor matrix to expanded factor matrix
            memcpy(M->U[n], U[n], current_dim * rank * sizeof(FType));

            for (int r = 0; r < rank; ++r) {
                M->U[n][current_dim * rank + r] = 0.0;
            }
            M->dims[n] = current_dim + 1;
        }
    }
    M->lambda = (FType *) AlignedMalloc(rank * sizeof (FType));
    assert(M->lambda != NULL);
    memcpy(M->lambda, lambda, sizeof(FType) * rank);
    DestroyKruskalModel(*M_);
    *M_ = M;
}
*/
void GrowKruskalModel(IType *dims, KruskalModel **M_, FillValueType FillValueType_)
{
    IType mode = (*M_)->mode;
    IType rank = (*M_)->rank;
    IType * old_dims = (*M_)->dims; // previous dimension sizes
    FType ** U = (*M_)->U;
    FType * lambda = (*M_)->lambda;

    assert(rank >= 1);
    assert(mode >= 1);

    KruskalModel *M = (KruskalModel *)AlignedMalloc(sizeof(KruskalModel));
    assert(M != NULL);

    M->mode = mode;
    M->rank = rank;
    M->dims = (IType *) AlignedMalloc(mode * sizeof(IType));
    assert(M->dims != NULL);
    memcpy(M->dims, dims, sizeof(IType) * mode); // copy new dims to new kruskal

    M->U = (FType **)AlignedMalloc(mode * sizeof(FType *));
    assert(M->U != NULL);
    for (int n = 0; n < mode; ++n) {
        M->U[n] = (FType *)AlignedMalloc(dims[n] * rank * sizeof(FType));
        assert(M->U[n] != NULL);

        // Copy values from previous factor matrix to expanded factor matrix
        memcpy(M->U[n], U[n], old_dims[n] * rank * sizeof(FType));

        // Fill exceeding values with zeros
        int added_nrows = dims[n] - old_dims[n];
        if (added_nrows > 0) { // If we need to add more rows
            int seed = time(NULL);
            srand(seed);
            #pragma omp parallel
            {
                unsigned int local_seed = seed + omp_get_thread_num();
                #pragma omp for simd schedule(static)
                for (int r = 0; r < added_nrows * rank; ++r) {
                    if (FillValueType_ == FILL_RANDOM) {
                        M->U[n][old_dims[n] * rank + r] = (FType) rand_r (&local_seed) / RAND_MAX;;
                    } 
                    else if (FillValueType_ == FILL_ZEROS) {
                        M->U[n][old_dims[n] * rank + r] = 0.0;
                    }
                }
            }
        }
    }
    M->lambda = (FType *) AlignedMalloc(rank * sizeof (FType));
    assert(M->lambda != NULL);
    memcpy(M->lambda, lambda, sizeof(FType) * rank);
    DestroyKruskalModel(*M_);
    *M_ = M;
}

void CopyKruskalModel(KruskalModel **prev_M_, KruskalModel **M_)
{
    IType mode = (*M_)->mode;
    IType rank = (*M_)->rank;
    IType * dims = (*M_)->dims;
    FType ** U = (*M_)->U;
    FType * lambda = (*M_)->lambda;

    // Check source Kruskal model is not empty
    assert(rank >= 1);
    assert(mode >= 1);

    KruskalModel *M = (KruskalModel *)AlignedMalloc(sizeof(KruskalModel));
    assert(M != NULL);

    M->mode = mode;
    M->rank = rank;
    M->dims = (IType *) AlignedMalloc(mode * sizeof(IType));
    assert(M->dims != NULL);
    memcpy(M->dims, dims, sizeof(IType) * mode);

    M->U = (FType **)AlignedMalloc(mode * sizeof(FType *));
    assert(M->U != NULL);
    for (int n = 0; n < mode; ++n) {
        M->U[n] = (FType *)AlignedMalloc(dims[n] * rank * sizeof(FType));
        assert(M->U[n] != NULL);
        memcpy(M->U[n], U[n], dims[n] * rank * sizeof(FType));
    }
    M->lambda = (FType *) AlignedMalloc(rank * sizeof (FType));
    assert(M->lambda != NULL);
    memcpy(M->lambda, lambda, sizeof(FType) * rank);

    *prev_M_ = M;
}

void GrowTimeFactorMatrix(KruskalModel **M_, int streaming_mode) {
    int current_dim = (*M_)->dims[streaming_mode];
    int rank = (*M_)->rank;
    FType * tmp = (FType *)AlignedMalloc((current_dim + 1) * sizeof(FType));
    memcpy(tmp, (*M_)->U[streaming_mode], sizeof(FType) * current_dim * rank);
    AlignedFree((*M_)->U[streaming_mode]);

    // Set Time Factor to 0 for newly added row
    for (int r = 0; r < rank; ++r) {
        tmp[current_dim * rank + r] = 0.0;
    }

    (*M_)->U[streaming_mode] = tmp;
    (*M_)->dims[streaming_mode] = current_dim + 1;
}

void KruskalModelRandomInit(KruskalModel *M, unsigned int seed)
{
    for (IType i = 0; i < M->rank; i++) {
        M->lambda[i] = (FType) 1.0;
    }

    srand(seed);
    for (int n = 0; n < M->mode; n++) {
        #pragma omp parallel
        {
            unsigned int local_seed = seed + omp_get_thread_num();
            #pragma omp for simd schedule(static)
            for (IType i = 0; i < M->dims[n] * M->rank; i++) {
                #if DEBUG == 1
                M->U[n][i] = (FType) 1.0;
                #else
                M->U[n][i] = (FType) rand_r (&local_seed) / RAND_MAX;
                #endif
            }
        }
    }
}

void KruskalModelZeroInit(KruskalModel *M)
{
    for (IType i = 0; i < M->rank; i++) {
        M->lambda[i] = (FType) 0.0;
    }

    for (int n = 0; n < M->mode; n++) {
        #pragma omp parallel
        {
            #pragma omp for simd schedule(static)
            for (IType i = 0; i < M->dims[n] * M->rank; i++) {
                M->U[n][i] = (FType) 0.0;
            }
        }
    }
}


void KruskalModelNormalize(KruskalModel *M)
{
    for (int n = 0; n < M->mode; n++) {
        // For each factor
        IType dim = M->dims[n];
        for (IType j = 0; j < M->rank; j++) {
            // Calculate the norm for this column
            FType tmp = 0.0;
            for (IType k = 0; k < dim; k++) {
                #if ROW
                tmp = tmp + fabs(M->U[n][k * M->rank + j]);
                #else
                tmp = tmp + fabs(M->U[n][j * dim + k]);
                #endif
            }
            // Normalize the elements 
            for (IType k = 0; k < dim; k++) {
                #if ROW
                M->U[n][k * M->rank + j] = M->U[n][k * M->rank + j] / tmp;
                #else
                M->U[n][j * dim + k] = M->U[n][j * dim + k] / tmp;
                #endif
            }
            // Absorb the norm into lambda
            M->lambda[j] = M->lambda[j] * tmp;
        }
    }
}

static void inline Mat2Norm(IType dim, IType rank, FType * vals, FType * lambda, FType ** scratchpad)
{
    IType nthreads = omp_get_max_threads();
    // Find the max value in each column and store it in lambda
    #pragma omp parallel proc_bind(close)
     {
         IType tid = omp_get_thread_num();
         FType * _lambda = scratchpad[tid];

        #pragma omp for schedule(static) 
        for(IType i = 0; i < dim; i++) {
            #pragma omp simd
            for(IType j = 0; j < rank; j++) {
                _lambda[j] += vals[i * rank + j] * vals[i * rank + j];
            }
        }

        #pragma omp for reduction(+: lambda[:rank]) schedule(static)
        for (IType t = 0; t < nthreads; ++t) {
            #pragma omp simd
            for (IType j = 0; j < rank; ++j) {
                lambda[j] += scratchpad[t][j];
            }
        }
    }

    #pragma omp for schedule(static)
    for(IType j=0; j < rank; ++j) {
      lambda[j] = sqrt(lambda[j]);
    }

    #pragma omp parallel for schedule(static)
    for(IType i = 0; i < dim; i++) {
        #pragma omp simd
        for(IType j = 0; j < rank; j++) {
            vals[i * rank + j] /= lambda[j];
        }
    }

}

static void inline MatMaxNorm(IType dim, IType rank, FType * vals, FType * lambda, FType ** scratchpad)
{
    IType nthreads = omp_get_max_threads();

    // Find the max value in each column and store it in lambda
    #pragma omp parallel proc_bind(close)
     {
         IType tid = omp_get_thread_num();
         FType * _lambda = scratchpad[tid];

        #pragma omp for schedule(static) 
        for(IType i = 0; i < dim; i++) {
            #pragma omp simd
            for(IType j = 0; j < rank; j++) {
                _lambda[j] = std::max(_lambda[j], vals[i * rank + j]);
            }
        }

        // If any entry is less than 1, set it to 1
        #pragma omp simd
        for(IType i = 0; i < rank; i++) {
            _lambda[i] = std::max(_lambda[i], 1.);
        }

        #pragma omp for reduction(max: lambda[:rank]) schedule(static)
        for (IType t = 0; t < nthreads; ++t) {
          #pragma omp simd
          for (IType j = 0; j < rank; ++j) {
              lambda[j] = std::max(lambda[j], scratchpad[t][j]);
          }
        }
    }

    #pragma omp parallel for schedule(static)
    for(IType i = 0; i < dim; i++) {
        #pragma omp simd
        for(IType j = 0; j < rank; j++) {
            vals[i * rank + j] /= lambda[j];
        }
    }
}

void KruskalModelNorm(KruskalModel* M, IType mode, mat_norm_type which, FType ** scratchpad)
{
    IType dim = M->dims[mode];
    IType rank = M->rank;
    FType * vals = M->U[mode];
    FType * lambda = M->lambda;

    IType nthreads = omp_get_max_threads();

    // Initialize lambda scratchpad
    #pragma omp parallel for schedule(static)
    for (IType t = 0; t < nthreads; ++t) {
        #pragma omp simd
        for (IType r = 0; r < rank; ++r) {
            scratchpad[t][r] = 0.0;
        }
    }
    #pragma omp simd
    for (IType r = 0; r < rank; ++r) {
        lambda[r] = 0.0;
    }

    // Call normalization accordingly...
    switch (which) {
    case MAT_NORM_2:
        Mat2Norm(dim, rank, vals, lambda, scratchpad);
        break;
    case MAT_NORM_MAX:
        MatMaxNorm(dim, rank, vals, lambda, scratchpad);
        break;

    default:
        abort();
    }
}


void DestroyKruskalModel(KruskalModel *M)
{
    AlignedFree(M->dims);
    for (int n = 0; n < M->mode; n++) {
        AlignedFree(M->U[n]);
    }
    AlignedFree(M->U);
    AlignedFree(M->lambda);
    AlignedFree(M);
}

void RedistributeLambda (KruskalModel *M, int n)
{
    FType *U = M->U[n];
    IType rank = M->rank;
    FType *lambda = M->lambda;
    IType dim = M->dims[n];

    for(IType r = 0; r < rank; r++) {
        for(IType i = 0; i < dim; i++) {
            #if ROW
            U[i * rank + r] = U[i * rank + r] * lambda[r];
            #else
            U[r * dim + i] = U[r * dim + i] * lambda[r];
            #endif
        }
        lambda[r] = 1.0;
    }
}

double KruskalTensorFit()
{
  return 0.0;
}
