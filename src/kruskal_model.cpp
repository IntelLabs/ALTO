#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <omp.h>

#include "kruskal_model.hpp"

void update_gram(FType* gram, KruskalModel* M, int mode)
{
  MKL_INT m = M->dims[mode];
  MKL_INT n = M->rank;
  MKL_INT lda = n;
  MKL_INT ldc = n;
  FType alpha = 1.0;
  FType beta = 0.0;

  CBLAS_ORDER layout = CblasRowMajor;
  CBLAS_UPLO uplo = CblasLower;
  CBLAS_TRANSPOSE trans = CblasTrans;
  SYRK(layout, uplo, trans, n, m, alpha, M->U[mode], lda, beta, gram, ldc);
}


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


void KruskalModelRandomInit(KruskalModel *M, unsigned int seed)
{
    for (IType i = 0; i < M->rank; i++) {
        M->lambda[i] = (FType) 1.0;
    }

    srand(seed);

    for (int n = 0; n < M->mode; n++) {
        //#pragma omp parallel for simd
        for (IType i = 0; i < M->dims[n] * M->rank; i++) {
            // M->U[n][i] = (FType) 1.0;
            // M->U[n][i] = (FType) 0.000000001;
            M->U[n][i] = (FType) rand() / RAND_MAX;
        }
    }
}


void KruskalModelNormalize(KruskalModel *M)
{
    IType rank = M->rank;
    FType * col_norm = (FType *) AlignedMalloc(sizeof(FType) * rank);

    for (int n = 0; n < M->mode; n++) {
        // For each factor
        IType dim = M->dims[n];
        FType *U = M->U[n];

        #pragma omp simd
        for (IType j = 0; j < rank; ++j) {
            col_norm[j] = 0.0;
        }

        // Calculate the norm for this column
        #pragma omp parallel for schedule(static) reduction(+: col_norm[:rank])
        for (IType i = 0; i < dim; i++) {
            #pragma omp simd
            for(IType j = 0; j < rank; j++) {
                #if ROW
                col_norm[j] += fabs(U[i * rank + j]);
                #else
                col_norm[j] += fabs(U[j * dim + i]);
                #endif
            }
        }

        // Normalize the elements
        #pragma omp parallel for schedule(static)
        for (IType i = 0; i < dim; i++) {
            #pragma omp simd
            for(IType j = 0; j < rank; j++) {
                #if ROW
                U[i * rank + j] /=  col_norm[j];
                #else
                U[j * dim + i] /=  col_norm[j];
                #endif
            }
        }

        #pragma omp simd
        for (IType j = 0; j < rank; j++) {
            // Absorb the norm into lambda
            M->lambda[j] *= col_norm[j];
        }
    }
    free(col_norm);
}

void KruskalModelNormalizeMode(KruskalModel *M, int n)
{
    IType rank = M->rank;
    IType dim = M->dims[n];
    FType *U = M->U[n];
    FType * col_norm = (FType *) AlignedMalloc(sizeof(FType) * rank);

    #pragma omp simd
    for (IType j = 0; j < rank; ++j) {
        col_norm[j] = 0.0;
    }

    // Calculate the norm for this column
    #pragma omp parallel for schedule(static) reduction(+: col_norm[:rank])
    for (IType i = 0; i < dim; i++) {
        #pragma omp simd
        for(IType j = 0; j < rank; j++) {
            #if ROW
            col_norm[j] += fabs(U[i * rank + j]);
            #else
            col_norm[j] += fabs(U[j * dim + i]);
            #endif
        }
    }

    // Normalize the elements
    #pragma omp parallel for schedule(static)
    for (IType i = 0; i < dim; i++) {
        #pragma omp simd
        for(IType j = 0; j < rank; j++) {
            #if ROW
            U[i * rank + j] /=  col_norm[j];
            #else
            U[j * dim + i] /=  col_norm[j];
            #endif
        }
    }

    #pragma omp simd
    for (IType j = 0; j < rank; j++) {
        // Absorb the norm into lambda
        M->lambda[j] *= col_norm[j];
    }
    free(col_norm);
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
            _lambda[i] = std::max(_lambda[i], (FType)1.);
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

    // Initialize lambda scracthpad
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

    #pragma omp parallel for schedule(static)
    for(IType i = 0; i < dim; i++) {
        #pragma omp simd
        for(IType j = 0; j < rank; j++) {
            #if ROW
            U[i * rank + j] *= lambda[j];
            #else
            U[j * dim + i] *= lambda[j];
            #endif
        }
    }

    #pragma omp simd
    for(IType j = 0; j < rank; j++) {
        lambda[j] = 1.0;
    }
}

double KruskalTensorFit()
{
  return 0.0;
}
