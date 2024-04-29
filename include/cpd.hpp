#ifndef CPD_HPP_
#define CPD_HPP_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <omp.h>

#ifdef MKL
#include <mkl.h>
#else
#include <cblas.h>
#include <lapacke.h>
typedef size_t MKL_INT;
#endif

#include "common.hpp"
#include "sptensor.hpp"
#include "kruskal_model.hpp"
#include "alto.hpp"

// Adaptive Linearized Tensor Order (ALTO) APIs
template <typename LIT>
void cpd_alto(AltoTensor<LIT>* AT, KruskalModel* M, int max_iters, double epsilon);

template <typename LIT>
double cpd_fit_alto(AltoTensor<LIT>* AT, KruskalModel* M, FType** grams, FType* U_mttkrp, FType normAT);

// Reference COO implementations
void cpd(SparseTensor* X, KruskalModel* M, int max_iters, double epsilon);
double cpd_fit(SparseTensor* X, KruskalModel* M, FType** grams, FType* U_mttkrp);
void mttkrp_par(SparseTensor* X, KruskalModel* M, IType mode, omp_lock_t* writelocks);
void mttkrp(SparseTensor* X, KruskalModel* M, IType mode);

// CPD kernels
static void pseudo_inverse(FType** grams, KruskalModel* M, IType mode);
static void destroy_grams(FType** grams, KruskalModel* M);
static void init_grams(FType*** grams, KruskalModel* M);

template <typename LIT>
void cpd_alto(AltoTensor<LIT>* AT, KruskalModel* M, int max_iters, double epsilon)
{
  fprintf(stdout, "Running ALTO CP-ALS with %d max iterations and %.2e epsilon\n",
          max_iters, epsilon);
#ifdef MKL
  mkl_set_dynamic(1);
#endif

  int nmodes = AT->nmode;
  IType* dims = AT->dims;
  IType rank = M->rank;

  // Set up temporary data structures
  IType nthreads = omp_get_max_threads();
  FType* scratch = (FType*) AlignedMalloc(sizeof(FType) * dims[nmodes - 1] * rank);
  assert(scratch);
  FType ** lambda_sp = (FType **) AlignedMalloc(sizeof(FType*) * nthreads);
  assert(lambda_sp);
  #pragma omp parallel for
  for (IType t = 0; t < nthreads; ++t) {
    lambda_sp[t] = (FType *) AlignedMalloc(sizeof(FType) * rank);
  }

  // Keep track of the fit for convergence check
  double fit = 0.0;
  double prev_fit = 0.0;

  // Compute ttnormsq to later compute fit
  FType normAT = 0.0;
  ValType* vals = AT->vals;
  IType nnz = AT->nnz;

  #pragma omp parallel for reduction(+:normAT) schedule(static)
  for(IType i = 0; i < nnz; ++i) {
    normAT += (FType)vals[i] * (FType)vals[i];
  }

  // Compute initial A**T * A for every mode
  FType** grams;
  init_grams(&grams, M);

  // Create local fiber copies
  FType ** ofibs = NULL;
  create_da_mem(-1, rank, AT, &ofibs);

  // Timers
  double wtime_tot = omp_get_wtime();
  double wtime_mttkrp_tot = 0.0, wtime_pseudoinv_tot = 0.0;
  double wtime_copy_tot = 0.0, wtime_norm_tot = 0.0;
  double wtime_update_tot = 0.0, wtime_fit_tot = 0.0;

  int i_ = max_iters;
  for(int i = 0; i < max_iters; i++) {
    double wtime_it = omp_get_wtime();
    double wtime_mttkrp = 0.0, wtime_pseudoinv = 0.0;
    double wtime_copy = 0.0, wtime_norm = 0.0;
    double wtime_update = 0.0, wtime_fit = 0.0;

    for(int j = 0; j < AT->nmode; j++) {
      double wtime_tmp;
      wtime_tmp = omp_get_wtime();
      ParMemset(M->U[j], 0, sizeof(FType) * dims[j] * rank);
      wtime_copy += (omp_get_wtime() - wtime_tmp);

      // MTTKRP
      wtime_tmp = omp_get_wtime();
      mttkrp_alto_par(j, M->U, rank, AT, NULL, ofibs);
      wtime_mttkrp += omp_get_wtime() - wtime_tmp;

      // If it is the last mode, save the MTTKRP result for fit calculation.
      wtime_tmp = omp_get_wtime();
      if(j == nmodes - 1) {
        ParMemcpy(scratch, M->U[j], sizeof(FType) * dims[j] * rank);
      }
      wtime_copy += omp_get_wtime() - wtime_tmp;

      // Pseudo inverse
      wtime_tmp = omp_get_wtime();
      pseudo_inverse(grams, M, j);
      wtime_pseudoinv += omp_get_wtime() - wtime_tmp;

      // Normalize columns
      wtime_tmp = omp_get_wtime();
      if(i == 0) {
        KruskalModelNorm(M, j, MAT_NORM_2, lambda_sp);
      } else {
        KruskalModelNorm(M, j, MAT_NORM_MAX, lambda_sp);
      }
      wtime_norm += omp_get_wtime() - wtime_tmp;

      // Update the Gram matrices
      wtime_tmp = omp_get_wtime();
      update_gram(grams[j], M, j);
      // PrintFPMatrix("Grams", rank, rank, grams[j], rank);
      wtime_update += omp_get_wtime() - wtime_tmp;

      // PrintFPMatrix("Lambda", 1, rank, M->lambda, rank);
    } // for each mode

    // Calculate fit
    wtime_fit = omp_get_wtime();
    fit = cpd_fit_alto(AT, M, grams, scratch, normAT);
    wtime_fit_tot       += omp_get_wtime() - wtime_fit;

    wtime_mttkrp_tot    += wtime_mttkrp;
    wtime_pseudoinv_tot += wtime_pseudoinv;
    wtime_copy_tot      += wtime_copy;
    wtime_norm_tot      += wtime_norm;
    wtime_update_tot    += wtime_update;
    wtime_it            = omp_get_wtime() - wtime_it;

    printf("it: %d\t fit: %g\t fit-delta: %g\ttime(for MTTKRP): %.4f s (%.4f s)\n", i, fit,
           fabs(prev_fit - fit), wtime_it, wtime_mttkrp);
    // if fit - oldfit < epsilon, quit
    if((i > 0) && (fabs(prev_fit - fit) < epsilon)) {
      i_ = i+1;
      break;
    }

    prev_fit = fit;
  } // for max_iters

  wtime_tot = omp_get_wtime() - wtime_tot;
  printf("Total time (for MTTKRP):\t %.4f s (%.4f s)\n", wtime_tot, wtime_mttkrp_tot);
  printf("Total     MTTKRP    PseudoInv MemCopy   Normalize Update    Fit\n");
  printf("%07.4f   %07.4f   %07.4f   %07.4f   %07.4f   %07.4f   %07.4f\n",
            wtime_tot, wtime_mttkrp_tot, wtime_pseudoinv_tot, wtime_copy_tot, wtime_norm_tot, wtime_update_tot, wtime_fit_tot
  );

  printf("Per iteration\n%07.4f   %07.4f   %07.4f   %07.4f   %07.4f   %07.4f   %07.4f\n",
         wtime_tot/i_, wtime_mttkrp_tot/i_, wtime_pseudoinv_tot/i_, wtime_copy_tot/i_, wtime_norm_tot/i_, wtime_update_tot/i_, wtime_fit_tot/i_
  );

  // cleanup
  #pragma omp parallel for
  for (IType t = 0; t < nthreads; ++t) {
    free(lambda_sp[t]);
  }
  free(lambda_sp);
  free(scratch);
  destroy_da_mem(AT, ofibs, rank, -1);
  destroy_grams(grams, M);
}


template <typename LIT>
double cpd_fit_alto(AltoTensor<LIT>* AT, KruskalModel* M, FType** grams, FType* U_mttkrp, FType normAT)
{
  // Calculate inner product between AT and M
  // This can be done via sum(sum(P.U{dimorder(end)} .* U_mttkrp) .* lambda');
  IType rank = M->rank;
  IType nmodes = AT->nmode;
  IType* dims = AT->dims;

  FType* accum = (FType*) AlignedMalloc(sizeof(FType) * rank);
  assert(accum);
  memset(accum, 0, sizeof(FType) * rank);

  // Computing the inner product for M->U and U_mttkrp
  #pragma omp parallel for reduction(+: accum[:rank]) schedule(static)
  for(IType i = 0; i < dims[nmodes - 1]; ++i) {
    #pragma omp simd
    for(IType j = 0; j < rank; ++j) {
      accum[j] += M->U[nmodes - 1][i * rank + j] * U_mttkrp[i * rank + j];
    }
  }

  FType inner_prod = 0.0;
  #pragma omp simd
  for(IType i = 0; i < rank; ++i) {
    inner_prod += accum[i] * M->lambda[i];
  }

  // Calculate norm of factor matrices
  // This can be done via taking the hadamard product between all the gram
  // matrices, and then summing up all the elements and taking the square root
  // of the absolute value
  FType* tmp_gram = (FType*) AlignedMalloc(rank * rank * sizeof(FType));
  assert(tmp_gram);

  #pragma omp parallel for schedule(static)
  #pragma unroll
  for(IType i = 0; i < rank; ++i) {
    #pragma omp simd
    for(IType j = 0; j < i + 1; ++j) {
      tmp_gram[i * rank + j] = M->lambda[i] * M->lambda[j];
    }
  }

  // Calculate the hadamard product between all the Gram matrices
  for(IType i = 0; i < nmodes; ++i) {
    #pragma omp parallel for schedule(static)
    for(IType j = 0; j < rank; ++j) {
      #pragma omp simd
      for(IType k = 0; k < j + 1; ++k) {
        tmp_gram[j * rank + k] *= grams[i][j * rank + k];
      }
    }
  }

  FType normU = 0.0;
  #pragma unroll
  for(IType i = 0; i < rank; ++i) {
    #pragma omp simd
    for(IType j = 0; j < i; ++j) {
      normU += tmp_gram[i * rank + j] * 2;
    }
  }
  #pragma omp simd
  for (IType i = 0; i < rank; ++i) {
      normU += tmp_gram[i * rank + i];
  }

  normU = fabs(normU);

  // Calculate residual using the above
  FType norm_residual = normAT + normU - 2 * inner_prod;
  if (norm_residual > 0.0) {
      norm_residual = sqrt(norm_residual);
  }
  FType ret = 1 - (norm_residual / sqrt(normAT));

  // free memory
  free(accum);
  free(tmp_gram);

  return ret;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void cpd(SparseTensor* X, KruskalModel* M, int max_iters, double epsilon)
{
  fprintf(stdout, "Running CP-ALS with %d max iterations and %.2e epsilon\n",
          max_iters, epsilon);

  int nmodes = X->nmodes;
  IType* dims = X->dims;
  IType rank = M->rank;

  // set up temporary data structures
  FType* scratch = (FType*) AlignedMalloc(sizeof(FType) * dims[nmodes - 1] *
                                          rank);
  assert(scratch);
  IType nthreads = omp_get_max_threads();
  FType ** lambda_sp = (FType **) AlignedMalloc(sizeof(FType*) * nthreads);
  assert(lambda_sp);
  #pragma omp parallel for
  for (IType t = 0; t < nthreads; ++t) {
    lambda_sp[t] = (FType *) AlignedMalloc(sizeof(FType) * rank);
    assert(lambda_sp[t]);
  }

  // set up OpenMP locks
  IType max_mode_len = 0;
  for(int i = 0; i < M->mode; i++) {
    if(max_mode_len < M->dims[i]) {
        max_mode_len = M->dims[i];
    }
  }
  omp_lock_t* writelocks = (omp_lock_t*) AlignedMalloc(sizeof(omp_lock_t) *
                                                       max_mode_len);
  assert(writelocks);
  for(IType i = 0; i < max_mode_len; i++) {
    omp_init_lock(&(writelocks[i]));
  }

  // keep track of the fit for convergence check
  double fit = 0.0;
  double prev_fit = 0.0;

  // compute initial A**T * A for every mode
  FType** grams;
  init_grams(&grams, M);

  for(int i = 0; i < max_iters; i++) {
    for(int j = 0; j < X->nmodes; j++) {
      // MTTKRP
      memset(M->U[j], 0, sizeof(FType) * dims[j] * rank);
      mttkrp_par(X, M, j, writelocks);
      // mttkrp(X, M, j);
      // if it is the last mode, save the MTTKRP result for fit calculation
      if(j == nmodes - 1) {
        memcpy(scratch, M->U[j], sizeof(FType) * dims[j] * rank);
      }

      pseudo_inverse(grams, M, j);

      // Normalize columns
      if(i == 0) {
        KruskalModelNorm(M, j, MAT_NORM_2, lambda_sp);
      } else {
        KruskalModelNorm(M, j, MAT_NORM_MAX, lambda_sp);
      }

      // Update the Gram matrices
      update_gram(grams[j], M, j);
      // PrintFPMatrix("Grams", rank, rank, grams[j], rank);

      // PrintFPMatrix("Lambda", 1, rank, M->lambda, rank);
    } // for each mode

    // calculate fit
    fit = cpd_fit(X, M, grams, scratch);

    // if fit - oldfit < epsilon, quit
    if((i > 0) && (fabs(prev_fit - fit) < epsilon)) {
      printf("it: %d\t fit: %g\t fit-delta: %g\n", i, fit,
             fabs(prev_fit - fit));
      break;
    } else {
      printf("it: %d\t fit: %g\t fit-delta: %g\n", i, fit,
             fabs(prev_fit - fit));
    }
    prev_fit = fit;
  } // for max_iters

  // cleanup
  #pragma omp parallel for
  for (IType t = 0; t < nthreads; ++t) {
    free(lambda_sp[t]);
  }
  free(lambda_sp);
  free(scratch);
  for(IType i = 0; i < max_mode_len; i++) {
    omp_destroy_lock(&(writelocks[i]));
  }
  free(writelocks);
  destroy_grams(grams, M);
}

double cpd_fit(SparseTensor* X, KruskalModel* M, FType** grams, FType* U_mttkrp)
{
  // Calculate inner product between X and M
  // This can be done via sum(sum(P.U{dimorder(end)} .* U_mttkrp) .* lambda');
  IType rank = M->rank;
  IType nmodes = X->nmodes;
  IType* dims = X->dims;

  FType* accum = (FType*) AlignedMalloc(sizeof(FType) * rank);
  assert(accum);
  memset(accum, 0, sizeof(FType*) * rank);
  #if 0
  for(IType i = 0; i < dims[nmodes - 1]; i++) {
    for(IType j = 0; j < rank; j++) {
      accum[j] += M->U[nmodes - 1][i * rank + j] * U_mttkrp[i * rank + j];
    }
  }
  #else
  #pragma omp parallel for schedule(static)
  for(IType j = 0; j < rank; j++) {
    for(IType i = 0; i < dims[nmodes - 1]; i++) {
      accum[j] += M->U[nmodes - 1][i * rank + j] * U_mttkrp[i * rank + j];
    }
  }
  #endif

  FType inner_prod = 0.0;
  for(IType i = 0; i < rank; i++) {
    inner_prod += accum[i] * M->lambda[i];
  }

  // Calculate norm(X)^2
  IType nnz = X->nnz;
  FType* vals = X->vals;
  FType normX = 0.0;
  #pragma omp parallel for reduction(+:normX) schedule(static)
  for(IType i = 0; i < nnz; i++) {
    normX += vals[i] * vals[i];
  }

  // Calculate norm of factor matrices
  // This can be done via taking the hadamard product between all the gram
  // matrices, and then summing up all the elements and taking the square root
  // of the absolute value
  FType* tmp_gram = (FType*) AlignedMalloc(sizeof(FType) * rank * rank);
  assert(tmp_gram);
  #pragma omp parallel for schedule(dynamic)
  for(IType i = 0; i < rank; i++) {
    for(IType j = 0; j < i + 1; j++) {
      tmp_gram[i * rank + j] = M->lambda[i] * M->lambda[j];
    }
  }

  // calculate the hadamard product between all the Gram matrices
  for(IType i = 0; i < nmodes; i++) {
    #pragma omp parallel for schedule(dynamic)
    for(IType j = 0; j < rank; j++) {
      for(IType k = 0; k < j + 1; k++) {
        tmp_gram[j * rank + k] *= grams[i][j * rank + k];
      }
    }
  }

  FType normU = 0.0;
  for(IType i = 0; i < rank; i++) {
    for(IType j = 0; j < i; j++) {
      normU += tmp_gram[i * rank + j] * 2;
    }
    normU += tmp_gram[i * rank + i];
  }
  normU = fabs(normU);

  // Calculate residual using the above
  FType norm_residual = normX + normU - 2 * inner_prod;
  if (norm_residual > 0.0) {
    norm_residual = sqrt(norm_residual);
  }
  FType ret = 1 - (norm_residual / sqrt(normX));

  // free memory
  free(accum);
  free(tmp_gram);

  return ret;
}

void mttkrp_par(SparseTensor* X, KruskalModel* M, IType mode, omp_lock_t* writelocks)
{
  IType nmodes = X->nmodes;
  IType nnz = X->nnz;
  IType** cidx = X->cidx;
  IType rank = M->rank;

  int max_threads = omp_get_max_threads();
  FType* rows = (FType*) AlignedMalloc(sizeof(FType) * rank * max_threads);
  assert(rows);

  #pragma omp parallel
  {
    // get thread ID
    int tid = omp_get_thread_num();
    FType* row = &(rows[tid * rank]);

    #pragma omp for schedule(static)
    for(IType i = 0; i < nnz; i++) {
      // initialize temporary accumulator
      for(IType r = 0; r < rank; r++) {
        row[r] = X->vals[i];
      }

      // calculate mttkrp for the current non-zero
      for(IType m = 0; m < nmodes; m++) {
        if(m != mode) {
          IType row_id = cidx[m][i];
          for(IType r = 0; r < rank; r++) {
            row[r] *= M->U[m][row_id * rank + r];
          }
        }
      }

      // update destination row
      IType row_id = cidx[mode][i];
      omp_set_lock(&(writelocks[row_id]));
      for(IType r = 0; r < rank; r++) {
        M->U[mode][row_id * rank + r] += row[r];
      }
      omp_unset_lock(&(writelocks[row_id]));
    } // for each nonzero
  } // #pragma omp parallel


  // free memory
  free(rows);
}

void mttkrp(SparseTensor* X, KruskalModel* M, IType mode)
{
  IType nmodes = X->nmodes;
  //IType* dims = X->dims;
  IType nnz = X->nnz;
  IType** cidx = X->cidx;
  IType rank = M->rank;

  FType row[rank];

  for(IType i = 0; i < nnz; i++) {
    // initialize temporary accumulator
    for(IType r = 0; r < rank; r++) {
      row[r] = X->vals[i];
    }

    // calculate mttkrp for the current non-zero
    for(IType m = 0; m < nmodes; m++) {
      if(m != mode) {
        IType row_id = cidx[m][i];
        for(IType r = 0; r < rank; r++) {
          row[r] *= M->U[m][row_id * rank + r];
        }
      }
    }

    // update destination row
    IType row_id = cidx[mode][i];
    for(IType r = 0; r < rank; r++) {
      M->U[mode][row_id * rank + r] += row[r];
    }
  } // for each nonzero

  // PrintFPMatrix("MTTKRP", dims[mode], rank, M->U[mode], rank);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
static void pseudo_inverse(FType** grams, KruskalModel* M, IType mode)
{
  IType rank = M->rank;
  IType nmodes = (IType) M->mode;

  // Calculate V
  IType m = 0;
  if(mode == 0) {
    m = 1;
  }
  memcpy(grams[mode], grams[m], sizeof(FType) * rank * rank);

  #pragma unroll
  for(IType i = m + 1; i < nmodes; i++) {
    if(i != mode) {
      #pragma omp simd
      for(IType j = 0; j < rank * rank; j++) {
        grams[mode][j] *= grams[i][j];
      }
    }
  }
  // PrintFPMatrix("V", rank, rank, grams[mode], rank);

  FType* scratch = (FType*) AlignedMalloc(sizeof(FType) * rank * rank);
  assert(scratch);

  // Do manual transpose for gram matrix so that we use col_major instead of row_major
  double * vals = (double*) AlignedMalloc(sizeof(double) * rank * rank);
  assert(vals);

  #pragma unroll
  for (IType i = 0; i < rank; ++i) {
    #pragma omp simd
    for (IType j = 0; j < rank; ++j) {
      vals[j * rank + i] = grams[mode][i * rank + j];
    }
  }

  memcpy(grams[mode], vals, rank * rank * sizeof(FType));
  // Backup grams[mode] in case Cholesky fails
  memcpy(scratch, vals, sizeof(FType) * rank * rank);

  free(vals);

  // Try using Cholesky to find the pseudoinvser of V
  // Setup parameters for LAPACK calls
  // convert IType to int
  char uplo = 'L';
  lapack_int _rank = (lapack_int)rank;
  lapack_int I = (lapack_int)M->dims[mode];
  lapack_int info;

  POTRF(&uplo, &_rank, grams[mode], &_rank, &info);

  if(info == 0) {
    // Cholesky was successful - use it to find the pseudo_inverse and multiply
    // it with the MTTKRP result
    POTRS(&uplo, &_rank, &I, grams[mode], &_rank,
          M->U[mode], &_rank, &info);

  } else {
    // Otherwise use rank-deficient solver, GELSY
    // Restore V
    memcpy(grams[mode], scratch, sizeof(FType) * rank * rank);
    //PrintFPMatrix("gram matrix when fallback", rank, rank, grams[mode], rank);
    // Fill up the upper part
    #pragma unroll
    for(IType i = 0; i < rank; i++) {
      #pragma omp simd
      for(IType j = i; j < rank; j++) {
        grams[mode][i * rank + j] = grams[mode][j * rank + i];
      }
    }

    // Use a rank-deficient solver
    lapack_int* jpvt = (lapack_int*) AlignedMalloc(sizeof(lapack_int) * rank);
    memset(jpvt, 0, sizeof(lapack_int) * rank);
    lapack_int lwork = -1;
    FType work_qr;
    lapack_int ret_rank;
    lapack_int info_dgelsy;
    FType rcond = -1.0f;//1.1e-16;

    GELSY(&_rank, &_rank, &I, grams[mode], &_rank, M->U[mode], &_rank,
          jpvt, &rcond, &ret_rank, &work_qr, &lwork, &info_dgelsy);
    FType* work = (FType*) AlignedMalloc(sizeof(FType) * work_qr);
    GELSY(&_rank, &_rank, &I, grams[mode], &_rank, M->U[mode], &_rank,
          jpvt, &rcond, &ret_rank, work, &lwork, &info_dgelsy);

    fprintf(stderr, "\t Mode %llu Min Norm Solve: %d\nDGELSS effective rank: %d\n", mode, info_dgelsy, ret_rank);
    free(work);
    free(jpvt);
  }

  // PrintFPMatrix("Factor Matrix", M->dims[mode], rank, M->U[mode], rank);

  // Restore gram
  memcpy(grams[mode], scratch, sizeof(FType) * rank * rank);

  // cleanup
  free(scratch);
}

static void destroy_grams(FType** grams, KruskalModel* M)
{
  for(int i = 0; i < M->mode; i++) {
    free(grams[i]);
  }
  free(grams);
}

static void init_grams(FType*** grams, KruskalModel* M)
{
  IType rank = M->rank;
  FType** _grams = (FType**) AlignedMalloc(sizeof(FType*) * M->mode);
  assert(_grams);
  for(int i = 0; i < M->mode; i++) {
    _grams[i] = (FType*) AlignedMalloc(sizeof(FType) * rank * rank);
    assert(_grams[i]);
  }

  for(int i = 0; i < M->mode; i++) {
    update_gram(_grams[i], M, i);
  }

  *grams = _grams;
}

#endif // CPD_HPP_
