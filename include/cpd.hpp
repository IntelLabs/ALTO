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
#include "stream_matrix.hpp"
#include "util.hpp"
#include "kruskal_model.hpp"
#include "alto.hpp"

// #define DEBUG 1

// Adaptive Linearized Tensor Order (ALTO) APIs
template <typename LIT>
void cpd_alto(AltoTensor<LIT>* AT, KruskalModel* M, int max_iters, double epsilon);

template <typename LIT>
double cpd_fit_alto(AltoTensor<LIT>* AT, KruskalModel* M, Matrix ** grams, FType* U_mttkrp, FType normAT);

// Reference COO implementations
void cpd(SparseTensor* X, KruskalModel* M, int max_iters, double epsilon);
void streaming_cpd(SparseTensor* X, KruskalModel* M, KruskalModel * prev_M, Matrix** grams, int max_iters, double epsilon, int streaming_mode, int iter);

template <typename LIT>
void streaming_cpd_alto(AltoTensor<LIT>* AT, KruskalModel* M, KruskalModel * prev_M, Matrix** grams, int max_iters, double epsilon, int streaming_mode, int iter);

double cpd_fit(SparseTensor* X, KruskalModel* M, Matrix ** grams, FType* U_mttkrp);
void mttkrp_par(SparseTensor* X, KruskalModel* M, IType mode, omp_lock_t* writelocks);
void mttkrp(SparseTensor* X, KruskalModel* M, IType mode);

// CPD kernels
static void pseudo_inverse(Matrix ** grams, KruskalModel* M, IType mode);

// More explicit version for streaming cpd
static void _pseudo_inverse(Matrix * gram, KruskalModel* M, IType mode);
static void destroy_grams(Matrix ** grams, KruskalModel* M);
static void init_grams(Matrix *** grams, KruskalModel* M);
static void copy_upper_tri(Matrix * M);

static void update_gram(Matrix * gram, KruskalModel* M, int mode);

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
  FType* vals = AT->vals;
  IType nnz = AT->nnz;

  #pragma omp parallel for reduction(+:normAT) schedule(static)
  for(IType i = 0; i < nnz; ++i) {
    normAT += vals[i] * vals[i];
  }

  // Compute initial A**T * A for every mode
  Matrix** grams;
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
double cpd_fit_alto(AltoTensor<LIT>* AT, KruskalModel* M, Matrix** grams, FType* U_mttkrp, FType normAT)
{
  // Calculate inner product between AT and M
  // This can be done via sum(sum(P.U{dimorder(end)} .* U_mttkrp) .* lambda');
  IType rank = M->rank;
  IType nmodes = AT->nmode;
  IType* dims = AT->dims;

  FType* accum = (FType*) AlignedMalloc(sizeof(FType) * rank);
  assert(accum);
  memset(accum, 0, sizeof(FType*) * rank);

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
  FType* tmp_gram = (FType*) AlignedMalloc(sizeof(FType) * rank * rank);
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
        tmp_gram[j * rank + k] *= grams[i]->vals[j * rank + k];
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
  FType ret = (norm_residual / sqrt(normAT));

  // free memory
  free(accum);
  free(tmp_gram);

  return ret;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename LIT>
void streaming_cpd_alto(
  AltoTensor<LIT>* AT, KruskalModel* M, KruskalModel* prev_M, Matrix** grams,
  int max_iters, double epsilon, 
  int streaming_mode, int iteration)
{
  fprintf(stdout, "Running ALTO CP-Stream with %d max iterations and %.2e epsilon\n",
          max_iters, epsilon);

  // Timing stuff 
	InitTSC();
	uint64_t ticks_start = 0;
	uint64_t ticks_end = 0;

  double t_sm_mttkrp = 0.0;
  double t_sm_backsolve = 0.0;
  double t_m_mttkrp = 0.0;
  double t_m_backsolve = 0.0;
  double t_prepare_alto = 0.0;
  double t_fit = 0.0;
  double t_aux = 0.0;
  double t_norm = 0.0;

  int num_inner_iter = 0;

  int nmodes = AT->nmode;
  IType* dims = AT->dims;
  IType rank = M->rank;

  IType nthreads = omp_get_max_threads();

  // Lambda scratchpad
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

  // Compute ttnormsq to later compute fit
  FType normAT = 0.0;
  FType* vals = AT->vals;
  IType nnz = AT->nnz;

  #pragma omp parallel for reduction(+:normAT) schedule(static)
  for(IType i = 0; i < nnz; ++i) {
    normAT += vals[i] * vals[i];
  }

  // keep track of the fit for convergence check
  double fit = 0.0;
  double prev_fit = 0.0;

  double delta = 0.0;
  double prev_delta = 0.0;

  Matrix * old_gram = zero_mat(rank, rank);

  BEGIN_TIMER(&ticks_start);

  // Create local fiber copies
  FType ** ofibs = NULL;
  create_da_mem(-1, rank, AT, &ofibs);
  
  END_TIMER(&ticks_end);
    
  ELAPSED_TIME(ticks_start, ticks_end, &t_prepare_alto);

  // Copy G_t-1 at the begining
  memcpy(old_gram->vals, grams[streaming_mode]->vals, rank * rank * sizeof(*grams[streaming_mode]->vals));

  int tmp_iter = 0;
  for(int i = 0; i < max_iters; i++) {    
    delta = 0.0;
    // Solve for time mode (s_t)
    // set to zero
    memset(M->U[streaming_mode], 0, sizeof(FType) * rank);
    // MTTKRP for s_t
    // mttkrp_par(X, M, streaming_mode, writelocks);
    BEGIN_TIMER(&ticks_start);
    mttkrp_alto_par(streaming_mode, M->U, rank, AT, NULL, ofibs);
		END_TIMER(&ticks_end);
		ELAPSED_TIME(ticks_start, ticks_end, &t_sm_mttkrp);

    BEGIN_TIMER(&ticks_start);
    pseudo_inverse(grams, M, streaming_mode);
    END_TIMER(&ticks_end);
    
    ELAPSED_TIME(ticks_start, ticks_end, &t_sm_backsolve);

    // PrintMatrix("gram mat for streaming mode", grams[streaming_mode]);
    copy_upper_tri(grams[streaming_mode]);
    // Copy newly computed gram matrix G_t to old_gram
    // memcpy(old_gram->vals, grams[streaming_mode]->vals, rank * rank * sizeof(*grams[streaming_mode]->vals));
    // PrintMatrix("gram mat for streaming mode", grams[streaming_mode]);
    // exit(1);

    // Accumulate new time slice into temporal Gram matrix
    // Update grams
    for (int m = 0; m < rank; ++m) {
      for (int n = 0; n < rank; ++n) {
          // Hard coded forgetting factor?
          grams[streaming_mode]->vals[m + n * rank] = old_gram->vals[m + n * rank] + M->U[streaming_mode][m] * M->U[streaming_mode][n];
      }
    }
    // PrintFPMatrix("streaming mode factor matrix", M->U[streaming_mode], rank, 1);
    // PrintMatrix("gram mat after updating", grams[streaming_mode]);
    // exit(1);

    // set up temporary data structures
    FType* scratch = (FType*) AlignedMalloc(sizeof(FType) * dims[nmodes - 1] *
                                          rank);
    assert(scratch);

    for(int j = 0; j < AT->nmode; j++) {
      if (j == streaming_mode) continue;
      // Create buffer for factor matrix
      FType * fm_buf = (FType*) AlignedMalloc(sizeof(FType) * dims[j] * rank);

      // Copy original A(n) to fm_buf
      memcpy(fm_buf, M->U[j], sizeof(FType) * dims[j] * rank);

      // MTTKRP
      memset(M->U[j], 0, sizeof(FType) * dims[j] * rank);
      // MTTKRP results are written in M->U[j]

      BEGIN_TIMER(&ticks_start);
      mttkrp_alto_par(j, M->U, rank, AT, NULL, ofibs);
      END_TIMER(&ticks_end);
      ELAPSED_TIME(ticks_start, ticks_end, &t_m_mttkrp);      

      // mttkrp(X, M, j);
      // if it is the last mode, save the MTTKRP result for fit calculation
      if(j == nmodes - 1) {
        memcpy(scratch, M->U[j], sizeof(FType) * dims[j] * rank);
      }

      BEGIN_TIMER(&ticks_start);

      // add historical
      Matrix * historical = zero_mat(rank, rank);
      Matrix * ata_buf = zero_mat(rank, rank);

      // Starts with mu * G_t-1
      memcpy(ata_buf->vals, old_gram->vals, rank * rank * sizeof(*ata_buf->vals));

      // Copmute A_t-1 * A_t for all other modes
      for (int m = 0; m < nmodes; ++m) {
        if ((m == j) || (m == streaming_mode)) {
          continue;
        }
        // Check previous factor matrix has same dimension size as current factor matrix
        // this should be handled when new tensor is being fed in..
        assert(prev_M->dims[m] == M->dims[m]);

        // int m = prev_M->dims[m];
        // int n = (int)rank;
        // int k = (int)(M->dims[m]);
        // int l = (int)rank;
        my_matmul(prev_M->U[m], true, M->U[m], false, historical->vals, prev_M->dims[m], rank, M->dims[m], rank, 0.0);

        for (int x = 0; x < rank * rank; ++x) {
          ata_buf->vals[x] *= historical->vals[x];
        }
      }
      // END: Updating ata_buf (i.e. aTa matrices for all factor matrices)

      // A(n) (ata_buf)
      my_matmul(prev_M->U[j], false, ata_buf->vals, false, M->U[j], prev_M->dims[j], rank, rank, rank, 1.0);    
      END_TIMER(&ticks_end);
      ELAPSED_TIME(ticks_start, ticks_end, &t_aux);      

      BEGIN_TIMER(&ticks_start);
      pseudo_inverse(grams, M, j);
      END_TIMER(&ticks_end);
      ELAPSED_TIME(ticks_start, ticks_end, &t_m_backsolve);

      // Normalize columns
      // printf("Lambda before norm\n");
      // for (int ii = 0; ii < M->rank; ++ii) {
      //   printf("%f\t", M->lambda[ii]);
      // }
      
      // printf("\n");
      BEGIN_TIMER(&ticks_start);
      if(i == 0) {
        KruskalModelNorm(M, j, MAT_NORM_2, lambda_sp);
      } else {
        KruskalModelNorm(M, j, MAT_NORM_MAX, lambda_sp);
      }
      END_TIMER(&ticks_end);
      ELAPSED_TIME(ticks_start, ticks_end, &t_norm);
      // printf("Lambda after norm\n");
      // for (int ii = 0; ii < M->rank; ++ii) {
      //   printf("%f\t", M->lambda[i]);
      // }
      // printf("\n");
      // Update the Gram matrices
      BEGIN_TIMER(&ticks_start);

      update_gram(grams[j], M, j);

      // Copy old factor matrix to new
      memcpy(prev_M->U[j], M->U[j], sizeof(FType) * rank * M->dims[j]);
      // PrintFPMatrix("Grams", rank, rank, grams[j], rank);
      // PrintFPMatrix("Lambda", 1, rank, M->lambda, rank);

      END_TIMER(&ticks_end);
      ELAPSED_TIME(ticks_start, ticks_end, &t_aux); 
      int factor_mat_size = rank * M->dims[j];
      delta += mat_norm_diff(prev_M->U[j], M->U[j], factor_mat_size) / (mat_norm(M->U[j], factor_mat_size) + 1e-12);

      free(fm_buf);
    } // for each mode


    for (IType x = 0; x < rank * rank; ++x) {
      grams[streaming_mode]->vals[x] *= 0.95;
    }
    // calculate fit
    // fit = cpd_fit(X, M, grams, scratch);
    BEGIN_TIMER(&ticks_start);
    
    fit = cpd_fit_alto(AT, M, grams, scratch, normAT);
		END_TIMER(&ticks_end);
		ELAPSED_TIME(ticks_start, ticks_end, &t_fit);

    free(scratch);

    // printf("it: %d delta: %e prev_delta: %e (%e diff)\n", i, delta, prev_delta, fabs(delta - prev_delta));
    /*
    if ((i > 0) && fabs(prev_delta - delta) < epsilon) {
      prev_delta = 0.0;
      break;
    } else {
      prev_delta = delta;
    }
    */
    tmp_iter = i;

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
  num_inner_iter += tmp_iter;

  // cleanup
  #pragma omp parallel for
  for (IType t = 0; t < nthreads; ++t) {
    free(lambda_sp[t]);
  }
  free(lambda_sp);
  for(IType i = 0; i < max_mode_len; i++) {
    omp_destroy_lock(&(writelocks[i]));
  }
  free(writelocks);
  printf("Time: %d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", num_inner_iter, t_sm_mttkrp, t_sm_backsolve, t_m_mttkrp, t_m_backsolve, t_fit, t_aux, t_norm, t_prepare_alto);

}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void streaming_cpd(
  SparseTensor* X, KruskalModel* M, KruskalModel* prev_M, Matrix** grams,
  int max_iters, double epsilon, 
  int streaming_mode, int iteration)
{
  fprintf(stdout, "Running CP-Stream (iter: %d) with %d max iterations and %.2e epsilon\n",
          iteration, max_iters, epsilon);

  // Timing stuff 
	InitTSC();
	uint64_t ticks_start = 0;
	uint64_t ticks_end = 0;

  double t_sm_mttkrp = 0.0;
  double t_sm_backsolve = 0.0;
  double t_m_mttkrp = 0.0;
  double t_m_backsolve = 0.0;
  double t_fit = 0.0;
  double t_aux = 0.0;
  double t_norm = 0.0;

  int num_inner_iter = 0;
  
  int nmodes = X->nmodes;
  IType* dims = X->dims;
  IType rank = M->rank;

  IType nthreads = omp_get_max_threads();

  // Lambda scratchpad
  FType ** lambda_sp = (FType **) AlignedMalloc(sizeof(FType*) * nthreads);
  assert(lambda_sp);
  #pragma omp parallel for
  for (IType t = 0; t < nthreads; ++t) {
    lambda_sp[t] = (FType *) AlignedMalloc(sizeof(FType) * rank);
    assert(lambda_sp[t]);
  }
  
  if (iteration == 0) {
    for (int m = 0; m < nmodes; ++m) {
      if (m == streaming_mode) continue;
      KruskalModelNorm(M, m, MAT_NORM_2, lambda_sp);
    }
  }

#if DEBUG == 1
  PrintKruskalModel(M);
#endif
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

  double delta = 0.0;
  double prev_delta = 0.0;

  Matrix * old_gram = zero_mat(rank, rank);

  // Copy G_t-1 at the begining
  memcpy(old_gram->vals, grams[streaming_mode]->vals, rank * rank * sizeof(*grams[streaming_mode]->vals));

  int tmp_iter = 0;
  for(int i = 0; i < max_iters; i++) {    
    delta = 0.0;
    // Solve for time mode (s_t)
    // set to zero
    memset(M->U[streaming_mode], 0, sizeof(FType) * rank);

    BEGIN_TIMER(&ticks_start);
    mttkrp_par(X, M, streaming_mode, writelocks);
		END_TIMER(&ticks_end);

#if DEBUG == 1
    PrintFPMatrix("mttkrp for s_t", M->U[streaming_mode], 1, rank);
#endif
		ELAPSED_TIME(ticks_start, ticks_end, &t_sm_mttkrp);
    // MTTKRP for s_t
    BEGIN_TIMER(&ticks_start);

    // Init gram matrix aTa for all other modes

    // _pseudo_inverse(grams, M, streaming_mode);
    pseudo_inverse(grams, M, streaming_mode);
		END_TIMER(&ticks_end);
		ELAPSED_TIME(ticks_start, ticks_end, &t_sm_backsolve);

#if DEBUG == 1
    printf("s_t\n");
    for (int r = 0; r < rank; ++r) { 
      printf("%e\t", M->U[streaming_mode][r]);
    }
    printf("\n");
#endif
    // PrintMatrix("gram mat for streaming mode", grams[streaming_mode]);
    copy_upper_tri(grams[streaming_mode]);
    // Copy newly computed gram matrix G_t to old_gram
    // memcpy(old_gram->vals, grams[streaming_mode]->vals, rank * rank * sizeof(*grams[streaming_mode]->vals));
    // PrintMatrix("gram mat for streaming mode", grams[streaming_mode]);
    // exit(1);

    // Accumulate new time slice into temporal Gram matrix
    // Update grams
    for (int m = 0; m < rank; ++m) {
      for (int n = 0; n < rank; ++n) {
          // Hard coded forgetting factor?
          grams[streaming_mode]->vals[m + n * rank] = old_gram->vals[m + n * rank] + M->U[streaming_mode][m] * M->U[streaming_mode][n];
      }
    }
    // PrintFPMatrix("streaming mode factor matrix", M->U[streaming_mode], rank, 1);
    // PrintMatrix("gram mat after updating", grams[streaming_mode]);
    // exit(1);

    // set up temporary data structures
    FType* scratch = (FType*) AlignedMalloc(sizeof(FType) * dims[nmodes - 1] *
                                          rank);
    assert(scratch);

    for(int j = 0; j < X->nmodes; j++) {
      if (j == streaming_mode) continue;
      // Create buffer for factor matrix
      FType * fm_buf = (FType*) AlignedMalloc(sizeof(FType) * dims[j] * rank);

      // Copy original A(n) to fm_buf
      memcpy(fm_buf, M->U[j], sizeof(FType) * dims[j] * rank);

      // MTTKRP
      memset(M->U[j], 0, sizeof(FType) * dims[j] * rank);
      BEGIN_TIMER(&ticks_start);
      mttkrp_par(X, M, j, writelocks);
		  END_TIMER(&ticks_end);
		  ELAPSED_TIME(ticks_start, ticks_end, &t_m_mttkrp);
      // MTTKRP results are written in M->U[j]
      // mttkrp(X, M, j);
      // if it is the last mode, save the MTTKRP result for fit calculation
      if(j == nmodes - 1) {
        memcpy(scratch, M->U[j], sizeof(FType) * dims[j] * rank);
      }

      BEGIN_TIMER(&ticks_start);
      // add historical
      Matrix * historical = zero_mat(rank, rank);
      Matrix * ata_buf = zero_mat(rank, rank);

      // Starts with mu * G_t-1
      memcpy(ata_buf->vals, old_gram->vals, rank * rank * sizeof(*ata_buf->vals));

      // Copmute A_t-1 * A_t for all other modes
      for (int m = 0; m < nmodes; ++m) {
        if ((m == j) || (m == streaming_mode)) {
          continue;
        }
        // Check previous factor matrix has same dimension size as current factor matrix
        // this should be handled when new tensor is being fed in..
        assert(prev_M->dims[m] == M->dims[m]);

        // int m = prev_M->dims[m];
        // int n = (int)rank;
        // int k = (int)(M->dims[m]);
        // int l = (int)rank;
        my_matmul(prev_M->U[m], true, M->U[m], false, historical->vals, prev_M->dims[m], rank, M->dims[m], rank, 0.0);

        for (int x = 0; x < rank * rank; ++x) {
          ata_buf->vals[x] *= historical->vals[x];
        }
      }
      // END: Updating ata_buf (i.e. aTa matrices for all factor matrices)

      // A(n) (ata_buf)
      my_matmul(prev_M->U[j], false, ata_buf->vals, false, M->U[j], prev_M->dims[j], rank, rank, rank, 1.0);    
		  END_TIMER(&ticks_end);
		  ELAPSED_TIME(ticks_start, ticks_end, &t_aux);

      BEGIN_TIMER(&ticks_start);
      pseudo_inverse(grams, M, j);
		  END_TIMER(&ticks_end);
		  ELAPSED_TIME(ticks_start, ticks_end, &t_m_backsolve);

      // Normalize columns
      // printf("Lambda before norm\n");
      // for (int ii = 0; ii < M->rank; ++ii) {
      //   printf("%f\t", M->lambda[ii]);
      // }
      
      // printf("\n");

      BEGIN_TIMER(&ticks_start);
      if(i == 0) {
        KruskalModelNorm(M, j, MAT_NORM_2, lambda_sp);
      } else {
        KruskalModelNorm(M, j, MAT_NORM_2, lambda_sp);
      }
      END_TIMER(&ticks_end);
      ELAPSED_TIME(ticks_start, ticks_end, &t_norm);
      // printf("Lambda after norm\n");
      // for (int ii = 0; ii < M->rank; ++ii) {
      //   printf("%f\t", M->lambda[i]);
      // }
      // printf("\n");
      // Update the Gram matrices
      BEGIN_TIMER(&ticks_start);
      update_gram(grams[j], M, j);

      // Copy old factor matrix to new
      // memcpy(prev_M->U[j], M->U[j], sizeof(FType) * rank * M->dims[j]);
      // PrintFPMatrix("Grams", rank, rank, grams[j], rank);
      // PrintFPMatrix("Lambda", 1, rank, M->lambda, rank);
      END_TIMER(&ticks_end);
      ELAPSED_TIME(ticks_start, ticks_end, &t_aux);      
      
      int factor_mat_size = rank * M->dims[j];
      delta += mat_norm_diff(prev_M->U[j], M->U[j], factor_mat_size) / (mat_norm(M->U[j], factor_mat_size) + 1e-12);

      free(fm_buf);
    } // for each mode

    // PrintKruskalModel(M);

    // calculate fit
    BEGIN_TIMER(&ticks_start);
    fit = cpd_fit(X, M, grams, scratch);
		END_TIMER(&ticks_end);
		ELAPSED_TIME(ticks_start, ticks_end, &t_fit);

    // PrintMatrix("gram matrix for streaming mode", grams[streaming_mode]);
    free(scratch);

#if 1
    printf("it: %d delta: %e prev_delta: %e (%e diff)\n", i, delta, prev_delta, fabs(delta - prev_delta));
    
    if ((i > 0) && fabs(prev_delta - delta) < epsilon) {
      prev_delta = 0.0;
      break;
    } else {
      prev_delta = delta;
    }
#else
    // if fit - oldfit < epsilon, quit
    tmp_iter = i;
    
    if((i > 0) && (fabs(prev_fit - fit) < epsilon)) {
      printf("inner-it: %d\t fit: %g\t fit-delta: %g\n", i, fit,
             fabs(prev_fit - fit));
      break;
    } else {
      printf("inner-it: %d\t fit: %g\t fit-delta: %g\n", i, fit,
             fabs(prev_fit - fit));
    }
    prev_fit = fit;
  #endif
  } // for max_iters
  num_inner_iter += tmp_iter;

  // Copy new into old factor matrix
  for(int j = 0; j < X->nmodes; j++) {
    memcpy(prev_M->U[j], M->U[j], sizeof(FType) * rank * M->dims[j]);
  }

  // incorporate forgetting factor
  for (IType x = 0; x < rank * rank; ++x) {
    grams[streaming_mode]->vals[x] *= 0.95;
  }

  // cleanup
  #pragma omp parallel for
  for (IType t = 0; t < nthreads; ++t) {
    free(lambda_sp[t]);
  }
  free(lambda_sp);
  for(IType i = 0; i < max_mode_len; i++) {
    omp_destroy_lock(&(writelocks[i]));
  }
  free(writelocks);

  printf("Time: %d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", num_inner_iter, t_sm_mttkrp, t_sm_backsolve, t_m_mttkrp, t_m_backsolve, t_fit, t_aux, t_norm);
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
  Matrix ** grams;
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

double cpd_fit(SparseTensor* X, KruskalModel* M, Matrix** grams, FType* U_mttkrp)
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
        tmp_gram[j * rank + k] *= grams[i]->vals[j * rank + k];
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
  // printf("normX: %f, normU: %f, inner_prod: %f\n", normX, normU, inner_prod);
  if (norm_residual > 0.0) {
    norm_residual = sqrt(norm_residual);
  }
  FType ret = norm_residual / sqrt(normX);

  // free memory
  free(accum);
  free(tmp_gram);

  return ret;
}

double streaming_cpd_fit(SparseTensor* X, KruskalModel* M, Matrix** grams, FType* U_mttkrp, int streaming_mode)
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
        tmp_gram[j * rank + k] *= grams[i]->vals[j * rank + k];
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
  printf("normX: %f, normU: %f, inner_prod: %f\n", normX, normU, inner_prod);
  if (norm_residual > 0.0) {
    norm_residual = sqrt(norm_residual);
  }
  FType ret = norm_residual / sqrt(normX);

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

static void _pseudo_inverse(Matrix * gram, KruskalModel * M, IType mode) {
  IType rank = M->rank;
  IType nmodes = (IType) M->mode;

  // Maybe perform transpose for gram matrix?
  // Do manual transpose for gram matrix so that we use col_major instead of row_major
  double * vals = (double*) AlignedMalloc(sizeof(double) * rank * rank);
  assert(vals);

  // Store original gram matrix just in case we need to use fallback
  FType* scratch = (FType*) AlignedMalloc(sizeof(FType) * rank * rank);
  assert(scratch);

  #pragma unroll
  for (IType i = 0; i < rank; ++i) {
    #pragma omp simd 
    for (IType j = 0; j < rank; ++j) {
      vals[j * rank + i] = gram->vals[i * rank + j];
    }
  }

  memcpy(scratch, vals, sizeof(FType) * rank * rank);


  free(vals);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
static void pseudo_inverse(Matrix ** grams, KruskalModel* M, IType mode)
{
  IType rank = M->rank;
  IType nmodes = (IType) M->mode;

  // Calculate V
  IType m = 0;
  if(mode == 0) {
    m = 1;
  }

  memcpy(grams[mode]->vals, grams[m]->vals, sizeof(FType) * rank * rank);
  #pragma unroll
  for(IType i = m + 1; i < nmodes; i++) {
    if(i != mode) {
      #pragma omp simd
      for(IType j = 0; j < rank * rank; j++) {
        grams[mode]->vals[j] *= grams[i]->vals[j];
      }
    }
  }
  // PrintFPMatrix("V", rank, rank, grams[mode], rank);

  FType* scratch = (FType*) AlignedMalloc(sizeof(FType) * rank * rank);
  assert(scratch);

  memcpy(scratch, grams[mode]->vals, sizeof(FType) * rank * rank);

  // Do manual transpose for gram matrix so that we use col_major instead of row_major
  
  // double * vals = (double*) AlignedMalloc(sizeof(double) * rank * rank);
  // assert(vals);

  // #pragma unroll
  // for (IType i = 0; i < rank; ++i) {
  //   #pragma omp simd 
  //   for (IType j = 0; j < rank; ++j) {
  //     vals[j * rank + i] = grams[mode]->vals[i * rank + j];
  //   }
  // }
  
  // memcpy(grams[mode]->vals, vals, rank * rank * sizeof(FType));
  
  // Backup grams[mode] in case Cholesky fails

  // free(vals);

// Apply frobenious norm
// This stabilizes (?) the cholesky factorization of the matrix
// For now just use a generic value (1e-3)

for (int r = 0; r < rank; ++r) {
  grams[mode]->vals[r * rank + r] += 1e-3;
}

#if DEBUG == 1
  PrintMatrix("A matrix", grams[mode]);
#endif
  // Try using Cholesky to find the pseudoinvsere of V
  // Setup parameters for LAPACK calls
  // convert IType to int
  char uplo = 'L';
  lapack_int _rank = (lapack_int)rank;
  lapack_int I = (lapack_int)M->dims[mode];
  lapack_int info;
  DPOTRF(&uplo, &_rank, grams[mode]->vals, &_rank, &info);
  
  if(info == 0) {
#if DEBUG == 1
    PrintMatrix("cholesky", grams[mode]);
    PrintFPMatrix("rhs", M->U[mode], I, rank);
#endif    
    // Cholesky was successful - use it to find the pseudo_inverse and multiply
    // it with the MTTKRP result
    POTRS(&uplo, &_rank, &I, grams[mode]->vals, &_rank,
          M->U[mode], &_rank, &info);

#if DEBUG == 1
    PrintMatrix("after - cholesky", grams[mode]);
    PrintFPMatrix("after - rhs", M->U[mode], I, rank);
#endif
  } else {
    // Otherwise use rank-deficient solver, GELSY
    // Restore V
    memcpy(grams[mode]->vals, scratch, sizeof(FType) * rank * rank);
    //PrintFPMatrix("gram matrix when fallback", rank, rank, grams[mode], rank);
    // Fill up the upper part
    #pragma unroll
    for(IType i = 0; i < rank; i++) {
      #pragma omp simd
      for(IType j = i; j < rank; j++) {
        grams[mode]->vals[i * rank + j] = grams[mode]->vals[j * rank + i];
      }
    }

    // Use a rank-deficient solver
    lapack_int* jpvt = (lapack_int*) AlignedMalloc(sizeof(lapack_int) * rank);
    memset(jpvt, 0, sizeof(lapack_int) * rank);
    lapack_int lwork = -1;
    double work_qr;
    lapack_int ret_rank;
    lapack_int info_dgelsy;
    double rcond = -1.0f;//1.1e-16;
    
    GELSY(&_rank, &_rank, &I, grams[mode]->vals, &_rank, M->U[mode], &_rank,
          jpvt, &rcond, &ret_rank, &work_qr, &lwork, &info_dgelsy);
    double* work = (double*) AlignedMalloc(sizeof(double) * work_qr);
    GELSY(&_rank, &_rank, &I, grams[mode]->vals, &_rank, M->U[mode], &_rank,
          jpvt, &rcond, &ret_rank, work, &lwork, &info_dgelsy);
    
    fprintf(stderr, "\t Mode %llu Min Norm Solve: %d\nDGELSS effective rank: %d\n", mode, info_dgelsy, ret_rank);
    free(work);
    free(jpvt);
  }

  // cleanup
  free(scratch);
}

static void destroy_grams(Matrix ** grams, KruskalModel* M)
{
  for(int i = 0; i < M->mode; i++) {
    free_mat(grams[i]);
  }
  free(grams);
}

static void init_grams(Matrix *** grams, KruskalModel* M)
{
  IType rank = M->rank;
  Matrix ** _grams = (Matrix**) AlignedMalloc(sizeof(Matrix*) * M->mode);
  assert(_grams);
  for(int i = 0; i < M->mode; i++) {
    _grams[i] = init_mat(rank, rank);
    assert(_grams[i]);
  }

  for(int i = 0; i < M->mode; i++) {
    update_gram(_grams[i], M, i);
  }

  *grams = _grams;
}


static void update_gram(Matrix * gram, KruskalModel* M, int mode)
{
  MKL_INT m = M->dims[mode];
  MKL_INT n = M->rank;
  MKL_INT lda = n;
  MKL_INT ldc = n;
  FType alpha = 1.0;
  FType beta = 0.0;

  CBLAS_ORDER layout = CblasRowMajor;
  CBLAS_UPLO uplo = CblasUpper;
  CBLAS_TRANSPOSE trans = CblasTrans;
  SYRK(layout, uplo, trans, n, m, alpha, M->U[mode], lda, beta, gram->vals, ldc);
}

static void copy_upper_tri(Matrix * M) {
  IType const I = M->I;
  IType const J = M->J;
  FType * const vals = M->vals;

  #pragma omp parallel for schedule(static, 1) if(I > 50)
  for(IType i=1; i < I; ++i) {
    for(IType j=0; j < i; ++j) {  
      vals[j + (i*J)] = vals[i + (j*J)];
    }
  }  
};



#endif // CPD_HPP_
