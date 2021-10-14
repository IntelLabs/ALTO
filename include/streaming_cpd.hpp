#ifndef STREAMING_CPD_HPP_
#define STREAMING_CPD_HPP_

#include "common.hpp"
#include "sptensor.hpp"
#include "kruskal_model.hpp"
#include "stream_matrix.hpp"
#include "streaming_sptensor.hpp"
#include "util.hpp"
#include "alto.hpp"

// Signatures
void cp_stream(
    SparseTensor* X, int rank, int max_iters, int streaming_mode, 
    FType epsilon, IType seed);

void cp_stream_iter(SparseTensor* X, KruskalModel* M, KruskalModel * prev_M, 
    Matrix** grams, int max_iters, double epsilon, 
    int streaming_mode, int iter);

template <typename LIT>
void streaming_cpd_alto(AltoTensor<LIT>* AT, KruskalModel* M, KruskalModel * prev_M, Matrix** grams, int max_iters, double epsilon, int streaming_mode, int iter);

// More explicit version for streaming cpd
static void pseudo_inverse_stream(Matrix ** grams, KruskalModel* M, IType mode, IType stream_mode);


// Implementations
void cp_stream(SparseTensor* X, int rank, int max_iters, int streaming_mode, 
    FType epsilon, IType seed) {
    // Define timers (common)
    double t_preprocess = 0.0;
    double tot_preprocess = 0.0;


    // Step 1. Preprocess SparseTensor * X
    printf("Processing Streaming Sparse Tensor\n");
    StreamingSparseTensor sst(X, streaming_mode);
    sst.print_tensor_info();
        
    // Step 2. Prepare variables and load first time batch
    // Instantiate kruskal models
    KruskalModel * M; // Keeps track of current factor matrices
    KruskalModel * prev_M; // Keeps track of previous factor matrices
        
    Matrix ** grams;
    // concatencated s_t's
    Matrix * global_time = zero_mat(1, rank);

    int it = 0;

    while(!sst.last_batch()) {   
        // Get next time batch
        SparseTensor * t_batch = sst.next_batch();
        PrintTensorInfo(rank, max_iters, t_batch);

        if (it == 0) {
            CreateKruskalModel(t_batch->nmodes, t_batch->dims, rank, &M);
            CreateKruskalModel(t_batch->nmodes, t_batch->dims, rank, &prev_M);
            KruskalModelRandomInit(M, (unsigned int)seed);
            KruskalModelZeroInit(prev_M);

            // Override values for M->U[stream_mode] with last row of global_time matrix
            M->U[streaming_mode] = &(global_time->vals[it*rank]);


            // TODO: specify what "grams" exactly
            init_grams(&grams, M);
        } else {
            GrowKruskalModel(t_batch->dims, &M, FILL_RANDOM); // Expands the kruskal model to accomodate new dimensions
            GrowKruskalModel(t_batch->dims, &prev_M, FILL_ZEROS); // Expands the kruskal model to accomodate new dimensions
            for (int j = 0; j < M->mode; ++j) {
                if (j != streaming_mode) {
                    update_gram(grams[j], M, j);
                }
            }
        }

        GrowKruskalModel(t_batch->dims, &M, FILL_RANDOM); // Expands the kruskal model to accomodate new dimensions
        GrowKruskalModel(t_batch->dims, &prev_M, FILL_ZEROS); // Expands the kruskal model to accomodate new dimensions
        for (int j = 0; j < M->mode; ++j) {
            if (j != streaming_mode) {
                update_gram(grams[j], M, j);
            }
        }

        cp_stream_iter(
            t_batch, M, prev_M, grams, 
            max_iters, epsilon, streaming_mode, it);

        // TODO: Save checkpoints

        // Copy latest
        CopyKruskalModel(&prev_M, &M);

        DestroySparseTensor(t_batch);
        ++it;
    }

    DestroySparseTensor(X);
    destroy_grams(grams, M);
    DestroyKruskalModel(M);
    DestroyKruskalModel(prev_M);
    return;    
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

  // Copy G_t-1 at the begining
  memcpy(old_gram->vals, grams[streaming_mode]->vals, rank * rank * sizeof(*grams[streaming_mode]->vals));

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
		AGG_ELAPSED_TIME(ticks_start, ticks_end, &t_sm_mttkrp);

    BEGIN_TIMER(&ticks_start);
    pseudo_inverse_stream(grams, M, streaming_mode, streaming_mode);

    END_TIMER(&ticks_end);
    
    AGG_ELAPSED_TIME(ticks_start, ticks_end, &t_sm_backsolve);

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
      AGG_ELAPSED_TIME(ticks_start, ticks_end, &t_m_mttkrp);      

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
      AGG_ELAPSED_TIME(ticks_start, ticks_end, &t_aux);      

      BEGIN_TIMER(&ticks_start);
      pseudo_inverse_stream(
        grams, M, j, streaming_mode);
      END_TIMER(&ticks_end);
      AGG_ELAPSED_TIME(ticks_start, ticks_end, &t_m_backsolve);

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
      AGG_ELAPSED_TIME(ticks_start, ticks_end, &t_norm);
      // printf("Lambda after norm\n");
      // for (int ii = 0; ii < M->rank; ++ii) {
      //   printf("%f\t", M->lambda[i]);
      // }
      // printf("\n");
      // Update the Gram matrices
      BEGIN_TIMER(&ticks_start);

      update_gram(grams[j], M, j);
      // PrintFPMatrix("Grams", rank, rank, grams[j], rank);
      // PrintFPMatrix("Lambda", 1, rank, M->lambda, rank);

      END_TIMER(&ticks_end);
      AGG_ELAPSED_TIME(ticks_start, ticks_end, &t_aux); 
      int factor_mat_size = rank * M->dims[j];
      delta += mat_norm_diff(prev_M->U[j], M->U[j], factor_mat_size) / (mat_norm(M->U[j], factor_mat_size) + 1e-12);

      free(fm_buf);
    } // for each mode

    // calculate fit
    // fit = cpd_fit(X, M, grams, scratch);
    BEGIN_TIMER(&ticks_start);
    
    fit = cpd_fit_alto(AT, M, grams, scratch, normAT);
		END_TIMER(&ticks_end);
		AGG_ELAPSED_TIME(ticks_start, ticks_end, &t_fit);

    free(scratch);
    
    tmp_iter = i;
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
  for(int j = 0; j < AT->nmode; j++) {
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
  printf("%d\t%llu\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", iteration, AT->nnz, num_inner_iter, t_sm_mttkrp, t_sm_backsolve, t_m_mttkrp, t_m_backsolve, t_fit, t_aux, t_norm);

}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void cp_stream_iter(
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
		AGG_ELAPSED_TIME(ticks_start, ticks_end, &t_sm_mttkrp);
    // MTTKRP for s_t
    BEGIN_TIMER(&ticks_start);

    // Init gram matrix aTa for all other modes

    pseudo_inverse_stream(
      grams, M, streaming_mode, streaming_mode);
		END_TIMER(&ticks_end);
		AGG_ELAPSED_TIME(ticks_start, ticks_end, &t_sm_backsolve);

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
#if DEBUG == 1    
    PrintMatrix("gram mat after updating s_t", grams[streaming_mode]);
#endif
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
#if DEBUG == 1    
    char str[512];
    sprintf(str, "it: %d: M[%d] before mttkrp", i, j);
    PrintFPMatrix(str, M->U[j], M->dims[j], rank);
    memset(str, 0, 512);
#endif
      mttkrp_par(X, M, j, writelocks);


#if DEBUG == 1
      sprintf(str, "mttkrp output for M[%d]", j);
      PrintFPMatrix(str, M->U[j], M->dims[j], rank);
#endif      
		  END_TIMER(&ticks_end);
		  AGG_ELAPSED_TIME(ticks_start, ticks_end, &t_m_mttkrp);
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
#if DEBUG == 1
      sprintf(str, "ata buf for M[%d]", j);
      PrintMatrix(str, ata_buf);
      // END: Updating ata_buf (i.e. aTa matrices for all factor matrices)

      // A(n) (ata_buf)
      memset(str, 0, 512);
      sprintf(str, "prev_M for mode %d", j);
      PrintFPMatrix(str, prev_M->U[j], prev_M->dims[j], rank);
      memset(str, 0, 512);

      sprintf(str, "mttkrp part for mode %d", j);
      PrintFPMatrix(str, M->U[j], M->dims[j], rank);
      memset(str, 0, 512);
#endif            

      my_matmul(prev_M->U[j], false, ata_buf->vals, false, M->U[j], prev_M->dims[j], rank, rank, rank, 1.0);    
		  END_TIMER(&ticks_end);
		  AGG_ELAPSED_TIME(ticks_start, ticks_end, &t_aux);

      BEGIN_TIMER(&ticks_start);
      pseudo_inverse_stream(
      grams, M, j, streaming_mode);
		  END_TIMER(&ticks_end);
		  AGG_ELAPSED_TIME(ticks_start, ticks_end, &t_m_backsolve);
#if DEBUG == 1
      sprintf(str, "updated factor matrix for mode %d", j);
      PrintFPMatrix(str, M->U[j], M->dims[j], rank);
      memset(str, 0, 512);

      PrintKruskalModel(M);

      if (j == 1) exit(1);
#endif      

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
        // MAT_NORM_MAX doesn't affect lambda as much
        KruskalModelNorm(M, j, MAT_NORM_MAX, lambda_sp);
      }
      END_TIMER(&ticks_end);
      AGG_ELAPSED_TIME(ticks_start, ticks_end, &t_norm);
      // printf("Lambda after norm\n");
      // for (int ii = 0; ii < M->rank; ++ii) {
      //   printf("%f\t", M->lambda[i]);
      // }
      // printf("\n");
      // Update the Gram matrices
      BEGIN_TIMER(&ticks_start);
      update_gram(grams[j], M, j);

      // PrintFPMatrix("mttkrp output for M[j]", M->U[j], M->dims[j], rank);
      // PrintMatrix("updated gram matrix for mode j", grams[j]);
      // Copy old factor matrix to new
      // memcpy(prev_M->U[j], M->U[j], sizeof(FType) * rank * M->dims[j]);
      // PrintFPMatrix("Grams", rank, rank, grams[j], rank);
      // PrintFPMatrix("Lambda", 1, rank, M->lambda, rank);
      END_TIMER(&ticks_end);
      AGG_ELAPSED_TIME(ticks_start, ticks_end, &t_aux);      
      
      int factor_mat_size = rank * M->dims[j];
      delta += mat_norm_diff(prev_M->U[j], M->U[j], factor_mat_size) / (mat_norm(M->U[j], factor_mat_size) + 1e-12);

      free(fm_buf);
    } // for each mode

    // PrintKruskalModel(M);

    // calculate fit
    BEGIN_TIMER(&ticks_start);
    fit = cpd_fit(X, M, grams, scratch);
		END_TIMER(&ticks_end);
		AGG_ELAPSED_TIME(ticks_start, ticks_end, &t_fit);

    // PrintMatrix("gram matrix for streaming mode", grams[streaming_mode]);
    free(scratch);

    tmp_iter = i;
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

  // // Copy new into old factor matrix
  // CopyKruskalModel(prev_M, M)
  // for(int j = 0; j < X->nmodes; j++) {
  //   memcpy(prev_M->U[j], M->U[j], sizeof(FType) * rank * M->dims[j]);
  // }

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

  printf("%d\t%llu\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", iteration, X->nnz, num_inner_iter, t_sm_mttkrp, t_sm_backsolve, t_m_mttkrp, t_m_backsolve, t_fit, t_aux, t_norm);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Need a separate version so that we can selectively apply frob reg to stream mode only
static void pseudo_inverse_stream(
  Matrix ** grams, KruskalModel* M, IType mode, IType stream_mode)
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

  FType* scratch = (FType*) AlignedMalloc(sizeof(FType) * rank * rank);
  assert(scratch);

  memcpy(scratch, grams[mode]->vals, sizeof(FType) * rank * rank);

// Apply frobenious norm
// This stabilizes (?) the cholesky factorization of the matrix
// For now just use a generic value (1e-3)
if (mode == stream_mode) {
  for (int r = 0; r < rank; ++r) {
    grams[mode]->vals[r * rank + r] += 1e-3;
  }
}

#if DEBUG == 1
    PrintMatrix("before cholesky or factorization", grams[mode]);
    PrintFPMatrix("before solve: rhs", M->U[mode], (int)M->dims[mode] , rank);
#endif    

  // Try using Cholesky to find the pseudoinvsere of V
  // Setup parameters for LAPACK calls
  // convert IType to int
  char uplo = 'L';
  lapack_int _rank = (lapack_int)rank;
  lapack_int I = (lapack_int)M->dims[mode];
  lapack_int info;
  POTRF(&uplo, &_rank, grams[mode]->vals, &_rank, &info);
  
  if(info == 0) {
    lapack_int s_info = 0;

#if DEBUG == 1
    printf("uplo: %c, lda: %d, nrhs: %d, ldb: %d, info: %d\n", uplo, _rank, I, _rank, s_info);
#endif
    // printf("\n\n");
    // for (int i = 0; i < M->dims[mode] * rank; ++i) {
    //   printf("%e\t", M->U[mode][i]);
    // }
    // printf("\n\n");
    // for (int i = 0; i < rank * rank; ++i) {
    //   printf("%e\t", grams[mode]->vals[i]);
    // }
    // printf("\n\n");
    POTRS(&uplo, &_rank, &I, grams[mode]->vals, &_rank,
          M->U[mode], &_rank, &s_info);

  } else {
    fprintf(stderr, "ALTO: DPOTRF returned %d, Solving using GELSS\n", info);
    
    // Otherwise use rank-deficient solver, GELSY
    // Restore V
    memcpy(grams[mode]->vals, scratch, sizeof(FType) * rank * rank);

    //PrintFPMatrix("gram matrix when fallback", rank, rank, grams[mode], rank);
    // Fill up the upper part
    // #pragma unroll
    // for(IType i = 0; i < rank; i++) {
    //   #pragma omp simd
    //   for(IType j = i; j < rank; j++) {
    //     grams[mode]->vals[i * rank + j] = grams[mode]->vals[j * rank + i];
    //   }
    // }

    // Use a rank-deficient solver
    lapack_int* jpvt = (lapack_int*) AlignedMalloc(sizeof(lapack_int) * rank);

    FType * conditions = (FType *) AlignedMalloc(rank * sizeof(FType));
    memset(jpvt, 0, sizeof(lapack_int) * rank);
    lapack_int lwork = -1;
    double work_qr;
    lapack_int ret_rank;
    lapack_int info_dgelsy;
    double rcond = -1.0f;//1.1e-16;

/* Exactly the same for both !!!
      for (int n = 0; n < rank * rank; ++n) {
        printf("%e\t", grams[mode]->vals[n]);
      }
      printf("\n\n");

      for (int n = 0; n < rank * I; ++n) {
        printf("%e\t", M->U[mode][n]);
      }
*/
    GELSS(&_rank, &_rank, &I,
        grams[mode]->vals, &_rank, 
        M->U[mode], &_rank,
        conditions, &rcond, &ret_rank, 
        &work_qr, &lwork, &info_dgelsy);

    lwork = (lapack_int) work_qr;
    double* work = (double*) AlignedMalloc(sizeof(double) * lwork);

    GELSS(&_rank, &_rank, &I, 
          grams[mode]->vals, &_rank, 
          M->U[mode], &_rank,
          conditions, &rcond, &ret_rank, 
          work, &lwork, &info_dgelsy);

    if (info_dgelsy) {
      PrintMatrix("gram matrix", grams[mode]);
      PrintFPMatrix("rhs", M->U[mode], I, rank);
      fprintf(stderr, "\tDGELSS failed!! Mode %llu Min Norm Solve: %d\nDGELSS effective rank: %d\n", mode, info_dgelsy, ret_rank);    
      exit(1);
    }

#if DEBUG == 1
    printf("uplo: %c, lda: %d, nrhs: %d, ldb: %d, info: %d\n", uplo, _rank, I, _rank, info_dgelsy);
    fprintf(stderr, "\t Mode %llu Min Norm Solve: %d\nDGELSS effective rank: %d\n", mode, info_dgelsy, ret_rank);
#endif    
    free(work);
    free(jpvt);
  }

  // Cholesky was successful - use it to find the pseudo_inverse and multiply
  // it with the MTTKRP result
  // SPLATT still calls this function regardless of the factorization output

#if DEBUG == 1
    PrintMatrix("after cholesky or factorization", grams[mode]);
    PrintFPMatrix("after solve - rhs", M->U[mode], I, rank);
#endif

  // cleanup
  free(scratch);
}

#endif // STREAMING_CPD_HPP_