#include "cpstream.hpp"

#include <vector>
#include <algorithm>

using namespace std;
#define DEBUG 1

#if DEBUG == 1
    char _str[512];
#endif

// Implementations
void cpstream(
    SparseTensor* X,
    int rank,
    int max_iters,
    int streaming_mode,
    FType epsilon,
    IType seed,
    bool use_alto) {
    // Define timers (common)
    double t_preprocess = 0.0;
    double tot_preprocess = 0.0;

    IType nmodes = X->nmodes;

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
    Matrix * local_time = zero_mat(1, rank);
    StreamMatrix * global_time = new StreamMatrix(rank);

#if 1
    // Hypersparse ALS specific
    SparseCPGrams * scpgrams = InitSparseCPGrams(nmodes, rank);

    RowSparseMatrix ** A_nz_prev = (RowSparseMatrix**) AlignedMalloc(nmodes * sizeof(RowSparseMatrix*));
    RowSparseMatrix ** A_nz = (RowSparseMatrix**) AlignedMalloc(nmodes * sizeof(RowSparseMatrix*));

    Matrix ** Q_Phi_inv;

    int it = 0;
    while (!sst.last_batch()) {
        SparseTensor * t_batch = sst.next_batch();

        if (it == 0) {
            CreateKruskalModel(t_batch->nmodes, t_batch->dims, rank, &M);
            CreateKruskalModel(t_batch->nmodes, t_batch->dims, rank, &prev_M);
            KruskalModelRandomInit(M, (unsigned int)seed);
            KruskalModelZeroInit(prev_M);

            // Override values for M->U[stream_mode] with last row of local_time matrix
            M->U[streaming_mode] = local_time->vals;

            // TODO: specify what "grams" exactly
            init_grams(&grams, M);

            for (int i = 0; i < rank * rank; ++i) {
                grams[streaming_mode]->vals[i] = 0.0;
            }
        } else {
            GrowKruskalModel(t_batch->dims, &M, FILL_RANDOM); // Expands the kruskal model to accomodate new dimensions
            GrowKruskalModel(t_batch->dims, &prev_M, FILL_ZEROS); // Expands the kruskal model to accomodate new dimensions

            for (int j = 0; j < M->mode; ++j) {
                if (j != streaming_mode) {
                    update_gram(grams[j], M, j);
                }
            }
        }

        // TODO: We need to compute c, h
        // before we start the iteration
        // Do we need to pass in A_nz, A
        spcpstream_iter(
          t_batch,
          M, prev_M, A_nz, A_nz_prev, grams, scpgrams,
          max_iters, epsilon, streaming_mode, it, use_alto
        );

        // Compute A_z..  (outside of the iteration)

        // Update M based on A_nz and A_z
        // Copy M -> prev_M
        // Copy A_nz -> A_nz_prev
        // Copy scpgrams->c_nz -> scpgrams->c_nz_prev


        DestroySparseTensor(t_batch);
        ++it;
    }
    DeleteSparseCPGrams(scpgrams, X->nmodes);

    return;
    /*
    RowSparseMatrix ** A_nz_prev;
    RowSparseMatrix ** A_nz;
    */

#else

   // Matrix ** c_nz_prev = (Matrix**)malloc()

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

            // Override values for M->U[stream_mode] with last row of local_time matrix
            M->U[streaming_mode] = local_time->vals;

            // TODO: specify what "grams" exactly
            init_grams(&grams, M);

            for (int i = 0; i < rank * rank; ++i) {
                grams[streaming_mode]->vals[i] = 0.0;
            }
        } else {
            GrowKruskalModel(t_batch->dims, &M, FILL_RANDOM); // Expands the kruskal model to accomodate new dimensions
            GrowKruskalModel(t_batch->dims, &prev_M, FILL_ZEROS); // Expands the kruskal model to accomodate new dimensions

            for (int j = 0; j < M->mode; ++j) {
                if (j != streaming_mode) {
                    update_gram(grams[j], M, j);
                }
            }
        }

        // TODO: Can convert it to ALTO outside of the iter function
        // Since our plan is to remove COO mttkrp completely

        // Set s_t to the latest row of global time stream matrix
        cpstream_iter(
            t_batch, M, prev_M, grams,
            max_iters, epsilon, streaming_mode, it, use_alto);


#if DEBUG == 1
        exit(1);
#endif
        // spcpstream_iter(t_batch, M, prev_M, grams,
        //     max_iters, epsilon, streaming_mode, it, use_alto);

        // Save checkpoints

        // Copy latest
        CopyKruskalModel(&prev_M, &M);

        // Increase global time stream matrix
        DestroySparseTensor(t_batch);
        ++it;
        // printf("it: %d\n", it);
        global_time->grow_zero(it);
        memcpy(&(global_time->mat()->vals[rank * (it-1)]), M->U[streaming_mode], rank * sizeof(FType));

        // Compute fit??
    }

    // Compute final fit
    // Construct final kruskal model: factored
    KruskalModel * factored;
    CreateKruskalModel(X->nmodes, X->dims, rank, &factored);

    // Copy factor matrix values
    for (int m = 0; m < X->nmodes; ++m) {
        if (m == streaming_mode) {
            // Copy from global_time Stream matrix
            memcpy(factored->U[m], global_time->mat()->vals, rank * X->dims[m] * sizeof(FType));
        }
        else {
            // Are there cases where X->dims[m] > M->dims[m]?
            memcpy(factored->U[m], M->U[m], M->dims[m] * rank * sizeof(FType));
        }
    }
    memcpy(factored->lambda, M->lambda, rank * sizeof(FType));

    PrintKruskalModelInfo(factored);

#if DEBUG==1
    PrintKruskalModel(factored);
#endif
    DestroySparseTensor(X);
    destroy_grams(grams, M);
    DestroyKruskalModel(M);
    DestroyKruskalModel(prev_M);
    return;
#endif // spcp-stream dev mode
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void cpstream_iter(
  SparseTensor* X, KruskalModel* M, KruskalModel* prev_M, Matrix** grams,
  int max_iters, double epsilon,
  int streaming_mode, int iteration, bool use_alto)
{
    fprintf(stdout,
        "Running CP-Stream (%s, iter: %d) with %d max iterations and %.2e epsilon\n",
        use_alto ? "ALTO" : "Non ALTO", iteration, max_iters, epsilon);

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

    // Needed to measure alto time
    double t_alto = 0.0;

    int num_inner_iter = 0;

    int nmodes = X->nmodes;
    IType* dims = X->dims;
    IType rank = M->rank;

    IType nthreads = omp_get_max_threads();

    // In case we use alto format
    AltoTensor<LIType> * AT;
    FType ** ofibs = NULL;
    // If we're using ALTO format
    if (use_alto) {
        BEGIN_TIMER(&ticks_start);

        int num_partitions = omp_get_max_threads();
        // int num_partitions = 1;
        int nnz_ptrn = (X->nnz + num_partitions - 1) / num_partitions;

        if (X->nnz < num_partitions) {
            num_partitions = 1;
        }
        else {
            while (nnz_ptrn < omp_get_max_threads()) {
                // Insufficient nnz per partition
                printf("Insufficient nnz per partition: %d ... Reducing # of partitions... \n", nnz_ptrn);
                num_partitions /= 2;
                nnz_ptrn = (X->nnz + num_partitions - 1) / num_partitions;
            }
        }
        create_alto(X, &AT, num_partitions);

        // Create local fiber copies
        create_da_mem(-1, rank, AT, &ofibs);
        END_TIMER(&ticks_end);
        ELAPSED_TIME(ticks_start, ticks_end, &t_alto);
    }
    // end if use alto

    // Compute ttnormsq to later compute fit
    FType normAT = 0.0;
    FType * vals = X->vals;
    IType nnz = X->nnz;

    #pragma omp parallel for reduction(+:normAT) schedule(static)
    for(IType i = 0; i < nnz; ++i) {
        normAT += vals[i] * vals[i];
    }

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
        for (int r = 0; r < rank; ++r) {
            // Just normalize the columns and reset the lambda
            M->lambda[r] = 1.0;
        }
    }

#if DEBUG == 1
    // PrintKruskalModel(M);
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

    int tmp_iter = 0;
    for(int i = 0; i < max_iters; i++) {
        delta = 0.0;
        // Solve for time mode (s_t)
        // set to zero
        memset(M->U[streaming_mode], 0, sizeof(FType) * rank);

        BEGIN_TIMER(&ticks_start);

        if (use_alto) {
            mttkrp_alto_par(streaming_mode, M->U, rank, AT, NULL, ofibs);
        }
        else {
            mttkrp_par(X, M, streaming_mode, writelocks);
        }
        END_TIMER(&ticks_end);

#if DEBUG == 1
    PrintFPMatrix("mttkrp before s_t", M->U[streaming_mode], 1, rank);
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
    memcpy(old_gram->vals, grams[streaming_mode]->vals, rank * rank * sizeof(*grams[streaming_mode]->vals));
    // PrintMatrix("gram mat for streaming mode", grams[streaming_mode]);
    // exit(1);

#if DEBUG == 1
    PrintMatrix("gram mat before updating s_t", grams[streaming_mode]);
#endif

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

    // For all other modes
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
        sprintf(str, "M[%d] before mttkrp", j);
        PrintFPMatrix(str, M->U[j], M->dims[j], rank);
        memset(str, 0, 512);
#endif

        if (use_alto) {
            mttkrp_alto_par(j, M->U, rank, AT, NULL, ofibs);
      }
        else {
            mttkrp_par(X, M, j, writelocks);
        }

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
            matmul(prev_M->U[m], true, M->U[m], false, historical->vals, prev_M->dims[m], rank, M->dims[m], rank, 0.0);

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

        matmul(prev_M->U[j], false, ata_buf->vals, false, M->U[j], prev_M->dims[j], rank, rank, rank, 1.0);
        END_TIMER(&ticks_end);
        AGG_ELAPSED_TIME(ticks_start, ticks_end, &t_aux);

#if DEBUG == 1
        sprintf(str, "after add historical for for mode %d", j);
        PrintFPMatrix(str, M->U[j], M->dims[j], rank);
        memset(str, 0, 512);
#endif

        BEGIN_TIMER(&ticks_start);
        pseudo_inverse_stream(
            grams, M, j, streaming_mode);
        END_TIMER(&ticks_end);
        AGG_ELAPSED_TIME(ticks_start, ticks_end, &t_m_backsolve);

#if DEBUG == 1
        sprintf(str, "ts: %d, it: %d: updated factor matrix for mode %d", iteration, i, j);
        PrintFPMatrix(str, M->U[j], M->dims[j], rank);
        memset(str, 0, 512);

        // PrintKruskalModel(M);
#endif

        // Normalize columns
        // printf("Lambda before norm\n");
        // for (int ii = 0; ii < M->rank; ++ii) {
        //   printf("%f\t", M->lambda[ii]);
        // }

        // printf("\n");
        BEGIN_TIMER(&ticks_start);
        if(i == 0) {
        // KruskalModelNorm(M, j, MAT_NORM_2, lambda_sp);
        } else {
        // MAT_NORM_MAX doesn't affect lambda as much
        // KruskalModelNorm(M, j, MAT_NORM_MAX, lambda_sp);
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

#if DEBUG == 1
    // PrintKruskalModel(M);
#endif

    // calculate fit

    BEGIN_TIMER(&ticks_start);
    fit = cpstream_fit(X, M, grams, scratch);
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

    printf("%d\t%llu\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", iteration, X->nnz, num_inner_iter, t_sm_mttkrp, t_sm_backsolve, t_m_mttkrp, t_m_backsolve, t_fit, t_aux, t_norm, t_alto);
}

// add reverse idx
void nonzero_slices(
    SparseTensor * const tt, const IType mode,
    vector<size_t> &nz_rows,
    vector<size_t> &idx,
    vector<int> &ridx,
    vector<size_t> &buckets)
{
  idxsort_hist(tt, mode, idx, buckets);
  size_t num_bins = buckets.size() - 1;

  for (IType i = 0; i < num_bins; i++) {
    nz_rows.push_back(tt->cidx[mode][idx[buckets[i]]]);
  }
  // Create array for reverse indices
  // We traverse through all rows i
  // if it is a non zero row then add i to ridx array
  // if not, push value -1, which means invalid
  // For example if I = 10: [0, 1, 2, 3, 4, 5, ... 9] and non zero rows are [2, 4, 5]
  // then ridx would have [-1, -1, 0, -1, 1, 2, -1, ...]
  IType _ptr = 0;
  for (IType i = 0; i < tt->dims[mode]; i++) {
    if (nz_rows[_ptr] == i) {
      ridx.push_back(_ptr);
      _ptr++;
    } else {
      ridx.push_back(-1);
    }
  }
}

void spcpstream_iter(SparseTensor* X, KruskalModel* M, KruskalModel * prev_M,
    RowSparseMatrix ** A_nz, RowSparseMatrix ** A_nz_prev,
    Matrix** grams, SparseCPGrams* scpgrams, int max_iters, double epsilon,
    int streaming_mode, int iteration, bool use_alto)
{
  /* Unpack stuff */
  // basic params
  int nmodes = X->nmodes;
  IType* dims = X->dims;
  IType rank = M->rank;
  IType nthreads = omp_get_max_threads();

  // Unpack scpgrams;
  Matrix ** c = scpgrams->c;
  Matrix ** h = scpgrams->h;

  Matrix ** c_nz = scpgrams->c_nz;
  Matrix ** h_nz = scpgrams->h_nz;

  Matrix ** c_z = scpgrams->c_z;
  Matrix ** h_z = scpgrams->h_z;

  Matrix ** c_prev = scpgrams->c_prev;
  Matrix ** c_z_prev = scpgrams->c_z_prev;
  Matrix ** c_nz_prev = scpgrams->c_nz_prev;
  /* End: Unpack stuff */

  /* Init variables */
  int num_inner_iter = 0;
  // keep track of the fit for convergence check
  double fit = 0.0, prev_fit = 0.0, delta = 0.0, prev_delta = 0.0;

  // Used to store non_zero row informatoin for all modes
  vector<vector<size_t>> nz_rows((size_t)nmodes, vector<size_t> (0, 0));
  vector<vector<size_t>> buckets((size_t)nmodes, vector<size_t> (0, 0));
  vector<vector<size_t>> idx((size_t)nmodes, vector<size_t> (0, 0));
  // For storing mappings of indices in I to indices in rowind
  vector<vector<int>> ridx((size_t)nmodes, vector<int> (0, 0));

  Matrix * Q = init_mat(rank, rank);
  Matrix * Phi = init_mat(rank, rank);
  Matrix * old_gram = zero_mat(rank, rank);

  /* End: Init variables */

  /* Housekeeping - Generic, repetitive code */
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
  // Lambda scratchpad
  FType ** lambda_sp = (FType **) AlignedMalloc(sizeof(FType*) * nthreads);
  assert(lambda_sp);
  #pragma omp parallel for
  for (IType t = 0; t < nthreads; ++t) {
      lambda_sp[t] = (FType *) AlignedMalloc(sizeof(FType) * rank);
      assert(lambda_sp[t]);
  }
  /* End: Housekeeping */

#if DEBUG == 1
  PrintSparseTensor(X);
#endif

  // ==== Step 0. ==== Normalize factor matrices for first iteration
  if (iteration == 0) {
      for (int m = 0; m < nmodes; ++m) {
          if (m == streaming_mode) continue;
          KruskalModelNorm(M, m, MAT_NORM_2, lambda_sp);
      }
      for (int r = 0; r < rank; ++r) {
          // Just normalize the columns and reset the lambda
          M->lambda[r] = 1.0;
      }
  }

  // ==== Step 1. ==== Updating c[m], h[m], A_nz_prev[m]
  for (IType m = 0; m < nmodes; ++m) {
    if (m == streaming_mode) continue;

    // Identify nonzero slices for all modes
    nonzero_slices(
      X, m, nz_rows[m], idx[m], ridx[m], buckets[m]
    );

    size_t nnzr = nz_rows[m].size();
    size_t * rowind = &nz_rows[m][0];

    // Use two times for prev_M and M
    Matrix * _fm = init_mat(M->dims[m], M->rank);

    memcpy(_fm->vals, prev_M->U[m], prev_M->dims[m] * rank * sizeof(FType));
    // Updating A_nz_prev[m]
    A_nz_prev[m] = convert_to_rspmat(_fm, nnzr, rowind);
    // Updating h[m]
    matmul(
      prev_M->U[m], true,
      M->U[m], false, h[m]->vals,
      prev_M->dims[m], rank, M->dims[m], rank, 0.0);
    // Updating c[m]
    memcpy(_fm->vals, M->U[m], M->dims[m] * rank * sizeof(FType));
    mat_aTa(_fm, c[m]);

    free_mat(_fm);
#if 0
    printf("non zero rows for mode %llu\n", m);
    for (int i = 0; i < nnzr; ++i) {
      printf("%zu ", rowind[i]);
    }
    printf("\n");

    printf("idx for mode %llu\n", m);
    for (int i = 0; i < X->nnz; ++i) {
      printf("%zu ", idx[m][i]);
    }
    printf("\n");

    printf("ridx for mode %llu\n", m);
    for (int i = 0; i < ridx[m].size(); ++i) {
      printf("%d ", ridx[m][i]);
    }
    printf("\n");
#endif
  }

  // ==== Step 2. ==== Compute c_z_prev using c_prev, c_nz_prev - it > 0
  if (iteration > 0) {
    for (IType m = 0; m < nmodes; ++m) {
      if (m == streaming_mode) continue;
      Matrix * _fm = init_mat(M->dims[m], rank);
      memcpy(_fm->vals, M->U[m], M->dims[m] * rank * sizeof(FType));
      mataTa_idx_based(_fm, nz_rows[m], c_nz_prev[m]);

      for (IType i = 0; i < rank * rank; ++i) {
        c_z_prev[m]->vals[i] = c_prev[m]->vals[i] - c_nz_prev[m]->vals[i];
      }
      free_mat(_fm);
    }
  }

  // ==== Step 3. ==== Set A_nz for all modes
  for (IType m = 0; m < nmodes; ++m) {
    if (m == streaming_mode) {
      A_nz[m] = rspmat_init(1, rank, 1);
    } else {
      Matrix * _fm = init_mat(M->dims[m], rank);
      memcpy(_fm->vals, M->U[m], M->dims[m] * rank * sizeof(FType));
      A_nz[m] = convert_to_rspmat(_fm, nz_rows[m].size(), &nz_rows[m][0]);
      free_mat(_fm);
    }
  }


  // ==== Step 4. ==== Inner iteration for-loop
  int tmp_iter = 0; // To log number of iterations until convergence
  for (int i = 0; i < max_iters; i++) {
    delta = 0.0; // Reset to 0.0 for every iteration

    // ==== Step 4-1. ===== Compute s_t
    FType * s_t = M->U[streaming_mode];
    memset(s_t, 0, rank * sizeof(FType));
    mttkrp_par(X, M, streaming_mode, writelocks); // Use regular mttkrp

#if DEBUG == 1
    PrintFPMatrix("s_t before solve", s_t, 1, rank);
#endif

    pseudo_inverse_stream(grams, M, streaming_mode, streaming_mode);
    // Update A_nz[streaming_mode]
    memcpy(A_nz[streaming_mode]->mat->vals, s_t, rank * sizeof(FType));

#if DEBUG == 1
    PrintFPMatrix("s_t after solve", M->U[streaming_mode], 1, rank);
#endif

    // ==== Step 4-2. ==== Compute G_t-1(old_gram), G_t-1 + ssT (grams[streaming_mode])
    // Update gram matrix
    copy_upper_tri(grams[streaming_mode]);
    // Copy newly computed gram matrix G_t to old_gram
    memcpy(old_gram->vals, grams[streaming_mode]->vals, rank * rank * sizeof(*grams[streaming_mode]->vals));

#if DEBUG == 1
    PrintMatrix("gram mat before updating s_t", grams[streaming_mode]);
#endif

    // Accumulate new time slice into temporal Gram matrix
    // Update grams
    for (int m = 0; m < rank; ++m) {
      for (int n = 0; n < rank; ++n) {
          // Hard coded forgetting factor?
          grams[streaming_mode]->vals[m + n * rank] += s_t[m] * s_t[n];
      }
    }

#if DEBUG == 1
    PrintMatrix("gram mat after updating s_t", grams[streaming_mode]);
#endif
    // Maintainence; updating c and h gram matrices
    // to incorporate latest s_t, this may not be needed
    // If we clean out the Matrix ** grams part
    // This just makes it more explicit
    memcpy(h[streaming_mode]->vals, old_gram->vals, rank * rank * sizeof(FType));
    memcpy(c[streaming_mode]->vals, grams[streaming_mode]->vals, rank * rank * sizeof(FType));

    // ==== Step 4-3. ==== Compute for all other modes
    for(int m = 0; m < X->nmodes; m++) {
      if (m == streaming_mode) continue;

      // ==== Step 4-3-1. ==== Compute Phi[m], Q[m]
      // h[sm], c[sm] each contains old_gram, old_gram * s*s.T
      mat_form_gram(h, Q, nmodes, m);
      mat_form_gram(c, Phi, nmodes, m);

#if DEBUG == 1
      memset(_str, 0, 512);
      sprintf(_str, "Before mttkrp: %d, A_nz[%d]", iteration, m);
      PrintRowSparseMatrix(_str, A_nz[m]);
#endif

#if 0 // In case we need to check input to rowsparse_mttkrp
  // Print all input rsp_mats
      for (int _m = 0; _m < nmodes; ++_m) {
        PrintRowSparseMatrix("rsp mat", A_nz[_m]);
      }
#endif
      // ==== Step 4-3-2. ==== Compute rowsparse MTTKRP for mode m
      RowSparseMatrix * mttkrp_res = rowsparse_mttkrp(
        X, A_nz, m, streaming_mode,
        idx[m], ridx, buckets[m]);

#if DEBUG == 1
      memset(_str, 0, 512);
      sprintf(_str, "After mttkrp: %d, A_nz[%d]", iteration, m);
      PrintRowSparseMatrix(_str, mttkrp_res);
#endif

      // ==== Step 4-3-3 ==== Add historical (mttkrp_res + A_nz_prev[m] * Q[m])
      // TODO: ????? Do we update A_nz_prev between iterations or is A_nz_prev static for
      // current time slice
      RowSparseMatrix * A_nz_prev_Q = rsp_mat_mul(A_nz_prev[m], Q);
      rsp_mat_add(mttkrp_res, A_nz_prev_Q);

      // ==== Step 4-3-4 Solve for A_nz ====
      Matrix * _Phi = init_mat(rank, rank);
      memcpy(_Phi->vals, Phi->vals, rank * rank * sizeof(FType));
      pseudo_inverse(_Phi, mttkrp_res->mat);

      // ==== Step 4-3-5 Update h_nz[m] c_nz[m] ====
      rsp_mataTb(mttkrp_res, mttkrp_res, c_nz[m]);
      rsp_mataTb(A_nz_prev[m], mttkrp_res, h_nz[m]);

      // ==== Step 4-3-6 Solve for zero slices (h_z[m], c_z[m]) ====
      memcpy(_Phi->vals, Phi->vals, rank * rank * sizeof(FType));
      Matrix * _Q = init_mat(rank, rank);
      memcpy(_Q->vals, Q->vals, rank * rank * sizeof(FType));

      pseudo_inverse(_Phi, _Q); // _Q now is Q_Phi_inv

      matmul(c_nz_prev[m], false, _Q, false, h_z[m], 0.0);
      matmul(_Q, true, h_z[m], false, c_z[m], 0.0);

      // ==== Step 4-3-7 Update h[m], c[m] ====
      for (int i = 0; i < rank * rank; ++i) {
        c[m]->vals[i] = c_nz[m]->vals[i] + c_z[m]->vals[i];
        h[m]->vals[i] = h_nz[m]->vals[i] + h_z[m]->vals[i];
      }

      // ==== Step 4-3-8 Compute delta ====
      FType tr_c = mat_trace(c[m]);
      FType tr_h = mat_trace(h[m]);
      FType tr_c_prev = mat_trace(c_prev[m]);

      delta += sqrt(fabs(((tr_c + tr_c_prev - 2.0 * tr_h) / (tr_c + 1e-12))));

#if DEBUG == 1
      PrintRowSparseMatrix("After solve", mttkrp_res);
#endif

      free_mat(_Phi);
      free_mat(_Q);
      rspmat_free(mttkrp_res);
      rspmat_free(A_nz_prev_Q);
    }
    printf("Delta: %f\n", delta);
    exit(1);
  } // end for loop: max_iters

  // ==== Step 5. ==== Update factor matrices M->U[m]
  // ==== Step 5-1. ==== Compute A_z[m]
  // ==== Step 5-2. ==== Update M->U[m] = A_z[m] + A_nz[m]
  for (int m = 0; m < nmodes; ++m) {
    if (m == streaming_mode) continue; // Already done in previous steps
    // else update non_zero rows first
    // update zero rows next
  }

  // ==== Step 6. ==== Housekeeping...? 
  // Copy M -> prevM
  // Apply forgetting factor ..?

  /* Cleaning up */
  free_mat(Q);
  free_mat(Phi);
  free_mat(old_gram);

  /* End: Cleaning up */
  return;
};



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Need a separate version so that we can selectively apply frob reg to stream mode only
static void pseudo_inverse_stream(
  Matrix ** grams, KruskalModel* M, IType mode, IType stream_mode)
{
  IType rank = M->rank;
  IType nmodes = (IType) M->mode;

    // Instantiate Phi = ((*) AtA) (*) Time
    Matrix * Phi = zero_mat(rank, rank);

    for (int i = 0; i < rank * rank; ++i) {
      Phi->vals[i] = 1.0;
    }

  // Calculate V
  IType m = 0;
  if(mode == 0) {
    m = 1;
  }

  memcpy(Phi->vals, grams[m]->vals, sizeof(FType) * rank * rank);
  #pragma unroll
  for(IType i = m + 1; i < nmodes; i++) {
    if(i != mode) {
      #pragma omp simd
      for(IType j = 0; j < rank * rank; j++) {
        Phi->vals[j] *= grams[i]->vals[j];
      }
    }
  }

  FType* scratch = (FType*) AlignedMalloc(sizeof(FType) * rank * rank);
  assert(scratch);

  memcpy(scratch, Phi->vals, sizeof(FType) * rank * rank);

// Apply frobenious norm
// This stabilizes (?) the cholesky factorization of the matrix
// For now just use a generic value (1e-3)
  for (int r = 0; r < rank; ++r) {
    Phi->vals[r * rank + r] += 1e-12;
  }

// if (mode == stream_mode) {
//   for (int r = 0; r < rank; ++r) {
//     Phi->vals[r * rank + r] += 1e-3;
//   }
// }

#if DEBUG == 1
    PrintMatrix("before cholesky or factorization", Phi);
    // PrintFPMatrix("before solve: rhs", M->U[mode], (int)M->dims[mode] , rank);
#endif

  // Try using Cholesky to find the pseudoinvsere of V
  // Setup parameters for LAPACK calls
  // convert IType to int
  char uplo = 'L';
  lapack_int _rank = (lapack_int)rank;
  lapack_int I = (lapack_int)M->dims[mode];
  lapack_int info;
  POTRF(&uplo, &_rank, Phi->vals, &_rank, &info);

  if(info == 0) {
    lapack_int s_info = 0;

#if DEBUG == 1
    // printf("uplo: %c, lda: %d, nrhs: %d, ldb: %d, info: %d\n", uplo, _rank, I, _rank, s_info);
#endif
    // printf("\n\n");
    // for (int i = 0; i < M->dims[mode] * rank; ++i) {
    //   printf("%e\t", M->U[mode][i]);
    // }
    // printf("\n\n");
    // for (int i = 0; i < rank * rank; ++i) {
    //   printf("%e\t", Phi->vals[i]);
    // }
    // printf("\n\n");
    POTRS(&uplo, &_rank, &I, Phi->vals, &_rank,
          M->U[mode], &_rank, &s_info);

  } else {
    fprintf(stderr, "ALTO: DPOTRF returned %d, Solving using GELSS\n", info);

    // Otherwise use rank-deficient solver, GELSY
    // Restore V
    memcpy(Phi->vals, scratch, sizeof(FType) * rank * rank);

    //PrintFPMatrix("gram matrix when fallback", rank, rank, Phi, rank);
    // Fill up the upper part
    // #pragma unroll
    // for(IType i = 0; i < rank; i++) {
    //   #pragma omp simd
    //   for(IType j = i; j < rank; j++) {
    //     Phi->vals[i * rank + j] = Phi->vals[j * rank + i];
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
        printf("%e\t", Phi->vals[n]);
      }
      printf("\n\n");

      for (int n = 0; n < rank * I; ++n) {
        printf("%e\t", M->U[mode][n]);
      }
*/
    GELSS(&_rank, &_rank, &I,
        Phi->vals, &_rank,
        M->U[mode], &_rank,
        conditions, &rcond, &ret_rank,
        &work_qr, &lwork, &info_dgelsy);

    lwork = (lapack_int) work_qr;
    double* work = (double*) AlignedMalloc(sizeof(double) * lwork);

    GELSS(&_rank, &_rank, &I,
          Phi->vals, &_rank,
          M->U[mode], &_rank,
          conditions, &rcond, &ret_rank,
          work, &lwork, &info_dgelsy);

    if (info_dgelsy) {
      PrintMatrix("gram matrix", Phi);
      PrintFPMatrix("rhs", M->U[mode], I, rank);
      fprintf(stderr, "\tDGELSS failed!! Mode %llu Min Norm Solve: %d\nDGELSS effective rank: %d\n", mode, info_dgelsy, ret_rank);
      exit(1);
    }

#if DEBUG == 1
    // printf("uplo: %c, lda: %d, nrhs: %d, ldb: %d, info: %d\n", uplo, _rank, I, _rank, info_dgelsy);
    // fprintf(stderr, "\t Mode %llu Min Norm Solve: %d\nDGELSS effective rank: %d\n", mode, info_dgelsy, ret_rank);
#endif
    free(work);
    free(jpvt);
  }

  // Cholesky was successful - use it to find the pseudo_inverse and multiply
  // it with the MTTKRP result
  // SPLATT still calls this function regardless of the factorization output

#if DEBUG == 1
    // PrintMatrix("after cholesky or factorization", Phi);
    // PrintFPMatrix("after solve - rhs", M->U[mode], I, rank);
#endif

  // cleanup
  free(scratch);
}

static double cpstream_fit(SparseTensor* X, KruskalModel* M, Matrix** grams, FType* U_mttkrp)
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

SparseCPGrams * InitSparseCPGrams(IType nmodes, IType rank) {
  SparseCPGrams * grams = (SparseCPGrams *) malloc(sizeof(SparseCPGrams));

  assert(grams);

  grams->c_nz_prev = (Matrix **) malloc(nmodes * sizeof(Matrix *));
  grams->c_z_prev = (Matrix **) malloc(nmodes * sizeof(Matrix *));

  grams->c_nz = (Matrix **) malloc(nmodes * sizeof(Matrix *));
  grams->c_z = (Matrix **) malloc(nmodes * sizeof(Matrix *));

  grams->h_nz = (Matrix **) malloc(nmodes * sizeof(Matrix *));
  grams->h_z = (Matrix **) malloc(nmodes * sizeof(Matrix *));

  grams->c = (Matrix **) malloc(nmodes * sizeof(Matrix *));
  grams->h = (Matrix **) malloc(nmodes * sizeof(Matrix *));

  grams->c_prev = (Matrix **) malloc(nmodes * sizeof(Matrix *));

  for (IType m = 0; m < nmodes; ++m) {
    grams->c_nz_prev[m] = ones_mat(rank, rank);
    grams->c_z_prev[m] = ones_mat(rank, rank);

    grams->c_nz[m] = ones_mat(rank, rank);
    grams->c_z[m] = ones_mat(rank, rank);

    grams->h_nz[m] = ones_mat(rank, rank);
    grams->h_z[m] = ones_mat(rank, rank);

    grams->c[m] = ones_mat(rank, rank);
    grams->h[m] = ones_mat(rank, rank);

    grams->c_prev[m] = ones_mat(rank, rank);
  }

  return grams;
}

void DeleteSparseCPGrams(SparseCPGrams * grams, IType nmodes) {
  for (IType m = 0; m < nmodes; ++m) {
    free_mat(grams->c_nz_prev[m]);
    free_mat(grams->c_z_prev[m]);

    free_mat(grams->c_nz[m]);
    free_mat(grams->c_z[m]);

    free_mat(grams->h_nz[m]);
    free_mat(grams->h_z[m]);

    free_mat(grams->c[m]);
    free_mat(grams->h[m]);

    free_mat(grams->c_prev[m]);
  }

  free(grams->c_nz_prev);
  free(grams->c_z_prev);

  free(grams->c_nz);
  free(grams->c_z);

  free(grams->h_nz);
  free(grams->h_z);

  free(grams->c);
  free(grams->h);

  free(grams->c_prev);

  free(grams);
  return;
}