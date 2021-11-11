#include "cpstream.hpp"

#include <vector>
#include <algorithm>

using namespace std;
// #define DEBUG 1
#define TIMER 1

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
    bool use_alto,
    bool use_spcpstream) {

    // Define timers (cpstream)
    uint64_t ts = 0;
    uint64_t te = 0;
    double t_preprocess = 0.0;
    double t_iteration = 0.0;
    double t_postprocess = 0.0;


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

    SparseCPGrams * scpgrams;
    
    if (use_spcpstream) {
        // Hypersparse ALS specific
        scpgrams = InitSparseCPGrams(nmodes, rank);
    }

    fprintf(stderr, "==== Executing %s (%s) ====\n", use_spcpstream ? "spCPSTREAM" : "CPSTREAM", use_alto ? "ALTO" : "non-ALTO");

    int it = 0;
    while (!sst.last_batch()) {
#if TIMER == 1
        BEGIN_TIMER(&ts);
#endif
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
#if TIMER == 1
        END_TIMER(&te);
        ELAPSED_TIME(ts, te, &t_preprocess);
#endif

#if TIMER == 1
        BEGIN_TIMER(&ts);
#endif
        if (use_spcpstream) {
            spcpstream_iter(
              t_batch, M, prev_M, grams, scpgrams,
              max_iters, epsilon, streaming_mode, it, use_alto);
        }
        else {
            // TODO: Can convert it to ALTO outside of the iter function
            // Since our plan is to remove COO mttkrp completely
            // Set s_t to the latest row of global time stream matrix
            cpstream_iter(
                t_batch, M, prev_M, grams,
                max_iters, epsilon, streaming_mode, it, use_alto);
        }
#if TIMER == 1
        END_TIMER(&te);
        ELAPSED_TIME(ts, te, &t_iteration);
#endif
        ++it; // increment of it has to precede global_time memcpy

#if TIMER == 1
        BEGIN_TIMER(&ts);
#endif
        // Copy M -> prev_M
        CopyKruskalModel(&prev_M, &M);
        DestroySparseTensor(t_batch);
        global_time->grow_zero(it);
        memcpy(&(global_time->mat()->vals[rank * (it-1)]), M->U[streaming_mode], rank * sizeof(FType));

#if TIMER == 1
        END_TIMER(&te);
        ELAPSED_TIME(ts, te, &t_postprocess);
#endif

#if TIMER == 1
    fprintf(stderr, "timing CPSTREAM (#it, pre, iter, post)\n");
    fprintf(stderr, "%d\t%f\t%f\t%f\n", it, t_preprocess, t_iteration, t_postprocess);
#endif
    }
    // Step N: Compute final fit
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

    double const final_err = cpd_error(X, factored);

    fprintf(stdout, "final fit error: %f\n", final_err);
#if DEBUG == 1
    PrintKruskalModel(factored);
#endif

    // Clean up 
    DestroySparseTensor(X);
    destroy_grams(grams, M);
    DestroyKruskalModel(M);
    DestroyKruskalModel(prev_M);
    DestroyKruskalModel(factored);
    if (use_spcpstream) {
        DeleteSparseCPGrams(scpgrams, nmodes);
    }
    delete global_time;

    return;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void cpstream_iter(
  SparseTensor* X, KruskalModel* M, KruskalModel* prev_M, Matrix** grams,
  int max_iters, double epsilon,
  int streaming_mode, int iteration, bool use_alto)
{
    fprintf(stderr,
        "Running CP-Stream (%s, iter: %d) with %d max iterations and %.2e epsilon\n",
        use_alto ? "ALTO" : "Non ALTO", iteration, max_iters, epsilon);

    // Timing stuff
    uint64_t ts = 0;
    uint64_t te = 0;

    double t_mttkrp_sm = 0.0;
    double t_mttkrp_om = 0.0;
    double t_bs_sm = 0.0;
    double t_bs_om = 0.0;

    double t_add_historical = 0.0;
    double t_memset = 0.0;

    double t_conv_check = 0.0;
    double t_gram_mat = 0.0;
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
        BEGIN_TIMER(&ts);

        int num_partitions = omp_get_max_threads();
        // int num_partitions = 1;
        int nnz_ptrn = (X->nnz + num_partitions - 1) / num_partitions;

        if (X->nnz < num_partitions) {
            num_partitions = 1;
        }
        else {
            while (nnz_ptrn < omp_get_max_threads()) {
                // Insufficient nnz per partition
                fprintf(stderr, "Insufficient nnz per partition: %d ... Reducing # of partitions... \n", nnz_ptrn);
                num_partitions /= 2;
                nnz_ptrn = (X->nnz + num_partitions - 1) / num_partitions;
            }
        }
        create_alto(X, &AT, num_partitions);

        // Create local fiber copies
        create_da_mem(-1, rank, AT, &ofibs);
        END_TIMER(&te);
        ELAPSED_TIME(ts, te, &t_alto);
    }
    // end - if use alto

    // Lambda scratchpad
    FType ** lambda_sp = (FType **) AlignedMalloc(sizeof(FType*) * nthreads);
    assert(lambda_sp);
    #pragma omp parallel for
    for (IType t = 0; t < nthreads; ++t) {
        lambda_sp[t] = (FType *) AlignedMalloc(sizeof(FType) * rank);
        assert(lambda_sp[t]);
    }

    // REFACTOR: This should go outside of inner iteraion
    if (iteration == 0) {
        for (int m = 0; m < nmodes; ++m) {
            if (m == streaming_mode) continue;
            KruskalModelNorm(M, m, MAT_NORM_2, lambda_sp);
            update_gram(grams[m], M, m);
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

    // keep track of delta for convergence check
    double delta = 0.0;
    double prev_delta = 0.0;

    Matrix * old_gram = zero_mat(rank, rank);

    int tmp_iter = 0;
    for(int i = 0; i < max_iters; i++) {
        delta = 0.0;
        // Solve for time mode (s_t)
        // set to zero
        memset(M->U[streaming_mode], 0, sizeof(FType) * rank);
        BEGIN_TIMER(&ts);
        if (use_alto) {
            mttkrp_alto_par(streaming_mode, M->U, rank, AT, NULL, ofibs);
        }
        else {
            mttkrp_par(X, M, streaming_mode, writelocks);
        }
        END_TIMER(&te);
        AGG_ELAPSED_TIME(ts, te, &t_mttkrp_sm);

#if DEBUG == 1
        PrintFPMatrix("mttkrp before s_t", M->U[streaming_mode], 1, rank);
#endif

        BEGIN_TIMER(&ts);
        // Init gram matrix aTa for all other modes
        pseudo_inverse_stream(
          grams, M, streaming_mode, streaming_mode);
        END_TIMER(&te);
        AGG_ELAPSED_TIME(ts, te, &t_bs_sm);

#if DEBUG == 1
        PrintFPMatrix("s_t: after solve", M->U[streaming_mode], rank, 1);
#endif

        copy_upper_tri(grams[streaming_mode]);
        // Copy newly computed gram matrix G_t to old_gram
        memcpy(old_gram->vals, grams[streaming_mode]->vals, rank * rank * sizeof(*grams[streaming_mode]->vals));

#if DEBUG == 1
        PrintMatrix("gram mat before updating s_t", grams[streaming_mode]);
#endif
        BEGIN_TIMER(&ts);
        // Accumulate new time slice into temporal Gram matrix
        // Update grams
        for (int m = 0; m < rank; ++m) {
          for (int n = 0; n < rank; ++n) {
              grams[streaming_mode]->vals[m + n * rank] = old_gram->vals[m + n * rank] + M->U[streaming_mode][m] * M->U[streaming_mode][n];
          }
        }
        END_TIMER(&te);
        AGG_ELAPSED_TIME(ts, te, &t_gram_mat);

#if DEBUG == 1
        PrintMatrix("gram mat after updating s_t", grams[streaming_mode]);
#endif
        // For all other modes
        for(int j = 0; j < X->nmodes; j++) {
            if (j == streaming_mode) continue;

            BEGIN_TIMER(&ts);
            memset(M->U[j], 0, sizeof(FType) * dims[j] * rank);
            END_TIMER(&te);
            AGG_ELAPSED_TIME(ts, te, &t_memset);


#if DEBUG == 1
            char str[512];
            sprintf(str, "M[%d] before mttkrp", j);
            PrintFPMatrix(str, M->U[j], M->dims[j], rank);
            memset(str, 0, 512);
#endif

            BEGIN_TIMER(&ts);
            if (use_alto) {
                mttkrp_alto_par(j, M->U, rank, AT, NULL, ofibs);
            } else {
                mttkrp_par(X, M, j, writelocks);
            }
            END_TIMER(&te);
            AGG_ELAPSED_TIME(ts, te, &t_mttkrp_om);

            // add historical
            BEGIN_TIMER(&ts);
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
                // assert(prev_M->dims[m] == M->dims[m]);
                matmul(prev_M->U[m], true, M->U[m], false, 
                  historical->vals, prev_M->dims[m], rank, M->dims[m], rank, 0.0);

                #pragma omp parallel for schedule(static)
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

            END_TIMER(&te);
            AGG_ELAPSED_TIME(ts, te, &t_add_historical);

#if DEBUG == 1
            sprintf(str, "after add historical for for mode %d", j);
            PrintFPMatrix(str, M->U[j], M->dims[j], rank);
            memset(str, 0, 512);
#endif

            BEGIN_TIMER(&ts);
            pseudo_inverse_stream(
                grams, M, j, streaming_mode);
            END_TIMER(&te);
            AGG_ELAPSED_TIME(ts, te, &t_bs_om);

#if DEBUG == 1
            sprintf(str, "ts: %d, it: %d: updated factor matrix for mode %d", iteration, i, j);
            PrintFPMatrix(str, M->U[j], M->dims[j], rank);
            memset(str, 0, 512);
#endif

            BEGIN_TIMER(&ts);
            update_gram(grams[j], M, j);
            END_TIMER(&te);
            AGG_ELAPSED_TIME(ts, te, &t_gram_mat);

            int factor_mat_size = rank * M->dims[j];

            BEGIN_TIMER(&ts);
            delta += mat_norm_diff(prev_M->U[j], M->U[j], factor_mat_size) / (mat_norm(M->U[j], factor_mat_size) + 1e-12);
            END_TIMER(&te);
            AGG_ELAPSED_TIME(ts, te, &t_conv_check);
        } // for each mode

#if DEBUG == 1
        // PrintKruskalModel(M);
#endif
        tmp_iter = i;

#if 1
        fprintf(stderr, "it: %d delta: %e prev_delta: %e (%e diff)\n", i, delta, prev_delta, fabs(delta - prev_delta));
#endif
        if ((i > 0) && fabs(prev_delta - delta) < epsilon) {
            prev_delta = 0.0;
            break;
        } else {
            prev_delta = delta;
        }
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

    fprintf(stderr, "timing CPSTREAM-ITER\n");
    fprintf(stderr, "#ts\t#nnz\t#it\talto\tmttkrp_sm\tbs_sm\tmem set\tmttkrp_om\thist\tbs_om\tupd_gram\tconv_check\n");
    fprintf(stderr, "%d\t%llu\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", 
        iteration+1, X->nnz, num_inner_iter+1, 
        t_alto, t_mttkrp_sm, t_bs_sm, 
        t_memset, t_mttkrp_om, t_add_historical, 
        t_bs_om, t_gram_mat, t_conv_check);
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

vector<size_t> zero_slices(const IType I, vector<size_t> &nz_rows)
{
  vector<size_t> zero_slices;

  IType cnt = 0;
  for (IType i = 0; i < I; i++) {
    if (cnt >= nz_rows.size()) {
      zero_slices.push_back(i);
    } else  {
      if (i == nz_rows.at(cnt)) {
        // If i is a index for a non-zero row, skip
        cnt += 1;
      } else {
        zero_slices.push_back(i);
      }
    }
  }
  return zero_slices;
}

void spcpstream_iter(SparseTensor* X, KruskalModel* M, KruskalModel * prev_M,
    Matrix** grams, SparseCPGrams* scpgrams, int max_iters, double epsilon,
    int streaming_mode, int iteration, bool use_alto)
{
  /* Timing stuff */
  uint64_t ts = 0;
  uint64_t te = 0;

  double t_mttkrp_sm = 0.0;
  double t_mttkrp_om = 0.0;
  double t_bs_sm = 0.0;
  double t_bs_om = 0.0;

  double t_add_historical = 0.0;
  double t_memset = 0.0;

  double t_conv_check = 0.0;
  double t_gram_mat = 0.0;
  double t_norm = 0.0;

  // Needed to measure alto time
  double t_alto = 0.0;
  
  // spcpstream specific
  double t_row_op = 0.0; // row related operations
  double t_mat_conversion = 0.0; // fm <-> rsp mat conversion op
  double t_upd_fm = 0.0; // update full factor matrix
  /* End - Timing stuff */

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

  // Needed to formulate full-sized factor matrix within the convergence loop
  // The zero rows still change inbetween iterations due to Q and Phi changing
  RowSparseMatrix ** A_nz = (RowSparseMatrix**) AlignedMalloc(nmodes * sizeof(RowSparseMatrix*));
  RowSparseMatrix ** A_nz_prev = (RowSparseMatrix**) AlignedMalloc(nmodes * sizeof(RowSparseMatrix*));

  // Q * Phi^-1 is needed to update A_z[m] after inner convergence
  Matrix ** Q_Phi_inv = (Matrix **) AlignedMalloc(nmodes * sizeof(Matrix *));
  for (int m = 0; m < nmodes; ++m) {
    Q_Phi_inv[m] = init_mat(rank, rank);
  }
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

  // In case we use alto format
  AltoTensor<LIType> * AT;
  FType ** ofibs = NULL;
  // if using ALTO
  if (use_alto) {
    BEGIN_TIMER(&ts);
    int num_partitions = omp_get_max_threads();
    // int num_partitions = 1;
    int nnz_ptrn = (X->nnz + num_partitions - 1) / num_partitions;

    if (X->nnz < num_partitions) {
        num_partitions = 1;
    }
    else {
        while (nnz_ptrn < omp_get_max_threads()) {
            // Insufficient nnz per partition
            fprintf(stderr, "Insufficient nnz per partition: %d ... Reducing # of partitions... \n", nnz_ptrn);
            num_partitions /= 2;
            nnz_ptrn = (X->nnz + num_partitions - 1) / num_partitions;
        }
    }
    create_alto(X, &AT, num_partitions);

    // Create local fiber copies
    create_da_mem(-1, rank, AT, &ofibs);
    END_TIMER(&te);
    ELAPSED_TIME(ts, te, &t_alto);

    END_TIMER(&te); ELAPSED_TIME(ts, te, &t_alto);
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
    BEGIN_TIMER(&ts);
    // Identify nonzero slices for all modes
    nonzero_slices(
      X, m, nz_rows[m], idx[m], ridx[m], buckets[m]
    );
    END_TIMER(&te);
    AGG_ELAPSED_TIME(ts, te, &t_row_op);

    size_t nnzr = nz_rows[m].size();
    size_t * rowind = &nz_rows[m][0];

    // temp mat to Use two times for prev_M and M
    Matrix _fm; // temp mat to compute A_nz_prev and c[m]
    mat_hydrate(&_fm, prev_M->U[m], prev_M->dims[m], rank);

    BEGIN_TIMER(&ts);
    // Updating A_nz_prev[m]
    A_nz_prev[m] = convert_to_rspmat(&_fm, nnzr, rowind);
    END_TIMER(&te);
    AGG_ELAPSED_TIME(ts, te, &t_mat_conversion);

    // Updating c[m]
    mat_hydrate(&_fm, M->U[m], M->dims[m], rank);

    BEGIN_TIMER(&ts);
    mat_aTa(&_fm, c[m]);
    END_TIMER(&te);
    AGG_ELAPSED_TIME(ts, te, &t_gram_mat);

    // Updating h[m]
    BEGIN_TIMER(&ts);
    matmul(
      prev_M->U[m], true,
      M->U[m], false, h[m]->vals,
      prev_M->dims[m], rank, M->dims[m], rank, 0.0);
    END_TIMER(&te);
    AGG_ELAPSED_TIME(ts, te, &t_add_historical);
  }

  BEGIN_TIMER(&ts);
  // ==== Step 2. ==== Compute c_z_prev using c_prev, c_nz_prev - it > 0
  if (iteration > 0) {
    for (IType m = 0; m < nmodes; ++m) {
      if (m == streaming_mode) continue;
      Matrix _fm;
      mat_hydrate(&_fm, prev_M->U[m], prev_M->dims[m], rank);
      mataTa_idx_based(&_fm, nz_rows[m], c_nz_prev[m]);
      for (IType i = 0; i < rank * rank; ++i) {
        c_z_prev[m]->vals[i] = c_prev[m]->vals[i] - c_nz_prev[m]->vals[i];
      }
    }
  }
  END_TIMER(&te);
  AGG_ELAPSED_TIME(ts, te, &t_gram_mat);

  BEGIN_TIMER(&ts);
  // ==== Step 3. ==== Set A_nz for all modes
  for (IType m = 0; m < nmodes; ++m) {
    if (m == streaming_mode) {
      A_nz[m] = rspmat_init(1, rank, 1);
      A_nz[m]->rowind[0] = 0; // this never changes
      for (int r = 0; r < rank; ++r) {
        A_nz[m]->mat->vals[r] = 0.0;
      }
    } else {
      Matrix _fm;
      mat_hydrate(&_fm, M->U[m], M->dims[m], rank);
      A_nz[m] = convert_to_rspmat(&_fm, nz_rows[m].size(), &nz_rows[m][0]);
    }
  }
  END_TIMER(&te);
  AGG_ELAPSED_TIME(ts, te, &t_mat_conversion);

  // ==== Step 4. ==== Inner iteration for-loop
  int tmp_iter = 0; // To log number of iterations until convergence
  for (int i = 0; i < max_iters; i++) {
    delta = 0.0; // Reset to 0.0 for every iteration

    // ==== Step 4-1. ===== Compute s_t
    FType * s_t = A_nz[streaming_mode]->mat->vals;
    memset(s_t, 0, rank * sizeof(FType));
    
    BEGIN_TIMER(&ts);
    if (use_alto) {
      rowsparse_mttkrp_alto_par(streaming_mode, A_nz, ridx, rank, AT, NULL, ofibs);
    } else {
      // The parallel rowsparse_mttkrp is currently messy..
      RowSparseMatrix * mttkrp_res = rowsparse_mttkrp(
        X, A_nz, streaming_mode, streaming_mode, idx[streaming_mode], ridx, buckets[streaming_mode]);
      memcpy(A_nz[streaming_mode]->mat->vals, mttkrp_res->mat->vals, sizeof(FType) * rank);
      rspmat_free(mttkrp_res);
    }
    END_TIMER(&te);
    AGG_ELAPSED_TIME(ts, te, &t_mttkrp_sm);

#if DEBUG == 1
    PrintFPMatrix("s_t before solve", s_t, 1, rank);
#endif

    memcpy(M->U[streaming_mode], A_nz[streaming_mode]->mat->vals, rank * sizeof(FType));

    BEGIN_TIMER(&ts);
    pseudo_inverse_stream(c /* used to be grams */, M, streaming_mode, streaming_mode);
    END_TIMER(&te);
    AGG_ELAPSED_TIME(ts, te, &t_bs_sm);

    // Update A_nz[streaming_mode] and M->U[streaming_mode]
    memcpy(A_nz[streaming_mode]->mat->vals, M->U[streaming_mode], rank * sizeof(FType));

#if DEBUG == 1
    PrintFPMatrix("s_t after solve", A_nz[streaming_mode]->mat->vals, 1, rank);
#endif

    // ==== Step 4-2. ==== Compute G_t-1(old_gram), G_t-1 + ssT (grams[streaming_mode])
    // Update gram matrix
    copy_upper_tri(grams[streaming_mode]);
    // Copy newly computed gram matrix G_t to old_gram
    memcpy(old_gram->vals, grams[streaming_mode]->vals, rank * rank * sizeof(*grams[streaming_mode]->vals));

#if DEBUG == 1
    PrintMatrix("gram mat before updating s_t", grams[streaming_mode]);
#endif

    BEGIN_TIMER(&ts);
    // Accumulate new time slice into temporal Gram matrix
    // Update grams
    for (int m = 0; m < rank; ++m) {
      for (int n = 0; n < rank; ++n) {
          // Hard coded forgetting factor?
          grams[streaming_mode]->vals[m + n * rank] += s_t[m] * s_t[n];
      }
    }
    END_TIMER(&te);
    AGG_ELAPSED_TIME(ts, te, &t_gram_mat);

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

      BEGIN_TIMER(&ts);
      // ==== Step 4-3-1. ==== Compute Phi[m], Q[m]
      // h[sm], c[sm] each contains old_gram, old_gram * s*s.T
      mat_form_gram(h, Q, nmodes, m);
      mat_form_gram(c, Phi, nmodes, m);
      BEGIN_TIMER(&te); AGG_ELAPSED_TIME(ts, te, &t_gram_mat);

#if DEBUG == 1
      memset(_str, 0, 512);
      sprintf(_str, "Before mttkrp: %d, A_nz[%d]", iteration, m);
      PrintRowSparseMatrix(_str, A_nz[m]);
#endif

      BEGIN_TIMER(&ts);
      // ==== Step 4-3-2. ==== Compute rowsparse MTTKRP for mode m
      RowSparseMatrix * mttkrp_res;

      if (use_alto) {
        for (int jj = 0; jj < A_nz[m]->nnzr * rank; ++jj) {
          A_nz[m]->mat->vals[jj] = 0.0;
        };
        rowsparse_mttkrp_alto_par(m, A_nz, ridx, rank, AT, NULL, ofibs);
        mttkrp_res = A_nz[m];
      } 
      else {
        mttkrp_res = rowsparse_mttkrp(
          X, A_nz, m, streaming_mode,
          idx[m], ridx, buckets[m]);
      }
      END_TIMER(&te); AGG_ELAPSED_TIME(ts, te, &t_mttkrp_om);

#if DEBUG == 1
      memset(_str, 0, 512);
      sprintf(_str, "After mttkrp: %d, A_nz[%d]", iteration, m);
      PrintRowSparseMatrix(_str, mttkrp_res);

      for (int mm = 0; mm < nmodes; ++mm) {
        fprintf(stderr, "mode: %d\n", mm);
        PrintMatrix("c", c[mm]);
      }
      PrintMatrix("Phi matrix", Phi);
#endif

      // fprintf(stderr, "add_hist\n");      
      BEGIN_TIMER(&ts);
      // ==== Step 4-3-3 ==== Add historical (mttkrp_res + A_nz_prev[m] * Q[m])
      // TODO: ????? Do we update A_nz_prev between iterations or is A_nz_prev static for
      // current time slice
      RowSparseMatrix * A_nz_prev_Q = rsp_mat_mul(A_nz_prev[m], Q);
      rsp_mat_add(mttkrp_res, A_nz_prev_Q);
      END_TIMER(&te); AGG_ELAPSED_TIME(ts, te, &t_add_historical);

      BEGIN_TIMER(&ts);
      // ==== Step 4-3-4 Solve for A_nz ====
      Matrix * _Phi = init_mat(rank, rank);
      memcpy(_Phi->vals, Phi->vals, rank * rank * sizeof(FType));
      pseudo_inverse(_Phi, mttkrp_res->mat);
      END_TIMER(&te); AGG_ELAPSED_TIME(ts, te, &t_bs_om);


      // ===== Step 4-3-5 =====
      BEGIN_TIMER(&ts);
      if (!use_alto) {
        memcpy(A_nz[m]->mat->vals, mttkrp_res->mat->vals, nz_rows[m].size() * rank * sizeof(FType));
        memcpy(A_nz[m]->rowind, mttkrp_res->rowind, sizeof(size_t) * nz_rows[m].size());
      }
      END_TIMER(&te); AGG_ELAPSED_TIME(ts, te, &t_mttkrp_om); // Hard to categorize but it technically should be boiled in actually mttkrp op

#if DEBUG == 1
      PrintRowSparseMatrix("After solve", A_nz[m]);
#endif

      // ==== Step 4-3-6 Update h_nz[m] c_nz[m] ====
      rsp_mataTb(mttkrp_res, mttkrp_res, c_nz[m]);
      rsp_mataTb(A_nz_prev[m], mttkrp_res, h_nz[m]);
      END_TIMER(&te); AGG_ELAPSED_TIME(ts, te, &t_gram_mat);

      // fprintf(stderr, "gram_mat\n");      

      // ==== Step 4-3-7 Solve for zero slices (h_z[m], c_z[m]) ====
      // TODO: We don't need to to cholesky factorization for Phi twice!!!
      memcpy(_Phi->vals, Phi->vals, rank * rank * sizeof(FType));
      memcpy(Q_Phi_inv[m]->vals, Q->vals, rank * rank * sizeof(FType));

      BEGIN_TIMER(&ts);
      pseudo_inverse(_Phi, Q_Phi_inv[m]); // _Q now is Q_Phi_inv
      END_TIMER(&te); AGG_ELAPSED_TIME(ts, te, &t_bs_om);

      BEGIN_TIMER(&ts);
      matmul(c_z_prev[m], false, Q_Phi_inv[m], false, h_z[m], 0.0);
      matmul(Q_Phi_inv[m], true, h_z[m], false, c_z[m], 0.0);
      END_TIMER(&te); AGG_ELAPSED_TIME(ts, te, &t_gram_mat);

      BEGIN_TIMER(&ts);
      // ==== Step 4-3-8 Update h[m], c[m] ====
      for (int i = 0; i < rank * rank; ++i) {
        c[m]->vals[i] = c_nz[m]->vals[i] + c_z[m]->vals[i];
        h[m]->vals[i] = h_nz[m]->vals[i] + h_z[m]->vals[i];
      }
      END_TIMER(&te); AGG_ELAPSED_TIME(ts, te, &t_gram_mat);

      BEGIN_TIMER(&ts);
      // ==== Step 4-3-9 Compute delta ====
      FType tr_c = mat_trace(c[m]);
      FType tr_h = mat_trace(h[m]);
      FType tr_c_prev = mat_trace(c_prev[m]);

      delta += sqrt(fabs(((tr_c + tr_c_prev - 2.0 * tr_h) / (tr_c + 1e-12))));
      END_TIMER(&te);
      AGG_ELAPSED_TIME(ts, te, &t_conv_check);

      free_mat(_Phi);
      if (use_alto) {
        // rspmat_free(mttkrp_res);
        mttkrp_res = NULL;
      } else {
        rspmat_free(mttkrp_res);
      }
      rspmat_free(A_nz_prev_Q);
    } // for each non-streaming mode
    
    // May compute fit here - probably not the best idea due to being slow
    
    tmp_iter = i;
    fprintf(stderr, "it: %d delta: %e prev_delta: %e (%e diff)\n", i, delta, prev_delta, fabs(delta - prev_delta));

    if ((i > 0) && fabs(prev_delta - delta) < epsilon) {
      prev_delta = 0.0;
      break;
    } else {
      prev_delta = delta;
    }
  } // end for loop: max_iters

  num_inner_iter += tmp_iter; // track number of iterations per time-slice

  // ==== Step 5. ==== Update factor matrices M->U[m]
  for (int m = 0; m < nmodes; ++m) {
    if (m == streaming_mode) continue;

    BEGIN_TIMER(&ts);
    // ==== Step 5-1. ==== Update M->U[m] = A_z[m] + A_nz[m]
    // Updating prev_A_z[m]
    std::vector<size_t> z_rows = zero_slices(X->dims[m], nz_rows[m]);
    END_TIMER(&te); AGG_ELAPSED_TIME(ts, te, &t_row_op);
    size_t nzr = z_rows.size();

    BEGIN_TIMER(&ts);
    Matrix _fm;
    mat_hydrate(&_fm, prev_M->U[m], prev_M->dims[m], rank);
    RowSparseMatrix * prev_A_z = convert_to_rspmat(&_fm, nzr, &z_rows[0]);
    RowSparseMatrix * prev_A_z_Q_Phi_inv = rsp_mat_mul(prev_A_z, Q_Phi_inv[m]);
    END_TIMER(&te); AGG_ELAPSED_TIME(ts, te, &t_mat_conversion);

    BEGIN_TIMER(&ts);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nzr; ++i) {
      size_t ridx = z_rows.at(i);
      memcpy(&(M->U[m][ridx * rank]), &(prev_A_z_Q_Phi_inv->mat->vals[i * rank]), sizeof(FType) * rank);

      // for (int r = 0; r < rank; ++r) {
      //   M->U[m][ridx * rank + r] = prev_A_z_Q_Phi_inv->mat->vals[i * rank + r];
      // }
    }
    rspmat_free(prev_A_z_Q_Phi_inv);
    rspmat_free(prev_A_z);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < A_nz[m]->nnzr; ++i) {
      size_t ridx = A_nz[m]->rowind[i];
      memcpy(&(M->U[m][ridx * rank]), &(A_nz[m]->mat->vals[i * rank]), sizeof(FType) * rank);
      // for (int r = 0; r < rank; ++r) {
      //   M->U[m][ridx * rank + r] = A_nz[m]->mat->vals[i * rank + r];
      // }      
    }
    END_TIMER(&te); AGG_ELAPSED_TIME(ts, te, &t_upd_fm);
  }

  // ==== Step 6. ==== Housekeeping...? 
  // ==== Step 6. ==== Apply forgetting factor
  for (IType x = 0; x < rank * rank; ++x) {
    grams[streaming_mode]->vals[x] *= 0.95;
  }

  for (IType m = 0; m < nmodes; ++m) {
    if (m == streaming_mode) continue;
    // Copy all c's to prev_c's
    memcpy(c_prev[m]->vals, c[m]->vals, rank * rank * sizeof(FType));
    memcpy(c_nz_prev[m]->vals, c_nz[m]->vals, rank * rank * sizeof(FType));
  }
  // Apply forgetting factor ..?
  // ==== Step 7. Cleaning up ====
  for (int m = 0; m < nmodes; ++m) {
    free_mat(Q_Phi_inv[m]);
    rspmat_free(A_nz[m]);
    rspmat_free(A_nz_prev[m]);
  }

  free_mat(Q);
  free_mat(Phi);
  free_mat(old_gram);

  #pragma omp parallel for
  for (IType t = 0; t < nthreads; ++t) {
      free(lambda_sp[t]);
  }
  free(lambda_sp);

  for(IType i = 0; i < max_mode_len; i++) {
      omp_destroy_lock(&(writelocks[i]));
  }
  free(writelocks);

  if (use_alto) {
      destroy_da_mem(AT, ofibs, rank, -1);
      destroy_alto(AT);
  }
  /* End: Cleaning up */

  fprintf(stdout, "timing SPCPSTREAM-ITER\n");
  fprintf(stdout, "#ts\t#nnz\t#it\talto\tmttkrp_sm\tbs_sm\tmemset\tmttkrp_om\thist\tbs_om\tupd_gram\tconv_check\trow op\tmat conv\tupd fm\n");
  fprintf(stdout, "%d\t%llu\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", 
      iteration+1, X->nnz, num_inner_iter+1, 
      t_alto, t_mttkrp_sm, t_bs_sm, 
      t_memset, t_mttkrp_om, t_add_historical, 
      t_bs_om, t_gram_mat, t_conv_check, 
      t_row_op, t_mat_conversion, t_upd_fm);
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
  SparseCPGrams * grams = (SparseCPGrams *) AlignedMalloc(sizeof(SparseCPGrams));

  assert(grams);

  grams->c_nz_prev = (Matrix **) AlignedMalloc(nmodes * sizeof(Matrix *));
  grams->c_z_prev = (Matrix **) AlignedMalloc(nmodes * sizeof(Matrix *));

  grams->c_nz = (Matrix **) AlignedMalloc(nmodes * sizeof(Matrix *));
  grams->c_z = (Matrix **) AlignedMalloc(nmodes * sizeof(Matrix *));

  grams->h_nz = (Matrix **) AlignedMalloc(nmodes * sizeof(Matrix *));
  grams->h_z = (Matrix **) AlignedMalloc(nmodes * sizeof(Matrix *));

  grams->c = (Matrix **) AlignedMalloc(nmodes * sizeof(Matrix *));
  grams->h = (Matrix **) AlignedMalloc(nmodes * sizeof(Matrix *));

  grams->c_prev = (Matrix **) AlignedMalloc(nmodes * sizeof(Matrix *));

  for (IType m = 0; m < nmodes; ++m) {
    grams->c_nz_prev[m] = zero_mat(rank, rank);
    grams->c_z_prev[m] = zero_mat(rank, rank);

    grams->c_nz[m] = zero_mat(rank, rank);
    grams->c_z[m] = zero_mat(rank, rank);

    grams->h_nz[m] = zero_mat(rank, rank);
    grams->h_z[m] = zero_mat(rank, rank);

    grams->c[m] = zero_mat(rank, rank);
    grams->h[m] = zero_mat(rank, rank);

    grams->c_prev[m] = zero_mat(rank, rank);
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


double cpd_error(SparseTensor * tensor, KruskalModel * factored) {
    // set up OpenMP locks
    IType max_mode_len = 0;
    IType min_mode_len = tensor->dims[0];
    IType min_mode_idx = 0;

    for(int i = 0; i < factored->mode; i++) {
        if(max_mode_len < factored->dims[i]) {
            max_mode_len = factored->dims[i];
        } 
        if(min_mode_len > factored->dims[i]) {
            min_mode_len = factored->dims[i]; // used to compute mttkrp
            min_mode_idx = i;
        }
    }
    IType nrows = factored->dims[min_mode_idx];
    IType rank = factored->rank;
    omp_lock_t* writelocks = (omp_lock_t*) AlignedMalloc(sizeof(omp_lock_t) *
                                                        max_mode_len);
    assert(writelocks);
    for(IType i = 0; i < max_mode_len; i++) {
        omp_init_lock(&(writelocks[i]));
    }
    // MTTKRP 
    // Copy original matrix
    Matrix * smallmat = mat_fillptr(factored->U[min_mode_idx], nrows, rank);
    for (int i = 0; i < nrows * rank; ++i) {
      factored->U[min_mode_idx][i] = 0.0;
    }
    
    mttkrp_par(tensor, factored, min_mode_idx, writelocks);
    FType * mttkrp = (FType *) malloc(nrows * rank * sizeof(FType));
    memcpy(mttkrp, factored->U[min_mode_idx], nrows * rank * sizeof(FType));

    // Restore factored->U[min_mode_idx]
    memcpy(factored->U[min_mode_idx], smallmat->vals, nrows * rank * sizeof(FType));

    for(IType i = 0; i < max_mode_len; i++) {
        omp_destroy_lock(&(writelocks[i]));
    }
    // inner product between tensor and factored
    double inner = 0;
    #pragma omp parallel reduction(+:inner)
    {
      int const tid = omp_get_thread_num();
      FType * accumF = (FType *) malloc(rank * sizeof(*accumF));

      for(IType r=0; r < rank; ++r) {
        accumF[r] = 0.;
      }

      /* Hadamard product with newest factor and previous MTTKRP */
      #pragma omp for schedule(static)
      for(IType i=0; i < nrows; ++i) {
        FType const * const smallmat_row = &(smallmat->vals[i * rank]);
        FType const * const mttkrp_row = mttkrp + (i*rank);
        for(IType r=0; r < rank; ++r) {
          accumF[r] += smallmat_row[r] * mttkrp_row[r];
        }
      }

      /* accumulate everything into 'inner' */
      for(IType r=0; r < rank; ++r) {
        inner += accumF[r] * factored->lambda[r];
      }

      free(accumF);
    } /* end omp parallel -- reduce myinner */

    // Compute ttnormsq to later compute fit
    FType Xnormsq = 0.0;
    FType* vals = tensor->vals;
    IType nnz = tensor->nnz;

    #pragma omp parallel for reduction(+:Xnormsq) schedule(static)
    for(IType i = 0; i < nnz; ++i) {
      Xnormsq += vals[i] * vals[i];
    }
    free(mttkrp);
    double const Znormsq = kruskal_norm(factored);
    double const residual = sqrt(Xnormsq + Znormsq - (2 * inner));
    
    fprintf(stderr, "Xnormsq: %f, Znormsq: %f, inner: %f\n", Xnormsq, Znormsq, inner);
    double const err = residual / sqrt(Xnormsq);
    return err;
}
