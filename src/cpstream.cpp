#include "cpstream.hpp"

#include <vector>
#include <algorithm>

using namespace std;
// #define DEBUG 1

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
    SparseCPGrams * scpgrams = InitSparseCPGrams(X->nmodes, rank);
    DeleteSparseCPGrams(scpgrams, X->nmodes);

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

        spcpstream_iter(
          t_batch,
          M, prev_M, grams, scpgrams,
          max_iters, epsilon, streaming_mode, it, use_alto
        );


        DestroySparseTensor(t_batch);
        ++it;
    }

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


        spcpstream_iter(t_batch, M, prev_M, grams, 
            max_iters, epsilon, streaming_mode, it, use_alto);

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


void idxsort_hist(
  const SparseTensor * const tt, 
  const IType mode, 
  vector<size_t>& idx, 
  vector<size_t>& buckets) {
  if (idx.size() != tt->nnz) {
    idx.resize(tt->nnz);
  }

  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < tt->nnz; ++i) {
    idx[i] = i;
  }

  std::sort(idx.begin(), idx.end(), [&](size_t x, size_t y) {
      return (tt->cidx[mode][x] < tt->cidx[mode][y]);});

  IType* hists;
  IType num_bins;
  size_t* counts;

  #pragma omp parallel
 {
   int nthreads = omp_get_num_threads();
   int tid = omp_get_thread_num();
   size_t start = (tt->nnz * tid) / nthreads;
   size_t end = (tt->nnz * (tid+1)) / nthreads;


   size_t my_first = std::numeric_limits<size_t>::max();
   size_t my_last = std::numeric_limits<size_t>::max();
   size_t my_unique = 0;
   if (start == 0 && start < end) { // first thread with elements
     ++start;
     my_first = 0;
     my_unique = 1;
   }

  for (size_t i = start; i < end; ++i) {
    if (tt->cidx[mode][idx[i]] != tt->cidx[mode][idx[i-1]]) {
      if (my_first > i) {
        my_first = i;
      }
      ++my_unique;
      my_last = i;
    }
  }

  // get total count and prefix sum of `my_unique`
  // TODO: this impleementation is pretty ad-hoc and has a lot of potential
  // to be improved (if necessary - currently this is not even close to a
  // bottleneck)
#pragma omp single
  {
    counts = (size_t*) malloc(nthreads*sizeof(size_t));
  }
#pragma omp barrier
  counts[tid] = my_unique; // FIXME: lots of false sharing here
  // prefix
#pragma omp barrier

  size_t my_prefix = 0;
  size_t num_unique = 0;
  for (int i = 0; i < tid; ++i) {
    my_prefix += counts[i];
  }
  for (int i = 0; i < nthreads; ++i) {
    num_unique += counts[i];
  }

  // create bucket offset array from sorted order
  #pragma omp single
  {
    num_bins = num_unique;
    buckets.resize(num_bins+1);
    buckets[0] = 0;
    buckets[num_bins] = tt->nnz;
  }
  #pragma omp barrier

  if (my_first <= my_last && my_last < tt->nnz) {
    IType c = my_prefix;
    buckets[c] = my_first;
    for (size_t i = my_first+1; i < my_last+1; ++i) {
      if (tt->cidx[mode][idx[i]] != tt->cidx[mode][idx[i-1]]) {
        ++c;
        buckets[c] = i;
      }
    }
  }
 }
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
    Matrix** grams, SparseCPGrams* scpgrams, int max_iters, double epsilon, 
    int streaming_mode, int iter, bool use_alto) {

      PrintSparseTensor(X);

      IType nmodes = X->nmodes;
      // Identify the non zero rows

      vector<vector<size_t>> nz_rows((size_t)nmodes, 
                                              vector<size_t> (0, 0));
      vector<vector<size_t>> buckets((size_t)nmodes, 
                                              vector<size_t> (0, 0));
      vector<vector<size_t>> idx((size_t)nmodes, 
                                          vector<size_t> (0, 0));
      // For storing mappings of indices in I to indices in rowind
      // Without this we had to traverse through all rowind to find the 
      // matching index
      vector<vector<int>> ridx((size_t)nmodes, 
                                        vector<int> (0, 0));


      for (IType m = 0; m < nmodes; ++m) {
        if (m == streaming_mode) continue;

        nonzero_slices(
          X, m, nz_rows[m], idx[m], ridx[m], buckets[m]
        );
        IType nnzr = nz_rows[m].size();
        size_t * rowind = &nz_rows[m][0];

        printf("non zero rows for mode %d\n", m);
        for (int i = 0; i < nnzr; ++i) {
          printf("%d ", rowind[i]);
        }
        printf("\n");
      }

      if (iter == 1) exit(1);

      // What stuf
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
    grams->c_nz_prev[m] = init_mat(rank, rank);
    grams->c_z_prev[m] = init_mat(rank, rank);

    grams->c_nz[m] = init_mat(rank, rank);
    grams->c_z[m] = init_mat(rank, rank);

    grams->h_nz[m] = init_mat(rank, rank);
    grams->h_z[m] = init_mat(rank, rank);

    grams->c[m] = init_mat(rank, rank);
    grams->h[m] = init_mat(rank, rank);

    grams->c_prev[m] = init_mat(rank, rank);    
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