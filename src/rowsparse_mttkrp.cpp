#include "rowsparse_mttkrp.hpp"
#include <vector>

using namespace std;

#define MAX_NMODES 8

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



RowSparseMatrix * rowsparse_mttkrp(
    SparseTensor * X, 
    RowSparseMatrix ** rsp_mats, 
    IType mode, 
    IType stream_mode,
    std::vector<size_t>& idx, 
    std::vector<std::vector<int>>& ridx, 
    std::vector<size_t>& buckets)
{
    IType const I = X->dims[mode];
    IType const nfactors = rsp_mats[0]->J;

    size_t* const hist = buckets.data() + 1;
    size_t num_bins = buckets.size() - 1;

    /* allocate and initialize the row-sparse output matrix */
    RowSparseMatrix * M = rspmat_init(I, nfactors, num_bins);

    FType * const outmat = M->mat->vals;

    // timer_fstart(&clear_output);
    #pragma omp parallel for schedule(static)
    for(IType x=0; x < num_bins * nfactors; ++x) {
        outmat[x] = 0.;
    }

    #pragma omp parallel for schedule(static)
    // Write row index for rsp matrix
    for (IType i = 0; i < num_bins; ++i) {
        M->rowind[i] = (size_t)X->cidx[mode][idx[buckets[i]]];
    }
  
    IType const nmodes = X->nmodes;
    FType * mvals[MAX_NMODES];
    for(IType m=0; m < nmodes; ++m) {
        mvals[m] = rsp_mats[m]->mat->vals;
    }

    FType const * const vals = X->vals;

    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int tid = omp_get_thread_num();

        FType * accum = (FType*)AlignedMalloc(nfactors * sizeof(FType));
        FType * row_accum = (FType*)AlignedMalloc(nfactors * sizeof(FType));

        /* stream through buckets of nnz */
        #pragma omp for schedule(static,1)
        for(IType hi = 0; hi < num_bins; ++hi) {
            IType start = hi == 0 ? 0 : hist[hi-1];
            IType end = hist[hi];
            if (start == end) continue;
            memset(row_accum, 0, nfactors*sizeof(FType));

            IType oidx = hi;
            for (IType i = start; i < end; ++i) {

              /* initialize with value */
              for(IType f=0; f < nfactors; ++f) {
                  accum[f] = vals[idx[i]];
              }

              for(IType m=0; m < nmodes; ++m) {
                if(m == mode) {
                  continue;
                }

                IType m_idx;

                // Given the 'raw' index of the nonzero we need 
                // to find the corresponding rowind from A_nz[m]
                // FYI X->cidx[m] is the array that contains the indices of non zeros in m-th mode
                if (m == stream_mode) m_idx = 0; // Because time-mode has only one row
                else {
                  m_idx = ridx[m][X->cidx[m][idx[i]]];
                }

                FType const * const inrow = mvals[m] + (m_idx * nfactors);
                for(IType f=0; f < nfactors; ++f) {
                  accum[f] *= inrow[f];
                }
              }

              for (IType f=0; f < nfactors; ++f) {
                row_accum[f] += accum[f];
              }
            }

          FType * const outrow = outmat + (oidx * nfactors);
          for (IType f = 0; f < nfactors; ++f) {
            outrow[f] += row_accum[f];
          }
        }
      free(accum);
      free(row_accum);
    } /* end omp parallel */
  return M;
}
    
