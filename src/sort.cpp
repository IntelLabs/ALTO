#include "sort.hpp"

/**
 * idx = idx2*dim1 + idx1
 * -> ret = idx1*dim2 + idx2
 *        = (idx%dim1)*dim2 + idx/dim1
 */
static inline IType p_transpose_idx(
    IType const idx,
    IType const dim1,
    IType const dim2)
{
  return idx%dim1*dim2 + idx/dim1;
}

static inline int p_ttqcmp2(
  IType const * const ind0,
  IType const * const ind1,
  IType const i,
  IType const j[2])
{
  if(ind0[i] < j[0]) {
    return -1;
  } else if(j[0] < ind0[i]) {
    return 1;
  }
  if(ind1[i] < j[1]) {
    return -1;
  } else if(j[1] < ind1[i]) {
    return 1;
  }

  return 0;
}

static inline int p_ttqcmp3(
  IType const * const ind0,
  IType const * const ind1,
  IType const * const ind2,
  IType const i,
  IType const j[3])
{
  if(ind0[i] < j[0]) {
    return -1;
  } else if(j[0] < ind0[i]) {
    return 1;
  }
  if(ind1[i] < j[1]) {
    return -1;
  } else if(j[1] < ind1[i]) {
    return 1;
  }
  if(ind2[i] < j[2]) {
    return -1;
  } else if(j[2] < ind2[i]) {
    return 1;
  }

  return 0;
}

static inline int p_ttqcmp(
  SparseTensor const * const tt,
  IType const * const cmplt,
  IType const i,
  IType const j[MAX_NUM_MODES])
{
  for(IType m=0; m < tt->nmodes; ++m) {
    if(tt->cidx[cmplt[m]][i] < j[cmplt[m]]) {
      return -1;
    } else if(j[cmplt[m]] < tt->cidx[cmplt[m]][i]) {
      return 1;
    }
  }
  return 0;
}

/**
* @brief Swap nonzeros i and j.
*
* @param tt The tensor to operate on.
* @param i The first nonzero to swap.
* @param j The second nonzero to swap with.
*/
static inline void p_ttswap(
  SparseTensor * const tt,
  IType const i,
  IType const j)
{
  FType vtmp = tt->vals[i];
  tt->vals[i] = tt->vals[j];
  tt->vals[j] = vtmp;

  IType itmp;
  for(IType m=0; m < tt->nmodes; ++m) {
    itmp = tt->cidx[m][i];
    tt->cidx[m][i] = tt->cidx[m][j];
    tt->cidx[m][j] = itmp;
  }
}


static void p_tt_quicksort3(
  SparseTensor * const tt,
  IType const * const cmplt,
  IType const start,
  IType const end)
{
  FType vmid;
  IType imid[3];

  IType * const ind0 = tt->cidx[cmplt[0]];
  IType * const ind1 = tt->cidx[cmplt[1]];
  IType * const ind2 = tt->cidx[cmplt[2]];
  FType * const vals = tt->vals;


    size_t i = start+1;
    size_t j = end-1;
    size_t k = start + ((end - start) / 2);

    /* grab pivot */
    vmid = vals[k];
    vals[k] = vals[start];
    imid[0] = ind0[k];
    imid[1] = ind1[k];
    imid[2] = ind2[k];
    ind0[k] = ind0[start];
    ind1[k] = ind1[start];
    ind2[k] = ind2[start];

    while(i < j) {
      /* if tt[i] > mid  -> tt[i] is on wrong side */
      if(p_ttqcmp3(ind0,ind1,ind2,i,imid) == 1) {
        /* if tt[j] <= mid  -> swap tt[i] and tt[j] */
        if(p_ttqcmp3(ind0,ind1,ind2,j,imid) < 1) {
          FType vtmp = vals[i];
          vals[i] = vals[j];
          vals[j] = vtmp;
          IType itmp = ind0[i];
          ind0[i] = ind0[j];
          ind0[j] = itmp;
          itmp = ind1[i];
          ind1[i] = ind1[j];
          ind1[j] = itmp;
          itmp = ind2[i];
          ind2[i] = ind2[j];
          ind2[j] = itmp;
          ++i;
        }
        --j;
      } else {
        /* if tt[j] > mid  -> tt[j] is on right side */
        if(p_ttqcmp3(ind0,ind1,ind2,j,imid) == 1) {
          --j;
        }
        ++i;
      }
    }

    /* if tt[i] > mid */
    if(p_ttqcmp3(ind0,ind1,ind2,i,imid) == 1) {
      --i;
    }
    vals[start] = vals[i];
    vals[i] = vmid;
    ind0[start] = ind0[i];
    ind1[start] = ind1[i];
    ind2[start] = ind2[i];
    ind0[i] = imid[0];
    ind1[i] = imid[1];
    ind2[i] = imid[2];

    if(i > start + 1) {
      p_tt_quicksort3(tt, cmplt, start, i);
    }
    ++i; /* skip the pivot element */
    if(end - i > 1) {
      p_tt_quicksort3(tt, cmplt, i, end);
    }
}


/**
* @brief Perform quicksort on a n-mode tensor between start and end.
*
* @param tt The tensor to sort.
* @param cmplt Mode permutation used for defining tie-breaking order.
* @param start The first nonzero to sort.
* @param end The last nonzero to sort.
*/
static void p_tt_quicksort(
  SparseTensor * const tt,
  IType const * const cmplt,
  IType const start,
  IType const end)
{
  FType vmid;
  IType imid[MAX_NUM_MODES];

  IType * ind;
  FType * const vals = tt->vals;
  IType const nmodes = tt->nmodes;

    size_t i = start+1;
    size_t j = end-1;
    size_t k = start + ((end - start) / 2);

    /* grab pivot */
    vmid = vals[k];
    vals[k] = vals[start];
    for(IType m=0; m < nmodes; ++m) {
      ind = tt->cidx[m];
      imid[m] = ind[k];
      ind[k] = ind[start];
    }

    while(i < j) {
      /* if tt[i] > mid  -> tt[i] is on wrong side */
      if(p_ttqcmp(tt,cmplt,i,imid) == 1) {
        /* if tt[j] <= mid  -> swap tt[i] and tt[j] */
        if(p_ttqcmp(tt,cmplt,j,imid) < 1) {
          p_ttswap(tt,i,j);
          ++i;
        }
        --j;
      } else {
        /* if tt[j] > mid  -> tt[j] is on right side */
        if(p_ttqcmp(tt,cmplt,j,imid) == 1) {
          --j;
        }
        ++i;
      }
    }

    /* if tt[i] > mid */
    if(p_ttqcmp(tt,cmplt,i,imid) == 1) {
      --i;
    }
    vals[start] = vals[i];
    vals[i] = vmid;
    for(IType m=0; m < nmodes; ++m) {
      ind = tt->cidx[m];
      ind[start] = ind[i];
      ind[i] = imid[m];
    }

    if(i > start + 1) {
      p_tt_quicksort(tt, cmplt, start, i);
    }
    ++i; /* skip the pivot element */
    if(end - i > 1) {
      p_tt_quicksort(tt, cmplt, i, end);
    }
}


static void p_tt_quicksort2(
  SparseTensor * const tt,
  IType const * const cmplt,
  IType const start,
  IType const end)
{
  FType vmid;
  IType imid[2];

  IType * const ind0 = tt->cidx[cmplt[0]];
  IType * const ind1 = tt->cidx[cmplt[1]];
  FType * const vals = tt->vals;

    size_t i = start+1;
    size_t j = end-1;
    size_t k = start + ((end - start) / 2);

    /* grab pivot */
    vmid = vals[k];
    vals[k] = vals[start];
    imid[0] = ind0[k];
    imid[1] = ind1[k];
    ind0[k] = ind0[start];
    ind1[k] = ind1[start];

    while(i < j) {
        /* if tt[i] > mid  -> tt[i] is on wrong side */
        if(p_ttqcmp2(ind0,ind1,i,imid) == 1) {
        /* if tt[j] <= mid  -> swap tt[i] and tt[j] */
        if(p_ttqcmp2(ind0,ind1,j,imid) < 1) {
            FType vtmp = vals[i];
            vals[i] = vals[j];
            vals[j] = vtmp;
            IType itmp = ind0[i];
            ind0[i] = ind0[j];
            ind0[j] = itmp;
            itmp = ind1[i];
            ind1[i] = ind1[j];
            ind1[j] = itmp;
            ++i;
        }
        --j;
        } else {
        /* if tt[j] > mid  -> tt[j] is on right side */
        if(p_ttqcmp2(ind0,ind1,j,imid) == 1) {
            --j;
        }
        ++i;
        }
    }

    /* if tt[i] > mid */
    if(p_ttqcmp2(ind0,ind1,i,imid) == 1) {
        --i;
    }
    vals[start] = vals[i];
    vals[i] = vmid;
    ind0[start] = ind0[i];
    ind1[start] = ind1[i];
    ind0[i] = imid[0];
    ind1[i] = imid[1];

    if(i > start + 1) {
        p_tt_quicksort2(tt, cmplt, start, i);
    }
    ++i; /* skip the pivot element */
    if(end - i > 1) {
        p_tt_quicksort2(tt, cmplt, i, end);
    }

}


static void p_counting_sort_hybrid(
    SparseTensor * const tt,
    IType * const cmplt)
{
  IType m = cmplt[0];
  IType nslices = tt->dims[m];

  IType * new_ind[MAX_NUM_MODES];

  for(IType i = 0; i < tt->nmodes; ++i) {
    if(i != m) {
      new_ind[i] = (IType*)malloc(tt->nnz * sizeof(**new_ind));
    }
  }
  FType * new_vals = (FType*)malloc(tt->nnz * sizeof(*new_vals));

  IType * histogram_array = (IType*)malloc(
      (nslices * omp_get_max_threads() + 1) * sizeof(*histogram_array));
  #pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();

    IType * histogram = histogram_array + (nslices * tid);
    memset(histogram, 0, nslices * sizeof(IType));

    IType j_per_thread = (tt->nnz + nthreads - 1)/nthreads;
    IType jbegin = SS_MIN(j_per_thread*tid, tt->nnz);
    IType jend = SS_MIN(jbegin + j_per_thread, tt->nnz);

    /* count */
    for(IType j = jbegin; j < jend; ++j) {
      IType idx = tt->cidx[m][j];
      ++histogram[idx];
    }

    #pragma omp barrier

    /* prefix sum */
    for(IType j = (tid*nslices) + 1; j < (tid+1) * nslices; ++j) {
      IType transpose_j = p_transpose_idx(j, nthreads, nslices);
      IType transpose_j_minus_1 = p_transpose_idx(j - 1, nthreads, nslices);

      histogram_array[transpose_j] += histogram_array[transpose_j_minus_1];
    }

    #pragma omp barrier
    #pragma omp master
    {
      for(int t = 1; t < nthreads; ++t) {
        IType j0 = (nslices*t) - 1, j1 = nslices * (t+1) - 1;
        IType transpose_j0 = p_transpose_idx(j0, nthreads, nslices);
        IType transpose_j1 = p_transpose_idx(j1, nthreads, nslices);

        histogram_array[transpose_j1] += histogram_array[transpose_j0];
      }
    }
    #pragma omp barrier

    if (tid > 0) {
      IType transpose_j0 = p_transpose_idx(nslices*tid - 1, nthreads, nslices);

      for(IType j = tid*nslices; j < (tid+1) * nslices - 1; ++j) {

        IType transpose_j = p_transpose_idx(j, nthreads, nslices);

        histogram_array[transpose_j] += histogram_array[transpose_j0];
      }
    }

    #pragma omp barrier


    /* now copy values into new structures (but not the mode we are sorting */
    for(IType j_off = 0; j_off < (jend-jbegin); ++j_off) {
      /* we are actually going backwards */
      IType const j = jend - j_off - 1;

      IType idx = tt->cidx[m][j];
      --histogram[idx];

      IType offset = histogram[idx];

      new_vals[offset] = tt->vals[j];
      for(IType mode=0; mode < tt->nmodes; ++mode) {
        if(mode != m) {
          new_ind[mode][offset] = tt->cidx[mode][j];
        }
      }
    }
  } /* omp parallel */
  for(IType i = 0; i < tt->nmodes; ++i) {
    if(i != m) {
      free(tt->cidx[i]);
      tt->cidx[i] = new_ind[i];
    }
  }
  free(tt->vals);
  tt->vals = new_vals;


  histogram_array[nslices] = tt->nnz;

  /* for 3/4D, we can use quicksort on only the leftover modes */
  if(tt->nmodes == 3) {
    #pragma omp parallel for schedule(dynamic)
    for(IType i = 0; i < nslices; ++i) {
      p_tt_quicksort2(tt, cmplt+1, histogram_array[i], histogram_array[i + 1]);
      for(IType j = histogram_array[i]; j < histogram_array[i + 1]; ++j) {
        tt->cidx[m][j] = i;
      }
    }

  } else if(tt->nmodes == 4) {

    #pragma omp parallel for schedule(dynamic)
    for(IType i = 0; i < nslices; ++i) {
      p_tt_quicksort3(tt, cmplt+1, histogram_array[i], histogram_array[i + 1]);
      for(IType j = histogram_array[i]; j < histogram_array[i + 1]; ++j) {
        tt->cidx[m][j] = i;
      }
    }
  } else {
    /* shift cmplt left one time, then do normal quicksort */
    IType saved = cmplt[0];
    memmove(cmplt, cmplt+1, (tt->nmodes - 1) * sizeof(*cmplt));
    cmplt[tt->nmodes-1] = saved;

    #pragma omp parallel for schedule(dynamic)
    for(IType i = 0; i < nslices; ++i) {
      p_tt_quicksort(tt, cmplt, histogram_array[i], histogram_array[i + 1]);
      for(IType j = histogram_array[i]; j < histogram_array[i + 1]; ++j) {
        tt->cidx[m][j] = i;
      }
    }

    /* undo cmplt changes */
    saved = cmplt[tt->nmodes-1];
    memmove(cmplt+1, cmplt, (tt->nmodes - 1) * sizeof(*cmplt));
    cmplt[0] = saved;
  }

  free(histogram_array);
}


void tt_sort(
  SparseTensor * const tt,
  IType const mode,
  IType * dim_perm)
{
  tt_sort_range(tt, mode, dim_perm, 0, tt->nnz);
}

void tt_sort_range(
  SparseTensor * const tt,
  IType const mode,
  IType * dim_perm,
  IType const start,
  IType const end)
{
  IType * cmplt;
  if(dim_perm == NULL) {
    cmplt = (IType*) malloc(tt->nmodes * sizeof(IType));
    cmplt[0] = mode;
    for(IType m=1; m < tt->nmodes; ++m) {
      cmplt[m] = (mode + m) % tt->nmodes;
    }
  } else {
    cmplt = dim_perm;
  }

  if(start == 0 && end == tt->nnz) {
      p_tt_quicksort(tt, cmplt, start, end);
      // Causes bugs in certain datasets
      // p_counting_sort_hybrid(tt, cmplt);
  /* sort a subtensor */
  } else {
    if(tt->nmodes == 3) {
      p_tt_quicksort3(tt, cmplt, start, end);
    } else {
      p_tt_quicksort(tt, cmplt, start, end);
    }
  }

  if(dim_perm == NULL) {
    free(cmplt);
  }
 
}