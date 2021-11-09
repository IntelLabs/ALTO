#include "mttkrp.hpp"
#include <assert.h>
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
