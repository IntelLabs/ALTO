#ifndef ROWSPARSE_MTTKRP_HPP_
#define ROWSPARSE_MTTKRP_HPP_


#include "rowsparse_matrix.hpp"
#include "sptensor.hpp"

#include <vector>

using namespace std;

void idxsort_hist(
  const SparseTensor * const tt, 
  const IType mode, 
  vector<size_t>& idx, 
  vector<size_t>& buckets);

RowSparseMatrix * rowsparse_mttkrp(
    SparseTensor * X, 
    RowSparseMatrix ** rsp_mats, 
    IType mode, 
    IType stream_mode,
    std::vector<size_t>& idx, 
    std::vector<std::vector<int>>& ridx, 
    std::vector<size_t>& buckets);

#endif