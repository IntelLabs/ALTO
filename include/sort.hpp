#ifndef SPLATT_SORT_H
#define SPLATT_SORT_H

#include "common.hpp"
#include "sptensor.hpp"

void tt_sort(
  SparseTensor * const tt,
  IType const mode,
  IType * dim_perm);


void tt_sort_range(
  SparseTensor * const tt,
  IType const mode,
  IType * dim_perm,
  IType const start,
  IType const end);

#endif
