#include "util.hpp"

Permutation * perm_alloc(
    IType const * const dims, int const nmodes) {
    Permutation * perm = (Permutation *) malloc(sizeof(Permutation));

    for (int m = 0; m < MAX_NUM_MODES; ++m) {
        if (m < nmodes) {
            perm->perms[m] = (IType *) malloc(dims[m] * sizeof(IType));
            perm->iperms[m] = (IType *) malloc(dims[m] * sizeof(IType));
        } else {
            perm->perms[m] = NULL;
            perm->iperms[m] = NULL;
        }
    }
    return perm;
}

FType rand_val()
{
  FType v =  3.0 * ((FType) rand() / (FType) RAND_MAX);
  if(rand() % 2 == 0) {
    v *= -1;
  }
  return v;
}

void fill_rand(FType * vals, IType num_el) {
    srand(44);
    for(IType i=0; i < num_el; ++i) {
        vals[i] = rand_val();
    } 
}

IType argmin_elem(
  IType const * const arr,
  IType const N)
{
  IType mkr = 0;
  for(IType i=1; i < N; ++i) {
    if(arr[i] < arr[mkr]) {
      mkr = i;
    }
  }
  return mkr;
}
