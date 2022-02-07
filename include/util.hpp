#ifndef UTIL_HPP_
#define UTIL_HPP_

#include "common.hpp"
#include "assert.h"
#include <math.h>

struct permutation_struct {
    IType * perms[MAX_NUM_MODES];
    IType * iperms[MAX_NUM_MODES];
}; 
typedef struct permutation_struct Permutation;

Permutation * perm_alloc(
    IType const * const dims, int const nmodes);

FType rand_val();
void fill_rand(FType * vals, IType num_el, unsigned int seed);

IType argmin_elem(
  IType const * const arr,
  IType const N);

#endif // UTIL_HPP_