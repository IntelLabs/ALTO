#ifndef UTIL_HPP_
#define UTIL_HPP_

#include "common.hpp"

struct permutation_struct {
    IType * perms[MAX_NUM_MODES];
    IType * iperms[MAX_NUM_MODES];
}; 
typedef struct permutation_struct Permutation;

Permutation * perm_alloc(
    IType const * const dims, int const nmodes);

#endif // UTIL_HPP_
