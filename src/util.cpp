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
