#ifndef UTIL_HPP_
#define UTIL_HPP_

#include "common.hpp"
#include "stream_matrix.hpp"

struct permutation_struct {
    IType * perms[MAX_NUM_MODES];
    IType * iperms[MAX_NUM_MODES];
}; 
typedef struct permutation_struct Permutation;

Permutation * perm_alloc(
    IType const * const dims, int const nmodes);

Matrix * init_mat(IType nrows, IType ncols);
Matrix * rand_mat(IType nrows, IType ncols);
Matrix * zero_mat(IType nrows, IType ncols);
FType rand_val();
void fill_rand(FType * vals, IType num_el);

void free_mat(Matrix * mat);

#endif // UTIL_HPP_
