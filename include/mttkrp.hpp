#ifndef MTTKRP_HPP_
#define MTTKRP_HPP_

// #include "common.hpp"
#include "sptensor.hpp"
#include "kruskal_model.hpp"
#include "omp.h"

void mttkrp_par(SparseTensor* X, KruskalModel* M, IType mode, omp_lock_t* writelocks);
void mttkrp(SparseTensor* X, KruskalModel* M, IType mode);

#endif