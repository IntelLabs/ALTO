#ifndef STREAMING_CPD_HPP_
#define STREAMING_CPD_HPP_

#include "common.hpp"
#include "sptensor.hpp"
#include "kruskal_model.hpp"
#include "stream_matrix.hpp"
#include "util.hpp"
#include "alto.hpp"


// Streaming CPD kernels
void streaming_cpd(SparseTensor* X, KruskalModel* M, KruskalModel * prev_M, Matrix** grams, int max_iters, double epsilon, int streaming_mode, int iter);

template <typename LIT>
void streaming_cpd_alto(AltoTensor<LIT>* AT, KruskalModel* M, KruskalModel * prev_M, Matrix** grams, int max_iters, double epsilon, int streaming_mode, int iter);

// More explicit version for streaming cpd
static void _pseudo_inverse(Matrix * gram, KruskalModel* M, IType mode);


#endif // STREAMING_CPD_HPP_