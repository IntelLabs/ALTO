#ifndef CPSTREAM_HPP_
#define CPSTREAM_HPP_

#include "common.hpp"
#include "sptensor.hpp"
#include "kruskal_model.hpp"
#include "stream_matrix.hpp"
#include "streaming_sptensor.hpp"
#include "util.hpp"
#include "alto.hpp"
#include "gram.hpp"
#include "mttkrp.hpp"

// Signatures
void cpstream(
    SparseTensor* X, 
    int rank, 
    int max_iters, 
    int streaming_mode, 
    FType epsilon, 
    IType seed, 
    bool use_alto);

void cpstream_iter(SparseTensor* X, KruskalModel* M, KruskalModel * prev_M, 
    Matrix** grams, int max_iters, double epsilon, 
    int streaming_mode, int iter, bool use_alto);

void spcpstream_iter(SparseTensor* X, KruskalModel* M, KruskalModel * prev_M, 
    Matrix** grams, int max_iters, double epsilon, 
    int streaming_mode, int iter, bool use_alto);

// More explicit version for streaming cpd
void pseudo_inverse_stream(Matrix ** grams, KruskalModel* M, IType mode, IType stream_mode);

// Specifically for cpstream
static double cpstream_fit(SparseTensor* X, KruskalModel* M, Matrix ** grams, FType* U_mttkrp);

#endif // CPSTREAM_HPP_
