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

#include <vector>

using namespace std;

// Structs
struct sparse_cp_grams {
    Matrix ** c_nz_prev;
    Matrix ** c_z_prev;

    Matrix ** c_nz;
    Matrix ** c_z;

    Matrix ** h_nz;
    Matrix ** h_z;

    Matrix ** c;
    Matrix ** h;

    Matrix ** c_prev;
}; typedef struct sparse_cp_grams SparseCPGrams;

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
    Matrix** grams, SparseCPGrams * scpgrams, int max_iters, double epsilon, 
    int streaming_mode, int iter, bool use_alto);

// More explicit version for streaming cpd
void pseudo_inverse_stream(Matrix ** grams, KruskalModel* M, IType mode, IType stream_mode);

// Specifically for cpstream
static double cpstream_fit(SparseTensor* X, KruskalModel* M, Matrix ** grams, FType* U_mttkrp);

SparseCPGrams * InitSparseCPGrams(IType nmodes, IType rank);
void DeleteSparseCPGrams(SparseCPGrams * grams, IType nmodes);

void nonzero_slices(
    SparseTensor * const tt, const IType mode,
    vector<size_t>& nz_rows,
    vector<size_t>& idx,
    vector<int>& ridx,
    vector<size_t>& buckets);


#endif // CPSTREAM_HPP_
