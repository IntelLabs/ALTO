#include "streaming_sptensor.hpp"
#include "algorithm"
#include "assert.h"

// Debugging purposes
void print_sptensor(SparseTensor *sp, int nnz) {
    printf("Sparse tensor has %d nnzs\n", sp->nnz);
    
    for (int n = 0; n < sp->nmodes; ++n) {
        printf("Dim #%d: %d\n", n, sp->dims[n]);
    }
    
    for (int n = 0; n < nnz; ++n) {
        // print dims
        printf("(");
        for (int i = 0; i < sp->nmodes; ++i) {
            printf("%d ", sp->cidx[i][n]);
        }
        printf(")");
        
        printf(" : %f\n", sp->vals[n]);
    } 
}

/**
 * Sorts sparse tensor by stream mode
 * Preprocessing for streaming tensor decomposition
 */
void tensor_sort(SparseTensor * sp, int mode) {
    int * perm = (int*)AlignedMalloc(sizeof(int) * sp->nnz);
    // Init permutation vector
    for (int i = 0; i < sp->nnz; ++i) {
        perm[i] = i;
    }

    std::sort(perm, perm+sp->nnz, [&](const int& a, const int& b) {
        return (sp->cidx[mode][a] < sp->cidx[mode][b]);
    });

    // for (int ii = 0; ii < sp->nnz; ++ii) {
    //     printf("%d\n", perm[ii]);
    // }

    // sort sp based on permutation
    for (int i = 0; i < sp->nnz; ++i) {
        int swp_idx = perm[i];
        while (swp_idx < i) {
            swp_idx = perm[swp_idx];
        };
        int tmp;
        float tmp_vals;
        for (int m = 0; m < sp->nmodes; ++m) {
            tmp = sp->cidx[m][swp_idx];
            sp->cidx[m][swp_idx] = sp->cidx[m][i];
            sp->cidx[m][i] = tmp;
        };
        // swap vals
        tmp_vals = sp->vals[swp_idx];
        sp->vals[swp_idx] = sp->vals[i];
        sp->vals[i] = tmp_vals;
    };

}

StreamingSparseTensor::StreamingSparseTensor(
    SparseTensor * sp,
    IType stream_mode
) : _stream_mode(stream_mode),
    _batch_num(0),
    _nnz_ptr(0)
{
    _tensor = sp; // Load tensor

    // First sort tensor based on streaming mode
    tensor_sort(_tensor, _stream_mode);

};

SparseTensor * StreamingSparseTensor::next_batch() {
    
    IType const * const streaming_cidx = _tensor->cidx[_stream_mode];
    // If we're already at the end
    if (_nnz_ptr == _tensor->nnz) {
        return NULL;
    }

    // Find starting nnz
    IType start_nnz = _nnz_ptr;
    while ((start_nnz < _tensor->nnz) && (streaming_cidx[start_nnz] < _batch_num)) {
        ++start_nnz;
    }

    // Find ending nnz
    IType end_nnz = start_nnz;
    while ((end_nnz < _tensor->nnz) && (streaming_cidx[end_nnz] < _batch_num + 1)) {
        ++end_nnz;
    }

    IType nnz = end_nnz - start_nnz;

    // Make sure we don't have empty batches
    assert(nnz > 0);

    SparseTensor * t_batch = AllocSparseTensor(nnz, _tensor->nmodes - 1); // Since we're dismissing the streaming mode
    memcpy(t_batch->vals, &(_tensor->vals[start_nnz]), nnz * sizeof(*(t_batch->vals)));
    
    // Need to figure out how to modify the dims and cidx for the batch tensors
    // SPLATT recomputes it in a unique way - should we follow?
    int new_mode_idx = 0;

    for (int m = 0; m < _tensor->nmodes; ++m) {
        if (m == _stream_mode) continue;
        else {
            t_batch->dims[new_mode_idx] = _tensor->dims[m];
            memcpy(t_batch->cidx[new_mode_idx], &(_tensor->cidx[m][start_nnz]), nnz * sizeof(*(_tensor->cidx[m])));
            ++new_mode_idx;
        }
    }

    _nnz_ptr = end_nnz;
    ++_batch_num;

    return t_batch;
}

bool StreamingSparseTensor::last_batch()
{
    return _nnz_ptr == _tensor->nnz;
};