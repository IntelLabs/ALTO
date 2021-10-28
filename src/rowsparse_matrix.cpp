#include "rowsparse_matrix.hpp"

RowSparseMatrix * rspmat_init(IType nrows, IType ncols, IType nnzr)
{
    RowSparseMatrix * rsp_mat = (RowSparseMatrix*) malloc(sizeof(RowSparseMatrix));
    rsp_mat->I = nrows;
    rsp_mat->J = ncols;
    rsp_mat->nnzr = nnzr;
    rsp_mat->mat = init_mat(nnzr, ncols);
    rsp_mat->rowind = (size_t *)malloc(nnzr * sizeof(size_t));
    return rsp_mat;
}


void rspmat_free(RowSparseMatrix* rsp_mat)
{
    if (rsp_mat == NULL) return;
    if (rsp_mat->rowind == NULL) return;

    free_mat(rsp_mat->mat);
    free(rsp_mat->rowind);
    free(rsp_mat);
    return;
}

RowSparseMatrix * convert_to_rspmat(Matrix * fm, size_t nnzr, size_t * rowind)
{
    RowSparseMatrix * rsp_mat = rspmat_init(fm->I, fm->J, nnzr);
    IType I = fm->I;
    IType J = fm->J;

    #pragma omp parallel for schedule(static, 1) if(nnzr > 50)
    for (IType i = 0; i < nnzr; i++) {
        memcpy(
            &rsp_mat->mat->vals[i * J],
            &fm->vals[rowind[i] * J],
            sizeof(*fm->vals) * J);
    }
    // Returned RSP matrix has same row indices as rowind
    memcpy(rsp_mat->rowind, rowind, sizeof(size_t) * nnzr);
    return rsp_mat;
}

void rsp_mataTb(RowSparseMatrix* A, RowSparseMatrix* B, Matrix * dest)
{
    assert(A->I == B->I);
    assert(A->J == B->J);
    // aTb op is used when a and b have identical row indices
    // The result gram matrix therefore doesn't have to consider the 
    // row indices of A and B
    assert(A->nnzr == B->nnzr);
    matmul(A->mat, true, B->mat, false, dest, 0.0);
    return;
}

void rsp_mat_add(RowSparseMatrix* A, RowSparseMatrix* B)
{
    assert(A->I == B->I);
    assert(A->J == B->J);
    assert(A->nnzr == B->nnzr);
    IType I = A->I;
    IType J = A->J;
    IType nnzr = A->nnzr;

    // Need to compare the row indices as well
    FType * A_vals = A->mat->vals;
    FType const * const B_vals = B->mat->vals;

    #pragma omp parallel for schedule(static, 1) if(nnzr > 50)
    for (IType i = 0; i < nnzr; i++) {
        for (IType j = 0; j < J; j++) {
            A_vals[i * J + j] += B_vals[i * J + j];
        }
    }
    return;
}

RowSparseMatrix* rsp_mat_mul(RowSparseMatrix * A, Matrix * B)
{
    IType nrows = A->I;
    IType ncols = A->J;
    IType nnzr = A->nnzr;

    RowSparseMatrix * rsp_mat = rspmat_init(nrows, ncols, nnzr);
    memcpy(rsp_mat->rowind, A->rowind, sizeof(size_t) * nnzr);

    // Use conventional matmul for A and B and write to C
    matmul(A->mat, false, B, false, rsp_mat->mat, 0.0);

    // A and C have same row indices
    return rsp_mat;
}

void PrintRowSparseMatrix(char *name, RowSparseMatrix * rsp_mat)
{
    size_t m = rsp_mat->I;
    size_t n = rsp_mat->J;
  
	fprintf(stderr,"Row sparse matrix: %s:\n", name);
    // Print row_ind
    PrintIntMatrix("Non-zero row indices", rsp_mat->rowind, 1, rsp_mat->nnzr);

    for (size_t i = 0; i < m; i++) {
        fprintf(stderr, "[");
        for (size_t j = 0; j < n; j++) {
        	fprintf(stderr,"%e", rsp_mat->mat->vals[i * n + j]);
            if (j != n-1) fprintf(stderr, ", ");
            if (j == n-1) fprintf(stderr, "],");
        }

        fprintf(stderr,"\n");
    }
    fprintf(stderr,"\n");    
}

// TODO: Replace with BLAS kernels
void mataTa_idx_based(
    Matrix * A,
    std::vector<size_t>& idx,
    Matrix * dest)
{
    assert(dest->I == dest->J);
    size_t _size = dest->I;
    #pragma omp parallel for schedule(static) 
    for (IType i = 0; i < _size; i ++) {
        for (IType j = 0; j < _size; j ++) {
            FType tmp = 0.0;
            for (IType k = 0; k < idx.size(); k++) {
                IType row_idx = idx.at(k);
                tmp += A->vals[row_idx * _size + j] * A->vals[row_idx * _size + i];
            }
            dest->vals[i * _size + j] = tmp;
        }
    }
}