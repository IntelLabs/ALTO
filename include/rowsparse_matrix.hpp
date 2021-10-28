#ifndef ROWSPARSE_MATRIX_HPP_
#define ROWSPARSE_MATRIX_HPP_

#include "util.hpp"
#include "matrix.hpp"
#include <vector>

#ifdef MKL
#include <mkl.h>
#else
#include <cblas.h>
#include <lapacke.h>
typedef size_t MKL_INT;
#endif

/**
 * @brief Row-Sparse Matrix
 *
 * row-sparse matrix I x J where `nnzr` is the number of nonzero rows
 * represented by a `nnzr x J` matrix and a `nnzr x 1` vector of row indeces
 */
struct rowsparse_matrix_struct {
  IType I;
  IType J;
  IType nnzr;
  size_t * rowind;
  Matrix * mat;
};
typedef struct rowsparse_matrix_struct RowSparseMatrix;

/**********************************************************************
*                     Row-Spare Matrix Operations                    *
**********************************************************************/

/**
 * @brief Allocate a (uninitialized) Row-Sparse Matrix
 *
 * @param nrows     Number of rows
 * @param ncols     Number of columns
 * @param nnzr      Number of nonzero rows (only these are saved)
 *
 * @return The newly allocated Row SParse Matrix
 */
RowSparseMatrix * rspmat_init(IType nrows, IType ncols, IType nnzr);

/**
* @brief Free a row-sparse matrix allocated with rspmat_alloc().
*
* This also frees the given matrix pointer!
*
* @param mat The row-sparse matrix to be freed.
 */
void rspmat_free(RowSparseMatrix* mat);

/**
 * @brief converts a full matrix to rsp_matrix format
 */
RowSparseMatrix * convert_to_rspmat(
    Matrix * fm, size_t nnzr, size_t * rowind);

/**
 * @brief computes aTa for row sparse matrices
 */
void rsp_mataTb(RowSparseMatrix* A, RowSparseMatrix* B, Matrix * dest);

/**
 * @brief sum (A+=B) of two row sparse matrices
 */
void rsp_mat_add(RowSparseMatrix* A, RowSparseMatrix* B);

/**
 * @brief product (A*=B) of two row sparse matrices
 */
RowSparseMatrix* rsp_mat_mul(RowSparseMatrix* A, Matrix * B);

/**
 *  @brief ATA operation based on subset of rows
 */
void mataTa_idx_based(
    Matrix * A,
    std::vector<size_t>& idx,
    Matrix * dest);

void PrintRowSparseMatrix(char *name, RowSparseMatrix * mat);

#endif