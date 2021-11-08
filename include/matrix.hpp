#ifndef MATRIX_HPP_
#define MATRIX_HPP_

#include "util.hpp"

#ifdef MKL
#include <mkl.h>
#else
#include <cblas.h>
#include <lapacke.h>
typedef size_t MKL_INT;
#endif

struct matrix_struct {
  IType I;
  IType J;
  FType *vals;
  int rowmajor;
};
typedef struct matrix_struct Matrix;

Matrix * init_mat(IType nrows, IType ncols);
Matrix * rand_mat(IType nrows, IType ncols);
Matrix * zero_mat(IType nrows, IType ncols);
Matrix * ones_mat(IType nrows, IType ncols);
Matrix * mat_fillptr(FType * vals, IType nrows, IType ncols);
void mat_hydrate(Matrix * mat, FType * vals, IType nrows, IType ncols);

void free_mat(Matrix * mat);

void matmul(
  FType * const A,
  bool transA,
  FType * const B,
  bool transB,
  FType  * const C, 
  int m, int n, int k, int l, FType beta);

// TODO: Figure out better way
void matmul(
  Matrix const * const A,
  bool transA,
  Matrix const * const B,
  bool transB,
  Matrix  * const C, FType beta);

double mat_norm_diff(FType * matA, FType * matB, IType size);
double mat_norm(FType * mat, IType size);

void mat_form_gram(Matrix ** aTa, Matrix * out_mat, IType nmodes, IType mode);

void PrintMatrix(char *name, Matrix * M);

void PrintFPMatrix(char *name, FType * a, size_t m, size_t n);

void PrintIntMatrix(char *name, size_t * a, size_t m, size_t n);

void copy_upper_tri(Matrix * M);

void pseudo_inverse(Matrix * A, Matrix * B);

void mat_aTa(Matrix const * const A, Matrix * const ret);
 
FType mat_trace(Matrix * mat);


#endif // MATRIX_HPP_
