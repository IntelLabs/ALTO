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

void free_mat(Matrix * mat);
void my_matmul(
  FType * const A,
  bool transA,
  FType * const B,
  bool transB,
  FType  * const C, 
  int m, int n, int k, int l, FType beta);

double mat_norm_diff(FType * matA, FType * matB, IType size);
double mat_norm(FType * mat, IType size);

void mat_form_gram(Matrix ** aTa, Matrix * out_mat, IType nmodes, IType mode);

void PrintMatrix(char *name, Matrix * M);

#endif // MATRIX_HPP_
