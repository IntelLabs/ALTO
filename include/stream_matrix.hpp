#ifndef STREAM_MATRIX_HPP_
#define STREAM_MATRIX_HPP_

#include "common.hpp"
#include "matrix.hpp"

class StreamMatrix
{
public:
  StreamMatrix(IType rank);
  ~StreamMatrix();

  void grow_zero(IType nrows);
  void grow_rand(IType nrows, unsigned int seed);

  inline Matrix * mat() { return _mat; };
  inline FType * vals() { return _mat->vals; };
  inline IType num_rows() { return _nrows; };
  inline IType num_cols() { return _ncols; };

private:
  IType _nrows;
  IType _ncols;
  IType _row_capacity;

  void grow(IType nrows);

  Matrix * _mat;
};

#endif