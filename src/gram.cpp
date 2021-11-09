#include "gram.hpp"

void destroy_grams(Matrix ** grams, KruskalModel* M)
{
  for(int i = 0; i < M->mode; i++) {
    free_mat(grams[i]);
  }
  free(grams);
}

void init_grams(Matrix *** grams, KruskalModel* M)
{
  IType rank = M->rank;
  Matrix ** _grams = (Matrix**) AlignedMalloc(sizeof(Matrix*) * M->mode);
  assert(_grams);
  for(int i = 0; i < M->mode; i++) {
    _grams[i] = init_mat(rank, rank);
    assert(_grams[i]);
  }

  for(int i = 0; i < M->mode; i++) {
    update_gram(_grams[i], M, i);
  }

  *grams = _grams;
}


void update_gram(Matrix * gram, KruskalModel* M, int mode)
{
  MKL_INT m = M->dims[mode];
  MKL_INT n = M->rank;
  MKL_INT lda = n;
  MKL_INT ldc = n;
  FType alpha = 1.0;
  FType beta = 0.0;

  CBLAS_ORDER layout = CblasRowMajor;
  CBLAS_UPLO uplo = CblasUpper;
  CBLAS_TRANSPOSE trans = CblasTrans;
  SYRK(layout, uplo, trans, n, m, alpha, M->U[mode], lda, beta, gram->vals, ldc);
}