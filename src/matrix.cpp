#include "matrix.hpp"

Matrix * init_mat(IType nrows, IType ncols) {
    Matrix * mat = (Matrix *) AlignedMalloc(sizeof(Matrix));
    mat->I = nrows;
    mat->J = ncols;
    mat->vals = (FType *) AlignedMalloc(sizeof(FType*) * nrows * ncols);
    mat->rowmajor = 1;

    return mat;
};

Matrix * rand_mat(IType nrows, IType ncols) {
    Matrix * mat = init_mat(nrows, ncols);
    for (int i = 0; i < mat->I; ++i) {
        for (int j = 0; j < mat->J; ++j) {
            mat->vals[j + (i * ncols)] = ((FType) rand() / (FType) RAND_MAX);
        }
    }
    return mat;
};

Matrix * zero_mat(IType nrows, IType ncols) {
    Matrix * mat = init_mat(nrows, ncols);
    for (int i = 0; i < mat->I; ++i) {
        for (int j = 0; j < mat->J; ++j) {
            mat->vals[j + (i * ncols)] = 0.0;
        }
    }
    return mat;
};

Matrix * ones_mat(IType nrows, IType ncols) {
    Matrix * mat = init_mat(nrows, ncols);
    for (int i = 0; i < mat->I; ++i) {
        for (int j = 0; j < mat->J; ++j) {
            mat->vals[j + (i * ncols)] = 1.0;
        }
    }
    return mat;
};

void grow_mat(IType nrows, IType ncols) {
    // Add implementation
}

void free_mat(Matrix * mat) {
    if (mat == NULL) return;
    free(mat->vals);
    free(mat);
};

// This matmul function has to accomodate 
// the Kruskal model which keeps track of the factor matrix in double pointer format
// rather than using the Matrix struct
void matmul(
  FType * A,
  bool transA,
  FType * B,
  bool transB,
  FType * C,
  int m, int n, int k, int l, FType beta) {

    int const M = transA ? n : m;
    int const N = transB ? k : l;
    int const K = transA ? m : n;
    int const LDA = n;
    int const LDB = l;
    int const LDC = N;

    assert(K == (int)(transB ? l : k));
    /* TODO precision! (double vs not) */
    cblas_dgemm(
        CblasRowMajor,
        transA ? CblasTrans : CblasNoTrans,
        transB ? CblasTrans : CblasNoTrans,
        M, N, K,
        1.,
        A, LDA,
        B, LDB,
        beta,
        C, LDC);
}

void matmul(
  Matrix const * const A,
  bool transA,
  Matrix const * const B,
  bool transB,
  Matrix  * const C, FType beta = 0.0) {
    int const M = transA ? A->J : A->I;
    int const N = transB ? B->I : B->J;
    int const K = transA ? A->I : A->J;
    int const LDA = A->J;
    int const LDB = B->J;
    int const LDC = N;

    assert(K == (int)(transB ? B->J : B->I));
    assert((int)(C->I * C->J) <= M*N);

    /* TODO precision! (double vs not) */
    cblas_dgemm(
        CblasRowMajor,
        transA ? CblasTrans : CblasNoTrans,
        transB ? CblasTrans : CblasNoTrans,
        M, N, K,
        1.,
        A->vals, LDA,
        B->vals, LDB,
        beta,
        C->vals, LDC);
}

double mat_norm_diff(FType * matA, FType * matB, IType size) {
    double norm = 0.0;
    for (int i = 0; i < size; ++i) {
        FType diff = matA[i] - matB[i];
        norm += diff * diff;
    }
    return sqrt(norm);
}

double mat_norm(FType * mat, IType size) {
    double norm = 0.0;
    for (int i = 0; i < size; ++i) {
        norm += mat[i] * mat[i];
    }
    return sqrt(norm);
}

/* Computes the elementwise product for all other aTa matrices besidse the mode */
void mat_form_gram(Matrix ** aTa, Matrix * out_mat, IType nmodes, IType mode) {
    // Init gram matrix
    for (IType i = 0; i < out_mat->I * out_mat->J; ++i) {
        out_mat->vals[i] = 1.0;
    }
    
    for (IType m = 0; m < nmodes; ++m) {
        if (m == mode) continue;
        for (IType i = 0; i < out_mat->I * out_mat->J; ++i) {
            out_mat->vals[i] *= aTa[m]->vals[i];        
        }
    }
};

void PrintMatrix(char *name, Matrix * M)
{
  size_t m = M->I;
  size_t n = M->J;
  
	fprintf(stderr,"%s:\n", name);
    for (size_t i = 0; i < m; i++) {
        fprintf(stderr, "[");
        for (size_t j = 0; j < n; j++) {
        	fprintf(stderr,"%e", M->vals[i * n + j]);
            if (j != n-1) fprintf(stderr, ", ");
            if (j == n-1) fprintf(stderr, "],");
        }

        fprintf(stderr,"\n");
    }
    fprintf(stderr,"\n");
}

void PrintFPMatrix(char *name, FType * a, size_t m, size_t n)
{
	fprintf(stderr,"%s:\n", name);
    for (size_t i = 0; i < m; i++) {
        fprintf(stderr, "[");
        for (size_t j = 0; j < n; j++) {
        	fprintf(stderr,"%e", a[i * n + j]);
          if (j != n-1) fprintf(stderr, ", ");
          if (j == n-1) fprintf(stderr, "],");
        }
        fprintf(stderr,"\n");
    }
    fprintf(stderr,"\n");
}


void PrintIntMatrix(char *name, size_t * a, size_t m, size_t n)
{
	fprintf(stderr,"%s:\n", name);
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
        	fprintf(stderr,"%zu ", a[i * n + j]);
        }
        fprintf(stderr,"\n");
    }
    fprintf(stderr,"\n");
}

void copy_upper_tri(Matrix * M) {
  IType const I = M->I;
  IType const J = M->J;
  FType * const vals = M->vals;

  #pragma omp parallel for schedule(static, 1) if(I > 50)
  for(IType i=1; i < I; ++i) {
    for(IType j=0; j < i; ++j) {  
      vals[j + (i*J)] = vals[i + (j*J)];
    }
  }  
};

/**
 * @brief
 * Solves AX=B using Cholesky factorizatin
 * Applies frobenius regularization and assumes cholesky factorization
 * always works, Solution is updated in B
 */
void pseudo_inverse(Matrix * A, Matrix * B)
{
    // Cholesky factorization requires a square matrix
    assert(A->I == A->J);
    // Housekeeping
    assert(A->I == B->J);

    // Set up variables for POTRF call
    char uplo = 'L';
    lapack_int I = (lapack_int)A->I;
    lapack_int J = (lapack_int)B->I;
    lapack_int info;
    lapack_int s_info;

    // Apply frobenius regularization (1e-12)
    // This helps stablity 
    for (int i = 0; i < I; ++i) {
        A->vals[i * I + i] += 1e-12;
    }

    POTRF(&uplo, &I, A->vals, &I, &info);

    if (info != 0) {
        // Loud failure message
        fprintf(stderr, "ALTO: Cholesky factorization failed. No fallback implemented!");
    }
    else {
        POTRS(&uplo, &I, &J, A->vals, &I,
          B->vals, &I, &s_info);    
    }
}

void mat_aTa(
  Matrix const * const A,
  Matrix * const ret)
{
    MKL_INT m = A->I;
    MKL_INT n = A->J;
    MKL_INT lda = n;
    MKL_INT ldc = n;
    FType alpha = 1.0;
    FType beta = 0.0;
    
    CBLAS_ORDER layout = CblasRowMajor;
    CBLAS_UPLO uplo = CblasUpper;
    CBLAS_TRANSPOSE trans = CblasTrans;
    SYRK(layout, uplo, trans, n, m, alpha, A->vals, lda, beta, ret->vals, ldc);
}

FType mat_trace(Matrix * mat)
{
    assert(mat->I == mat->J);
    FType tr = 0.0;
    for (int i = 0; i < mat->I * mat->J; ++i) 
    {
        tr += mat->vals[i * mat->I + i];
    }
    return tr;
}