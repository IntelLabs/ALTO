#include "util.hpp"

Permutation * perm_alloc(
    IType const * const dims, int const nmodes) {
    Permutation * perm = (Permutation *) malloc(sizeof(Permutation));

    for (int m = 0; m < MAX_NUM_MODES; ++m) {
        if (m < nmodes) {
            perm->perms[m] = (IType *) malloc(dims[m] * sizeof(IType));
            perm->iperms[m] = (IType *) malloc(dims[m] * sizeof(IType));
        } else {
            perm->perms[m] = NULL;
            perm->iperms[m] = NULL;
        }
    }
    return perm;
}


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

void free_mat(Matrix * mat) {
    if (mat == NULL) return;
    free(mat->vals);
    free(mat);
};

FType rand_val()
{
  FType v =  3.0 * ((FType) rand() / (FType) RAND_MAX);
  if(rand() % 2 == 0) {
    v *= -1;
  }
  return v;
}


void fill_rand(FType * vals, IType num_el) {
    for(IType i=0; i < num_el; ++i) {
        vals[i] = rand_val();
    } 
}

// This matmul function has to accomodate 
// the Kruskal model which keeps track of the factor matrix in double pointer format
// rather than using the Matrix struct
void my_matmul(
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
            out_mat->vals[i] *= aTa[mode]->vals[i];        
        }
    }
};
