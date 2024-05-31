#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <omp.h>

#include "apr.hpp"
#include "alto.hpp"

double cpd_fit_(SparseTensor* X, KruskalModel* M, FType** grams, FType* U_mttkrp)
{
  // Calculate inner product between X and M
  // This can be done via sum(sum(P.U{dimorder(end)} .* U_mttkrp) .* lambda');
  IType rank = M->rank;
  IType nmodes = X->nmodes;
  IType* dims = X->dims;

  FType* accum = (FType*) AlignedMalloc(sizeof(FType) * rank);
  assert(accum);
  memset(accum, 0, sizeof(FType) * rank);
  #pragma omp parallel for schedule(static)
  for(IType j = 0; j < rank; j++) {
    for(IType i = 0; i < dims[nmodes - 1]; i++) {
      accum[j] += M->U[nmodes - 1][i * rank + j] * U_mttkrp[i * rank + j];
    }
  }

  FType inner_prod = 0.0;
  for(IType i = 0; i < rank; i++) {
    inner_prod += accum[i] * M->lambda[i];
  }

  // Calculate norm(X)^2
  IType nnz = X->nnz;
  FType* vals = X->vals;
  FType normX = 0.0;
  #pragma omp parallel for reduction(+:normX) schedule(static)
  for(IType i = 0; i < nnz; i++) {
    normX += vals[i] * vals[i];
  }

  // Calculate norm of factor matrices
  // This can be done via taking the hadamard product between all the gram
  // matrices, and then summing up all the elements and taking the square root
  // of the absolute value
  FType* tmp_gram = (FType*) AlignedMalloc(sizeof(FType) * rank * rank);
  assert(tmp_gram);
  #pragma omp parallel for schedule(dynamic)
  for(IType i = 0; i < rank; i++) {
    for(IType j = 0; j < i + 1; j++) {
      tmp_gram[i * rank + j] = M->lambda[i] * M->lambda[j];
    }
  }

  // calculate the hadamard product between all the Gram matrices
  for(IType i = 0; i < nmodes; i++) {
    #pragma omp parallel for schedule(dynamic)
    for(IType j = 0; j < rank; j++) {
      for(IType k = 0; k < j + 1; k++) {
        tmp_gram[j * rank + k] *= grams[i][j * rank + k];
      }
    }
  }

  FType normU = 0.0;
  for(IType i = 0; i < rank; i++) {
    for(IType j = 0; j < i; j++) {
      normU += tmp_gram[i * rank + j] * 2;
    }
    normU += tmp_gram[i * rank + i];
  }
  normU = fabs(normU);

  // Calculate residual using the above
  FType norm_residual = normX + normU - 2 * inner_prod;
  if (norm_residual > 0.0) {
    norm_residual = sqrt(norm_residual);
  }
  FType ret = 1 - (norm_residual / sqrt(normX));

  // free memory
  free(accum);
  free(tmp_gram);

  return ret;
}


/*
static void update_gram(FType* gram, KruskalModel* M, int mode)
{
  MKL_INT m = M->dims[mode];
  MKL_INT n = M->rank;
  MKL_INT lda = n;
  MKL_INT ldc = n;
  FType alpha = 1.0;
  FType beta = 0.0;

  CBLAS_ORDER layout = CblasRowMajor;
  CBLAS_UPLO uplo = CblasLower;
  CBLAS_TRANSPOSE trans = CblasTrans;
  SYRK(layout, uplo, trans, n, m, alpha, M->U[mode], lda, beta, gram, ldc);
}
 */

// Calculate the Log likelihood for the final decomposition
FType tt_logLikelihood(SparseTensor* X, KruskalModel* M, FType eps_div_zero)
{

	int nmodes = X->nmodes;
	IType rank = M->rank;
	FType* lambda = M->lambda;
	FType** U = M->U;

	// Absorb the lambda into the first mode
	RedistributeLambda (M, 0);

	// Initialize A
	FType* A = (FType*) AlignedMalloc(X->nnz * rank * sizeof(FType));
    #pragma omp parallel for simd schedule(static)
	for(IType i = 0; i < X->nnz * rank; i++) {
		A[i] = 1.0;
	}

	for(int n = 0; n < nmodes; n++) {
		FType* U_n = U[n];
        #pragma omp parallel for schedule(static)
		for(IType i = 0; i < X->nnz; i++) {
			IType row_index = X->cidx[n][i];
            #pragma omp simd
			for(IType r = 0 ; r < rank; r++) {
				A[i * rank + r] = A[i * rank + r] * U_n[row_index* rank + r];
			}
		}
	}

	FType logSum = 0.0;
    #pragma omp parallel for schedule(static) reduction(+:logSum)
	for(IType i = 0; i < X->nnz; i++) {
		FType tmp = 0.0;
        #pragma omp simd
		for(IType r = 0; r < rank; r++) {
			tmp += A[i * rank + r];
		}
        if (tmp < eps_div_zero)
            tmp = eps_div_zero;
		logSum += X->vals[i] * log(tmp);
	}

	FType factorSum = 0.0;
    #pragma omp parallel for simd schedule(static) reduction(+:factorSum)
	for(IType i = 0; i < X->dims[0] * rank; i++) {
		factorSum += U[0][i];
	}


	for(IType r = 0; r < rank; r++) {
		FType temp = 0.0;
        #pragma omp parallel for simd schedule(static) reduction(+:temp)
		for(IType i = 0; i < X->dims[0]; i++) {
			temp += fabs (U[0][i * rank + r]);
		}
        #pragma omp parallel for simd schedule(static)
		for(IType i = 0; i < X->dims[0]; i++) {
			U[0][i * rank + r] = U[0][i * rank + r] / temp;
		}
		lambda[r] = lambda[r] * temp;
	}
    AlignedFree(A);

	return (logSum - factorSum);
}


void CalculatePhi(FType* Phi, FType* Pi, SparseTensor* X, KruskalModel* M, int target_mode, FType eps_div_zero)
{
	IType nnz = X->nnz;
	IType rank = M->rank;
	IType** cidx = X->cidx;
	FType* vals = X->vals;
	FType** U = M->U;
	FType* B = U[target_mode];
	IType dim = X->dims[target_mode];

	// Initialize Phi to 0
	for(IType i = 0; i < dim * rank; i++) {
		Phi[i] = 0.0;
	}

	for(IType i = 0; i < nnz; i++) {
		IType row_index = cidx[target_mode][i];
		// dot-product between B and Pi, then compare to eps_div_zero
		FType v = 0.0;
		for(IType r = 0; r < rank; r++) {
			v += B[row_index * rank + r] * Pi[i * rank + r];
		}
		// divide nnz by this value
		v = vals[i] / fmax(v, eps_div_zero);

		// scale KR (i.e., Pi) by this value (i.e., v), and update Phi
		for(IType r = 0; r < rank; r++) {
			Phi[row_index * rank + r] += v * Pi[i * rank + r];
		}
	}
}


void CalculatePi(FType* Pi, SparseTensor* X, KruskalModel* M, int target_mode)
{
	IType nnz = X->nnz;
	IType rank = M->rank;
	int nmodes = X->nmodes;
	IType** cidx = X->cidx;
	FType** U = M->U;

	for(IType i = 0; i < nnz; i++) {
		// Initialize Pi to 1.0 for Hadamard product
		for(IType r = 0; r < rank; r++) {
			Pi[i * rank + r] = 1.0;
		}

		// Calculate Pi for each non-zero element
		for(int n = 0; n < nmodes; n++) {
			if(n != target_mode) {
				FType* A = U[n];
				IType row_index = cidx[n][i];
				for(IType r = 0; r < rank; r++) {
					Pi[i * rank + r] *= A[row_index * rank + r];
				}
			}
		}
	}

}


void cp_apr_mu_alto(SparseTensor* X, KruskalModel* M, int max_iters, int max_inner_iters, FType epsilon, FType eps_div_zero, FType kappa_tol, FType kappa)
{
	// timers
	double wtime_s, wtime_t;
	double wtime_pre = 0.0;
	double wtime_apr = 0.0;
	double wtime_apr_pi = 0.0;
	double wtime_apr_phi = 0.0;
	double wtime_post = 0.0;

	fprintf(stdout, "Running ALTO CP-APR with: \n");
	// max outer iterations
	fprintf(stdout, "\tmax iters:       %d\n", max_iters);
	// max inner iterations
	fprintf(stdout, "\tmax inner iters: %d\n", max_inner_iters);
	// tolerance on the overall KKT violation
	fprintf(stdout, "\ttolerance:       %.2e\n", epsilon);
	// safeguard against divide by zero
	fprintf(stdout, "\teps_div_zero:    %.2e\n", eps_div_zero);
	// tolerance on complementary slackness
	fprintf(stdout, "\tkapp_tol:        %.2e\n", kappa_tol);
	// offset to fix complementary slackness
	fprintf(stdout, "\tkapp:            %.2e\n", kappa);


	wtime_s = omp_get_wtime();
	// Tensor and factor matrices information
	int nmodes = X->nmodes;
	IType nnz = X->nnz;
	IType* dims = X->dims;
	IType rank = M->rank;
	FType** U = M->U;
	FType* lambda = M->lambda;

	// Intermediate matrices
	// Phi matrix is an intermediate matrix equal in size to the factor matrix
	FType** Phi = (FType**) AlignedMalloc(nmodes * sizeof(FType*));
	assert(Phi);
	for(int n = 0; n < nmodes; n++) {
		Phi[n] = (FType*) AlignedMalloc(dims[n] * rank * sizeof(FType));
		assert(Phi[n]);
	}
	// Pi matrix is the Khatri-Rap product between every OTHER factor matrix
	// Since it's extremely large, we do not compute it expliclity.
	// Instead, we calculate the required KR product for each non-zero element.
	// Thus, this Pi matrix is a nnz x R matrix
	FType* Pi = (FType*) AlignedMalloc(nnz * rank * sizeof(FType));
	assert(Pi);


	// Used for keeping track of convergence information
	FType* kktModeViolations = (FType*) AlignedMalloc(nmodes * sizeof(FType));
	assert(kktModeViolations);
	for(int n = 0; n < nmodes; n++) {
		kktModeViolations[n] = 0.0;
	}
	int* nViolations = (int*) AlignedMalloc(max_iters * sizeof(int));
	assert(nViolations);
	for(int i = 0; i < max_iters; i++) {
		nViolations[i] = 0;
	}
	FType* kktViolations = (FType*) AlignedMalloc(max_iters * sizeof(FType));
	assert(kktViolations);
	for(int i = 0; i < max_iters; i++) {
		kktViolations[i] = 0.0;
	}
	int* nInnerIters = (int*) AlignedMalloc(max_iters * sizeof(int));
	assert(nInnerIters);
	for(int i = 0; i < max_iters; i++) {
		nInnerIters[i] = 0;
	}
	wtime_pre += omp_get_wtime() - wtime_s;


	wtime_s = omp_get_wtime();
	fprintf(stdout, "CP_APR MU: \n");
	KruskalModelNormalize(M);
	// Iterate until covergence or max_iters reached
	int iter;
	for(iter = 0; iter < max_iters; iter++) {
		bool converged = true;
		for(int n = 0; n < nmodes; n++) {
			IType dim = dims[n];
			FType* B = U[n];
			FType* Phi_n = Phi[n];

			// Lines 4-5 in Algorithm 3 of the CP-APR paper
			// Calculate B <- (A^)n) + S) lambda
			if(iter > 0) {
				bool any_violation = false;
				for(IType i = 0; i < dim * rank; i++) {
					if((B[i] < kappa_tol) && (Phi_n[i] > 1.0)) {
						B[i] = B[i] + kappa;
						any_violation = true;
					}
				}
				if(any_violation) {
					nViolations[iter]++;
				}
			}
			// Redstribute Lambda
			RedistributeLambda(M, n);

			// Calculate Pi for every non-zero element
			wtime_t = omp_get_wtime();
			CalculatePi(Pi, X, M, n);
			wtime_apr_pi += omp_get_wtime() - wtime_t;

			// Iterate until convergence or max_inner_iters reached
			for(int inner = 0; inner < max_inner_iters; inner++) {
				nInnerIters[iter]++;

				wtime_t = omp_get_wtime();
				// Line 8 in Algorithm 3 of the CP-APR paper
				// Calculate Phi^(n)
				CalculatePhi(Phi_n, Pi, X, M, n, eps_div_zero);
				wtime_apr_phi += omp_get_wtime() - wtime_t;

				// Line 9-11 in Algorithm 3 of the CP-APR paper
				// Check for convergence
				FType kkt_violation = 0.0;
				for(IType i = 0; i < dim * rank; i++) {
					FType min_comp = fabs(fmin(B[i], 1.0 - Phi_n[i]));
					kkt_violation = fmax(kkt_violation, min_comp);
				}
				kktModeViolations[n] = kkt_violation;
				// calculate kkt_violation
				if(kkt_violation < epsilon) {
					break;
				} else {
					converged = false;
				}

				// Line 13 in Algorithm 3 of the CP-APR paper
				// Do the multiplicative update
				for(IType i = 0; i < dim * rank; i++) {
					B[i] *= Phi_n[i];
				}

				/*
				if(inner % 1 == 0) {
					fprintf (stdout, "\tMode = %d, Inner iter = %d, ",
							 n, inner);
					fprintf (stdout, " KKT violation == %.6e\n",
                             kktModeViolations[n]);
                }
				 */
			} // for(int inner = 0; inner < max_inner_iters; inner++)

			// Line 15-16 in Algorithm 3 of the CP-APR paper
			// Shift weights from mode n back to Lambda and update A^(n)
			// Basically, KruskalModelNormalize for only mode n
			for(IType r = 0; r < rank; r++) {
				FType tmp_sum = 0.0;
				for(IType i = 0; i < dim; i++) {
					tmp_sum += fabs(B[i * rank + r]);
				}
				for(IType i = 0; i < dim; i++) {
					B[i * rank + r] /= tmp_sum;
				}
				lambda[r] *= tmp_sum;
			}
		} // for(int n = 0; n < nmodes; n++)

		kktViolations[iter] = kktModeViolations[0];
		for(int n = 0; n < nmodes; n++) {
			kktViolations[iter] = fmax(kktViolations[iter],
										kktModeViolations[n]);
		}
		if((iter % 10) == 0) {
			fprintf (stdout, "Iter %d: Inner its = %d KKT violation = %.6e, ",
						iter, nInnerIters[iter], kktViolations[iter]);
			fprintf (stdout, "nViolations = %d\n", nViolations[iter]);
		}

		if(converged) {
			fprintf(stdout, "Exiting since all subproblems reached KKT tol\n");
			break;
		}
	} // for(iter = 0; iter < max_iters; iter++)
	wtime_apr += omp_get_wtime() - wtime_s;


	wtime_s = omp_get_wtime();
	// Calculate the Log likelihood fit
	int nTotalInner = 0;
	if(iter == max_iters) {
		iter = max_iters - 1;
	}
    assert(iter < max_iters);
	for(int i = 0; i < iter + 1; i++) {
		nTotalInner += nInnerIters[i];
	}
	FType obj = tt_logLikelihood(X, M, eps_div_zero);

	// Calculate the Gram matrices of the factor matrices and
	// last mode's MTTKRP
	// Then use cpd_fit to calculate the fit - this should give you the
	// least-squares fit
	FType** tmp_grams = (FType**) AlignedMalloc(nmodes * sizeof(FType*));
	assert(tmp_grams);
	for(int n = 0; n < nmodes; n++) {
		tmp_grams[n] = (FType*) AlignedMalloc(rank * rank * sizeof(FType));
		assert(tmp_grams[n]);
	}
	for(int n = 0; n < nmodes; n++) {
		update_gram(tmp_grams[n], M, n);
	}

	FType* tmp_mttkrp = (FType*) AlignedMalloc(dims[nmodes - 1] * rank *
												sizeof(FType));
	assert(tmp_mttkrp);
	for(IType i = 0; i < dims[nmodes - 1] * rank; i++) {
		tmp_mttkrp[i] = 0.0;
	}
	mttkrp_(X, M, nmodes - 1, tmp_mttkrp);
	double fit = cpd_fit_(X, M, tmp_grams, tmp_mttkrp);
	wtime_post += omp_get_wtime() - wtime_s;

	fprintf(stdout, "  Final log-likelihood = %e\n", obj);
	fprintf(stdout, "  Final least squares fit = %e\n", fit);
	fprintf(stdout, "  Final KKT violation = %7.7e\n", kktViolations[iter]);
	fprintf(stdout, "  Total outer iterations = %d\n", iter);
	fprintf(stdout, "  Total inner iterations = %d\n", nTotalInner);

	fprintf(stdout, "CP-APR pre-processing time  = %f (s)\n", wtime_pre);
	fprintf(stdout, "CP-APR main iteration time  = %f (s)\n", wtime_apr);
	fprintf(stdout, "CP-APR Pi accumulated time  = %f (s)\n", wtime_apr_pi);
	fprintf(stdout, "CP-APR Phi accumulated time = %f (s)\n", wtime_apr_phi);
	fprintf(stdout, "CP-APR post-processing time = %f (s)\n", wtime_post);
	fprintf(stdout, "CP-APR TOTAL time           = %f (s)\n",
			wtime_pre + wtime_apr + wtime_post);

	fprintf(stdout, ">%f,%f,%f,%f,%f,%d,%d,%e,%e\n",
			wtime_pre, wtime_apr, wtime_apr_pi, wtime_apr_phi, wtime_post,
			iter, nTotalInner, obj, fit);
    AlignedFree(nInnerIters);
    AlignedFree(nViolations);
    AlignedFree(kktModeViolations);
    AlignedFree(kktViolations);
    AlignedFree(tmp_mttkrp);
	AlignedFree(Pi);
	for(int n = 0; n < nmodes; n++) {
		AlignedFree(tmp_grams[n]);
		AlignedFree(Phi[n]);
    }
    AlignedFree(tmp_grams);
	AlignedFree(Phi);

}


void CalculatePhi_OTF(FType* Phi, SparseTensor* X, KruskalModel* M, int target_mode, FType eps_div_zero)
{
	IType nmodes = X->nmodes;
	IType nnz = X->nnz;
	IType rank = M->rank;
	IType** cidx = X->cidx;
	FType* vals = X->vals;
	FType** U = M->U;
	FType* B = U[target_mode];
	IType dim = X->dims[target_mode];

	// Initialize Phi to 0
	for(IType i = 0; i < dim * rank; i++) {
		Phi[i] = 0.0;
	}

	// temporary array to store the KRP
	FType row[rank];

	for(IType i = 0; i < nnz; i++) {
		IType row_index = cidx[target_mode][i];

		// For each non-zero element, calculate the required KRP
		for(IType r = 0; r < rank; r++) {
			row[r] = 1.0;
		}
		for(IType n = 0; n < nmodes; n++) {
			if(n != target_mode) {
				IType row_index_n = cidx[n][i];
				for(IType r = 0; r < rank; r++) {
					row[r] *= U[n][row_index_n * rank + r];
				}
			}
		}

		// Calculate B*KRP
		FType v = 0.0;
		for(IType r = 0; r < rank; r++) {
			v += B[row_index * rank + r] * row[r];
		}
		// divide nnz by this value
		v = vals[i] / fmax(v, eps_div_zero);

		// scale KR (i.e., row) by this value (i.e., v), and update Phi
		for(IType r = 0; r < rank; r++) {
			Phi[row_index * rank + r] += v * row[r];
		}
	}
}




void cp_apr_mu_alto_otf(SparseTensor* X, KruskalModel* M, int max_iters, int max_inner_iters, FType epsilon, FType eps_div_zero, FType kappa_tol, FType kappa)
{
    // timers
    double wtime_s, wtime_t;
    double wtime_pre = 0.0;
    double wtime_apr = 0.0;
    double wtime_apr_phi = 0.0;
    double wtime_post = 0.0;

	fprintf(stdout, "Running ALTO CP-APR with: \n");
	// max outer iterations
	fprintf(stdout, "\tmax iters:       %d\n", max_iters);
	// max inner iterations
	fprintf(stdout, "\tmax inner iters: %d\n", max_inner_iters);
	// tolerance on the overall KKT violation
	fprintf(stdout, "\ttolerance:       %.2e\n", epsilon);
	// safeguard against divide by zero
	fprintf(stdout, "\teps_div_zero:    %.2e\n", eps_div_zero);
	// tolerance on complementary slackness
	fprintf(stdout, "\tkappa_tol:       %.2e\n", kappa_tol);
	// offset to fix complementary slackness
	fprintf(stdout, "\tkappa:           %.2e\n", kappa);


    wtime_s = omp_get_wtime();
	// Tensor and factor matrices information
	int nmodes = X->nmodes;
	IType* dims = X->dims;
	IType rank = M->rank;
	FType** U = M->U;
	FType* lambda = M->lambda;

	// Intermediate matrices
	// Phi matrix is an intermediate matrix equal in size to the factor matrix
	FType** Phi = (FType**) AlignedMalloc(nmodes * sizeof(FType*));
	assert(Phi);
	for(int n = 0; n < nmodes; n++) {
		Phi[n] = (FType*) AlignedMalloc(dims[n] * rank * sizeof(FType));
		assert(Phi[n]);
	}


	// Used for keeping track of convergence information
	FType* kktModeViolations = (FType*) AlignedMalloc(nmodes * sizeof(FType));
	assert(kktModeViolations);
	for(int n = 0; n < nmodes; n++) {
		kktModeViolations[n] = 0.0;
	}
	int* nViolations = (int*) AlignedMalloc(max_iters * sizeof(int));
	assert(nViolations);
	for(int i = 0; i < max_iters; i++) {
		nViolations[i] = 0;
	}
	FType* kktViolations = (FType*) AlignedMalloc(max_iters * sizeof(FType));
	assert(kktViolations);
	for(int i = 0; i < max_iters; i++) {
		kktViolations[i] = 0.0;
	}
	int* nInnerIters = (int*) AlignedMalloc(max_iters * sizeof(int));
	assert(nInnerIters);
	for(int i = 0; i < max_iters; i++) {
		nInnerIters[i] = 0;
	}
    wtime_pre += omp_get_wtime() - wtime_s;


    wtime_s = omp_get_wtime();
	fprintf(stdout, "CP_APR MU (OTF): \n");
	KruskalModelNormalize(M);
	// Iterate until covergence or max_iters reached
	int iter;
	for(iter = 0; iter < max_iters; iter++) {
		bool converged = true;
		for(int n = 0; n < nmodes; n++) {
			IType dim = dims[n];
			FType* B = U[n];
			FType* Phi_n = Phi[n];

			// Lines 4-5 in Algorithm 3 of the CP-APR paper
			// Calculate B <- (A^)n) + S) lambda
			if(iter > 0) {
				bool any_violation = false;
				for(IType i = 0; i < dim * rank; i++) {
					if((B[i] < kappa_tol) && (Phi_n[i] > 1.0)) {
						B[i] = B[i] + kappa;
						any_violation = true;
					}
				}
				if(any_violation) {
					nViolations[iter]++;
				}
			}

			// Redstribute Lambda
			RedistributeLambda(M, n);

			// Iterate until convergence or max_inner_iters reached
			for(int inner = 0; inner < max_inner_iters; inner++) {
				nInnerIters[iter]++;

                wtime_t = omp_get_wtime();
				// Line 8 in Algorithm 3 of the CP-APR paper
				// Calculate Phi^(n)
				CalculatePhi_OTF(Phi_n, X, M, n, eps_div_zero);
                wtime_apr_phi += omp_get_wtime() - wtime_t;

				// Line 9-11 in Algorithm 3 of the CP-APR paper
				// Check for convergence
				FType kkt_violation = 0.0;
				for(IType i = 0; i < dim * rank; i++) {
					FType min_comp = fabs(fmin(B[i], 1.0 - Phi_n[i]));
					kkt_violation = fmax(kkt_violation, min_comp);
				}
				kktModeViolations[n] = kkt_violation;
				// calculate kkt_violation
				if(kkt_violation < epsilon) {
					break;
				} else {
					converged = false;
				}

				// Line 13 in Algorithm 3 of the CP-APR paper
				// Do the multiplicative update
				for(IType i = 0; i < dim * rank; i++) {
					B[i] *= Phi_n[i];
				}

				/*
				if(inner % 1 == 0) {
					fprintf (stdout, "\tMode = %d, Inner iter = %d, ",
							 n, inner);
					fprintf (stdout, " KKT violation == %.6e\n",
                             kktModeViolations[n]);
                }
				 */
			} // for(int inner = 0; inner < max_inner_iters; inner++)

			// Line 15-16 in Algorithm 3 of the CP-APR paper
			// Shift weights from mode n back to Lambda and update A^(n)
			// Basically, KruskalModelNormalize for only mode n
			for(IType r = 0; r < rank; r++) {
				FType tmp_sum = 0.0;
				for(IType i = 0; i < dim; i++) {
					tmp_sum += fabs(B[i * rank + r]);
				}
				for(IType i = 0; i < dim; i++) {
					B[i * rank + r] /= tmp_sum;
				}
				lambda[r] *= tmp_sum;
			}
		} // for(int n = 0; n < nmodes; n++)

		kktViolations[iter] = kktModeViolations[0];
		for(int n = 0; n < nmodes; n++) {
			kktViolations[iter] = fmax(kktViolations[iter],
										kktModeViolations[n]);
		}
		if((iter % 10) == 0) {
			fprintf (stdout, "Iter %d: Inner its = %d KKT violation = %.6e, ",
						iter, nInnerIters[iter], kktViolations[iter]);
			fprintf (stdout, "nViolations = %d\n", nViolations[iter]);
		}

		if(converged) {
			fprintf(stdout, "Exiting since all subproblems reached KKT tol\n");
			break;
		}
	} // for(iter = 0; iter < max_iters; iter++)
    wtime_apr += omp_get_wtime() - wtime_s;


    wtime_s = omp_get_wtime();
	// Calculate the Log likelihood fit
	int nTotalInner = 0;
	if(iter == max_iters) {
		iter = max_iters - 1;
	}
    assert(iter < max_iters);
	for(int i = 0; i < iter + 1; i++) {
		nTotalInner += nInnerIters[i];
	}
	FType obj = tt_logLikelihood(X, M, eps_div_zero);

	// Calculate the Gram matrices of the factor matrices and
	// last mode's MTTKRP
	// Then use cpd_fit to calculate the fit - this should give you the
	// least-squares fit
	FType** tmp_grams = (FType**) AlignedMalloc(nmodes * sizeof(FType*));
	assert(tmp_grams);
	for(int n = 0; n < nmodes; n++) {
		tmp_grams[n] = (FType*) AlignedMalloc(rank * rank * sizeof(FType));
		assert(tmp_grams[n]);
	}
	for(int n = 0; n < nmodes; n++) {
		update_gram(tmp_grams[n], M, n);
	}

	FType* tmp_mttkrp = (FType*) AlignedMalloc(dims[nmodes - 1] * rank *
												sizeof(FType));
	assert(tmp_mttkrp);
	for(IType i = 0; i < dims[nmodes - 1] * rank; i++) {
		tmp_mttkrp[i] = 0.0;
	}
	mttkrp_(X, M, nmodes - 1, tmp_mttkrp);
	double fit = cpd_fit_(X, M, tmp_grams, tmp_mttkrp);
    wtime_post += omp_get_wtime() - wtime_s;


	fprintf(stdout, "  Final log-likelihood = %e\n", obj);
	fprintf(stdout, "  Final least squares fit = %e\n", fit);
	fprintf(stdout, "  Final KKT violation = %7.7e\n", kktViolations[iter]);
	fprintf(stdout, "  Total outer iterations = %d\n", iter);
	fprintf(stdout, "  Total inner iterations = %d\n", nTotalInner);

    fprintf(stdout, "CP-APR pre-processing time  = %f (s)\n", wtime_pre);
    fprintf(stdout, "CP-APR main iteration time  = %f (s)\n", wtime_apr);
    fprintf(stdout, "CP-APR Phi accumulated time = %f (s)\n", wtime_apr_phi);
    fprintf(stdout, "CP-APR post-processing time = %f (s)\n", wtime_post);
    fprintf(stdout, "CP-APR TOTAL time           = %f (s)\n",
            wtime_pre + wtime_apr + wtime_post);

    fprintf(stdout, ">,%f,%f,%.2f,%f,%f,%d,%d,%e,%e\n",
            wtime_pre, wtime_apr, 0.0, wtime_apr_phi, wtime_post,
            iter, nTotalInner, obj, fit);
    AlignedFree(nInnerIters);
    AlignedFree(nViolations);
    AlignedFree(kktModeViolations);
    AlignedFree(kktViolations);
    AlignedFree(tmp_mttkrp);
	for(int n = 0; n < nmodes; n++) {
		AlignedFree(tmp_grams[n]);
		AlignedFree(Phi[n]);
    }
    AlignedFree(tmp_grams);
	AlignedFree(Phi);
}

/*

template <typename LIT>
void cp_apr_alto(AltoTensor<LIT>* AT, KruskalModel* M, int max_iters, int max_inner_iters, FType epsilon, FType eps_div_zero, FType kappa_tol, FType kappa)
{


}
 */
