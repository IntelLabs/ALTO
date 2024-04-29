#ifndef APR_HPP_
#define APR_HPP_

#include "common.hpp"
#include "sptensor.hpp"
#include "kruskal_model.hpp"
#include "alto.hpp"
#include "omp.h"

#define MIN_APR_REUSE 4
#define OTF
//#define PRE

void cp_apr_mu_alto(SparseTensor* X,
					KruskalModel* M,
					int max_iters,
					int max_inner_iters,
					FType epsilon,
					FType eps_div_zero,
					FType kappa_tol,
					FType kappa);

void cp_apr_mu_alto_otf(SparseTensor* X,
						KruskalModel* M,
						int max_iters,
						int max_inner_iters,
						FType epsilon,
						FType eps_div_zero,
						FType kappa_tol,
						FType kappa);

template <typename LIT>
static inline void
cp_apr_alto(AltoTensor<LIT>* AT,
				 KruskalModel* M,
				 int max_iters,
				 int max_inner_iters,
				 FType epsilon,
				 FType eps_div_zero,
				 FType kappa_tol,
				 FType kappa,
                 bool dobench);


template <typename LIT>
static inline void
CalculatePhi_alto_par(FType* Phi,
							AltoTensor<LIT>* AT,
							KruskalModel* M,
							int target_mode,
							FType eps_div_zero,
                            FType **ofibs,
                            IType rank,
                            bool usePrecomp);

template <typename LIT, typename ModeT, typename RankT>
static inline void
CalculatePhi_alto_da_mem_pull(FType* Phi,
							AltoTensor<LIT>* AT,
							KruskalModel* M,
							int target_mode,
							FType eps_div_zero,
                            FType **ofibs,
                            ModeT nmodes,
                            RankT rank,
                            bool usePrecomp);

template <typename LIT, typename ModeT, typename RankT>
static inline void
CalculatePhi_alto_atomic_cr(FType* Phi,
                            AltoTensor<LIT>* AT,
                            KruskalModel* M,
                            int target_mode,
                            FType eps_div_zero,
                            ModeT nmodes,
                            RankT rank,
                            bool usePrecomp);

template <typename LIT, typename ModeT, typename RankT>
static inline void
CalculatePhi_alto_atomic(FType* Phi,
                            AltoTensor<LIT>* AT,
                            KruskalModel* M,
                            int target_mode,
                            FType eps_div_zero,
                            ModeT nmodes,
                            RankT rank,
                            bool usePrecomp);


template <typename LIT>
static inline void
CalculatePhi_alto_OTF_base(FType* Phi,
							AltoTensor<LIT>* AT,
							KruskalModel* M,
							int target_mode,
							FType eps_div_zero);

template <typename LIT>
FType tt_logLikelihood(AltoTensor<LIT>* AT,
						KruskalModel* M,
                        FType eps_div_zero);

FType tt_logLikelihood(SparseTensor* X,
						KruskalModel* M,
                        FType eps_div_zero);

double cpd_fit_(SparseTensor* X,
				KruskalModel* M,
				FType** grams,
				FType* U_mttkrp);

template <typename LIT>
static inline void
mttkrp_(AltoTensor<LIT>* X,
			KruskalModel* M,
			int mode,
			FType* U_mttkrp);

static inline void
mttkrp_(SparseTensor* X,
			KruskalModel* M,
			int mode,
			FType* U_mttkrp);


template <typename LIT>
static inline void
mttkrp_(AltoTensor<LIT>* AT, KruskalModel* M, int mode, FType* U_mttkrp)
{
  // we only do one MTTKRP, therefore we go with the atomic update here
  int const nprtn = AT->nprtn;
  IType const nmodes = AT->nmode;
  IType const rank = M->rank;

  #pragma omp parallel for schedule(static,1)
  for (int p = 0; p < nprtn; ++p) {
    //local buffer
    LIT ALTO_MASKS[MAX_NUM_MODES];
    for (int n = 0; n < nmodes; ++n) {
        ALTO_MASKS[n] = AT->mode_masks[n];
    }

    FType row[rank];

    LIT* const idx = AT->idx;
    ValType* const vals = AT->vals;
    IType const nnz_s = AT->prtn_ptr[p];
    IType const nnz_e = AT->prtn_ptr[p + 1];

    for (IType i = nnz_s; i < nnz_e; ++i) {
      ValType const val = vals[i];
      LIT const alto_idx = idx[i];

      #pragma omp simd
      for (IType r = 0; r < rank; ++r) {
          row[r] = (FType)val;
      }

      for (int m = 0; m < nmodes; ++m) {
        if (m != mode) {
#ifndef ALT_PEXT
          IType const row_id = pext(alto_idx, ALTO_MASKS[m]);
#else
          IType const row_id = pext(alto_idx, ALTO_MASKS[m], AT->mode_pos[m]);
#endif
          #pragma omp simd
          for (IType r = 0; r < rank; r++) {
            row[r] *= M->U[m][row_id * rank + r];
          }
        }
      }

      // update destination row
#ifndef ALT_PEXT
      IType const row_id = pext(alto_idx, ALTO_MASKS[mode]);
#else
      IType const row_id = pext(alto_idx, ALTO_MASKS[mode], AT->mode_pos[mode]);
#endif
      for (IType r = 0; r < rank; ++r) {
        #pragma omp atomic update
        U_mttkrp[row_id * rank + r] += row[r];
      }
    } // nnzs
  } // prtns
}

static inline void
mttkrp_(SparseTensor* X, KruskalModel* M, int mode, FType* U_mttkrp)
{
  IType nmodes = X->nmodes;
  IType nnz = X->nnz;
  IType** cidx = X->cidx;
  IType rank = M->rank;

  FType row[rank];

  for(IType i = 0; i < nnz; i++) {
    // initialize temporary accumulator
    for(IType r = 0; r < rank; r++) {
      row[r] = X->vals[i];
    }

    // calculate mttkrp for the current non-zero
    for(IType m = 0; m < nmodes; m++) {
      if(m != mode) {
        IType row_id = cidx[m][i];
        for(IType r = 0; r < rank; r++) {
          row[r] *= M->U[m][row_id * rank + r];
        }
      }
    }

    // update destination row
    IType row_id = cidx[mode][i];
    for(IType r = 0; r < rank; r++) {
      // M->U[mode][row_id * rank + r] += row[r];
	  U_mttkrp[row_id * rank + r] += row[r];
    }
  } // for each nonzero
}


template <typename LIT, typename MT, IType... Ranks>
struct calcPhi_alto_rank_specializer;

template <typename LIT, typename MT>
struct calcPhi_alto_rank_specializer<LIT, MT, 0>
{
    void
    operator()(FType* Phi, AltoTensor<LIT>* AT, KruskalModel* M, int target_mode, FType eps_div_zero, FType** ofibs, IType fib_reuse, IType rank, MT nmodes, bool usePrecomp)
    {
        if (fib_reuse <= MIN_FIBER_REUSE) {
            // Use op_sched only if oidx is available
            if (AT->is_oidx)
                return CalculatePhi_alto_op_sched(Phi, AT, M, target_mode, eps_div_zero, nmodes, rank, usePrecomp);
            if (AT->cr_masks[target_mode])
                return CalculatePhi_alto_atomic_cr(Phi, AT, M, target_mode, eps_div_zero, nmodes, rank, usePrecomp);
            return CalculatePhi_alto_atomic(Phi, AT, M, target_mode, eps_div_zero, nmodes, rank, usePrecomp);
        }
        else
            return CalculatePhi_alto_da_mem_pull(Phi, AT, M, target_mode, eps_div_zero, ofibs, nmodes, rank, usePrecomp);
    }
};

template <typename LIT, typename MT, IType Head, IType... Tail>
struct calcPhi_alto_rank_specializer<LIT, MT, 0, Head, Tail...>
{
    void
    operator()(FType* Phi, AltoTensor<LIT>* AT, KruskalModel* M, int target_mode, FType eps_div_zero, FType** ofibs, IType fib_reuse, IType rank, MT nmodes, bool usePrecomp)
    {
        if (rank == Head) {
            using const_rank = std::integral_constant<IType, Head>;
            if (fib_reuse <= MIN_FIBER_REUSE) {
                // Use op_sched only if oidx is available
                if (AT->is_oidx)
                    return CalculatePhi_alto_op_sched(Phi, AT, M, target_mode, eps_div_zero, nmodes, const_rank(), usePrecomp);
                if (AT->cr_masks[target_mode])
                    return CalculatePhi_alto_atomic_cr(Phi, AT, M, target_mode, eps_div_zero, nmodes, const_rank(), usePrecomp);
                return CalculatePhi_alto_atomic(Phi, AT, M, target_mode, eps_div_zero, nmodes, const_rank(), usePrecomp);
            }
            else
                return CalculatePhi_alto_da_mem_pull(Phi, AT, M, target_mode, eps_div_zero, ofibs, nmodes, const_rank(), usePrecomp);
        }
        else
            return calcPhi_alto_rank_specializer<LIT, MT, 0, Tail...>()(Phi, AT, M, target_mode, eps_div_zero, ofibs, fib_reuse, rank, nmodes, usePrecomp);
    }
};


template <typename LIT, int... Modes>
struct calcPhi_alto_mode_specializer;

template <typename LIT>
struct calcPhi_alto_mode_specializer<LIT, 0>
{
    void
    operator()(FType* Phi, AltoTensor<LIT>* AT, KruskalModel* M, int target_mode, FType eps_div_zero, FType** ofibs, IType fib_reuse, IType rank, bool usePrecomp)
    { calcPhi_alto_rank_specializer<LIT, int, ALTO_RANKS_SPECIALIZED>()(Phi, AT, M, target_mode, eps_div_zero, ofibs, fib_reuse, rank, AT->nmode, usePrecomp); }
};

template <typename LIT, int Head, int... Tail>
struct calcPhi_alto_mode_specializer<LIT, 0, Head, Tail...>
{
    void
    operator()(FType* Phi, AltoTensor<LIT>* AT, KruskalModel* M, int target_mode, FType eps_div_zero, FType** ofibs, IType fib_reuse, IType rank, bool usePrecomp)
    {
        if (AT->nmode == Head) {
            using const_mode = std::integral_constant<int, Head>;
            calcPhi_alto_rank_specializer<LIT, const_mode, ALTO_RANKS_SPECIALIZED>()(Phi, AT, M, target_mode, eps_div_zero, ofibs, fib_reuse, rank, const_mode(), usePrecomp);
        }
        else
            calcPhi_alto_mode_specializer<LIT, 0, Tail...>()(Phi, AT, M, target_mode, eps_div_zero, ofibs, fib_reuse, rank, usePrecomp);
    }
};

template <typename LIT>
static inline void
CalculatePhi_alto_par(FType* Phi, AltoTensor<LIT>* AT, KruskalModel* M, int target_mode, FType eps_div_zero, FType** ofibs, IType rank, bool usePrecomp)
{
    IType fib_reuse = AT->nnz / AT->dims[target_mode];
    return calcPhi_alto_mode_specializer<LIT, ALTO_MODES_SPECIALIZED>()(Phi, AT, M, target_mode, eps_div_zero, ofibs, fib_reuse, rank, usePrecomp);
}


// NAIVE implementation for validation
template <typename LIT>
void CalculatePhi_alto_OTF_base(FType* Phi, AltoTensor<LIT>* AT, KruskalModel* M, int target_mode, FType eps_div_zero)
{
	// printf("in CalculatePhi_alto_OTF: %d\n", target_mode);
    IType nmodes = AT->nmode;
    IType nnz = AT->nnz;
    IType rank = M->rank;
    // IType** cidx = X->cidx;
    ValType* vals = AT->vals;
    FType** U = M->U;
    FType* B = U[target_mode];
    IType dim = AT->dims[target_mode];

    LIT ALTO_MASKS[MAX_NUM_MODES];
    for (int n = 0; n < nmodes; ++n) {
        ALTO_MASKS[n] = AT->mode_masks[n];
    }


    // Initialize Phi to 0
    for(IType i = 0; i < dim * rank; i++) {
        Phi[i] = 0.0;
    }

    // temporary array to store the KRP
    FType row[rank];

    for(IType i = 0; i < nnz; i++) {
        // IType row_index = cidx[target_mode][i];
        LIT alto_idx = AT->idx[i];
#ifndef ALT_PEXT
		IType row_index = pext(alto_idx, ALTO_MASKS[target_mode]);
#else
		IType row_index = pext(alto_idx, ALTO_MASKS[target_mode],
								AT->mode_pos[target_mode]);
#endif

        // For each non-zero element, calculate the required KRP
        for(IType r = 0; r < rank; r++) {
            row[r] = 1.0;
        }

        for(IType n = 0; n < nmodes; n++) {
            if(n != target_mode) {
                // IType row_index_n = cidx[n][i];
#ifndef ALT_PEXT
				IType row_index_n = pext(alto_idx, ALTO_MASKS[n]);
#else
				IType row_index_n = pext(alto_idx, ALTO_MASKS[n],
											AT->mode_pos[n]);
#endif

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

		// if(row_index == 0) printf("-- %f %f %f\n", vals[i], v, vals[i] / fmax(v, eps_div_zero));

        // divide nnz by this value
        v = (FType)vals[i] / fmax(v, eps_div_zero);

        // scale KR (i.e., row) by this value (i.e., v), and update Phi
        for(IType r = 0; r < rank; r++) {
            Phi[row_index * rank + r] += v * row[r];
        }
    }
}

template <typename LIT, typename ModeT, typename RankT>
static inline void
CalculatePhi_alto_da_mem_pull(FType* Phi, AltoTensor<LIT>* AT, KruskalModel* M, int target_mode, FType eps_div_zero, FType** ofibs, ModeT nmodes, RankT rank, bool usePrecomp)
{
    // printf("in CalculatePhi_alto_da_mem_pull: %d\n", target_mode);
    assert(nmodes == AT->nmode);
    ValType* vals = AT->vals;
    FType** U = M->U;
    FType* B = U[target_mode];
    IType dim = AT->dims[target_mode];

    int const nprtn = AT->nprtn;
    //printf("Mode %d, %d prtns\n", target_mode, nprtn);

    #pragma omp parallel proc_bind(close)
    {
        //local buffer
        LIT ALTO_MASKS[MAX_NUM_MODES];

        for (int n = 0; n < nmodes; ++n) {
            ALTO_MASKS[n] = AT->mode_masks[n];
        }

        #pragma omp for schedule(static, 1)
        for (int p = 0; p < nprtn; ++p) {
            // temporary array to store the KRP
            FType *row;
            if (!usePrecomp) {
                row = (FType*) AlignedMalloc(rank*sizeof(FType));
            }

            FType* const out = ofibs[p];
            Interval const intvl = AT->prtn_intervals[p * nmodes + target_mode];
            IType const offset = intvl.start;
            IType const stop = intvl.stop;

            memset(out, 0, (stop - offset + 1) * rank * sizeof(FType));

            IType const nnz_s = AT->prtn_ptr[p];
            IType const nnz_e = AT->prtn_ptr[p + 1];

            for (IType i = nnz_s; i < nnz_e; ++i) {
                LIT alto_idx = AT->idx[i];
#ifndef ALT_PEXT
                IType const row_index = pext(alto_idx, ALTO_MASKS[target_mode]);
#else
                IType const row_index = pext(alto_idx, ALTO_MASKS[target_mode], AT->mode_pos[target_mode]);
#endif
                if (usePrecomp) {
                    row = &AT->precomp_row[i * rank];
                } else {
                    // on-the-fly comp
                    // For each non-zero element, calculate the required KRP
                    for(IType r=0; r<rank; ++r) {
                        row[r] = 1.0;
                    }

                    for(IType n=0; n<nmodes; ++n) {
                        if(n != target_mode) {
    #ifndef ALT_PEXT
                            IType const row_index_n = pext(alto_idx, ALTO_MASKS[n]);
    #else
                            IType const row_index_n = pext(alto_idx, ALTO_MASKS[n], AT->mode_pos[n]);
    #endif
                            #pragma omp simd safelen(16)
                            for(IType r=0; r<rank; ++r) {
                                row[r] *= U[n][row_index_n * rank + r];
                            }
                        }
                    }
                } // usePrecomp

                // /*// Calculate B*KRP
                FType v = 0.0;
                #pragma omp simd safelen(16)
                for(IType r=0; r<rank; ++r) {
                     v += B[row_index * rank + r] * row[r];
                }

				// if(row_index == 0) printf("<<< %f %f %f\n", vals[i], v, vals[i] / fmax(v, eps_div_zero));

                // divide nnz by this value
                v = (FType)vals[i] / fmax(v, eps_div_zero);

                IType const row_index_o = row_index - offset;

                // scale KR (i.e., row) by this value (i.e., v), and update out
                #pragma omp simd safelen(16)
                for(IType r = 0; r < rank; r++) {
                    out[row_index_o * rank + r] += v * row[r];
                }// */
            } //nnzs
            if (!usePrecomp) AlignedFree(row);
        } // nprtn

        // pull-based accumulation
        #pragma omp for schedule(static)
        for (IType i = 0; i < dim; ++i) {
            #pragma omp simd
            for (IType r = 0; r < rank; r++) {
                Phi[i * rank + r] = 0.0;
            }
        	for (int p = 0; p < nprtn; p++) {
                Interval const intvl = AT->prtn_intervals[p * nmodes + target_mode];
                IType const offset = intvl.start;
                IType const stop = intvl.stop;

                if ((i >= offset) && (i <= stop)) {
                    FType* const out = ofibs[p];
                    IType const j = i - offset;
                    #pragma omp simd
                    for (IType r = 0; r < rank; r++) {
                        Phi[i * rank + r] += out[j * rank + r];
                    }
                }
            } //prtns
        } // num_fibs
    } // omp parallel
}

template <typename LIT, typename ModeT, typename RankT>
static inline void
CalculatePhi_alto_atomic(FType* Phi, AltoTensor<LIT>* AT, KruskalModel* M, int target_mode, FType eps_div_zero, ModeT nmodes, RankT rank, bool usePrecomp)
{
	//printf("in CalculatePhi_alto_atomic: %d\n", target_mode);
    assert(nmodes == AT->nmode);
    ValType* vals = AT->vals;
    FType** U = M->U;
    FType* B = U[target_mode];
    IType dim = AT->dims[target_mode];

    int const nprtn = AT->nprtn;
    //printf("Mode %d, %d prtns\n", target_mode, nprtn);

    #pragma omp parallel proc_bind(close)
    {
        //local buffer
        LIT ALTO_MASKS[MAX_NUM_MODES];

        for (int n = 0; n < nmodes; ++n) {
            ALTO_MASKS[n] = AT->mode_masks[n];
        }
		#pragma omp for simd schedule(static)
        for(IType i = 0; i < dim * rank; i++) {
        	Phi[i] = 0.0;
        }

        #pragma omp for schedule(static, 1)
        for (int p = 0; p < nprtn; ++p) {
            // temporary array to store the KRP
            FType *row;
            if (!usePrecomp) {
                row = (FType*) AlignedMalloc(rank * sizeof(FType));
            }

            IType const nnz_s = AT->prtn_ptr[p];
            IType const nnz_e = AT->prtn_ptr[p + 1];

            for (IType i = nnz_s; i < nnz_e; ++i) {
                LIT alto_idx = AT->idx[i];
#ifndef ALT_PEXT
                IType const row_index = pext(alto_idx, ALTO_MASKS[target_mode]);
#else
                IType const row_index = pext(alto_idx, ALTO_MASKS[target_mode], AT->mode_pos[target_mode]);
#endif
                if (usePrecomp) {
                    row = &AT->precomp_row[i * rank];
                } else {
                    // on-the-fly comp
                    // For each non-zero element, calculate the required KRP
                    for(IType r=0; r<rank; ++r) {
                        row[r] = 1.0;
                    }

                    for(IType n=0; n<nmodes; ++n) {
                        if(n != target_mode) {
    #ifndef ALT_PEXT
                            IType const row_index_n = pext(alto_idx, ALTO_MASKS[n]);
    #else
                            IType const row_index_n = pext(alto_idx, ALTO_MASKS[n], AT->mode_pos[n]);
    #endif
                            #pragma omp simd safelen(16)
                            for(IType r=0; r<rank; ++r) {
                                row[r] *= U[n][row_index_n * rank + r];
                            }
                        }
                    }
                } // usePrecomp

                // /*// Calculate B*KRP
                FType v = 0.0;
                #pragma omp simd safelen(16)
                for(IType r=0; r<rank; ++r) {
                     v += B[row_index * rank + r] * row[r];
                }

				// if(row_index == 0) printf("<<< %f %f %f\n", vals[i], v, vals[i] / fmax(v, eps_div_zero));

                // divide nnz by this value
                v = (FType)vals[i] / fmax(v, eps_div_zero);

                // scale KR (i.e., row) by this value (i.e., v), and update Phi atomically
                for(IType r = 0; r < rank; r++) {
                    #pragma omp atomic update
                    Phi[row_index * rank + r] += v * row[r];
                }
            } //nnzs
            if (!usePrecomp) AlignedFree(row);
        } // nprtn
    } // omp parallel
}


template <typename LIT, typename ModeT, typename RankT>
static inline void
CalculatePhi_alto_atomic_cr(FType* Phi, AltoTensor<LIT>* AT, KruskalModel* M, int target_mode, FType eps_div_zero, ModeT nmodes, RankT rank, bool usePrecomp)
{
	//printf("in CalculatePhi_alto_atomic_cr: %d\n", target_mode);
    assert(nmodes == AT->nmode);
    ValType* vals = AT->vals;
    FType** U = M->U;
    FType* B = U[target_mode];
    IType dim = AT->dims[target_mode];

    int const nprtn = AT->nprtn;
    LIT const cr_mask = AT->cr_masks[target_mode];
    //printf("Mode %d, %d prtns\n", target_mode, nprtn);

    #pragma omp parallel proc_bind(close)
    {
        //local buffer
        LIT ALTO_MASKS[MAX_NUM_MODES];

        for (int n = 0; n < nmodes; ++n) {
            ALTO_MASKS[n] = AT->mode_masks[n];
        }
		#pragma omp for simd schedule(static)
        for(IType i = 0; i < dim * rank; i++) {
        	Phi[i] = 0.0;
        }

        #pragma omp for schedule(static, 1)
        for (int p = 0; p < nprtn; ++p) {
            // temporary array to store the KRP
            FType *row;
            if (!usePrecomp) {
                row = (FType*) AlignedMalloc(rank*sizeof(FType));
            }

            IType const nnz_s = AT->prtn_ptr[p];
            IType const nnz_e = AT->prtn_ptr[p + 1];

            for (IType i = nnz_s; i < nnz_e; ++i) {
                LIT alto_idx = AT->idx[i];
#ifndef ALT_PEXT
                IType const row_index = pext(alto_idx, ALTO_MASKS[target_mode]);
#else
                IType const row_index = pext(alto_idx, ALTO_MASKS[target_mode], AT->mode_pos[target_mode]);
#endif
                if (usePrecomp) {
                    row = &AT->precomp_row[i * rank];
                } else {
                    // on-the-fly comp
                    // For each non-zero element, calculate the required KRP
                    for(IType r=0; r<rank; ++r) {
                        row[r] = 1.0;
                    }

                    for(IType n=0; n<nmodes; ++n) {
                        if(n != target_mode) {
    #ifndef ALT_PEXT
                            IType const row_index_n = pext(alto_idx, ALTO_MASKS[n]);
    #else
                            IType const row_index_n = pext(alto_idx, ALTO_MASKS[n], AT->mode_pos[n]);
    #endif
                            #pragma omp simd safelen(16)
                            for(IType r=0; r<rank; ++r) {
                                row[r] *= U[n][row_index_n * rank + r];
                            }
                        }
                    }
                } // usePrecomp

                // /*// Calculate B*KRP
                FType v = 0.0;
                #pragma omp simd safelen(16)
                for(IType r=0; r<rank; ++r) {
                     v += B[row_index * rank + r] * row[r];
                }

				// if(row_index == 0) printf("<<< %f %f %f\n", vals[i], v, vals[i] / fmax(v, eps_div_zero));

                // divide nnz by this value
                v = (FType)vals[i] / fmax(v, eps_div_zero);

                // scale KR (i.e., row) by this value (i.e., v), and update Phi atomically
                if (alto_idx & cr_mask) {
                    for(IType r = 0; r < rank; r++) {
                        #pragma omp atomic update
                        Phi[row_index * rank + r] += v * row[r];
                    }
                } else {
                    #pragma omp simd safelen(16)
                    for(IType r = 0; r < rank; r++) {
                        Phi[row_index * rank + r] += v * row[r];
                    }
                }
            } //nnzs
            if (!usePrecomp) AlignedFree(row);
        } // nprtn
    } // omp parallel
}

template <typename LIT, typename ModeT, typename RankT>
static inline void
CalculatePhi_alto_op_sched(FType* Phi, AltoTensor<LIT>* AT, KruskalModel* M, int target_mode, FType eps_div_zero, ModeT nmodes, RankT rank, bool usePrecomp)
{
	// printf("in CalculatePhi_alto_op_sched: %d\n", target_mode);
    assert(nmodes == AT->nmode);
    LIT* const idx = AT->idx;
    ValType* vals = AT->vals;
    IType* const mode_oidx = AT->oidx[target_mode];
    FType** U = M->U;
    FType* B = U[target_mode];
    IType dim = AT->dims[target_mode];

    int const nprtn = AT->nprtn;
    //printf("Mode %d, %d prtns\n", target_mode, nprtn);

    #pragma omp parallel proc_bind(close)
    {
        //local buffer
        LIT ALTO_MASKS[MAX_NUM_MODES];

        for (int n = 0; n < nmodes; ++n) {
            ALTO_MASKS[n] = AT->mode_masks[n];
        }

        #pragma omp for schedule(static, 1)
        for (int p = 0; p < nprtn; ++p) {
            IType const nnz_s = AT->prtn_ptr[p];
            IType const nnz_e = AT->prtn_ptr[p + 1];
            IType last_row;
#ifndef ALT_PEXT
            last_row = pext(idx[mode_oidx[nnz_s]], ALTO_MASKS[target_mode]);
#else
            last_row = pext(idx[mode_oidx[nnz_s]], ALTO_MASKS[target_mode], AT->mode_pos[target_mode]);
#endif
            for (IType r = 0; r < rank; ++r) {
                #pragma omp atomic write
                Phi[last_row * rank + r] = 0.0;
            }
#ifndef ALT_PEXT
            last_row = pext(idx[mode_oidx[nnz_e - 1]], ALTO_MASKS[target_mode]);
#else
            last_row = pext(idx[mode_oidx[nnz_e - 1]], ALTO_MASKS[target_mode], AT->mode_pos[target_mode]);
#endif
            for (IType r = 0; r < rank; ++r) {
                #pragma omp atomic write
                Phi[last_row * rank + r] = 0.0;
            }
        }
        #pragma omp barrier

        #pragma omp for schedule(static, 1)
        for (int p = 0; p < nprtn; ++p) {
            FType accum[rank]; //Allocate an auto array of variable size.
            memset(accum, 0, rank * sizeof(FType));
            // temporary array to store the KRP
            FType *row;
            if (!usePrecomp) {
                row = (FType*) AlignedMalloc(rank*sizeof(FType));
            }

            IType const nnz_s = AT->prtn_ptr[p];
            IType const nnz_e = AT->prtn_ptr[p + 1];
            IType last_row;
#ifndef ALT_PEXT
            last_row = pext(idx[mode_oidx[nnz_s]], ALTO_MASKS[target_mode]);
#else
            last_row = pext(idx[mode_oidx[nnz_s]], ALTO_MASKS[target_mode], AT->mode_pos[target_mode]);
#endif
            bool is_boundry = true;

            for (IType i = nnz_s; i < nnz_e; ++i) {
                LIT const alto_idx = idx[mode_oidx[i]];
                FType const val = (FType) vals[mode_oidx[i]];
#ifndef ALT_PEXT
                IType const row_index = pext(alto_idx, ALTO_MASKS[target_mode]);
#else
                IType const row_index = pext(alto_idx, ALTO_MASKS[target_mode], AT->mode_pos[target_mode]);
#endif
                if (usePrecomp) {
                    row = &AT->precomp_row[i * rank];
                } else {
                    // on-the-fly comp
                    // For each non-zero element, calculate the required KRP
                    #pragma omp simd safelen(16)
                    for(IType r=0; r<rank; ++r) {
                        row[r] = 1.0;
                    }

                    for(IType n=0; n<nmodes; ++n) {
                        if(n != target_mode) {
    #ifndef ALT_PEXT
                            IType const row_index_n = pext(alto_idx, ALTO_MASKS[n]);
    #else
                            IType const row_index_n = pext(alto_idx, ALTO_MASKS[n], AT->mode_pos[n]);
    #endif
                            #pragma omp simd safelen(16)
                            for(IType r=0; r<rank; ++r) {
                                row[r] *= U[n][row_index_n * rank + r];
                            }
                        }
                    }
                } // usePrecomp

                // /*// Calculate B*KRP
                FType v = 0.0;
                #pragma omp simd safelen(16)
                for(IType r=0; r<rank; ++r) {
                     v += B[row_index * rank + r] * row[r];
                }

				// if(row_index == 0) printf("<<< %f %f %f\n", vals[i], v, vals[i] / fmax(v, eps_div_zero));

                // divide nnz by this value
                v = val / fmax(v, eps_div_zero);

                // scale KR (i.e., row) by this value (i.e., v), and update Phi atomically
                // flush when index changes
                if(last_row != row_index) {
                    if (is_boundry) { // use atomics
                        for (IType r = 0; r < rank; ++r) {
                            #pragma omp atomic update
                            Phi[last_row * rank + r] += accum[r];
                        }
                        is_boundry = false;
                    } else {
                        #pragma omp simd safelen(16)
                        for (IType r = 0; r < rank; ++r) {
                            Phi[last_row * rank + r] = accum[r];
                            // Phi[last_row * rank + r] += accum[r];
                        }
                    }
                    memset(accum, 0, rank * sizeof(FType));
                } // index changes
                last_row = row_index;

                #pragma omp simd safelen(16)
                for (IType r = 0; r < rank; ++r) {
                    accum[r] += v * row[r];
                }

                // write the very last updates
                if (i == nnz_e - 1) {
                    for (IType r = 0; r < rank; ++r) {
                        #pragma omp atomic update
                        Phi[last_row * rank + r] += accum[r];
                    }
                }
            } //nnzs
            if (!usePrecomp) AlignedFree(row);
        } // nprtn
    } // omp parallel
}

// Calculate the Log likelihood for the final decomposition
template <typename LIT>
FType tt_logLikelihood(AltoTensor<LIT>* AT, KruskalModel* M, FType eps_div_zero)
{

	int nmodes = AT->nmode;
    IType nnz = AT->nnz;
	IType rank = M->rank;
	FType* lambda = M->lambda;
	FType** U = M->U;

    LIT* const idx = AT->idx;

	// Absorb the lambda into the first mode
	RedistributeLambda (M, 0);

#if 0
	// Initialize A
	FType* A = (FType*) AlignedMalloc(nnz * rank * sizeof(FType));
    #pragma omp parallel for simd schedule(static)
	for(IType i = 0; i < nnz * rank; i++) {
		A[i] = 1.0;
	}

	for(int n = 0; n < nmodes; n++) {
		FType* U_n = U[n];
        #pragma omp parallel for schedule(static)
		for(IType i = 0; i < nnz; i++) {
            LIT const alto_idx = idx[i];
#ifndef ALT_PEXT
            IType const row_index = pext(alto_idx, AT->mode_masks[n]) * rank;
#else
            IType const row_index = pext(alto_idx, AT->mode_masks[n], AT->mode_pos[n]) * rank;
#endif
            #pragma omp simd
			for(IType r = 0 ; r < rank; r++) {
				A[i * rank + r] = A[i * rank + r] * U_n[row_index + r];
			}
		}
	}

	FType logSum = 0.0;
    #pragma omp parallel for schedule(static) reduction(+:logSum)
	for(IType i = 0; i < nnz; i++) {
		FType tmp = 0.0;
        #pragma omp simd
		for(IType r = 0; r < rank; r++) {
			tmp += A[i * rank + r];
		}
        // TODO check if that's ok (to avoid -inf)
        if (tmp < eps_div_zero)
            tmp = eps_div_zero;
   		logSum += (FType)AT->vals[i] * log(tmp);
	}
    AlignedFree(A);
#else
	// sum over all nonzeros
	FType logSum = 0.0;
	// scratchpad for calculating KRP for each nonzero
	int num_threads = omp_get_max_threads();
	FType** gscratch = (FType**) AlignedMalloc(sizeof(FType*) * num_threads);
	assert(gscratch);
    #pragma omp parallel for
	for(int t = 0; t < num_threads; t++) {
		gscratch[t] = (FType*) AlignedMalloc(sizeof(FType) * rank);
		assert(gscratch[t]);
	}

    #pragma omp parallel
	{
		FType* scratch = gscratch[omp_get_thread_num()];
		#pragma omp for schedule(static) reduction(+:logSum)
		for(IType i = 0; i < nnz; i++) {
			// initialize scratch to 1.0
			#pragma omp simd
			for(IType r = 0; r < rank; r++) {
				scratch[r] = 1.0;
			}

			// Go through every mode and Hadmard product the i,j,k rows
			LIT const alto_idx = idx[i];
			for(int n = 0; n < nmodes; n++) {
				FType* U_n = U[n];
#ifndef ALT_PEXT
	            IType const row_index = pext(alto_idx, AT->mode_masks[n]) * rank;
#else
	            IType const row_index = pext(alto_idx, AT->mode_masks[n], AT->mode_pos[n]) * rank;
#endif
				#pragma omp simd
				for(IType r = 0; r < rank; r++) {
					scratch[r] *= U_n[row_index + r];
				}
			}

			FType tmp = 0.0;
			// #pragma omp simd
			for(IType r = 0; r < rank; r++) {
				tmp += scratch[r];
			}
			if(tmp < eps_div_zero) {
				tmp = eps_div_zero;
			}
			logSum += (FType) AT->vals[i] * log(tmp);
		}
	} // #pragma omp parallel
    #pragma omp parallel for
	for(int t = 0; t < num_threads; t++) {
		AlignedFree(gscratch[t]);
	}
	AlignedFree(gscratch);
#endif

	FType factorSum = 0.0;
    #pragma omp parallel for simd schedule(static) reduction(+:factorSum)
	for(IType i = 0; i < AT->dims[0] * rank; i++) {
		factorSum += U[0][i];
	}

	for(IType r = 0; r < rank; r++) {
		FType temp = 0.0;
        #pragma omp parallel for simd schedule(static) reduction(+:temp)
		for(IType i = 0; i < AT->dims[0]; i++) {
			temp += fabs (U[0][i * rank + r]);
		}
        #pragma omp parallel for simd schedule(static)
		for(IType i = 0; i < AT->dims[0]; i++) {
			U[0][i * rank + r] = U[0][i * rank + r] / temp;
		}
		lambda[r] = lambda[r] * temp;
	}

	return (logSum - factorSum);
}

template <typename LIT>
static inline void
cp_apr_alto(AltoTensor<LIT>* AT, KruskalModel* M, int max_iters, int max_inner_iters, FType epsilon, FType eps_div_zero, FType kappa_tol, FType kappa, bool dobench)
{

    // timers
    double wtime_s, wtime_t;
    double wtime_pre = 0.0;
    double wtime_apr = 0.0;
    double wtime_apr_phi = 0.0;
    double wtime_post = 0.0;

    fprintf(stdout, "Running ALTO CP-APR with: \n");
    // max outer iterations
    fprintf(stdout, "\tmax iters:           %d\n", max_iters);
    // max inner iterations
    fprintf(stdout, "\tmax inner iters:     %d\n", max_inner_iters);
    // tolerance on the overall KKT violation
    fprintf(stdout, "\ttolerance:           %.2e\n", epsilon);
    // safeguard against divide by zero
    fprintf(stdout, "\teps_div_zero:        %.2e\n", eps_div_zero);
    // tolerance on complementary slackness
    fprintf(stdout, "\tkappa_tol:           %.2e\n", kappa_tol);
    // offset to fix complementary slackness
    fprintf(stdout, "\tkappa:               %.2e\n", kappa);
    // data type sizes
    fprintf(stdout, "\tOrdinal Type:        int%lu\n", 8*sizeof(IType));
    fprintf(stdout, "\tSparse Val Type:     %s%lu\n", typeid(ValType) == typeid(double) || typeid(ValType) == typeid(float) ? "FP" : "int", 8*sizeof(ValType));
    fprintf(stdout, "\tKruskal Val Type:    FP%lu\n", 8*sizeof(FType));
#ifdef OTF
    fprintf(stdout, "\tPHI update mode:     OTF\n");
#endif
#ifdef PRE
    fprintf(stdout, "\tPHI update mode:     PRE\n");
#endif

    wtime_s = omp_get_wtime();
    // Tensor and factor matrices information
    int nmodes = AT->nmode;
    IType* dims = AT->dims;
    IType rank = M->rank;
    FType** U = M->U;
    FType* lambda = M->lambda;
    // LIT* idx = AT->idx;

    IType nthreads = omp_get_max_threads();
    // Create local fiber copies
    FType ** ofibs = NULL;
    create_da_mem(-1, rank, AT, &ofibs);

    bool needPhiPrecomp = true;
    for (int n=0; n<nmodes; ++n) {
        IType fib_reuse = AT->nnz / AT->dims[n];
        // So far, use the same threshold for deciding over pre-comp/on-the-fly-comp for Phi as for da_mem_pull/atomic approach choice
        if (fib_reuse <= MIN_APR_REUSE) {
            needPhiPrecomp = true;
        }
    }
#ifdef OTF
    needPhiPrecomp = false;
#ifdef PRE
    printf("Do not set both OTF and PRE. Exiting.\n");
    exit(1);
#endif
#endif
#ifdef PRE
    needPhiPrecomp = true;
#endif

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
    int* nInnerItersPerMode = (int*) AlignedMalloc(nmodes * sizeof(int));
    assert(nInnerItersPerMode);
    double * modeTimes = (double*) AlignedMalloc(nmodes * sizeof(double));
    assert(modeTimes);
    double * modePhiTimes = (double*) AlignedMalloc(nmodes * sizeof(double));
    assert(modePhiTimes);
    for(int i = 0; i < nmodes; i++) {
        nInnerItersPerMode[i] = 0;
        modeTimes[i] = 0.0;
        modePhiTimes[i] = 0.0;
    }
    int const nprtn = AT->nprtn;
    if (needPhiPrecomp) {
        AT->precomp_row = (FType*) AlignedMalloc(AT->nnz * rank * sizeof(FType));
        assert(AT->precomp_row);
    }

    wtime_pre += omp_get_wtime() - wtime_s;


    wtime_s = omp_get_wtime();
    fprintf(stdout, "CP_APR ALTO (OTF): \n");
    KruskalModelNormalize(M);
    // Iterate until covergence or max_iters reached
    int iter;
    double wtime_mod = omp_get_wtime();
    bool usePrecomp;
    for(iter = 0; iter < max_iters; iter++) {
        bool converged = false;
        for(int n = 0; n < nmodes; n++) {
            wtime_mod = omp_get_wtime();
            IType dim = dims[n];
            FType* B = U[n];
            FType* Phi_n = Phi[n];

            // based on re-use, do pre-computation or not
            ////////////////////////////////////////////////
            // EDIT HERE FOR A DIFFERENT HEURISTIC /////////
            // 1) heuristic equal to differentiation between pull-based and atomic approach for each mode:
            // IType fib_reuse = AT->nnz / AT->dims[n];
            // usePrecomp = fib_reuse <= MIN_APR_REUSE; //MIN_FIBER_REUSE;
            // 2) simple user-defined sttatic approach:
#ifdef OTF
            usePrecomp = false;
#endif
#ifdef PRE
            usePrecomp = true;     // or true
#endif
            // 3) like 1), but the same for all modes:
            //usePrecomp = needPhiPrecomp;
            ////////////////////////////////////////////////
            if (usePrecomp) {
                wtime_t = omp_get_wtime();
                // pre computation
                #pragma omp parallel for schedule(static,1)
                for (int p = 0; p < nprtn; ++p)
                {
                    Interval const intvl = AT->prtn_intervals[p * nmodes + n];
                    IType const offset = intvl.start;
                    IType const stop = intvl.stop;
                    IType const nnz_s = AT->prtn_ptr[p];
                    IType const nnz_e = AT->prtn_ptr[p + 1];

                    for (IType i = nnz_s; i < nnz_e; ++i)
                    {
                        LIT alto_idx = AT->idx[i];
                        for (IType r=0; r<rank; ++r) {
                            AT->precomp_row[i * rank + r] = 1.0;
                        }

                        for (IType n_sub=0; n_sub<nmodes; ++n_sub) {
                            if (n_sub != n) {
#ifndef ALT_PEXT
                                IType const row_index_n = pext(alto_idx, AT->mode_masks[n_sub]);
#else
                                IType const row_index_n = pext(alto_idx, AT->mode_masks[n_sub], AT->mode_pos[n_sub]);
#endif
#pragma omp simd safelen(16)
                                for (IType r=0; r<rank; ++r) {
                                    AT->precomp_row[i * rank + r] *= U[n_sub][row_index_n * rank + r];
                                }
                            }
                        } // n_sub
                    } // i
                } // partition
                double tmp = omp_get_wtime() - wtime_t;
				wtime_apr_phi += tmp;
                modePhiTimes[n] += tmp;
            } // phi_precomp

            // Lines 4-5 in Algorithm 3 of the CP-APR paper
            // Calculate B <- (A^)n) + S) lambda
            //TODO: optimize?
            if(iter > 0) {
                bool any_violation = false;

                #pragma omp parallel for schedule(static) reduction(||:any_violation)
                for(IType i = 0; i < dim * rank; i++) {
                    if((B[i] < kappa_tol) && (Phi_n[i] > 1.0)) {
                        B[i] += kappa;
                        any_violation |= true;
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
				// printf("inner iter: %d\n", inner);
                nInnerIters[iter]++;
                nInnerItersPerMode[n]++;

                wtime_t = omp_get_wtime();
                // Line 8 in Algorithm 3 of the CP-APR paper
                // Calculate Phi^(n)
                // CalculatePhi_OTF(Phi_n, AT, M, n, eps_div_zero);
				// CalculatePhi_alto_OTF_base(Phi_n, AT, M, n, eps_div_zero, ofibs);
                CalculatePhi_alto_par(Phi_n, AT, M, n, eps_div_zero, ofibs, rank, usePrecomp);

                double tmp = omp_get_wtime() - wtime_t;
				wtime_apr_phi += tmp;
                modePhiTimes[n] += tmp;
                // Line 9-11 in Algorithm 3 of the CP-APR paper
                // Check for convergence
                FType kkt_violation = 0.0;
                #pragma omp parallel for schedule(static) reduction(max:kkt_violation)
                for(IType i = 0; i < dim * rank; i++) {
                    FType min_comp = fabs(fmin(B[i], 1.0 - Phi_n[i]));
                    kkt_violation = fmax(kkt_violation, min_comp);
                }
                kktModeViolations[n] = kkt_violation;
                // calculate kkt_violation
                if ((kkt_violation < epsilon) && !dobench) {
                    break;
                } else {
                    converged = false;
                }

                // Line 13 in Algorithm 3 of the CP-APR paper
                // Do the multiplicative update
                #pragma omp parallel for simd schedule(static)
                for (IType i = 0; i < dim * rank; i++) {
                    B[i] *= Phi_n[i];
                }

                /*
                if (inner % 1 == 0) {
                    fprintf (stdout, "\tMode = %d, Inner iter = %d, ",
                             n, inner);
                    fprintf (stdout, " KKT violation == %.6e\n",
                             kktModeViolations[n]);
                }
                 */
            } // for (int inner = 0; inner < max_inner_iters; inner++)

            // Line 15-16 in Algorithm 3 of the CP-APR paper
            // Shift weights from mode n back to Lambda and update A^(n)
            // Basically, KruskalModelNormalize for only mode n
            KruskalModelNormalizeMode(M, n);
            modeTimes[n] += (omp_get_wtime() - wtime_mod);
        } // for (int n = 0; n < nmodes; n++)


        kktViolations[iter] = kktModeViolations[0];
        for (int n = 1; n < nmodes; n++) {
            kktViolations[iter] = fmax(kktViolations[iter],
                                        kktModeViolations[n]);
        }
        if ((iter % 10) == 0) {
            fprintf (stdout, "Iter %d: Inner its = %d KKT violation = %.6e, ",
                        iter, nInnerIters[iter], kktViolations[iter]);
            fprintf (stdout, "nViolations = %d\n", nViolations[iter]);
        }
        if ((std::abs(kktViolations[iter] - kktViolations[iter-1]) <= epsilon*epsilon) && !dobench) {
            converged = true;
            fprintf (stdout, "Iter %d: Inner its = %d KKT violation = %.6e, ",
                        iter, nInnerIters[iter], kktViolations[iter]);
            fprintf (stdout, "nViolations = %d\n", nViolations[iter]);
            printf("ErrorNorm delta below threshold. Assume convergence\n");
        }


        if (converged) {
            fprintf(stdout, "Exiting since all subproblems reached KKT tol\n");
            break;
        }
        if (iter == max_iters - 1) {
            fprintf (stdout, "Iter %d: Inner its = %d KKT violation = %.6e, ",
                        iter, nInnerIters[iter], kktViolations[iter]);
            fprintf (stdout, "nViolations = %d\n", nViolations[iter]);

        }
    } // for(iter = 0; iter < max_iters; iter++)
    wtime_apr += omp_get_wtime() - wtime_s;

    for (int n = 0; n < nmodes; n++) {
        AlignedFree(Phi[n]);
    }
    AlignedFree(Phi);
    if (needPhiPrecomp) AlignedFree(AT->precomp_row);

    wtime_s = omp_get_wtime();
    // Calculate the Log likelihood fit
    int nTotalInner = 0;
    for (int i = 0; i < iter; i++) {
        nTotalInner += nInnerIters[i];
    }
    FType obj = tt_logLikelihood(AT, M, eps_div_zero);

    // Calculate the Gram matrices of the factor matrices and
    // last mode's MTTKRP
    // Then use cpd_fit to calculate the fit - this should give you the
    // least-squares fit
    FType** tmp_grams = (FType**) AlignedMalloc(nmodes * sizeof(FType*));
    assert(tmp_grams);
    for (int n = 0; n < nmodes; n++) {
        tmp_grams[n] = (FType*) AlignedMalloc(rank * rank * sizeof(FType));
        assert(tmp_grams[n]);
    }
    for (int n = 0; n < nmodes; n++) {
        update_gram(tmp_grams[n], M, n);
    }

    FType* tmp_mttkrp = (FType*) AlignedMalloc(dims[nmodes - 1] * rank *
                                                sizeof(FType));
    assert(tmp_mttkrp);
    for(IType i = 0; i < dims[nmodes - 1] * rank; i++) {
        tmp_mttkrp[i] = 0.0;
    }
    mttkrp_(AT, M, nmodes - 1, tmp_mttkrp);
    // Compute ttnormsq to later compute fit
    FType normAT = 0.0;
    ValType* vals = AT->vals;
    IType nnz = AT->nnz;
    #pragma omp parallel for reduction(+:normAT) schedule(static)
    for (IType i = 0; i < nnz; ++i) {
        normAT += (FType)vals[i] * (FType)vals[i];
    }
    double fit = cpd_fit_alto(AT, M, tmp_grams, tmp_mttkrp, normAT);
    wtime_post += omp_get_wtime() - wtime_s;

    fprintf(stdout, "  Final log-likelihood = %e\n", obj);
    fprintf(stdout, "  Final least squares fit = %e\n", fit);
    fprintf(stdout, "  Final KKT violation = %7.7e\n", kktViolations[iter-1]);
    fprintf(stdout, "  Total outer iterations = %d\n", iter);
    fprintf(stdout, "  Total inner iterations = %d\n", nTotalInner);

    fprintf(stdout, "CP-APR pre-processing time  = %f (s)\n", wtime_pre);
    fprintf(stdout, "CP-APR main iteration time  = %f (s)\n", wtime_apr);
    for (int i=0; i<nmodes; ++i)
        fprintf(stdout, "\tTime Mode %d = %f (s)\n", i, modeTimes[i]);
    fprintf(stdout, "CP-APR Phi accumulated time = %f (s)\n", wtime_apr_phi);
    for (int i=0; i<nmodes; ++i)
        fprintf(stdout, "\tPhi Time Mode %d = %f (s)\n", i, modePhiTimes[i]);
    fprintf(stdout, "CP-APR post-processing time = %f (s)\n", wtime_post);
    fprintf(stdout, "CP-APR TOTAL time           = %f (s)\n",
            wtime_pre + wtime_apr + wtime_post);
    fprintf(stdout, "Time per iteration          = %f (s)\n\n",
            (wtime_pre + wtime_apr + wtime_post)/nTotalInner);

    fprintf(stdout, "outer-its  inner-its  its-mode1  its-mode2  its-mode3  its-mode4  its-mode5  ");
    fprintf(stdout, "LL           LSF         KKT         max-o-its  max-i-its  mem-mgmt\n");
    fprintf(stdout, "%-7d    %-7d    %-7d    %-7d    %-7d    %-7d    %-7d    %07.4e  %07.4e  %07.4e  %-7d    %-7d    %-7s\n",
            iter, nTotalInner, nInnerItersPerMode[0], nInnerItersPerMode[1],
            nInnerItersPerMode[2], nmodes > 3 ? nInnerItersPerMode[3] : 0, nmodes > 4 ? nInnerItersPerMode[4] : 0,
            obj, fit, kktViolations[iter-1], max_iters, max_inner_iters,
            needPhiPrecomp ? "pre" : needPhiPrecomp ? "dyn" : "otf");
    fprintf(stdout, "Total    Pre-proc  CP-APR   Phi      Post-proc  mode1    mode2    ");
    fprintf(stdout, "mode3    mode4    mode5    time-per-it\n");
    fprintf(stdout, "%07.4f  %07.4f   %07.4f  %07.4f  %07.4f    %07.4f  %07.4f  %07.4f  %07.4f  %07.4f  %07.4f\n",
            wtime_pre + wtime_apr + wtime_post, wtime_pre, wtime_apr, wtime_apr_phi, wtime_post,
            modeTimes[0], nmodes > 1 ? modeTimes[1] : 0, nmodes > 2 ? modeTimes[2] : 0, nmodes > 3 ? modeTimes[3] : 0,
            nmodes > 4 ? modeTimes[4] : 0, (wtime_pre + wtime_apr + wtime_post)/nTotalInner);

    // for (int n = 0; n < nmodes; n++) {
    //     AlignedFree(Phi[n]);
    // }
    // AlignedFree(Phi);

    // if (needPhiPrecomp) AlignedFree(AT->precomp_row);
}

#endif // APR_HPP_

