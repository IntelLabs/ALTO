#ifndef STREAMING_CPD_HPP_
#define STREAMING_CPD_HPP_

#include "common.hpp"
#include "streaming_sptensor.hpp"
#include "stream_matrix.hpp"
#include "kruskal_model.hpp"

#include "util.hpp"

class StreamingCPD {
    public:
        StreamingCPD(
            int rank,
            int nmodes,
            int max_iterations
        );
        ~StreamingCPD() {};

        KruskalModel * compute(float forgetting_factor);
        void init_factor_matrices();
        void init();
        void preprocess(SparseTensor * st, IType stream_mode);
        void update();
        void grow_factor_matrices(IType * new_dim_sizes);
        void compute_err(); // Compute error between factor matrices and orig. tensor

    private:
        SparseTensor * _st;
        int _rank;
        int _nmodes;
        int _max_iterations;

        FType * _colnorms;

        StreamMatrix * _global_time;
        StreamMatrix * _mttkrp_buf;
        StreamMatrix * _factor_matrices[MAX_NUM_MODES]; // A_n for all modes
        StreamMatrix * _prev_factor_matrices[MAX_NUM_MODES]; // A_n,t-1 for all modes

        Matrix ** mats_aTa;
        Matrix * gram;
};

#endif