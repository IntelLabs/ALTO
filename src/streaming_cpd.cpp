#include "streaming_cpd.hpp"

StreamingCPD::StreamingCPD(
    int rank,
    int nmodes,
    int max_iterations
) : _rank(rank), 
    _nmodes(nmodes),
    _max_iterations(max_iterations)
{
    printf("=== StreamingCPD initialized ===\n");
};

KruskalModel * StreamingCPD::compute(
    float forgetting_factor
) {
    int max_iterations = 100;
    for (int outer = 0; outer < max_iterations; ++outer) {
        // First solve for time mode
        // Solve for MTTKRP
        // mttkrp();
        // Multiply with GRAM matrix

    }

    // Solve for other modes

    return NULL;
};
void StreamingCPD::init()
{
    // Initialize everything needed for Streaming CPD
    _colnorms = (FType *) AlignedMalloc(sizeof(FType) * _rank);

    _global_time = new StreamMatrix(_rank);
    _mttkrp_buf = new StreamMatrix(_rank);

    for (int n = 0; n < _nmodes; ++n) {
        _factor_matrices[n] = new StreamMatrix(_rank);
        _prev_factor_matrices[n] = new StreamMatrix(_rank);
    };
};

void StreamingCPD::preprocess(SparseTensor * st, IType stream_mode) {
    int const nmodes = st->nmodes;
    IType const * dims = st->dims;
    IType const nnz = st->nnz;


    // for (int m = 0; m < nmodes; ++m) {
    //     if (m == stream_mode) {
    //         // Increase the size of global time matrix
    //         _global_time->grow_zero(_global_time->num_rows() + 1);
    //     } else {    
    //         printf("Increasing factor matrix mode- %d by %d", m, _factor_matrices[m]->num_rows());
    //         _factor_matrices[m]->grow_rand(st->dims[m]);
    //         _prev_factor_matrices[m]->grow_zero(st->dims[m]);        
    //         printf("to %d\n", _factor_matrices[m]->num_rows());            
    //     }
    // }
    // TODO: Compute aTa here as well? 
    return;
};

void StreamingCPD::update() {
    return;
};

void StreamingCPD::init_factor_matrices(

) {
    // For streaming mode

    // For non streaming modes
    
    return;
};