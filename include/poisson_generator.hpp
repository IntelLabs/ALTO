#ifndef POISSON_GENERATOR_HPP_
#define POISSON_GENERATOR_HPP_


#include "common.hpp"
#include "sptensor.hpp"
#include "kruskal_model.hpp"


typedef struct PoissonGenerator_
{
    int mode;
    IType *dims;
} PoissonGenerator;


void CreatePoissonGenerator(int mode, IType *dims, PoissonGenerator **pg_);

void DestroyPoissonGenerator(PoissonGenerator *pg);

void PoissonGeneratorRun(PoissonGenerator *pg, IType num_edges, IType rank,
                         KruskalModel **M_, SparseTensor **X_);


#endif // POISSON_GENERATOR_HPP_
