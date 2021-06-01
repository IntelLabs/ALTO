#ifndef RNG_HPP_
#define RNG_HPP_

#include "common.hpp"
#include <assert.h>


#define RNG_NN 312

typedef struct RNG_ {
    uint64_t mt[RNG_NN];
    int mti;
} RNG;


void rng_seed(RNG *rng, uint64_t seed);

uint64_t rng_rand64(RNG *rng);

#define rng_randfp64(rng) (((rng_rand64(rng)>>12)+0.5)*(1.0/4503599627370496.0))


#endif // RNG_HPP_
