#include "rng.hpp"

void CreateRNGStream(uint64_t seed, void **stream)
{
    RNG *rng = (RNG *)AlignedMalloc(sizeof(RNG));
    assert(rng != NULL);
    rng_seed(rng, seed);
    *stream = (void *)rng;
}


void DestroyRNGStream(void *stream)
{
    RNG *rng = (RNG *)stream;
    AlignedFree(rng);
}


void RNGStreamUniform(void * stream, double low, double high,
                      size_t len_v, double *v)
{
    RNG *rng = (RNG *)stream;
    for (size_t i = 0; i < len_v; i++) {
        v[i] = rng_randfp64(rng) * (high - low) + low;
    }
}


void RNGStream64(void * stream, size_t len_v, uint64_t *v)
{
    RNG *rng = (RNG *)stream;
    for (size_t i = 0; i < len_v; i++) {
        v[i] = rng_rand64(rng);
    }
}
