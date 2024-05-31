#ifndef RNG_STREAM_HPP_
#define RNG_STREAM_HPP_


#include "common.hpp"
#include "rng.hpp"

void CreateRNGStream(uint64_t seed, void **stream);

void DestroyRNGStream(void *stream);

template <typename FType>
void RNGStreamUniform(void * stream, double low, double high,
                      size_t len_v, FType *v);

void RNGStream64(void * stream, size_t len_v, uint64_t *v);

template <typename FType>
void RNGStreamUniform(void * stream, double low, double high,
                      size_t len_v, FType *v)
{
	RNG *rng = (RNG *)stream;
    for (size_t i = 0; i < len_v; i++) {
        v[i] = (FType)rng_randfp64(rng) * (high - low) + low;
    }
}

#endif // RNG_STREAM_HPP_
