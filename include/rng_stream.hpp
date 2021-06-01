#ifndef RNG_STREAM_HPP_
#define RNG_STREAM_HPP_


#include "common.hpp"


void CreateRNGStream(uint64_t seed, void **stream);

void DestroyRNGStream(void *stream);

void RNGStreamUniform(void * stream, double low, double high,
                      size_t len_v, double *v);

void RNGStream64(void * stream, size_t len_v, uint64_t *v);


#endif // RNG_STREAM_HPP_
