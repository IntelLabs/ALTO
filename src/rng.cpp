/* 
   A C-program for MT19937-64 (2004/9/29 version).
   Coded by Takuji Nishimura and Makoto Matsumoto.

   This is a 64-bit version of Mersenne Twister pseudorandom number
   generator.

   Before using, initialize the state by using init_genrand64(seed)  
   or init_by_array64(init_key, key_length).

   Copyright (C) 2004, Makoto Matsumoto and Takuji Nishimura,
   All rights reserved.                          

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.

     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.

     3. The names of its contributors may not be used to endorse or promote 
        products derived from this software without specific prior written 
        permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
   FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
   COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
   INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
   (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
   HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
   STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
   OF THE POSSIBILITY OF SUCH DAMAGE.

   References:
   T. Nishimura, ``Tables of 64-bit Mersenne Twisters''
     ACM Transactions on Modeling and 
     Computer Simulation 10. (2000) 348--357.
   M. Matsumoto and T. Nishimura,
     ``Mersenne Twister: a 623-dimensionally equidistributed
       uniform pseudorandom number generator''
     ACM Transactions on Modeling and 
     Computer Simulation 8. (Jan. 1998) 3--30.

   Any feedback is very welcome.
   http://www.math.hiroshima-u.ac.jp/~m-mat/MT/emt.html
   email: m-mat @ math.sci.hiroshima-u.ac.jp (remove spaces)
*/


#include <stdio.h>
#include "rng.hpp"


#define RNG_MM          156
#define RNG_MATRIX_A    0xB5026F5AA96619E9ULL
#define RNG_UM          0xFFFFFFFF80000000ULL
#define RNG_LM          0x000000007FFFFFFFULL


/* initializes mt[NN] with a seed */
void rng_seed(RNG *rng, uint64_t seed)
{
    rng->mt[0] = seed;
    for (rng->mti = 1; rng->mti < RNG_NN; rng->mti++) {
        rng->mt[rng->mti] =  rng->mti + 6364136223846793005ULL *
            (rng->mt[rng->mti - 1]^(rng->mt[rng->mti - 1] >> 62));
    }
}


/* generates a random number on [0, 2^64-1]-interval */
uint64_t rng_rand64(RNG *rng)
{
    uint64_t x, mag01[2] = {0ULL, RNG_MATRIX_A};
    int i;

    /* generate NN words at one time */
    if (rng->mti >= RNG_NN) {
        for (i = 0; i < RNG_NN - RNG_MM; i++) {
            x = (rng->mt[i]&RNG_UM)|(rng->mt[i + 1]&RNG_LM);
            rng->mt[i] = rng->mt[i +RNG_MM] ^ (x >> 1) ^
                        mag01[(int)(x&1ULL)];
        }
        for (; i < RNG_NN - 1; i++) {
            x = (rng->mt[i]&RNG_UM)|(rng->mt[i + 1]&RNG_LM);
            rng->mt[i] = rng->mt[i + (RNG_MM - RNG_NN)] ^
                    (x >> 1) ^ mag01[(int)(x&1ULL)];
        }
        x = (rng->mt[RNG_NN - 1]&RNG_UM)|(rng->mt[0]&RNG_LM);
        rng->mt[RNG_NN - 1] = rng->mt[RNG_MM - 1] ^ (x >> 1) ^
                        mag01[(int)(x&1ULL)];

        rng->mti = 0;
    }

    x = rng->mt[rng->mti++];
    x ^= (x >> 29) & 0x5555555555555555ULL;
    x ^= (x << 17) & 0x71D67FFFEDA60000ULL;
    x ^= (x << 37) & 0xFFF7EEE000000000ULL;
    x ^= (x >> 43);
 
    return x;
}
