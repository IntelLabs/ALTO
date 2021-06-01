#ifndef BITOPS_HPP_
#define BITOPS_HPP_

#if defined(__BMI2__)
#include <immintrin.h>
#endif

#ifndef ALT_PEXT
static inline unsigned long long //__attribute__((target("bmi2")))
pdep(unsigned long long x, unsigned long long y)
{ return _pdep_u64(x, y); }

static inline unsigned long long //__attribute__((target("bmi2")))
pext(unsigned long long x, unsigned long long y)
{ return _pext_u64(x, y); }

#else
static inline unsigned long long //__attribute__((target("default")))
pdep(unsigned long long x, unsigned long long y)
{
    unsigned long long res = 0;
    for (unsigned long long b=1; y!=0; b+=b) {
        if (x & b) {
            res |= y & (-y);
        }
        y &= (y - 1);
    }
    return res;
}

static inline unsigned long long //__attribute__((target("default")))
pext(unsigned long long x, unsigned long long y)
{
    unsigned long long res = 0;
    for (unsigned long long b=1; y!=0; b+=b) {
        if (x & y & -y) {
            res |= b;
        }
        y &= (y - 1);
    }
    return res;
}

static inline unsigned long long
pext(unsigned long long x, unsigned long long y, int pos)
{
    return (( x >> pos) & y);
}
#endif

static inline int
popcount(unsigned long long x)
{ return __builtin_popcountll(x); }

static inline int
clz(unsigned long long x)
{ return __builtin_clzll(x); }

#ifndef ALT_PEXT
static inline unsigned __int128 //__attribute__((target("bmi2")))
pdep(IType x, unsigned __int128 y)
{
  unsigned long long ylow = y & 0xffffffffffffffff;

  int shift = __builtin_popcountll(ylow);
  unsigned __int128 res = _pdep_u64(x >> shift, y >> 64);
  res = _pdep_u64(x, ylow) | (res << 64);
  return res;
}

static inline IType //__attribute__((target("bmi2")))
pext(unsigned __int128 x, unsigned __int128 y)
{
  unsigned long long ylow = y & 0xffffffffffffffff;
  unsigned long long xlow = x & 0xffffffffffffffff;

  int shift = __builtin_popcountll(ylow);
  return (_pext_u64(x >> 64, y >> 64) << shift) | _pext_u64(xlow, ylow);
}
#else
static inline unsigned __int128 //__attribute__((target("default")))
pdep(IType x, unsigned __int128 y)
{
  unsigned long long ylow = y & 0xffffffffffffffff;

  int shift = __builtin_popcountll(ylow);
  unsigned __int128 res = pdep((unsigned long long) (x >> shift), (unsigned long long) (y >> 64));
  res = pdep(x, ylow) | (res << 64);
  return res;
}

static inline IType //__attribute__((target("default")))
pext(unsigned __int128 x, unsigned __int128 y)
{
  unsigned long long ylow = y & 0xffffffffffffffff;
  unsigned long long xlow = x & 0xffffffffffffffff;

  int shift = __builtin_popcountll(ylow);
  return (pext((unsigned long long) (x >> 64), (unsigned long long) (y >> 64)) << shift) | pext(xlow, ylow);
}

static inline IType
pext(unsigned __int128 x, unsigned __int128 y, int pos)
{
    return (( x >> pos) & y);
}
#endif

static inline int
popcount(unsigned __int128 x)
{ return __builtin_popcountll(x >> 64) + __builtin_popcountll(x & 0xffffffffffffffff); }

static inline int
clz(unsigned __int128 x)
{
  unsigned long long xhi = x >> 64;
  if (!xhi)
    return 64 + __builtin_clzll(x & 0xffffffffffffffff);
  return __builtin_clzll(xhi);
}

#endif
