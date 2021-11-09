#include "common.hpp"

static double g_ticks_persecond = 0.0;

__attribute__ ((noinline))
     void ASMTrace(char *str)
{
    fprintf(stderr, "HERE %s.\n", str);
}


void InitTSC(void)
{
    uint64_t start_tick = ReadTSC();
    sleep(1);
    uint64_t end_tick = ReadTSC();

    g_ticks_persecond = (double) (end_tick - start_tick);
}


double ElapsedTime(uint64_t ticks)
{
    if (g_ticks_persecond == 0.0) {
        fprintf(stderr, "TSC timer has not been initialized.\n");
        return 0.0;
    }
    else {
        return (ticks / g_ticks_persecond);
    }
}

void ELAPSED_TIME(
  uint64_t start,
  uint64_t end,
  double* t_elapsed
)
{
#if TIME
  if (g_ticks_persecond == 0.0) {
    fprintf(stderr, "TSC timer has not been initialized.\n");
  } else {
    *t_elapsed = 0.0;
    *t_elapsed = ((end - start) / g_ticks_persecond);
  }
#endif
}

void AGG_ELAPSED_TIME(
  uint64_t start,
  uint64_t end,
  double* t_elapsed
)
{
#if TIME
  if (g_ticks_persecond == 0.0) {
    fprintf(stderr, "TSC timer has not been initialized.\n");
  } else {
    *t_elapsed += ((end - start) / g_ticks_persecond);
  }
#endif
}

void PRINT_TIMER(
  const char* message,
  double t
)
{
#if TIME
  printf("%s: %f s\n", message, t);
#endif
}