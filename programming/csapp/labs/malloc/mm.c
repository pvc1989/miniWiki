// #define DEBUG

// #define NAIVE
#ifdef NAIVE
#include "mm_naive.c"
#endif

#define IMPLICIT
#ifdef IMPLICIT
#include "mm_implicit.c"
#endif
