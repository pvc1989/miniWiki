#define NDEBUG
// #define NEXT_FIT

// #define NAIVE
#ifdef NAIVE
#include "mm_naive.c"
#endif

// #define IMPLICIT
#ifdef IMPLICIT
#include "mm_implicit.c"
#endif

#define EXPLICIT
#ifdef EXPLICIT
#include "mm_explicit.c"
#endif
