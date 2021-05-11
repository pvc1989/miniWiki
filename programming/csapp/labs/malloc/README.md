---
title: Malloc Lab
---

⚠️ The [*handout*](http://csapp.cs.cmu.edu/3e/malloclab-handout.tar) from the [*Student Site*](http://csapp.cs.cmu.edu/3e/labs.html) does not provide enough trace files. Fortunately, there is [*an updated version*](https://www.cs.cmu.edu/afs/cs/academic/class/15213-f10/www/labs/malloclab-handout.tar) available from the [*Fall 2010 Course Site*](http://www.cs.cmu.edu/afs/cs/academic/class/15213-f10/www/assignments.html).

# Build & Run

```shell
# Get the final score (might be slow):
make clean && make && ./mdriver -V
# Run a particular tracefile (12 times):
make clean && make && ./mdriver -V -f <tracefile>
# Run a particular tracefile only once:
make clean && make && ./mdriver -V -c <tracefile>
# If only mm.c and/or its dependencies have been modified, run this:
touch mm.c && make && ./mdriver -V -c <tracefile>
```

The `-Werror` option in `$(CFLAGS)` made the compiler treat warnings as errors. To build the project, we need to relax such errors back to warnings. This can be done by adding a single line in the `Makefile`, i.e.

```make
mdriver.o: mdriver.c fsecs.h fcyc.h clock.h memlib.h config.h mm.h driverlib.h
	$(CC) $(CFLAGS) -c $< -Wno-error=unused-result -Wno-error=unused-but-set-variable
```

# Trace File Format

A trace file is an ASCII file. It begins with a 4-line header:

```
<weight>          /* weight for this trace (0 or 1) */
<num_ids>         /* number of request id's */
<num_ops>         /* number of requests (operations) */
<sugg_heapsize>   /* suggested heap size (unused) */
```

The header is followed by `num_ops` text lines. Each line denotes either an allocate `[a]`, reallocate `[r]`, or free `[f]` request.
There is no support for `calloc`.
The `<alloc_id>` is an integer that uniquely identifies an allocate or reallocate request.  

```
a <id> <bytes>  /* ptr_<id> = malloc(<bytes>) */
r <id> <bytes>  /* realloc(ptr_<id>, <bytes>) */ 
f <id>          /* free(ptr_<id>) */
```

# `mm_naive.c`

- A block is allocated by simply incrementing the `brk` pointer.
- Blocks are never coalesced or reused, i.e. do nothing in `free()`.

```
Results for mm malloc:
   valid  util   ops    secs     Kops  trace
 * yes    23%    4805  0.000051 94653 ./traces/amptjp.rep
 * yes    19%    5032  0.000063 80269 ./traces/cccp.rep
 * yes     0%   14400  0.000214 67351 ./traces/coalescing-bal.rep
   yes   100%      15  0.000000 99906 ./traces/corners.rep
 * yes    30%    5683  0.000067 84602 ./traces/cp-decl.rep
 * yes    68%     118  0.000001180750 ./traces/hostname.rep
 * yes    65%   19405  0.000116167961 ./traces/login.rep
 * yes    75%     372  0.000002149859 ./traces/ls.rep
   yes    77%      17  0.000000138960 ./traces/malloc-free.rep
   yes    94%      10  0.000000 77513 ./traces/malloc.rep
 * yes    71%    1494  0.000020 75126 ./traces/perl.rep
 * yes    36%    4800  0.000043111876 ./traces/random.rep
 * yes    83%     147  0.000001138840 ./traces/rm.rep
   yes   100%      12  0.000000 33509 ./traces/short2.rep
 * yes    44%   57716  0.000310186211 ./traces/boat.rep
 * yes    25%     200  0.000001168223 ./traces/lrucd.rep
 * yes     0%  100000  0.001144 87405 ./traces/alaska.rep
 * yes    34%     200  0.000001235381 ./traces/nlydf.rep
 * yes    32%     200  0.000001206228 ./traces/qyqyc.rep
 * yes    28%     200  0.000001227634 ./traces/rulsr.rep
16        40%  214772  0.002035105545

Perf index = 0 (util) + 37 (thru) = 37/100
```

# `mm_implicit.c`

## First Fit: 40 (util) + 0 (thru)

```
Results for mm malloc:
   valid  util   ops    secs     Kops  trace
 * yes    99%    4805  0.012170   395 ./traces/amptjp.rep
 * yes    99%    5032  0.010735   469 ./traces/cccp.rep
 * yes    66%   14400  0.000110130843 ./traces/coalescing-bal.rep
   yes    96%      15  0.000000 46831 ./traces/corners.rep
 * yes    99%    5683  0.020750   274 ./traces/cp-decl.rep
 * yes    75%     118  0.000012  9967 ./traces/hostname.rep
 * yes    90%   19405  0.271233    72 ./traces/login.rep
 * yes    88%     372  0.000071  5263 ./traces/ls.rep
   yes    28%      17  0.000000 86359 ./traces/malloc-free.rep
   yes    34%      10  0.000000 48342 ./traces/malloc.rep
 * yes    86%    1494  0.001550   964 ./traces/perl.rep
 * yes    92%    4800  0.008296   579 ./traces/random.rep
 * yes    79%     147  0.000017  8858 ./traces/rm.rep
   yes    89%      12  0.000000 42148 ./traces/short2.rep
 * yes    56%   57716  2.628879    22 ./traces/boat.rep
 * yes    63%     200  0.000004 46040 ./traces/lrucd.rep
 * yes    86%  100000  0.005283 18928 ./traces/alaska.rep
 * yes    89%     200  0.000007 27735 ./traces/nlydf.rep
 * yes    57%     200  0.000005 38156 ./traces/qyqyc.rep
 * yes    68%     200  0.000004 53665 ./traces/rulsr.rep
16        81%  214772  2.959125    73

Perf index = 40 (util) + 0 (thru) = 40/100
```

## Next Fit: 34 (util) + 6 (thru)

```
Results for mm malloc:
   valid  util   ops    secs     Kops  trace
 * yes    90%    4805  0.003272  1469 ./traces/amptjp.rep
 * yes    92%    5032  0.002005  2509 ./traces/cccp.rep
 * yes    66%   14400  0.000118121641 ./traces/coalescing-bal.rep
   yes    96%      15  0.000000 60481 ./traces/corners.rep
 * yes    94%    5683  0.007933   716 ./traces/cp-decl.rep
 * yes    75%     118  0.000002 77276 ./traces/hostname.rep
 * yes    90%   19405  0.006772  2866 ./traces/login.rep
 * yes    88%     372  0.000006 65355 ./traces/ls.rep
   yes    28%      17  0.000000 61636 ./traces/malloc-free.rep
   yes    34%      10 -0.000054  -186 ./traces/malloc.rep
 * yes    81%    1494  0.000076 19741 ./traces/perl.rep
 * yes    90%    4800  0.005117   938 ./traces/random.rep
 * yes    79%     147  0.000003 46167 ./traces/rm.rep
   yes    89%      12  0.000000 33509 ./traces/short2.rep
 * yes    56%   57716  0.024308  2374 ./traces/boat.rep
 * yes    63%     200  0.000003 77147 ./traces/lrucd.rep
 * yes    77%  100000  0.002351 42534 ./traces/alaska.rep
 * yes    76%     200  0.000004 51944 ./traces/nlydf.rep
 * yes    57%     200  0.000003 73822 ./traces/qyqyc.rep
 * yes    68%     200  0.000003 68795 ./traces/rulsr.rep
16        78%  214772  0.051974  4132

Perf index = 34 (util) + 6 (thru) = 40/100
```

# `mm_explicit.c`

## First Fit: 32 (util) + 37 (thru)

```
Results for mm malloc:
   valid  util   ops    secs     Kops  trace
 * yes    89%    4805  0.000167 28700 ./traces/amptjp.rep
 * yes    92%    5032  0.000114 44019 ./traces/cccp.rep
 * yes    67%   14400  0.000125115161 ./traces/coalescing-bal.rep
   yes    94%      15  0.000000 43790 ./traces/corners.rep
 * yes    94%    5683  0.000261 21743 ./traces/cp-decl.rep
 * yes    75%     118  0.000001 99066 ./traces/hostname.rep
 * yes    88%   19405  0.000216 90016 ./traces/login.rep
 * yes    76%     372  0.000005 76787 ./traces/ls.rep
   yes    28%      17  0.000000 63957 ./traces/malloc-free.rep
   yes    34%      10  0.000000 68118 ./traces/malloc.rep
 * yes    81%    1494  0.000016 91321 ./traces/perl.rep
 * yes    88%    4800  0.000382 12562 ./traces/random.rep
 * yes    79%     147  0.000002 74549 ./traces/rm.rep
   yes    89%      12  0.000000 58323 ./traces/short2.rep
 * yes    56%   57716  0.000813 70948 ./traces/boat.rep
 * yes    63%     200  0.000002113386 ./traces/lrucd.rep
 * yes    77%  100000  0.002149 46533 ./traces/alaska.rep
 * yes    76%     200  0.000003 76199 ./traces/nlydf.rep
 * yes    57%     200  0.000002111075 ./traces/qyqyc.rep
 * yes    68%     200  0.000002115055 ./traces/rulsr.rep
16        77%  214772  0.004261 50409

Perf index = 32 (util) + 37 (thru) = 69/100
```

# `mm_segregated.c`

