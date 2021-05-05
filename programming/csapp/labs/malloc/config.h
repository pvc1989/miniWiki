#ifndef __CONFIG_H_
#define __CONFIG_H_

/*
 * config.h - malloc lab configuration file
 *
 * Copyright (c) 2002, R. Bryant and D. O'Hallaron, All rights reserved.
 * May not be used, modified, or copied without permission.
 */

/*
 * Config variables for submitting autoresults to the Autolab server
 * Note: The COURSE and LAB variables must match the directory names
 * on the Autolab server.
 */

/*
 * This is the default path where the driver will look for the
 * default tracefiles. You can override it at runtime with the -t flag.
 */
#define TRACEDIR "./traces/"

/*
 * This is the list of default tracefiles in TRACEDIR that the driver
 * will use for testing. Modify this if you want to add or delete
 * traces from the driver's test suite.
 *
 * The first four test correctness.  The last several test utilization
 * and performance.
 */
#define DEFAULT_TRACEFILES \
	"amptjp.rep", \
	"cccp.rep", \
	"coalescing-bal.rep", \
	"corners.rep", \
	"cp-decl.rep", \
	"hostname.rep", \
	"login.rep", \
	"ls.rep", \
	"malloc-free.rep", \
	"malloc.rep", \
	"perl.rep", \
	"random.rep", \
	"rm.rep", \
	"short2.rep", \
	"boat.rep",\
	"lrucd.rep", \
	"alaska.rep", \
	"nlydf.rep", \
	"qyqyc.rep", \
	"rulsr.rep"



/*
 * Students get 0 points for this point or below (ops / sec)
 */
#define MIN_SPEED       0E3


/*
 * Students get 0 points for this allocation fraction or below
 */
#define MIN_SPACE       0.60



/* 
 * Students can get more points for building faster allocators, up to
 * this point (in ops / sec)
 */
#define MAX_SPEED       25000E3

/* 
 * Students can get more points for building more efficient allocators,
 * up to this point (1 is perfect).
 */
#define MAX_SPACE       0.93

 /*
  * This constant determines the contributions of space utilization
  * (UTIL_WEIGHT) and throughput (1 - UTIL_WEIGHT) to the performance
  * index.
  */

#define UTIL_WEIGHT .63

/*
 * Alignment requirement in bytes (either 4 or 8)
 */
#define ALIGNMENT 8

/*
 * Maximum heap size in bytes
 */
#define MAX_HEAP (100*(1<<20))  /* 100 MB */

/*****************************************************************************
 * Set exactly one of these USE_xxx constants to "1" to select a timing method
 *****************************************************************************/
#define USE_FCYC   1   /* cycle counter w/K-best scheme (x86 & Alpha only) */
#define USE_ITIMER 0   /* interval timer (any Unix box) */
#define USE_GETTOD 0   /* gettimeofday (any Unix box) */

#endif /* __CONFIG_H */
