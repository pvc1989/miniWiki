---
title: Malloc Lab
---

⚠️ The [*handout*](http://csapp.cs.cmu.edu/3e/malloclab-handout.tar) from the [*Student Site*](http://csapp.cs.cmu.edu/3e/labs.html) does not provide enough trace files. Fortunately, there is [*an updated version*](https://www.cs.cmu.edu/afs/cs/academic/class/15213-f10/www/labs/malloclab-handout.tar) available from the [*Fall 2010 Course Site*](http://www.cs.cmu.edu/afs/cs/academic/class/15213-f10/www/assignments.html).

# Build & Run

```shell
make && ./mdriver -V
make && ./mdriver -f <tracefile> # Run a particular tracefile (12 times).
make && ./mdriver -c <tracefile> # Run a particular tracefile only once.
```

The `-Werror` option in `$(CFLAGS)` made the compiler treat warnings as errors. To build the project, we need to relax such errors back to warnings. This can be done by adding a single line in the `Makefile`, i.e.

```make
mdriver.o: mdriver.c fsecs.h fcyc.h clock.h memlib.h config.h mm.h driverlib.h
	$(CC) $(CFLAGS) -c $< -Wno-error=unused-result -Wno-error=unused-but-set-variable
```

# `mm-naive.c`

- A block is allocated by simply incrementing the `brk` pointer.
- Blocks are never coalesced or reused, i.e. do nothing in `free()`.

# `mm-implicit.c`

# `mm-explicit.c`

# `mm-segregated.c`

