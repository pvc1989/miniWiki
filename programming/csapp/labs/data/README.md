---
title: Data Lab
---

# Resources

- [`datalab.pdf`](http://csapp.cs.cmu.edu/3e/datalab.pdf)
- [`datalab-handout.tar`](http://csapp.cs.cmu.edu/3e/datalab-handout.tar)
  1. Run `tar xvf datalab-handout.tar` to unpack the project.
  1. Modify `bits.c` to solve the puzzles.
  1. Run `make` to build the project.
     - Install `gcc-multilib` (once only) if `fatal error: bits/libc-header-start.h: No such file or directory` occurs.

## Helpers

To make the working directory clean, it is recommended to add `btest`, `fshow` and `ishow` into a `.gitignore` file.

### `tests.c`

`tests.c` expresses the correct behavior of your functions.

### `fshow`, `ishow`

`fshow` (built from `fshow.c`) helps you understand the structure of floating point numbers.

### `btest`

`btest` (built from `btest.c`) checks the functional correctness of the functions in `bits.c`.

```shell
./btest -f bitXor  # tests only a single function.
```

### `dlc`

The executable `dlc` is an ANSI C compiler that you can use to check for compliance with the coding rules for each puzzle.

```shell
./dlc bits.c  # returns silently if there are no problems with your code.
./dlc -e bits.c  # prints the number of operators used by each function.
```

### `driver.pl`

The Perl script `driver.pl` is a driver program that uses `btest` and `dlc` to compute the correctness and performance points for your solution.

```shell
## Grading:
./driver.pl
## Install required modules (once only):
apt install cpanminus  # if `cpan` is not found
cpan App::cpanminus  # recommended in CPAN docs
cpanm Getopt::Std  # redundant, use as a check
```

# Hints

```c
(x == y) ==   (!(x ^ y)) ;
(x != y) == (!(!(x ^ y)));
(-x) == (~x + 1);
```

## `bitXor`

## `tmin`

The minimum two's complement (32-bit) integer is

```shell
./ishow 0x80000000
Hex = 0x80000000,       Signed = -2147483648,   Unsigned = 2147483648
```

## `isTmax`

The maximum two's complement (32-bit) integer is

```shell
./ishow 0x7FFFFFFF
Hex = 0x7fffffff,       Signed = 2147483647,    Unsigned = 2147483647
```

## `allOddBits`

The binary representation of `0xAA` is `10101010`.

## `negate`

`x + y == 0` implies `x + (y - 1) == 0xFFFFFFFF`.

## `isAsciiDigit`

```c
  0x30 <= x && x <= 0x39
== (x & 0xFFFFFF00 == 0)
&& (x & 0xF0 == 0x30)
&& (x & 0xF + 6 < 16);
```

## `conditional`

```c
int x_is_0 = !x ;  /* x == 0 ? 0x00000001 : 0x00000000 */
x_is_0 = ~x_is_0;  /* x == 0 ? 0xFFFFFFFE : 0xFFFFFFFF */
x_is_0 += 1;       /* x == 0 ? 0xFFFFFFFF : 0x00000000 */
```

## `isLessOrEqual`

```c
  x <= y
== (x == 1 << 31)
|| (x < 0 && 0 <= y)
|| (!((x ^ y) & (1 << 31)) && 0 <= y - x)
```

## `logicalNeg`

Compress all bits to the least significant one.

## `howManyBits`

If `x < 0`, find the first `0`. Else, find the first `0` in `~x`.
