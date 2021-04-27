---
title: Cache Lab
---

# Part A: Writing a Cache Simulator

## `cache_t`

```c
typedef struct {
  long tag;
  int valid, rank;
} line_t;

typedef struct {
  line_t** sets;
  long set_mask, tag_mask;
  int n_sets, n_lines, block_size;
  int n_hits, n_misses, n_evictions;
} cache_t;
```

## `getopt()`

```c
/*
  The following trivial example program uses getopt() to handle two  pro-
  gram  options:  -n, with no associated value; and -t val, which expects
  an associated value.
 */
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  int flags, opt;
  int nsecs, tfnd;

  nsecs = 0;
  tfnd = 0;
  flags = 0;
  while ((opt = getopt(argc, argv, "nt:")) != -1) {
    switch (opt) {
      case 'n':
        flags = 1;
        break;
      case 't':
        nsecs = atoi(optarg);
        tfnd = 1;
        break;
      default: /* '?' */
        fprintf(stderr, "Usage: %s [-t nsecs] [-n] name\n",
                argv[0]);
        exit(EXIT_FAILURE);
    }
  }

  printf("flags=%d; tfnd=%d; nsecs=%d; optind=%d\n",
         flags, tfnd, nsecs, optind);

  if (optind >= argc) {
    fprintf(stderr, "Expected argument after options\n");
    exit(EXIT_FAILURE);
  }

  printf("name argument = %s\n", argv[optind]);

  /* Other code omitted */

  exit(EXIT_SUCCESS);
}
```

See `man 3 getopt` for details.

# Part B: Optimizing Matrix Transpose

## $32\times32$

```c
void trans_32x32(int M, int N, int A[M][N], int B[M][N]);
/*
  &A[0][0] == 0x10c0a0
           == 0001 0000 1100 0000 1010 0000
              \______tag______/\_s__/\_b__/
  &A[i][j] == 0x10c0a0 + i * 32 * 4 + j * 4
           == 0x10c0a0 + i * 0x80 + j * 0x4
  &B[0][0] == 0x14c0a0
           == 0001 0100 1100 0000 1010 0000
              \______tag______/\_s__/\_b__/
 */
```

|   Set ID   | `0 <= col < 8` | `8 <= col < 16` | `16 <= col < 24` | `24 <= col < 32` |
| :--------: | :------------: | :-------------: | :--------------: | :--------------: |
| `row == 0` |      `05`      |      `06`       |       `07`       |       `08`       |
| `row == 1` |      `09`      |      `10`       |       `11`       |       `12`       |
| `row == 2` |      `13`      |      `14`       |       `15`       |       `16`       |
| `row == 3` |      `17`      |      `18`       |       `19`       |       `20`       |
| `row == 4` |      `21`      |      `22`       |       `23`       |       `24`       |
| `row == 5` |      `25`      |      `26`       |       `27`       |       `28`       |
| `row == 6` |      `29`      |      `30`       |       `31`       |       `00`       |
| `row == 7` |      `01`      |      `02`       |       `03`       |       `04`       |
| `row == 8` |      `05`      |      `06`       |       `07`       |       `08`       |
|   `...`    |     `...`      |      `...`      |      `...`       |      `...`       |

## $64\times64$

```c
void trans_64x64(int M, int N, int A[M][N], int B[M][N]);
/*
  &A[0][0] == 0x10d0a0
           == 0001 0000 1101 0000 1010 0000
              \______tag______/\_s__/\_b__/
  &A[i][j] == 0x10d0a0 + i * 64 * 4 + j * 4
           == 0x10d0a0 + i * 0x100 + j * 0x4
  &B[0][0] == 0x14d0a0
           == 0001 0100 1101 0000 1010 0000
              \______tag______/\_s__/\_b__/
 */
```

| `row` \ Set ID \ `col` | `[0, 8)` | `[8, 16)` | `[16, 24)` | `[24, 32)` | `[32, 40)` | `[40, 48)` | `[48, 56)` | `[56, 64)` |
| :--------------------: | :------: | :-------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: |
|          `0`           |   `05`   |   `06`    |    `07`    |    `08`    |    `09`    |    `10`    |    `11`    |    `12`    |
|          `1`           |   `13`   |   `14`    |    `15`    |    `16`    |    `17`    |    `18`    |    `19`    |    `20`    |
|          `2`           |   `21`   |   `22`    |    `23`    |    `24`    |    `25`    |    `26`    |    `27`    |    `28`    |
|          `3`           |   `29`   |   `30`    |    `31`    |    `00`    |    `01`    |    `02`    |    `03`    |    `04`    |
|          `4`           |   `05`   |   `06`    |    `07`    |    `08`    |    `09`    |    `10`    |    `11`    |    `12`    |
|          `5`           |   `13`   |   `14`    |    `15`    |    `16`    |    `17`    |    `18`    |    `19`    |    `20`    |
|          `6`           |   `21`   |   `22`    |    `23`    |    `24`    |    `25`    |    `26`    |    `27`    |    `28`    |
|          `7`           |   `29`   |   `30`    |    `31`    |    `00`    |    `01`    |    `02`    |    `03`    |    `04`    |
|          `8`           |   `05`   |   `06`    |    `07`    |    `08`    |    `09`    |    `10`    |    `11`    |    `12`    |
|         `...`          |  `...`   |   `...`   |   `...`    |   `...`    |   `...`    |   `...`    |   `...`    |   `...`    |

## $61\times67$

```c
void transpose_61x67(int M, int N, int A[N][M], int B[M][N])
{
    int i_min, j_min, k;
    int x0, x1, x2, x3, x4, x5, x6, x7;
    for (j_min = 0; j_min < 56; j_min += 8)
    {
        for (i_min = 0; i_min < 64; i_min += 8)
        {
            for (k = 0; k != 8; ++k) {
                /* use x0, ..., x7 as buffer */
            }
        }
    }
    for (i_min = 0; i_min < 64; ++i_min)
        for (j_min = 56; j_min < 61; ++j_min)
            B[j_min][i_min] = A[i_min][j_min];
    for (j_min = 0; j_min < 56; ++j_min)
        for (i_min = 64; i_min < 67; ++i_min)
            B[j_min][i_min] = A[i_min][j_min];
    for (j_min = 56; j_min < 61; ++j_min)
        for (i_min = 64; i_min < 67; ++i_min)
            B[j_min][i_min] = A[i_min][j_min];
}
```

