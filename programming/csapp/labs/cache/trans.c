/* 
 * trans.c - Matrix transpose B = A^T
 *
 * Each transpose function must have a prototype of the form:
 * void trans(int M, int N, int A[N][M], int B[M][N]);
 *
 * A transpose function is evaluated by counting the number of misses
 * on a 1KB direct mapped cache with a block size of 32 bytes.
 */ 
#include <stdio.h>
#include "cachelab.h"

int is_transpose(int M, int N, int A[N][M], int B[M][N]);

/* 
 * trans - A simple baseline transpose function, not optimized for the cache.
 */
char trans_desc[] = "Simple row-wise scan transpose";
void trans(int M, int N, int A[N][M], int B[M][N])
{
    int i, j, tmp;

    for (i = 0; i < N; i++) {
        for (j = 0; j < M; j++) {
            tmp = A[i][j];
            B[j][i] = tmp;
        }
    }    

}

/* 
 * You can define additional transpose functions below. We've defined
 * a simple one below to help you get started. 
 */ 

char transpose_32x32_desc[] = "Transpose 32x32";
void transpose_32x32(int M, int N, int A[N][M], int B[M][N])
{
    int i, j, i_min, j_min, stride = 8;
    for (i_min = 0; i_min != M; i_min += stride)
    {
        j_min = i_min;  /* only on diagonal blocks */
        {
            /* buffer them one-block right in B */
            for (i = i_min; i != i_min + stride; ++i)
                for (j = j_min; j != j_min + stride; ++j)
                    B[j][(i + 8) % M] = A[i][j];
            /* shift them back (one-block left) */
            for (i = i_min; i != i_min + stride; ++i)
                for (j = j_min; j != j_min + stride; ++j)
                    B[j][i] = B[j][(i + 8) % M];
        }
    }
    for (i_min = 0; i_min != M; i_min += stride)
    {
        for (j_min = 0; j_min != N; j_min += stride)
        {
            if (i_min == j_min) continue;  /* diagonal blocks already done */
            for (i = i_min; i != i_min + stride; ++i)
                for (j = j_min; j != j_min + stride; ++j)
                    B[j][i] = A[i][j];
        }
    }
}

void copy_1x8(int* src, int* dst)
{
    int x0, x1, x2, x3, x4, x5, x6, x7;
    // (possibly) 1 miss
    x0 = src[0]; x1 = src[1]; x2 = src[2]; x3 = src[3];
    x4 = src[4]; x5 = src[5]; x6 = src[6]; x7 = src[7];
    // (possibly) 1 miss
    dst[0] = x0; dst[1] = x1; dst[2] = x2; dst[3] = x3;
    dst[4] = x4; dst[5] = x5; dst[6] = x6; dst[7] = x7;
}

void swap(int* x, int* y)
{
    int temp = *x;
    *x = *y;
    *y = temp;
}

void transpose_4x4(int* row0, int* row1, int* row2, int* row3)
{
    swap(row0 + 1, row1 + 0);
    swap(row0 + 2, row2 + 0);
    swap(row0 + 3, row3 + 0);
    swap(row1 + 2, row2 + 1);
    swap(row1 + 3, row3 + 1);
    swap(row2 + 3, row3 + 2);
}
void swap_1x4(int* x, int* y) {
    int x0 = x[0], x1 = x[1], x2 = x[2], x3 = x[3];
    int y0 = y[0], y1 = y[1], y2 = y[2], y3 = y[3];  // (possibly) 1 miss
    y[0] = x0, y[1] = x1, y[2] = x2, y[3] = x3;
    x[0] = y0, x[1] = y1, x[2] = y2, x[3] = y3;      // (possibly) 1 miss
}

char transpose_64x64_desc[] = "Transpose 64x64";
void transpose_64x64(int M, int N, int A[N][M], int B[M][N])
{
    int i_min, j_min, k;
    for (j_min = 0; j_min != 64; j_min += 8)
    {
        for (i_min = 0; i_min != 56; i_min += 8)
        {
            // copy A[4:8)[0:8) to B[0:4)[8:16)
            for (k = 0; k != 4; ++k)  // 2 misses / iteration
                copy_1x8(&A[i_min + k + 4][j_min], &B[j_min + k][i_min + 8]);
            // copy A[0:4)[0:8) to B[0:4)[0:8)
            for (k = 0; k != 4; ++k)  // 2 misses / iteration
                copy_1x8(&A[i_min + k][j_min], &B[j_min + k][i_min]);
            // transpose each 4x4 block in B[0:4)[0:16)
            for (k = 0; k != 16; k += 4)  // 0 miss / iteration
                transpose_4x4(
                    &B[j_min + 0][i_min + k],
                    &B[j_min + 1][i_min + k],
                    &B[j_min + 2][i_min + k],
                    &B[j_min + 3][i_min + k]);
            // swap the off-diagonal 4x4 blocks
            for (k = 0; k != 4; ++k)  // 0 miss / iteration
                swap_1x4(&B[j_min + k][i_min + 4], &B[j_min + k][i_min + 8]);
            // copy B[0:4)[8:16) to B[4:8)[0:8)
            for (k = 0; k != 4; ++k)  // 1 miss / iteration
                copy_1x8(&B[j_min + k][i_min + 8], &B[j_min + k + 4][i_min]);
        }
        i_min = 56;
        if (j_min < 56)
        {
            // copy A[4:8)[0:8) to B[0:4)[8:16)
            for (k = 0; k != 4; ++k)  // 2 misses / iteration
                copy_1x8(&A[i_min + k + 4][j_min], &B[j_min + k + 8][0]);
            // copy A[0:4)[0:8) to B[0:4)[0:8)
            for (k = 0; k != 4; ++k)  // 2 misses / iteration
                copy_1x8(&A[i_min + k][j_min], &B[j_min + k][i_min]);
            // transpose each 4x4 block in B[0:4)[0:16)
            for (k = 0; k != 8; k += 4)  // 0 miss / iteration
            {
                transpose_4x4(
                    &B[j_min + 0][i_min + k],
                    &B[j_min + 1][i_min + k],
                    &B[j_min + 2][i_min + k],
                    &B[j_min + 3][i_min + k]);
                transpose_4x4(
                    &B[j_min +  8][k],
                    &B[j_min +  9][k],
                    &B[j_min + 10][k],
                    &B[j_min + 11][k]);
            }
            // swap the off-diagonal 4x4 blocks
            for (k = 0; k != 4; ++k)  // 0 miss / iteration
                swap_1x4(&B[j_min + k][i_min + 4], &B[j_min + k + 8][0]);
            // copy B[0:4)[8:16) to B[4:8)[0:8)
            for (k = 0; k != 4; ++k)  // 1 miss / iteration
                copy_1x8(&B[j_min + k + 8][0], &B[j_min + k + 4][i_min]);
        } else {
            for (k = 0; k != 4; ++k)  // 2 misses / iteration
                copy_1x8(&A[i_min + k][j_min], &B[j_min + k][i_min]);
            transpose_4x4(
                &B[j_min + 0][i_min + 0], &B[j_min + 1][i_min + 0],
                &B[j_min + 2][i_min + 0], &B[j_min + 3][i_min + 0]);
            transpose_4x4(
                &B[j_min + 0][i_min + 4], &B[j_min + 1][i_min + 4],
                &B[j_min + 2][i_min + 4], &B[j_min + 3][i_min + 4]);
            for (k = 4; k != 8; ++k)  // 2 misses / iteration
                copy_1x8(&A[i_min + k][j_min], &B[j_min + k][i_min]);
            transpose_4x4(
                &B[j_min + 4][i_min + 0], &B[j_min + 5][i_min + 0],
                &B[j_min + 6][i_min + 0], &B[j_min + 7][i_min + 0]);
            transpose_4x4(
                &B[j_min + 4][i_min + 4], &B[j_min + 5][i_min + 4],
                &B[j_min + 6][i_min + 4], &B[j_min + 7][i_min + 4]);
            for (k = 0; k != 4; ++k)  // 2 misses / iteration
                swap_1x4(&B[j_min + k + 4][i_min], &B[j_min + k][i_min + 4]);
        }
    }
}

/* 
 * transpose_submit - This is the solution transpose function that you
 *     will be graded on for Part B of the assignment. Do not change
 *     the description string "Transpose submission", as the driver
 *     searches for that string to identify the transpose function to
 *     be graded. 
 */
char transpose_submit_desc[] = "Transpose submission";
void transpose_submit(int M, int N, int A[N][M], int B[M][N])
{
    switch (M)
    {
    case 32:
      transpose_32x32(M, N, A, B);
      break;
    case 64:
      transpose_64x64(M, N, A, B);
      break;
    default:
      trans(M, N, A, B);
    }
}

/*
 * registerFunctions - This function registers your transpose
 *     functions with the driver.  At runtime, the driver will
 *     evaluate each of the registered functions and summarize their
 *     performance. This is a handy way to experiment with different
 *     transpose strategies.
 */
void registerFunctions()
{
    /* Register your solution function */
    registerTransFunction(transpose_submit, transpose_submit_desc); 

    /* Register any additional transpose functions */
    registerTransFunction(trans, trans_desc); 
    registerTransFunction(transpose_32x32, transpose_32x32_desc);
    registerTransFunction(transpose_64x64, transpose_64x64_desc);

}

/* 
 * is_transpose - This helper function checks if B is the transpose of
 *     A. You can check the correctness of your transpose by calling
 *     it before returning from the transpose function.
 */
int is_transpose(int M, int N, int A[N][M], int B[M][N])
{
    int i, j;

    for (i = 0; i < N; i++) {
        for (j = 0; j < M; ++j) {
            if (A[i][j] != B[j][i]) {
                return 0;
            }
        }
    }
    return 1;
}

