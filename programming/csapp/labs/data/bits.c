/* 
 * CS:APP Data Lab 
 * 
 * <Please put your name and userid here>
 * 
 * bits.c - Source file with your solutions to the Lab.
 *          This is the file you will hand in to your instructor.
 *
 * WARNING: Do not include the <stdio.h> header; it confuses the dlc
 * compiler. You can still use printf for debugging without including
 * <stdio.h>, although you might get a compiler warning. In general,
 * it's not good practice to ignore compiler warnings, but in this
 * case it's OK.  
 */

#if 0
/*
 * Instructions to Students:
 *
 * STEP 1: Read the following instructions carefully.
 */

You will provide your solution to the Data Lab by
editing the collection of functions in this source file.

INTEGER CODING RULES:
 
  Replace the "return" statement in each function with one
  or more lines of C code that implements the function. Your code 
  must conform to the following style:
 
  int Funct(arg1, arg2, ...) {
      /* brief description of how your implementation works */
      int var1 = Expr1;
      ...
      int varM = ExprM;

      varJ = ExprJ;
      ...
      varN = ExprN;
      return ExprR;
  }

  Each "Expr" is an expression using ONLY the following:
  1. Integer constants 0 through 255 (0xFF), inclusive. You are
      not allowed to use big constants such as 0xffffffff.
  2. Function arguments and local variables (no global variables).
  3. Unary integer operations ! ~
  4. Binary integer operations & ^ | + << >>
    
  Some of the problems restrict the set of allowed operators even further.
  Each "Expr" may consist of multiple operators. You are not restricted to
  one operator per line.

  You are expressly forbidden to:
  1. Use any control constructs such as if, do, while, for, switch, etc.
  2. Define or use any macros.
  3. Define any additional functions in this file.
  4. Call any functions.
  5. Use any other operations, such as &&, ||, -, or ?:
  6. Use any form of casting.
  7. Use any data type other than int.  This implies that you
     cannot use arrays, structs, or unions.

 
  You may assume that your machine:
  1. Uses 2s complement, 32-bit representations of integers.
  2. Performs right shifts arithmetically.
  3. Has unpredictable behavior when shifting if the shift amount
     is less than 0 or greater than 31.


EXAMPLES OF ACCEPTABLE CODING STYLE:
  /*
   * pow2plus1 - returns 2^x + 1, where 0 <= x <= 31
   */
  int pow2plus1(int x) {
     /* exploit ability of shifts to compute powers of 2 */
     return (1 << x) + 1;
  }

  /*
   * pow2plus4 - returns 2^x + 4, where 0 <= x <= 31
   */
  int pow2plus4(int x) {
     /* exploit ability of shifts to compute powers of 2 */
     int result = (1 << x);
     result += 4;
     return result;
  }

FLOATING POINT CODING RULES

For the problems that require you to implement floating-point operations,
the coding rules are less strict.  You are allowed to use looping and
conditional control.  You are allowed to use both ints and unsigneds.
You can use arbitrary integer and unsigned constants. You can use any arithmetic,
logical, or comparison operations on int or unsigned data.

You are expressly forbidden to:
  1. Define or use any macros.
  2. Define any additional functions in this file.
  3. Call any functions.
  4. Use any form of casting.
  5. Use any data type other than int or unsigned.  This means that you
     cannot use arrays, structs, or unions.
  6. Use any floating point data types, operations, or constants.


NOTES:
  1. Use the dlc (data lab checker) compiler (described in the handout) to 
     check the legality of your solutions.
  2. Each function has a maximum number of operations (integer, logical,
     or comparison) that you are allowed to use for your implementation
     of the function.  The max operator count is checked by dlc.
     Note that assignment ('=') is not counted; you may use as many of
     these as you want without penalty.
  3. Use the btest test harness to check your functions for correctness.
  4. Use the BDD checker to formally verify your functions
  5. The maximum number of ops for each function is given in the
     header comment for each function. If there are any inconsistencies 
     between the maximum ops in the writeup and in this file, consider
     this file the authoritative source.

/*
 * STEP 2: Modify the following functions according the coding rules.
 * 
 *   IMPORTANT. TO AVOID GRADING SURPRISES:
 *   1. Use the dlc compiler to check that your solutions conform
 *      to the coding rules.
 *   2. Use the BDD checker to formally verify that your solutions produce 
 *      the correct answers.
 */


#endif
//1
/* 
 * bitXor - x^y using only ~ and & 
 *   Example: bitXor(4, 5) = 1
 *   Legal ops: ~ &
 *   Max ops: 14
 *   Rating: 1
 */
int bitXor(int x, int y) {
  /*  ~(~(~x & y) & ~(x & ~y));  // 8 ops */
  return ~(x & y) & ~(~x & ~y);  /* 7 ops */
}
/* 
 * tmin - return minimum two's complement integer 
 *   Legal ops: ! ~ & ^ | + << >>
 *   Max ops: 4
 *   Rating: 1
 */
int tmin(void) {
  return 1 << 31;
}
//2
/*
 * isTmax - returns 1 if x is the maximum, two's complement number,
 *     and 0 otherwise 
 *   Legal ops: ! ~ & ^ | +
 *   Max ops: 10
 *   Rating: 1
 */
int isTmax(int x) {
  int c1 = !((x + 1) ^ (~x));  /* main cases: (x + 1) == ~x */
  int c2 = !(!(~x));  /* corner case: x != -1 */
  return c1 & c2;
}
/* 
 * allOddBits - return 1 if all odd-numbered bits in word set to 1
 *   where bits are numbered from 0 (least significant) to 31 (most significant)
 *   Examples allOddBits(0xFFFFFFFD) = 0, allOddBits(0xAAAAAAAA) = 1
 *   Legal ops: ! ~ & ^ | + << >>
 *   Max ops: 12
 *   Rating: 2
 */
int allOddBits(int x) {
  int mask = 0xAA;  /* bit pattern == 10101010, filter out even-numbered bits */
  x &= (x >> 16);
  x &= (x >>  8);
  return !((x & mask) ^ mask);  /* masked == mask */
}
/* 
 * negate - return -x 
 *   Example: negate(1) = -1.
 *   Legal ops: ! ~ & ^ | + << >>
 *   Max ops: 5
 *   Rating: 2
 */
int negate(int x) {
  return ~x + 1;
}
//3
/* 
 * isAsciiDigit - return 1 if 0x30 <= x <= 0x39 (ASCII codes for characters '0' to '9')
 *   Example: isAsciiDigit(0x35) = 1.
 *            isAsciiDigit(0x3a) = 0.
 *            isAsciiDigit(0x05) = 0.
 *   Legal ops: ! ~ & ^ | + << >>
 *   Max ops: 15
 *   Rating: 3
 */
int isAsciiDigit(int x) {
  int c0 = x & (~0xFF);  /* should be 0, if (x & 0xFFFFFF00) == 0 */
  int c1 = (x & 0xF0) ^ 0x30;     /* should be 0, if (x & 0xF0) == 0x30 */
  int c2 = ((x & 0xF) + 6) & 16;  /* should be 0, if (x & 0xF) + 6 < 16 */
  return !(c0 | c1 | c2);
}
/* 
 * conditional - same as x ? y : z 
 *   Example: conditional(2,4,5) = 4
 *   Legal ops: ! ~ & ^ | + << >>
 *   Max ops: 16
 *   Rating: 3
 */
int conditional(int x, int y, int z) {
  int x_is_0 = !x ;  /* x == 0 ? 0x00000001 : 0x00000000 */
  x_is_0 = ~x_is_0;  /* x == 0 ? 0xFFFFFFFE : 0xFFFFFFFF */
  x_is_0 += 1;       /* x == 0 ? 0xFFFFFFFF : 0x00000000 */
  return (y & (~x_is_0)) | (z & x_is_0);
}
/* 
 * isLessOrEqual - if x <= y  then return 1, else return 0 
 *   Example: isLessOrEqual(4,5) = 1.
 *   Legal ops: ! ~ & ^ | + << >>
 *   Max ops: 24
 *   Rating: 3
 */
int isLessOrEqual(int x, int y) {
  int int_min = 1 << 31;
  /* case 1: x == int_min */
  int x_eq_int_min = !(x ^ int_min);
  /* case 2: x < 0 && 0 <= y */
  int x_sign_bit = x & int_min;
  int y_sign_bit = y & int_min;
  int x_lt_zero_le_y = !(x_sign_bit ^ int_min) & !(y_sign_bit ^ 0);
  /* case 3: y - x >= 0 (no overflow) */
  int d = y + (~x + 1);  /* y - x */
  int d_ge_zero = ((~d) >> 31) & 1;
  int x_y_same_sign = !(x_sign_bit ^ y_sign_bit);
  /* return 1 iff any of them is 1 */
  return  x_eq_int_min | x_lt_zero_le_y | x_y_same_sign & d_ge_zero;
}
//4
/* 
 * logicalNeg - implement the ! operator, using all of 
 *              the legal operators except !
 *   Examples: logicalNeg(3) = 0, logicalNeg(0) = 1
 *   Legal ops: ~ & ^ | + << >>
 *   Max ops: 12
 *   Rating: 4 
 */
int logicalNeg(int x) {
  x |= (x >> 16);  // x & 0xFFFF != 0
  x |= (x >> 8);   // x & 0x00FF != 0
  x |= (x >> 4);   // x & 0x000F != 0
  x |= (x >> 2);   // x & 0x0002 != 0
  x |= (x >> 1);   // x & 0x0001 != 0
  return (x & 1) ^ 1;
}
/* howManyBits - return the minimum number of bits required to represent x in
 *             two's complement
 *  Examples: howManyBits(12) = 5
 *            howManyBits(298) = 10
 *            howManyBits(-5) = 4
 *            howManyBits(0)  = 1
 *            howManyBits(-1) = 1
 *            howManyBits(0x80000000) = 32
 *  Legal ops: ! ~ & ^ | + << >>
 *  Max ops: 90
 *  Rating: 4
 */
int howManyBits(int x) {
  int x_lt_0_mask = (x & (1 << 31)) >> 31;  /* x < 0 ? -1 : 0 */
  int y = (x & x_lt_0_mask) | (~x & ~x_lt_0_mask);  /* y = (x < 0 ? x : ~x) */
  /* find the highest `0` for `y` */
  int count = 1;  /* always has the leading sign bit */
  /* choose which 16 bits to go */
  int y_left = y >> 16;
  int y_left_has_0 = !(!(y_left + 1));  /* (y.left has 0) ? 1 : 0 */
  int y_left_has_0_mask = (y_left_has_0 << 31) >> 31;
  count += (y_left_has_0 << 4);  /* at least 16 trailing bits */
  y = (y_left & y_left_has_0_mask) | (y & ~y_left_has_0_mask);
  /* choose which 8 bits to go */
  y_left = y >> 8;
  y_left_has_0 = !(!(y_left + 1));  /* (y.left has 0) ? 1 : 0 */
  y_left_has_0_mask = (y_left_has_0 << 31) >> 31;
  count += (y_left_has_0 << 3);  /* at least 8 trailing bits */
  y = (y_left & y_left_has_0_mask) | (y & ~y_left_has_0_mask);
  /* choose which 4 bits to go */
  y_left = y >> 4;
  y_left_has_0 = !(!(y_left + 1));  /* (y.left has 0) ? 1 : 0 */
  y_left_has_0_mask = (y_left_has_0 << 31) >> 31;
  count += (y_left_has_0 << 2);  /* at least 4 trailing bits */
  y = (y_left & y_left_has_0_mask) | (y & ~y_left_has_0_mask);
  /* choose which 2 bits to go */
  y_left = y >> 2;
  y_left_has_0 = !(!(y_left + 1));  /* (y.left has 0) ? 1 : 0 */
  y_left_has_0_mask = (y_left_has_0 << 31) >> 31;
  count += (y_left_has_0 << 1);  /* at least 2 trailing bits */
  y = (y_left & y_left_has_0_mask) | (y & ~y_left_has_0_mask);
  /* choose which 1 bits to go */
  y_left = y >> 1;
  y_left_has_0 = !(!(y_left + 1));  /* (y.left has 0) ? 1 : 0 */
  y_left_has_0_mask = (y_left_has_0 << 31) >> 31;
  count += (y_left_has_0);  /* at least 1 trailing bit */
  y = (y_left & y_left_has_0_mask) | (y & ~y_left_has_0_mask);
  /* one more bit, if the last bit of y is 0 */
  count += !(y & 1);
  return count;
}
//float
/* 
 * floatScale2 - Return bit-level equivalent of expression 2*f for
 *   floating point argument f.
 * 
 *   Both the argument and result are passed as unsigned int's, but
 *   they are to be interpreted as the bit-level representation of
 *   single-precision floating point values.
 *   When argument is NaN, return argument
 * 
 *   Legal ops: Any integer/unsigned operations incl. ||, &&. also if, while
 *   Max ops: 30
 *   Rating: 4
 */
unsigned floatScale2(unsigned uf) {
  unsigned sign_bit = uf & 0x80000000;
  unsigned fraction = uf & 0x007FFFFF;
  unsigned exponent_mask = 0x7F800000;
  unsigned exponent = uf & exponent_mask;
  unsigned result;
  if (exponent == exponent_mask && fraction) {  /* `f` is `NaN` */
    result = uf;
  } else {
    if (exponent) {  /* `f` is normalized */
      if (exponent < exponent_mask) {  /* `2*f` does not overflow */
        result = (exponent + 0x00800000) | fraction;
      } else {  /* `2*f` overflows, return `inf` */
        result = exponent_mask;
      }
    } else {  /* `f` is denormalized */
     /* If the first bit of `fraction` is `1`,
        it will naturally become the last bit of `expotent`. */
      result = fraction << 1;
    }
    result |= sign_bit;
  }
  return result;
}
/* 
 * floatFloat2Int - Return bit-level equivalent of expression (int) f
 *   for floating point argument f.
 * 
 *   Argument is passed as unsigned int, but
 *   it is to be interpreted as the bit-level representation of a
 *   single-precision floating point value.
 *   Anything out of range (including NaN and infinity) should return
 *   0x80000000u.
 * 
 *   Legal ops: Any integer/unsigned operations incl. ||, &&. also if, while
 *   Max ops: 30
 *   Rating: 4
 */
int floatFloat2Int(unsigned uf) {
  unsigned result;
  unsigned sign_bit = uf & 0x80000000;
  unsigned fraction = uf & 0x007FFFFF;
  unsigned exponent_mask = 0x7F800000;
  unsigned expotent = uf & exponent_mask;
  int n_shifts = (expotent ? expotent >> 23 : 1) - 150/* 127 + 23 */;
  int n_bits = 24;
  int n_shifts_plus_n_bits = n_shifts + n_bits;
  unsigned x = fraction;
  if (expotent) {
    fraction += 0x00800000;  /* also for NaN or Inf */
  } else {
    n_bits = 0;
    while (x) {
      ++n_bits;
      x >>= 1;
    }
  }
  if (n_shifts_plus_n_bits <= 0) {  /* underflow */
    result = 0;
  } else if (31 < n_shifts_plus_n_bits) {  /* overflow */
    result = 0x80000000;
  } else {  /* ordinary case */
    if (n_shifts <= 0) {  /* -24 <= -n_bits < n_shifts <= 0 */
      n_shifts = -n_shifts;
      result = fraction >> n_shifts;
    } else if (n_shifts_plus_n_bits < 32) {  /* no overflow */
      /* assert(expotent && fraction); */
      result = fraction << n_shifts;
    }
    if (sign_bit) {
      result = -result;
    }
  }
  return result;
}
/* 
 * floatPower2 - Return bit-level equivalent of the expression 2.0^x
 *   (2.0 raised to the power x) for any 32-bit integer x.
 *
 *   The unsigned value that is returned should have the identical bit
 *   representation as the single-precision floating-point number 2.0^x.
 *   If the result is too small to be represented as a denorm, return
 *   0. If too large, return +INF.
 * 
 *   Legal ops: Any integer/unsigned operations incl. ||, &&. Also if, while 
 *   Max ops: 30 
 *   Rating: 4
 */
unsigned floatPower2(int x) {
    return 2;
}
