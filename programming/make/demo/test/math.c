#include <stdio.h>

#include "math.h"

int main() {
  printf("factorial(0) == %ld\n", factorial(0));
  printf("factorial(1) == %ld\n", factorial(1));
  printf("factorial(2) == %ld\n", factorial(2));
  printf("factorial(3) == %ld\n", factorial(3));
  printf("factorial(19) == %ld\n", factorial(19));
  printf("factorial(20) == %ld\n", factorial(20));
  printf("factorial(21) == %ld (overflowed)\n", factorial(21));
  printf("factorial(20) / factorial(19) == %ld\n",
          factorial(20) / factorial(19));
  printf("factorial(21) / factorial(20) == %ld (overflowed)\n",
          factorial(21) / factorial(20));
  return 0;
}
