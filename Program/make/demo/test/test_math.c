#include <stdio.h>

#include "../include/math.h"

int main() {
  printf("factorial(1) == %d\n", factorial(1));
  printf("factorial(2) == %d\n", factorial(2));
  printf("factorial(3) == %d\n", factorial(3));
  printf("factorial(12) == %d\n", factorial(12));
  printf("factorial(13) == %d\n", factorial(13));  // overflow
  printf("factorial(13) / factorial(12) == %d\n", factorial(13)/factorial(12));
  return 0;
}
