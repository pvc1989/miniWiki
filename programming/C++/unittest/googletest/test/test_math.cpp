#include "math.h"
#include "gtest/gtest.h"

TEST(FactorialTest, SmallIntegers) {
  EXPECT_EQ(factorial(1), 1);
  EXPECT_EQ(factorial(2), 2);
  EXPECT_EQ(factorial(3), 6);
}

TEST(FactorialTest, Overflow) {
  EXPECT_EQ(factorial(12), 479001600);
  EXPECT_EQ(factorial(13), 1932053504);
  EXPECT_NE(factorial(13)/factorial(12), 13);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
