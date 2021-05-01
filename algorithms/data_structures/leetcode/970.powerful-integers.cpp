/*
 * @lc app=leetcode id=970 lang=cpp
 *
 * [970] Powerful Integers
 *
 * https://leetcode.com/problems/powerful-integers/description/
 *
 * algorithms
 * Medium (40.03%)
 * Likes:    57
 * Dislikes: 34
 * Total Accepted:    31.5K
 * Total Submissions: 76.2K
 * Testcase Example:  '2\n3\n10'
 *
 * Given three integers x, y, and bound, return a list of all the powerful
 * integers that have a value less than or equal to bound.
 * 
 * An integer is powerful if it can be represented as x^i + y^j for some
 * integers i >= 0 and j >= 0.
 * 
 * You may return the answer in any order. In your answer, each value should
 * occur at most once.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: x = 2, y = 3, bound = 10
 * Output: [2,3,4,5,7,9,10]
 * Explanation:
 * 2 = 2^0 + 3^0
 * 3 = 2^1 + 3^0
 * 4 = 2^0 + 3^1
 * 5 = 2^1 + 3^1
 * 7 = 2^2 + 3^1
 * 9 = 2^3 + 3^0
 * 10 = 2^0 + 3^2
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: x = 3, y = 5, bound = 15
 * Output: [2,4,6,8,10,14]
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= x, y <= 100
 * 0 <= bound <= 10^6
 * 
 * 
 */

// @lc code=start
class Solution {
  vector<int> getPowers(int x, int bound) {
    auto x_powers = vector<int>();
    if (x == 1 && x < bound)
      x_powers.emplace_back(x);
    else {
      int x_power = 1;
      while (x_power < bound) {
        x_powers.emplace_back(x_power);
        x_power *= x;
      }
    }
    return x_powers;
  }
 public:
  vector<int> powerfulIntegers(int x, int y, int bound) {
    auto x_powers = getPowers(x, bound);
    auto y_powers = getPowers(y, bound);
    auto result = unordered_set<int>();
    for (auto x_power : x_powers) {
      for (auto y_power : y_powers) {
        auto sum = x_power + y_power;
        if (sum <= bound)
          result.emplace(sum);
        else
          break;
      }
    }
    return {result.begin(), result.end()};
  }
};
// @lc code=end

