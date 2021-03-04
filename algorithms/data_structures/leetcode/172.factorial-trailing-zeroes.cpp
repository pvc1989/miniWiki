/*
 * @lc app=leetcode id=172 lang=cpp
 *
 * [172] Factorial Trailing Zeroes
 *
 * https://leetcode.com/problems/factorial-trailing-zeroes/description/
 *
 * algorithms
 * Easy (38.58%)
 * Likes:    1202
 * Dislikes: 1355
 * Total Accepted:    247.5K
 * Total Submissions: 641.4K
 * Testcase Example:  '3'
 *
 * Given an integer n, return the number of trailing zeroes in n!.
 * 
 * Follow up: Could you write a solution that works in logarithmic time
 * complexity?
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: n = 3
 * Output: 0
 * Explanation: 3! = 6, no trailing zero.
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: n = 5
 * Output: 1
 * Explanation: 5! = 120, one trailing zero.
 * 
 * 
 * Example 3:
 * 
 * 
 * Input: n = 0
 * Output: 0
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 0 <= n <= 10^4
 * 
 * 
 */

// @lc code=start
class Solution {
  int count(int x, int y) {
    int n = 0;
    while (x >= y && x % y == 0) {
      ++n;
      x /= y;
    }
    return n;
  }
 public:
  int trailingZeroes(int n) {
    if (n == 0) return 0;
    int count_2 = 0, count_5 = 0;
    for (int i = 1; i <= n; ++i) {
      count_2 += count(i, 2);
      count_5 += count(i, 5);
    }
    return min(count_2, count_5);
  }
};
// @lc code=end

