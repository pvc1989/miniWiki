/*
 * @lc app=leetcode id=326 lang=cpp
 *
 * [326] Power of Three
 *
 * https://leetcode.com/problems/power-of-three/description/
 *
 * algorithms
 * Easy (42.10%)
 * Likes:    57
 * Dislikes: 11
 * Total Accepted:    328.6K
 * Total Submissions: 780.1K
 * Testcase Example:  '27'
 *
 * Given an integer n, return true if it is a power of three. Otherwise, return
 * false.
 * 
 * An integer n is a power of three, if there exists an integer x such that n
 * == 3^x.
 * 
 * 
 * Example 1:
 * Input: n = 27
 * Output: true
 * Example 2:
 * Input: n = 0
 * Output: false
 * Example 3:
 * Input: n = 9
 * Output: true
 * Example 4:
 * Input: n = 45
 * Output: false
 * 
 * 
 * Constraints:
 * 
 * 
 * -2^31 <= n <= 2^31 - 1
 * 
 * 
 * 
 * Follow up: Could you solve it without loops/recursion?
 */

// @lc code=start
class Solution {
 public:
  bool isPowerOfThree(int n) {
    // return (n > 0) && (1162261467 % n == 0);
    if (n <= 0)
      return false;
    while (n != 1) {
      if (n % 3)
        return false;
      else
        n /= 3;
    }
    return true;
  }
};
// @lc code=end

