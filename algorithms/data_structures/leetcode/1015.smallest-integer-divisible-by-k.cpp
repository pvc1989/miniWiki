/*
 * @lc app=leetcode id=1015 lang=cpp
 *
 * [1015] Smallest Integer Divisible by K
 *
 * https://leetcode.com/problems/smallest-integer-divisible-by-k/description/
 *
 * algorithms
 * Medium (42.11%)
 * Likes:    928
 * Dislikes: 765
 * Total Accepted:    52.6K
 * Total Submissions: 112.4K
 * Testcase Example:  '1'
 *
 * Given a positive integer k, you need to find the length of the smallest
 * positive integer n such that n is divisible by k, and n only contains the
 * digit 1.
 * 
 * Return the length of n. If there is no such n, return -1.
 * 
 * Note: n may not fit in a 64-bit signed integer.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: k = 1
 * Output: 1
 * Explanation: The smallest answer is n = 1, which has length 1.
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: k = 2
 * Output: -1
 * Explanation: There is no such positive integer n divisible by 2.
 * 
 * 
 * Example 3:
 * 
 * 
 * Input: k = 3
 * Output: 3
 * Explanation: The smallest answer is n = 111, which has length 3.
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= k <= 10^5
 * 
 * 
 */

// @lc code=start
class Solution {
 public:
  int smallestRepunitDivByK(int k) {
    int r = 0;
    /* The number of possible values of remainder is K. As a result, 
     * if the loop continues more than K times, and haven't stopped,
     * then we can conclude that remainder repeats.
     */
    for (int l = 1; l <= k; l++) {
      auto n = r * 10 + 1;
      r = n % k;
      if (r == 0) {
        return l;
      }
    }
    return -1;
  }
};
// @lc code=end

