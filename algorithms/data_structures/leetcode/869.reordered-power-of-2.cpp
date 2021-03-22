/*
 * @lc app=leetcode id=869 lang=cpp
 *
 * [869] Reordered Power of 2
 *
 * https://leetcode.com/problems/reordered-power-of-2/description/
 *
 * algorithms
 * Medium (54.26%)
 * Likes:    315
 * Dislikes: 112
 * Total Accepted:    22K
 * Total Submissions: 38.5K
 * Testcase Example:  '1'
 *
 * Starting with a positive integer N, we reorder the digits in any order
 * (including the original order) such that the leading digit is not zero.
 * 
 * Return true if and only if we can do this in a way such that the resulting
 * number is a power of 2.
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: 1
 * Output: true
 * 
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: 10
 * Output: false
 * 
 * 
 * 
 * Example 3:
 * 
 * 
 * Input: 16
 * Output: true
 * 
 * 
 * 
 * Example 4:
 * 
 * 
 * Input: 24
 * Output: false
 * 
 * 
 * 
 * Example 5:
 * 
 * 
 * Input: 46
 * Output: true
 * 
 * 
 * 
 * 
 * Note:
 * 
 * 
 * 1 <= N <= 10^9
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 */

// @lc code=start
class Solution {
  static string count(int n) {
    auto counter = string(10, '0');
    while (n) {
      ++counter[n % 10];
      n /= 10;
    }
    return counter;
  }

 public:
  bool reorderedPowerOf2(int n) {
    auto valid_counts = unordered_set<string>();
    for (int i = 1; i <= 1e9; i *= 2) {
      valid_counts.emplace(count(i));
    }
    return valid_counts.find(count(n)) != valid_counts.end();
  }
};
// @lc code=end

