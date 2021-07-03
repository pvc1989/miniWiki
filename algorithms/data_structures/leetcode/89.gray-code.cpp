/*
 * @lc app=leetcode id=89 lang=cpp
 *
 * [89] Gray Code
 *
 * https://leetcode.com/problems/gray-code/description/
 *
 * algorithms
 * Medium (51.19%)
 * Likes:    904
 * Dislikes: 1845
 * Total Accepted:    185.1K
 * Total Submissions: 360.8K
 * Testcase Example:  '2'
 *
 * An n-bit gray code sequence is a sequence of 2^n integers where:
 * 
 * 
 * Every integer is in the inclusive range [0, 2^n - 1],
 * The first integer is 0,
 * An integer appears no more than once in the sequence,
 * The binary representation of every pair of adjacent integers differs by
 * exactly one bit, and
 * The binary representation of the first and last integers differs by exactly
 * one bit.
 * 
 * 
 * Given an integer n, return any valid n-bit gray code sequence.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: n = 2
 * Output: [0,1,3,2]
 * Explanation:
 * The binary representation of [0,1,3,2] is [00,01,11,10].
 * - 00 and 01 differ by one bit
 * - 01 and 11 differ by one bit
 * - 11 and 10 differ by one bit
 * - 10 and 00 differ by one bit
 * [0,2,3,1] is also a valid gray code sequence, whose binary representation is
 * [00,10,11,01].
 * - 00 and 10 differ by one bit
 * - 10 and 11 differ by one bit
 * - 11 and 01 differ by one bit
 * - 01 and 00 differ by one bit
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: n = 1
 * Output: [0,1]
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= n <= 16
 * 
 * 
 */

// @lc code=start
class Solution {
 public:
  vector<int> grayCode(int n) {
    vector<int> prev, curr{0, 1};
    /* n == 1: { 0, 1 }
       n == 2: { 0, 1, 11, 10 }
       n == 3: { 0, 1, 11, 10, 110, 111, 101, 100 } */
    for (int curr_bits = 1; curr_bits < n; ++curr_bits) {
      swap(prev, curr);
      auto prev_size = (1 << curr_bits);
      curr.resize(prev_size*2);
      auto head_to_mid = curr.begin();
      auto tail_to_mid = curr.rbegin();
      for (int i = 0; i < prev_size; ++i) {
        head_to_mid[i] = prev[i];
        tail_to_mid[i] = prev[i] + prev_size;
      }
    }
    return curr;
  }
};
// @lc code=end

