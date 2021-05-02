/*
 * @lc app=leetcode id=696 lang=cpp
 *
 * [696] Count Binary Substrings
 *
 * https://leetcode.com/problems/count-binary-substrings/description/
 *
 * algorithms
 * Easy (58.13%)
 * Likes:    1371
 * Dislikes: 226
 * Total Accepted:    62.1K
 * Total Submissions: 105.7K
 * Testcase Example:  '"00110"'
 *
 * Give a string s, count the number of non-empty (contiguous) substrings that
 * have the same number of 0's and 1's, and all the 0's and all the 1's in
 * these substrings are grouped consecutively. 
 * 
 * Substrings that occur multiple times are counted the number of times they
 * occur.
 * 
 * Example 1:
 * 
 * Input: "00110011"
 * Output: 6
 * Explanation: There are 6 substrings that have equal number of consecutive
 * 1's and 0's: "0011", "01", "1100", "10", "0011", and "01".
 * Notice that some of these substrings repeat and are counted the number of
 * times they occur.
 * Also, "00110011" is not a valid substring because all the 0's (and 1's) are
 * not grouped together.
 * 
 * 
 * 
 * Example 2:
 * 
 * Input: "10101"
 * Output: 4
 * Explanation: There are 4 substrings: "10", "01", "10", "01" that have equal
 * number of consecutive 1's and 0's.
 * 
 * 
 * 
 * Note:
 * s.length will be between 1 and 50,000.
 * s will only consist of "0" or "1" characters.
 * 
 */

// @lc code=start
class Solution {
 public:
  int countBinarySubstrings(string s) {
    int total_count{0}, curr_count{0}, prev_count{0};
    char curr_char = s[0];
    for (auto c : s) {
      if (curr_char == c) {
        ++curr_count;
      }
      else {
        curr_char = c;
        prev_count = curr_count;
        curr_count = 1;
      }
      if (prev_count) {
        ++total_count;
        --prev_count;
      }
    }
    return total_count;
  }
};
// @lc code=end

