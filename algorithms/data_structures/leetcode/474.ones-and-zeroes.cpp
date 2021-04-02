/*
 * @lc app=leetcode id=474 lang=cpp
 *
 * [474] Ones and Zeroes
 *
 * https://leetcode.com/problems/ones-and-zeroes/description/
 *
 * algorithms
 * Medium (43.82%)
 * Likes:    1548
 * Dislikes: 272
 * Total Accepted:    65.6K
 * Total Submissions: 149K
 * Testcase Example:  '["10","0001","111001","1","0"]\n5\n3'
 *
 * You are given an array of binary strings strs and two integers m and n.
 * 
 * Return the size of the largest subset of strs such that there are at most m
 * 0's and n 1's in the subset.
 * 
 * A set x is a subset of a set y if all elements of x are also elements of
 * y.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: strs = ["10","0001","111001","1","0"], m = 5, n = 3
 * Output: 4
 * Explanation: The largest subset with at most 5 0's and 3 1's is {"10",
 * "0001", "1", "0"}, so the answer is 4.
 * Other valid but smaller subsets include {"0001", "1"} and {"10", "1", "0"}.
 * {"111001"} is an invalid subset because it contains 4 1's, greater than the
 * maximum of 3.
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: strs = ["10","0","1"], m = 1, n = 1
 * Output: 2
 * Explanation: The largest subset is {"0", "1"}, so the answer is 2.
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= strs.length <= 600
 * 1 <= strs[i].length <= 100
 * strs[i] consists only of digits '0' and '1'.
 * 1 <= m, n <= 100
 * 
 * 
 */

// @lc code=start
class Solution {
  int visit(char m_max, char n_max,
      array<array<int, 101>, 101>* index_to_count) {
    
    return count;
  }
 public:
  int findMaxForm(vector<string>& strs, int m, int n) {
    auto counter = vector<pair<char, char>>();
    for (auto& s : strs) {
      char n_nils = count(s.begin(), s.end(), '0');
      char n_ones = s.size() - n_nils;
      counter.emplace_back(n_nils, n_ones);
    }
    auto index_to_count = array<array<int, 101>, 101>();
    for (auto& row : index_to_count) {
      row.fill(-1);
    }
    return visit(counter, 0, m, n, &index_to_count);
  }
};
// @lc code=end

