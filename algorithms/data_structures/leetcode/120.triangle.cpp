/*
 * @lc app=leetcode id=120 lang=cpp
 *
 * [120] Triangle
 *
 * https://leetcode.com/problems/triangle/description/
 *
 * algorithms
 * Medium (45.98%)
 * Likes:    2888
 * Dislikes: 315
 * Total Accepted:    299.7K
 * Total Submissions: 646.1K
 * Testcase Example:  '[[2],[3,4],[6,5,7],[4,1,8,3]]'
 *
 * Given a triangle array, return the minimum path sum from top to bottom.
 * 
 * For each step, you may move to an adjacent number of the row below. More
 * formally, if you are on index i on the current row, you may move to either
 * index i or index i + 1 on the next row.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
 * Output: 11
 * Explanation: The triangle looks like:
 * ⁠  2
 * ⁠ 3 4
 * ⁠6 5 7
 * 4 1 8 3
 * The minimum path sum from top to bottom is 2 + 3 + 5 + 1 = 11 (underlined
 * above).
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: triangle = [[-10]]
 * Output: -10
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= triangle.length <= 200
 * triangle[0].length == 1
 * triangle[i].length == triangle[i - 1].length + 1
 * -10^4 <= triangle[i][j] <= 10^4
 * 
 * 
 * 
 * Follow up: Could you do this using only O(n) extra space, where n is the
 * total number of rows in the triangle?
 */

// @lc code=start
class Solution {
 public:
  int minimumTotal(vector<vector<int>>& triangle) {
    auto prev_sum = array<int, 200>();
    auto curr_sum = prev_sum;
    curr_sum[0] = triangle[0][0];
    for (int row = 1; row != triangle.size(); ++row) {
      swap(prev_sum, curr_sum);
      auto& curr_row = triangle[row];
      int col = 0;
      curr_sum[col] = curr_row[col] + prev_sum[col];
      while (++col < row)
        curr_sum[col] = curr_row[col] + min(prev_sum[col], prev_sum[col-1]);
      curr_sum[col] = curr_row[col] + prev_sum[col - 1];
    }
    return *min_element(curr_sum.begin(), curr_sum.begin() + triangle.size());
  }
};
// @lc code=end

