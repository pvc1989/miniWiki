/*
 * @lc app=leetcode id=1074 lang=cpp
 *
 * [1074] Number of Submatrices That Sum to Target
 *
 * https://leetcode.com/problems/number-of-submatrices-that-sum-to-target/description/
 *
 * algorithms
 * Hard (61.95%)
 * Likes:    875
 * Dislikes: 31
 * Total Accepted:    26.7K
 * Total Submissions: 42.7K
 * Testcase Example:  '[[0,1,0],[1,1,1],[0,1,0]]\n0'
 *
 * Given a matrix and a target, return the number of non-empty submatrices that
 * sum to target.
 * 
 * A submatrix x1, y1, x2, y2 is the set of all cells matrix[x][y] with x1 <= x
 * <= x2 and y1 <= y <= y2.
 * 
 * Two submatrices (x1, y1, x2, y2) and (x1', y1', x2', y2') are different if
 * they have some coordinate that is different: for example, if x1 != x1'.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: matrix = [[0,1,0],[1,1,1],[0,1,0]], target = 0
 * Output: 4
 * Explanation: The four 1x1 submatrices that only contain 0.
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: matrix = [[1,-1],[-1,1]], target = 0
 * Output: 5
 * Explanation: The two 1x2 submatrices, plus the two 2x1 submatrices, plus the
 * 2x2 submatrix.
 * 
 * 
 * Example 3:
 * 
 * 
 * Input: matrix = [[904]], target = 0
 * Output: 0
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= matrix.length <= 100
 * 1 <= matrix[0].length <= 100
 * -1000 <= matrix[i] <= 1000
 * -10^8 <= target <= 10^8
 * 
 * 
 */

// @lc code=start
class Solution {
 public:
  int numSubmatrixSumTarget(vector<vector<int>>& matrix, int target) {
    int n_rows = matrix.size(), n_cols = matrix[0].size(); 
    auto prefix_sum = array<array<int, 101>, 101>();
    for (int row = 0; row < n_rows; ++row) {
      for (int col = 0; col < n_cols; ++col) {
        prefix_sum[row + 1][col + 1] = prefix_sum[row + 1][col] +
            prefix_sum[row][col + 1] - prefix_sum[row][col] + matrix[row][col];
      }
    }
    int count = 0;
    for (int x1 = 0; x1 < n_rows; ++x1) {
      for (int x2 = x1 + 1; x2 <= n_rows; ++x2) {
        for (int y1 = 0; y1 < n_cols; ++y1) {
          for (int y2 = y1 + 1; y2 <= n_cols; ++y2) {
            count += (prefix_sum[x2][y2] + prefix_sum[x1][y1]
                    - prefix_sum[x2][y1] - prefix_sum[x1][y2] == target);
          }
        }
      }
    }
    return count;
  }
};
// @lc code=end

