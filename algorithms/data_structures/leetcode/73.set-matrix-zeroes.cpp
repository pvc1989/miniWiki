/*
 * @lc app=leetcode id=73 lang=cpp
 *
 * [73] Set Matrix Zeroes
 *
 * https://leetcode.com/problems/set-matrix-zeroes/description/
 *
 * algorithms
 * Medium (44.25%)
 * Likes:    3114
 * Dislikes: 354
 * Total Accepted:    400K
 * Total Submissions: 902.6K
 * Testcase Example:  '[[1,1,1],[1,0,1],[1,1,1]]'
 *
 * Given an m x n matrix. If an element is 0, set its entire row and column to
 * 0. Do it in-place.
 * 
 * Follow up:
 * 
 * 
 * A straight forward solution using O(mn) space is probably a bad idea.
 * A simple improvement uses O(m + n) space, but still not the best
 * solution.
 * Could you devise a constant space solution?
 * 
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: matrix = [[1,1,1],[1,0,1],[1,1,1]]
 * Output: [[1,0,1],[0,0,0],[1,0,1]]
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
 * Output: [[0,0,0,0],[0,4,5,0],[0,3,1,0]]
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * m == matrix.length
 * n == matrix[0].length
 * 1 <= m, n <= 200
 * -2^31 <= matrix[i][j] <= 2^31 - 1
 * 
 * 
 */

// @lc code=start
class Solution {
  int n_rows_, n_cols_;
  void cleanRow(vector<vector<int>>& matrix, int i_row) {
    auto &row = matrix[i_row];
    for (auto& x : row) {
      x = 0;
    }
  }
  void cleanCol(vector<vector<int>>& matrix, int i_col) {
    for (int i_row{0}; i_row != n_rows_; ++i_row) {
      matrix[i_row][i_col] = 0;
    }
  }
 public:
  void setZeroes(vector<vector<int>>& matrix) {
    n_rows_ = matrix.size();
    n_cols_ = matrix.front().size();
    auto clean_row = vector<bool>(n_rows_);
    auto clean_col = vector<bool>(n_cols_);
    for (int i_row{0}; i_row != n_rows_; ++i_row) {
      for (int i_col{0}; i_col != n_cols_; ++i_col) {
        if (matrix[i_row][i_col] == 0) {
          clean_row[i_row] = true;
          clean_col[i_col] = true;
        }
      }
    }
    for (int i_row{0}; i_row != n_rows_; ++i_row) {
      if (clean_row[i_row]) cleanRow(matrix, i_row);
    }
    for (int i_col{0}; i_col != n_cols_; ++i_col) {
      if (clean_col[i_col]) cleanCol(matrix, i_col);
    }
  }
};
// @lc code=end

