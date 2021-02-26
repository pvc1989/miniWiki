/*
 * @lc app=leetcode id=54 lang=cpp
 *
 * [54] Spiral Matrix
 *
 * https://leetcode.com/problems/spiral-matrix/description/
 *
 * algorithms
 * Medium (35.87%)
 * Likes:    3504
 * Dislikes: 641
 * Total Accepted:    466.4K
 * Total Submissions: 1.3M
 * Testcase Example:  '[[1,2,3],[4,5,6],[7,8,9]]'
 *
 * Given an m x n matrix, return all elements of the matrix in spiral order.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
 * Output: [1,2,3,6,9,8,7,4,5]
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
 * Output: [1,2,3,4,8,12,11,10,9,5,6,7]
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * m == matrix.length
 * n == matrix[i].length
 * 1 <= m, n <= 10
 * -100 <= matrix[i][j] <= 100
 * 
 * 
 */

// @lc code=start
class Solution {
 public:
  vector<int> spiralOrder(vector<vector<int>>& matrix) {
    auto output = vector<int>();
    int r_min{0}, r_max = matrix.size() - 1;
    int c_min{0}, c_max = matrix.front().size() - 1;
    int r{0}, c{-1}, direction{0};
    while (r_min <= r_max && c_min <= c_max) {
      switch (direction) {
      case 0:  // from left to right
        assert(c + 1 == c_min && r == r_min);
        while (c < c_max) {
          output.push_back(matrix[r][++c]);
        }
        ++r_min;
        direction = 1;
        break;
      case 1:  // from top to bottom
        assert(r + 1 == r_min && c == c_max);
        while (r < r_max) {
          output.push_back(matrix[++r][c]);
        }
        --c_max;
        direction = 2;
        break;
      case 2:  // from right to left
        assert(c - 1 == c_max && r == r_max);
        while (c_min < c) {
          output.push_back(matrix[r][--c]);
        }
        --r_max;
        direction = 3;
        break;
      case 3:  // from bottom to top
        assert(r - 1 == r_max && c == c_min);
        while (r_min < r) {
          output.push_back(matrix[--r][c]);
        }
        ++c_min;
        direction = 0;
        break;
      default:
        break;
      }
    }
    return output;
  }
};
// @lc code=end

