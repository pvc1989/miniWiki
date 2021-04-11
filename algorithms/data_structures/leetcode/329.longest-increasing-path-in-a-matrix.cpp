/*
 * @lc app=leetcode id=329 lang=cpp
 *
 * [329] Longest Increasing Path in a Matrix
 *
 * https://leetcode.com/problems/longest-increasing-path-in-a-matrix/description/
 *
 * algorithms
 * Hard (45.21%)
 * Likes:    2980
 * Dislikes: 54
 * Total Accepted:    206.3K
 * Total Submissions: 452.4K
 * Testcase Example:  '[[9,9,4],[6,6,8],[2,1,1]]'
 *
 * Given an m x n integers matrix, return the length of the longest increasing
 * path in matrix.
 * 
 * From each cell, you can either move in four directions: left, right, up, or
 * down. You may not move diagonally or move outside the boundary (i.e.,
 * wrap-around is not allowed).
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: matrix = [[9,9,4],[6,6,8],[2,1,1]]
 * Output: 4
 * Explanation: The longest increasing path is [1, 2, 6, 9].
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: matrix = [[3,4,5],[3,2,6],[2,2,1]]
 * Output: 4
 * Explanation: The longest increasing path is [3, 4, 5, 6]. Moving diagonally
 * is not allowed.
 * 
 * 
 * Example 3:
 * 
 * 
 * Input: matrix = [[1]]
 * Output: 1
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * m == matrix.length
 * n == matrix[i].length
 * 1 <= m, n <= 200
 * 0 <= matrix[i][j] <= 2^31 - 1
 * 
 * 
 */

// @lc code=start
class Solution {
  int n_rows_, n_cols_;
  void build(const vector<vector<int>>& matrix, int row, int col,
             int val_u, vector<int>* adj_u) {
    if (0 <= row && row < n_rows_ && 0 <= col && col < n_cols_ &&
        matrix[row][col] > val_u) {
      adj_u->emplace_back(row * n_cols_ + col);
    }
  }
  int visit(const vector<vector<int>>& adj, int u, vector<int>* depths) {
    auto depth = depths->at(u);
    if (depth == -1) {
      depth = 1;
      for (auto v : adj[u]) {
        depth = max(depth, 1 + visit(adj, v, depths));
      }
      depths->at(u) = depth;
    }
    return depth;
  }

 public:
  int longestIncreasingPath(vector<vector<int>>& matrix) {
    n_rows_ = matrix.size(), n_cols_ = matrix[0].size();
    int size = n_rows_ * n_cols_;
    // build adj
    auto adj = vector<vector<int>>(size);
    for (int u = 0; u != size; ++u) {
      auto& adj_u = adj[u];
      int row = u / n_cols_, col = u % n_cols_;
      int val = matrix[row][col];
      build(matrix, row + 1, col, val, &adj_u);
      build(matrix, row - 1, col, val, &adj_u);
      build(matrix, row, col + 1, val, &adj_u);
      build(matrix, row, col - 1, val, &adj_u);
    }
    // dp
    auto depths = vector<int>(size, -1);
    int max_depth = 1;
    for (int u = 0; u != size; ++u) {
      max_depth = max(max_depth, visit(adj, u, &depths));
    }
    return max_depth;
  }
};
// @lc code=end

