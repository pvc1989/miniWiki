/*
 * @lc app=leetcode id=417 lang=cpp
 *
 * [417] Pacific Atlantic Water Flow
 *
 * https://leetcode.com/problems/pacific-atlantic-water-flow/description/
 *
 * algorithms
 * Medium (42.59%)
 * Likes:    1863
 * Dislikes: 455
 * Total Accepted:    105.9K
 * Total Submissions: 246.3K
 * Testcase Example:  '[[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]'
 *
 * Given an m x n matrix of non-negative integers representing the height of
 * each unit cell in a continent, the "Pacific ocean" touches the left and top
 * edges of the matrix and the "Atlantic ocean" touches the right and bottom
 * edges.
 * 
 * Water can only flow in four directions (up, down, left, or right) from a
 * cell to another one with height equal or lower.
 * 
 * Find the list of grid coordinates where water can flow to both the Pacific
 * and Atlantic ocean.
 * 
 * Note:
 * 
 * 
 * The order of returned grid coordinates does not matter.
 * Both m and n are less than 150.
 * 
 * 
 * 
 * 
 * Example:
 * 
 * 
 * Given the following 5x5 matrix:
 * 
 * ⁠ Pacific ~   ~   ~   ~   ~ 
 * ⁠      ~  1   2   2   3  (5) *
 * ⁠      ~  3   2   3  (4) (4) *
 * ⁠      ~  2   4  (5)  3   1  *
 * ⁠      ~ (6) (7)  1   4   5  *
 * ⁠      ~ (5)  1   1   2   4  *
 * ⁠         *   *   *   *   * Atlantic
 * 
 * Return:
 * 
 * [[0, 4], [1, 3], [1, 4], [2, 2], [3, 0], [3, 1], [4, 0]] (positions with
 * parentheses in above matrix).
 * 
 * 
 * 
 * 
 */

// @lc code=start
class Solution {
  int n_rows, n_cols;
  vector<vector<bool>> bfs(const vector<vector<int>>& matrix,
                           vector<pair<int, int>>&& next_level) {
    auto visited = vector<vector<bool>>(n_rows, vector<bool>(n_cols));
    for (auto [row, col] : next_level) {
      visited[row][col] = true;
    }
    auto curr_level = vector<pair<int, int>>();
    while (!next_level.empty()) {
      swap(curr_level, next_level);
      next_level.clear();
      for (auto [row, col] : curr_level) {
        auto value = matrix[row][col];
        visit(matrix, row - 1, col, value, &visited, &next_level);
        visit(matrix, row + 1, col, value, &visited, &next_level);
        visit(matrix, row, col - 1, value, &visited, &next_level);
        visit(matrix, row, col + 1, value, &visited, &next_level);
      }
    }
    return visited;
  }
  void visit(const vector<vector<int>>& matrix, int row, int col, int value,
      vector<vector<bool>>* visited, vector<pair<int, int>>* next_level) {
    if (0 <= row && row < n_rows && 0 <= col && col < n_cols &&
        (*visited)[row][col] == false && value <= matrix[row][col]) {
      next_level->emplace_back(row, col);
      (*visited)[row][col] = true;
    }
  }
 public:
  vector<vector<int>> pacificAtlantic(vector<vector<int>>& matrix) {
    auto result = vector<vector<int>>();
    n_rows = matrix.size();
    if (n_rows == 0) { return result; }
    n_cols = matrix.front().size();
    if (n_cols == 0) { return result; }
    // run bfs for pacific
    auto p_sources = vector<pair<int, int>>();
    for (int col = 0; col != n_cols; ++col) {
      p_sources.emplace_back(0, col);
    }
    for (int row = 1; row != n_rows; ++row) {
      p_sources.emplace_back(row, 0);
    }
    auto p_reachable = bfs(matrix, move(p_sources));
    // run bfs for atlantic
    auto a_sources = vector<pair<int, int>>();
    auto row_max = n_rows - 1;
    for (int col = 0; col != n_cols; ++col) {
      a_sources.emplace_back(row_max, col);
    }
    auto col_max = n_cols - 1;
    for (int row = 0; row != row_max; ++row) {
      a_sources.emplace_back(row, col_max);
    }
    auto a_reachable = bfs(matrix, move(a_sources));
    // add commonly reachable cells to result
    for (int row = 0; row != n_rows; ++row) {
      for (int col = 0; col != n_cols; ++col) {
        if (p_reachable[row][col] && a_reachable[row][col]) {
          result.emplace_back(2, row);
          result.back().back() = col;
        }
      }
    }
    return result;
  }
};
// @lc code=end

