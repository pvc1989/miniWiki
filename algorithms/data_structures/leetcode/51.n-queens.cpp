/*
 * @lc app=leetcode id=51 lang=cpp
 *
 * [51] N-Queens
 *
 * https://leetcode.com/problems/n-queens/description/
 *
 * algorithms
 * Hard (49.99%)
 * Likes:    3195
 * Dislikes: 111
 * Total Accepted:    267.8K
 * Total Submissions: 518K
 * Testcase Example:  '4'
 *
 * The n-queens puzzle is the problem of placing n queens on an n x n
 * chessboard such that no two queens attack each other.
 * 
 * Given an integer n, return all distinct solutions to the n-queens puzzle.
 * 
 * Each solution contains a distinct board configuration of the n-queens'
 * placement, where 'Q' and '.' both indicate a queen and an empty space,
 * respectively.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: n = 4
 * Output: [[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
 * Explanation: There exist two distinct solutions to the 4-queens puzzle as
 * shown above
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: n = 1
 * Output: [["Q"]]
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= n <= 9
 * 
 * 
 */

// @lc code=start
class Solution {
  array<array<bool, 9>, 9> state_;
  int n_{0};

  bool valid(int i, int j) {
    return (0 <= i && i < n_) && (0 <= j && j < n_);
  }
  bool placeable(int i, int j) {
    for (int k = 0; k < n_; ++k) {
      if (state_[i][k] || state_[k][j])
        return false;
    }
    for (int k = 1; k <= j; ++k) {
      if (valid(i + k, j - k) && state_[i + k][j - k])
        return false;
      if (valid(i - k, j - k) && state_[i - k][j - k])
        return false;
    }
    return true;
  }
  void solve(int k, vector<vector<string>>* result) {
    if (k == n_) {
      auto board = vector<string>(n_, string(n_, '.'));
      for (int i = 0; i < n_; ++i)
        for (int j = 0; j < n_; ++j)
          if (state_[i][j])
            board[i][j] = 'Q';
      result->emplace_back(move(board));
    }
    else { // k < n
      for (int i = 0; i != n_; ++i) {
        if (placeable(i, k)) {
          state_[i][k] = true;
          solve(k+1, result);
          state_[i][k] = false;
        }
      }
    }
  }
 public:
  vector<vector<string>> solveNQueens(int n) {
    vector<vector<string>> result;
    n_ = n;
    for (int i = 0; i != n; ++i)
      for (int j = 0; j != n; ++j)
        state_[i][j] = false;
    solve(0, &result);
    return result;
  }
};
// @lc code=end

