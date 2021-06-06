/*
 * @lc app=leetcode id=52 lang=cpp
 *
 * [52] N-Queens II
 *
 * https://leetcode.com/problems/n-queens-ii/description/
 *
 * algorithms
 * Hard (60.41%)
 * Likes:    890
 * Dislikes: 187
 * Total Accepted:    166.7K
 * Total Submissions: 271.3K
 * Testcase Example:  '4'
 *
 * The n-queens puzzle is the problem of placing n queens on an n x n
 * chessboard such that no two queens attack each other.
 * 
 * Given an integer n, return the number of distinct solutions to the n-queens
 * puzzle.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: n = 4
 * Output: 2
 * Explanation: There are two distinct solutions to the 4-queens puzzle as
 * shown.
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: n = 1
 * Output: 1
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
  void solve(int k, int* count) {
    if (k == n_) {
      auto board = vector<string>(n_, string(n_, '.'));
      for (int i = 0; i < n_; ++i)
        for (int j = 0; j < n_; ++j)
          if (state_[i][j])
            board[i][j] = 'Q';
      ++*count;
    }
    else { // k < n
      for (int i = 0; i != n_; ++i) {
        if (placeable(i, k)) {
          state_[i][k] = true;
          solve(k+1, count);
          state_[i][k] = false;
        }
      }
    }
  }
 public:
  int totalNQueens(int n) {
    int count = 0;
    n_ = n;
    for (int i = 0; i != n; ++i)
      for (int j = 0; j != n; ++j)
        state_[i][j] = false;
    solve(0, &count);
    return count;
  }
};
// @lc code=end

