/*
 * @lc app=leetcode id=576 lang=cpp
 *
 * [576] Out of Boundary Paths
 *
 * https://leetcode.com/problems/out-of-boundary-paths/description/
 *
 * algorithms
 * Medium (36.46%)
 * Likes:    1173
 * Dislikes: 173
 * Total Accepted:    54.8K
 * Total Submissions: 138.9K
 * Testcase Example:  '2\n2\n2\n0\n0'
 *
 * There is an m x n grid with a ball. The ball is initially at the position
 * [startRow, startColumn]. You are allowed to move the ball to one of the four
 * adjacent cells in the grid (possibly out of the grid crossing the grid
 * boundary). You can apply at most maxMove moves to the ball.
 * 
 * Given the five integers m, n, maxMove, startRow, startColumn, return the
 * number of paths to move the ball out of the grid boundary. Since the answer
 * can be very large, return it modulo 10^9 + 7.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: m = 2, n = 2, maxMove = 2, startRow = 0, startColumn = 0
 * Output: 6
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: m = 1, n = 3, maxMove = 3, startRow = 0, startColumn = 1
 * Output: 12
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= m, n <= 50
 * 0 <= maxMove <= 50
 * 0 <= startRow < m
 * 0 <= startColumn < n
 * 
 */

// @lc code=start
#define NDEBUG
#ifdef NDEBUG
#define printf(...) 
#endif
class Solution {
  array<array<array<uint64_t, 51>, 51>, 51> exact_;
  int m_, n_;
  static constexpr uint64_t kM_ = 1000000007;

  bool valid(int i, int j) const {
    return 0 <= i && i < m_ && 0 <= j && j < n_;
  }
  void init_exact(int m, int n, int maxMove) {
    m_ = m--, n_ = n--;
    for (int row = 0; row < m_; ++row) {
      exact_[1][row][0] += 1;
      exact_[1][row][n] += 1;
    }
    for (int col = 0; col < n_; ++col) {
      exact_[1][0][col] += 1;
      exact_[1][m][col] += 1;
    }
    for (int mov = 2; mov <= maxMove; ++mov) {
      for (int row = 0; row < m_; ++row) {
        for (int col = 0; col < n_; ++col) {
          exact_[mov][row][col] = -1;
        }
      }
    }
  }
  uint64_t dp_exact(int mov, int row, int col) {
    uint64_t n_paths = 0;
    if (mov && valid(row, col)) {
      n_paths = exact_[mov][row][col];
      if (n_paths == -1) {
        n_paths = 0;
        --mov;
        n_paths += dp_exact(mov, row+1, col);
        n_paths += dp_exact(mov, row-1, col);
        n_paths += dp_exact(mov, row, col+1);
        n_paths += dp_exact(mov, row, col-1);
        n_paths %= kM_;
        exact_[mov+1][row][col] = n_paths;
      }
    }
    return n_paths;
  }
  
 public:
  int findPaths(int m, int n, int maxMove, int row, int col) {
    init_exact(m, n, maxMove);
    uint64_t n_paths = 0;
    for (int mov = 1; mov <= maxMove; ++mov) {
      n_paths += dp_exact(mov, row, col);
      n_paths %= kM_;
    }
    return n_paths;
  }
};
// @lc code=end

