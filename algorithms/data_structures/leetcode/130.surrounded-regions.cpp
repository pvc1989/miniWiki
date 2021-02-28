/*
 * @lc app=leetcode id=130 lang=cpp
 *
 * [130] Surrounded Regions
 *
 * https://leetcode.com/problems/surrounded-regions/description/
 *
 * algorithms
 * Medium (29.42%)
 * Likes:    2532
 * Dislikes: 749
 * Total Accepted:    285K
 * Total Submissions: 968.8K
 * Testcase Example:  '[["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]'
 *
 * Given an m x n matrix board containing 'X' and 'O', capture all regions
 * surrounded by 'X'.
 * 
 * A region is captured by flipping all 'O's into 'X's in that surrounded
 * region.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: board =
 * [["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]
 * Output:
 * [["X","X","X","X"],["X","X","X","X"],["X","X","X","X"],["X","O","X","X"]]
 * Explanation: Surrounded regions should not be on the border, which means
 * that any 'O' on the border of the board are not flipped to 'X'. Any 'O' that
 * is not on the border and it is not connected to an 'O' on the border will be
 * flipped to 'X'. Two cells are connected if they are adjacent cells connected
 * horizontally or vertically.
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: board = [["X"]]
 * Output: [["X"]]
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * m == board.length
 * n == board[i].length
 * 1 <= m, n <= 200
 * board[i][j] is 'X' or 'O'.
 * 
 * 
 */

// @lc code=start
class Solution {
  vector<int> curr_level_, next_level_;
  vector<vector<char>>* board_;
  int n_rows_, n_cols_;

private:
  int index(int row, int col) const {
    return row * n_cols_ + col;
  }
  void init(vector<vector<char>>* board) {
    for (int row = 0, col = 0; col != n_cols_; ++col) {
      if ((*board)[row][col] == 'O') { mark(row, col, board); }
    }
    for (int row = n_rows_ - 1, col = 0; col != n_cols_; ++col) {
      if ((*board)[row][col] == 'O') { mark(row, col, board); }
    }
    for (int row = n_rows_ - 2, col = 0; row != 0; --row) {
      if ((*board)[row][col] == 'O') { mark(row, col, board); }
    }
    for (int row = n_rows_ - 2, col = n_cols_ - 1; row != 0; --row) {
      if ((*board)[row][col] == 'O') { mark(row, col, board); }
    }
  }
  void mark(int row, int col, vector<vector<char>>* board) {
    if (0 <= row && row < n_rows_ && 0 <= col && col < n_cols_
        && (*board)[row][col] == 'O') {
      next_level_.push_back(index(row, col));
      (*board)[row][col] = 'M';  // marginal 'O'
    }
  }
  void mark(vector<vector<char>>* board) {
    while (!next_level_.empty()) {
      swap(next_level_, curr_level_);
      next_level_.clear();
      for (auto index : curr_level_) {
        int row = index / n_cols_;
        int col = index % n_cols_;
        mark(row + 1, col, board);
        mark(row - 1, col, board);
        mark(row, col + 1, board);
        mark(row, col - 1, board);
      }
    }
  }
  void flip(vector<vector<char>>* board) {
    for (int row = 0; row != n_rows_; ++row) {
      for (int col = 0; col != n_cols_; ++col) {
        switch ((*board)[row][col]) {
        case 'M':
          (*board)[row][col] = 'O';
          break;
        case 'O':
          (*board)[row][col] = 'X';
          break;
        default:
          break;
        }
      }
    }
  }

 public:
  void solve(vector<vector<char>>& board) {
    n_rows_ = board.size(), n_cols_ = board.front().size();
    if (n_rows_ == 1 || n_cols_ == 1) return;
    init(&board);  // push marginal O's into the 1st level
    mark(&board);  // run BFS on the board, edge := neighboring O's
    flip(&board);  // flip non-marginal O's
  }
};
// @lc code=end

