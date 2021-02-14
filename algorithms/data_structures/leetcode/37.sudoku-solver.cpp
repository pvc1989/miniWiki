/*
 * @lc app=leetcode id=37 lang=cpp
 *
 * [37] Sudoku Solver
 *
 * https://leetcode.com/problems/sudoku-solver/description/
 *
 * algorithms
 * Hard (46.37%)
 * Likes:    2489
 * Dislikes: 100
 * Total Accepted:    222.9K
 * Total Submissions: 480.6K
 * Testcase Example:  '[["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]]'
 *
 * Write a program to solve a Sudoku puzzle by filling the empty cells.
 * 
 * A sudoku solution must satisfy all of the following rules:
 * 
 * 
 * Each of the digits 1-9 must occur exactly once in each row.
 * Each of the digits 1-9 must occur exactly once in each column.
 * Each of the digits 1-9 must occur exactly once in each of the 9 3x3
 * sub-boxes of the grid.
 * 
 * 
 * The '.' character indicates empty cells.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: board =
 * [["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]]
 * Output:
 * [["5","3","4","6","7","8","9","1","2"],["6","7","2","1","9","5","3","4","8"],["1","9","8","3","4","2","5","6","7"],["8","5","9","7","6","1","4","2","3"],["4","2","6","8","5","3","7","9","1"],["7","1","3","9","2","4","8","5","6"],["9","6","1","5","3","7","2","8","4"],["2","8","7","4","1","9","6","3","5"],["3","4","5","2","8","6","1","7","9"]]
 * Explanation: The input board is shown above and the only valid solution is
 * shown below:
 * 
 * 
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * board.length == 9
 * board[i].length == 9
 * board[i][j] is a digit or '.'.
 * It is guaranteed that the input board has only one solution.
 * 
 * 
 */

// @lc code=start
class Solution {
  void Erase(char c, bitset<9> *candidates) {
    if (c != '.') candidates->reset(c - '1');
  }
  bool DFS(vector<pair<int, int>> const &rest, int n, vector<vector<char>> *char_board) {
    if (n >= rest.size()) return true;
    int row{rest[n].first}, col{rest[n].second};
    auto candidates = bitset<9>{"111111111"};
    for (int i = 0; i < 9; ++i) {
      Erase((*char_board)[row][i], &candidates);
      Erase((*char_board)[i][col], &candidates);
    }
    for (int i = row/3*3; i < row/3*3+3; ++i) {
      for (int j = col/3*3; j < col/3*3+3; ++j) {
        Erase((*char_board)[i][j], &candidates);
      }
    }
    for (int i = 0; i < 9; ++i) {
      if (candidates[i]) {
        (*char_board)[row][col] = i + '1';
        if (DFS(rest, n+1, char_board))
          return true;
      }
    }
    (*char_board)[row][col] = '.';
    return false;
  }
 public:
  void solveSudoku(vector<vector<char>>& char_board) {
    auto rest = vector<pair<int, int>>();
    for (int i = 0; i < 9; ++i) {
      for (int j = 0; j < 9; ++j) {
        if (char_board[i][j] == '.') {
          rest.emplace_back(i, j);
        }
      }
    }
    DFS(rest, 0, &char_board);
  }
};
// @lc code=end

