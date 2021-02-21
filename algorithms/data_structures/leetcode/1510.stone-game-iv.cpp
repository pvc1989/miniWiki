/*
 * @lc app=leetcode id=1510 lang=cpp
 *
 * [1510] Stone Game IV
 *
 * https://leetcode.com/problems/stone-game-iv/description/
 *
 * algorithms
 * Hard (58.84%)
 * Likes:    378
 * Dislikes: 25
 * Total Accepted:    25.8K
 * Total Submissions: 43.8K
 * Testcase Example:  '1\r'
 *
 * Alice and Bob take turns playing a game, with Alice starting first.
 * 
 * Initially, there are n stones in a pile.  On each player's turn, that player
 * makes a move consisting of removing any non-zero square number of stones in
 * the pile.
 * 
 * Also, if a player cannot make a move, he/she loses the game.
 * 
 * Given a positive integer n. Return True if and only if Alice wins the game
 * otherwise return False, assuming both players play optimally.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: n = 1
 * Output: true
 * Explanation: Alice can remove 1 stone winning the game because Bob doesn't
 * have any moves.
 * 
 * Example 2:
 * 
 * 
 * Input: n = 2
 * Output: false
 * Explanation: Alice can only remove 1 stone, after that Bob removes the last
 * one winning the game (2 -> 1 -> 0).
 * 
 * Example 3:
 * 
 * 
 * Input: n = 4
 * Output: true
 * Explanation: n is already a perfect square, Alice can win with one move,
 * removing 4 stones (4 -> 0).
 * 
 * 
 * Example 4:
 * 
 * 
 * Input: n = 7
 * Output: false
 * Explanation: Alice can't win the game if Bob plays optimally.
 * If Alice starts removing 4 stones, Bob will remove 1 stone then Alice should
 * remove only 1 stone and finally Bob removes the last one (7 -> 3 -> 2 -> 1
 * -> 0). 
 * If Alice starts removing 1 stone, Bob will remove 4 stones then Alice only
 * can remove 1 stone and finally Bob removes the last one (7 -> 6 -> 2 -> 1 ->
 * 0).
 * 
 * Example 5:
 * 
 * 
 * Input: n = 17
 * Output: false
 * Explanation: Alice can't win the game if Bob plays optimally.
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= n <= 10^5
 * 
 * 
 */

// @lc code=start
class Solution {
  bool lookup(vector<int> const &squares, int n, vector<char> *n_to_result) {
    auto result = (*n_to_result)[n];
    if (result == 0) {
      result = -1;
      for (auto x_times_x : squares) {
        auto new_n = n - x_times_x;
        if (new_n < 0) { break; }
        if (!lookup(squares, new_n, n_to_result)) { result = 1; break; }
      }
      (*n_to_result)[n] = result;
    }
    return result == 1;
  }
 public:
  bool winnerSquareGame(int n) {
    auto squares = vector<int>();
    int x = 0;
    do {
      ++x;
      squares.push_back(x * x);
    } while (squares.back() <= n);
    squares.pop_back();
    auto n_to_result = vector<char>(n+1);  // win := +1, lose := -1, todo := 0
    for (auto x_times_x : squares) {
      n_to_result[x_times_x] = 1;
    }
    return lookup(squares, n, &n_to_result);
  }
};
// @lc code=end

