/*
 * @lc app=leetcode id=877 lang=cpp
 *
 * [877] Stone Game
 *
 * https://leetcode.com/problems/stone-game/description/
 *
 * algorithms
 * Medium (66.48%)
 * Likes:    1000
 * Dislikes: 1253
 * Total Accepted:    79.8K
 * Total Submissions: 119.9K
 * Testcase Example:  '[5,3,4,5]'
 *
 * Alex and Lee play a game with piles of stones.  There are an even number of
 * piles arranged in a row, and each pile has a positive integer number of
 * stones piles[i].
 * 
 * The objective of the game is to end with the most stones.  The total number
 * of stones is odd, so there are no ties.
 * 
 * Alex and Lee take turns, with Alex starting first.  Each turn, a player
 * takes the entire pile of stones from either the beginning or the end of the
 * row.  This continues until there are no more piles left, at which point the
 * person with the most stones wins.
 * 
 * Assuming Alex and Lee play optimally, return True if and only if Alex wins
 * the game.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: piles = [5,3,4,5]
 * Output: true
 * Explanation: 
 * Alex starts first, and can only take the first 5 or the last 5.
 * Say he takes the first 5, so that the row becomes [3, 4, 5].
 * If Lee takes 3, then the board is [4, 5], and Alex takes 5 to win with 10
 * points.
 * If Lee takes the last 5, then the board is [3, 4], and Alex takes 4 to win
 * with 9 points.
 * This demonstrated that taking the first 5 was a winning move for Alex, so we
 * return true.
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 2 <= piles.length <= 500
 * piles.length is even.
 * 1 <= piles[i] <= 500
 * sum(piles) is odd.
 * 
 * 
 */

// @lc code=start
class Solution {
 public:
  bool stoneGame(vector<int>& piles) {
    auto curr_step_scores = vector<int>(piles.size() + 1);
    auto next_step_scores = vector<int>(piles.size() + 1);
    for (int length = 2; length <= piles.size(); length += 2) {
      swap(curr_step_scores, next_step_scores);
      for (int head = piles.size() - length; 0 <= head; --head) {
        auto tail = head + length - 1;  // range := [head, tail]
        curr_step_scores[head] = max(
            piles[head] + min(next_step_scores[head+1] - piles[tail],
                              next_step_scores[head+2] - piles[head+1]),
            piles[tail] + min(next_step_scores[head+1] - piles[head],
                              next_step_scores[head  ] - piles[tail-1]));
      }
    }
    return curr_step_scores.front() > 0;
  }
};
// @lc code=end

