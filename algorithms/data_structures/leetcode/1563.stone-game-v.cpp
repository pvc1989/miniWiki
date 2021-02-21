/*
 * @lc app=leetcode id=1563 lang=cpp
 *
 * [1563] Stone Game V
 *
 * https://leetcode.com/problems/stone-game-v/description/
 *
 * algorithms
 * Hard (40.06%)
 * Likes:    183
 * Dislikes: 45
 * Total Accepted:    8.1K
 * Total Submissions: 20.2K
 * Testcase Example:  '[6,2,3,4,5,5]'
 *
 * There are several stones arranged in a row, and each stone has an associated
 * value which is an integer given in the array stoneValue.
 * 
 * In each round of the game, Alice divides the row into two non-empty rows
 * (i.e. left row and right row), then Bob calculates the value of each row
 * which is the sum of the values of all the stones in this row. Bob throws
 * away the row which has the maximum value, and Alice's score increases by the
 * value of the remaining row. If the value of the two rows are equal, Bob lets
 * Alice decide which row will be thrown away. The next round starts with the
 * remaining row.
 * 
 * The game ends when there is only one stone remaining. Alice's is initially
 * zero.
 * 
 * Return the maximum score that Alice can obtain.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: stoneValue = [6,2,3,4,5,5]
 * Output: 18
 * Explanation: In the first round, Alice divides the row to [6,2,3], [4,5,5].
 * The left row has the value 11 and the right row has value 14. Bob throws
 * away the right row and Alice's score is now 11.
 * In the second round Alice divides the row to [6], [2,3]. This time Bob
 * throws away the left row and Alice's score becomes 16 (11 + 5).
 * The last round Alice has only one choice to divide the row which is [2],
 * [3]. Bob throws away the right row and Alice's score is now 18 (16 + 2). The
 * game ends because only one stone is remaining in the row.
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: stoneValue = [7,7,7,7,7,7,7]
 * Output: 28
 * 
 * 
 * Example 3:
 * 
 * 
 * Input: stoneValue = [4]
 * Output: 0
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= stoneValue.length <= 500
 * 1 <= stoneValue[i] <= 10^6
 * 
 * 
 */

// @lc code=start
class Solution {
  vector<int> const& convert(vector<int> *values) {
    for (auto riter = values->rbegin() + 1; riter != values->rend(); ++riter) {
      riter[0] += riter[-1];
    }
    return *values;
  }
  int getScore(vector<int> const &suffix_sum, int head, int tail,
      vector<vector<int>> *range_to_score) {
    auto score = (*range_to_score)[head][tail];
    if (score == -1) {
      score = 0;
      int half = head + 1;
      while (half < tail) {
        auto lepht_sum = suffix_sum[head] - suffix_sum[half];
        auto right_sum = suffix_sum[half] - suffix_sum[tail];
        if (lepht_sum < right_sum) {
          auto lepht_score = getScore(suffix_sum, head, half, range_to_score);
          score = max(score, lepht_sum + lepht_score);
        } else if (right_sum < lepht_sum) {
          auto right_score = getScore(suffix_sum, half, tail, range_to_score);
          score = max(score, right_sum + right_score);
        } else {  // right_sum == lepht_sum
          auto lepht_score = getScore(suffix_sum, head, half, range_to_score);
          auto right_score = getScore(suffix_sum, half, tail, range_to_score);
          score = max(score, right_sum + max(lepht_score, right_score));
        }
        ++half;
      }
      (*range_to_score)[head][tail] = score;
    }
    return score;
  }
 public:
  int stoneGameV(vector<int>& values) {
    auto size = values.size();
    values.push_back(0);
    auto const &suffix_sum = convert(&values);
    // range_to_score[head][tail] := max score in [head, tail)
    auto range_to_score = vector<vector<int>>(size, vector<int>(size + 1, -1));
    return getScore(suffix_sum, 0, size, &range_to_score);
  }
};
// @lc code=end

