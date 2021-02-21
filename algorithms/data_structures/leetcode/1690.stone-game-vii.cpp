/*
 * @lc app=leetcode id=1690 lang=cpp
 *
 * [1690] Stone Game VII
 *
 * https://leetcode.com/problems/stone-game-vii/description/
 *
 * algorithms
 * Medium (47.58%)
 * Likes:    182
 * Dislikes: 70
 * Total Accepted:    6.3K
 * Total Submissions: 13.3K
 * Testcase Example:  '[5,3,1,4,2]'
 *
 * Alice and Bob take turns playing a game, with Alice starting first.
 * 
 * There are n stones arranged in a row. On each player's turn, they can remove
 * either the leftmost stone or the rightmost stone from the row and receive
 * points equal to the sum of the remaining stones' values in the row. The
 * winner is the one with the higher score when there are no stones left to
 * remove.
 * 
 * Bob found that he will always lose this game (poor Bob, he always loses), so
 * he decided to minimize the score's difference. Alice's goal is to maximize
 * the difference in the score.
 * 
 * Given an array of integers stones where stones[i] represents the value of
 * the i^th stone from the left, return the difference in Alice and Bob's score
 * if they both play optimally.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: stones = [5,3,1,4,2]
 * Output: 6
 * Explanation: 
 * - Alice removes 2 and gets 5 + 3 + 1 + 4 = 13 points. Alice = 13, Bob = 0,
 * stones = [5,3,1,4].
 * - Bob removes 5 and gets 3 + 1 + 4 = 8 points. Alice = 13, Bob = 8, stones =
 * [3,1,4].
 * - Alice removes 3 and gets 1 + 4 = 5 points. Alice = 18, Bob = 8, stones =
 * [1,4].
 * - Bob removes 1 and gets 4 points. Alice = 18, Bob = 12, stones = [4].
 * - Alice removes 4 and gets 0 points. Alice = 18, Bob = 12, stones = [].
 * The score difference is 18 - 12 = 6.
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: stones = [7,90,5,1,100,10,10,2]
 * Output: 122
 * 
 * 
 * Constraints:
 * 
 * 
 * n == stones.length
 * 2 <= n <= 1000
 * 1 <= stones[i] <= 1000
 * 
 * 
 */

// @lc code=start
class Solution {
  vector<int> const* suffix_sum_;
  vector<int> const* convert(vector<int>* values) {
    for (auto riter = values->rbegin() + 1; riter != values->rend(); ++riter) {
      riter[0] += riter[-1];
    }
    return values;
  }
  int getRangeSum(int head, int tail) {
    return (*suffix_sum_)[head] - (*suffix_sum_)[tail];
  }
  vector<vector<int>> range_to_max_gap_;
  int getMaxGap(int head, int tail) {
    auto max_gap = range_to_max_gap_[head][tail];
    if (max_gap == INT_MIN) {
      if (head + 2 == tail) {
        auto half = head + 1;
        max_gap = max(getRangeSum(head, half), getRangeSum(half, tail));
      } else {
        // assert(head + 2 < tail);
        max_gap = max(
            getMinGap(head + 1, tail) + getRangeSum(head + 1, tail),
            getMinGap(head, tail - 1) + getRangeSum(head, tail - 1));
      }
      range_to_max_gap_[head][tail] = max_gap;
    }
    // printf("\nmax_gap[%d, %d) = %d", head, tail, max_gap);
    return max_gap;
  }
  vector<vector<int>> range_to_min_gap_;
  int getMinGap(int head, int tail) {
    auto min_gap = range_to_min_gap_[head][tail];
    if (min_gap == INT_MIN) {
      if (head + 2 == tail) {
        auto half = head + 1;
        min_gap = min(-getRangeSum(head, half), -getRangeSum(half, tail));
      } else {
        // assert(head + 2 < tail);
        min_gap = min(
            getMaxGap(head + 1, tail) - getRangeSum(head + 1, tail),
            getMaxGap(head, tail - 1) - getRangeSum(head, tail - 1));
      }
      range_to_min_gap_[head][tail] = min_gap;
    }
    // printf("\nmin_gap[%d, %d) = %d", head, tail, min_gap);
    return min_gap;
  }
 public:
  int stoneGameVII(vector<int>& stones) {
    auto size = stones.size();
    stones.push_back(0);
    suffix_sum_ = convert(&stones);
    range_to_max_gap_ = vector<vector<int>>(size, vector<int>(size+1, INT_MIN));
    range_to_min_gap_ = vector<vector<int>>(size, vector<int>(size+1, INT_MIN));
    return getMaxGap(0, size);
  }
};
// @lc code=end

