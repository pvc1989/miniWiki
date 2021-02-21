/*
 * @lc app=leetcode id=1140 lang=cpp
 *
 * [1140] Stone Game II
 *
 * https://leetcode.com/problems/stone-game-ii/description/
 *
 * algorithms
 * Medium (64.72%)
 * Likes:    713
 * Dislikes: 176
 * Total Accepted:    24.6K
 * Total Submissions: 38.1K
 * Testcase Example:  '[2,7,9,4,4]'
 *
 * Alice and Bob continue their games with piles of stones.  There are a number
 * of piles arranged in a row, and each pile has a positive integer number of
 * stones piles[i].  The objective of the game is to end with the most
 * stones. 
 * 
 * Alice and Bob take turns, with Alice starting first.  Initially, M = 1.
 * 
 * On each player's turn, that player can take all the stones in the first X
 * remaining piles, where 1 <= X <= 2M.  Then, we set M = max(M, X).
 * 
 * The game continues until all the stones have been taken.
 * 
 * Assuming Alice and Bob play optimally, return the maximum number of stones
 * Alice can get.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: piles = [2,7,9,4,4]
 * Output: 10
 * Explanation:  If Alice takes one pile at the beginning, Bob takes two piles,
 * then Alice takes 2 piles again. Alice can get 2 + 4 + 4 = 10 piles in total.
 * If Alice takes two piles at the beginning, then Bob can take all three piles
 * left. In this case, Alice get 2 + 7 = 9 piles in total. So we return 10
 * since it's larger. 
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: piles = [1,2,3,4,5,100]
 * Output: 104
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= piles.length <= 100
 * 1 <= piles[i] <= 10^4
 * 
 * 
 */

// @lc code=start
class Solution {
  vector<int> const& convert(vector<int> *piles) {
    for (auto riter = piles->rbegin() + 1; riter != piles->rend(); ++riter) {
      riter[0] += riter[-1];
    }
    return *piles;
  }
  int lookup(vector<int> const &suffix_sum, int head, int M,
             vector<vector<int>> *dp) {
    int N = suffix_sum.size();
    if (head == N) return 0;
    int max_stones = (*dp)[head][min(M, N)];
    if (max_stones == -1) {
      int X_max = min(2 * M, N - head);
      for (int X = 1; X <= X_max; ++X) {
        int stones = suffix_sum[head] - /* Bob plays optimally */
            lookup(suffix_sum, head + X, max(M, X), dp);
        max_stones = max(max_stones, stones);
        // printf("suffix_sum[%d] = %d, X = %d, stones = %d\n", head, suffix_sum[head], X, stones);
      }
      (*dp)[head][M] = max_stones;
      // printf("suffix_sum[%d] = %d, X_max = %d, max_stones = %d\n", head, suffix_sum[head], X_max, max_stones);
    }
    return max_stones;
  }
 public:
  int stoneGameII(vector<int> &piles) {
    auto N = piles.size();
    // dp[head][M] := maximum number of stones to get in [head, N)
    auto dp = vector<vector<int>>(N, vector<int>(N+ 1/* max_m */, -1));
    for (int M = 1; M <= N; ++M)
      dp.back()[M] = piles.back();
    auto const &suffix_sum = convert(&piles);
    return lookup(suffix_sum, 0, 1, &dp);
  }
};
// @lc code=end

