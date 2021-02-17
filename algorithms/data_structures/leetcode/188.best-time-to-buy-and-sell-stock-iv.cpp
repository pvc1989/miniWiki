/*
 * @lc app=leetcode id=188 lang=cpp
 *
 * [188] Best Time to Buy and Sell Stock IV
 *
 * https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/description/
 *
 * algorithms
 * Hard (29.54%)
 * Likes:    2229
 * Dislikes: 135
 * Total Accepted:    171K
 * Total Submissions: 578.5K
 * Testcase Example:  '2\n[2,4,1]'
 *
 * You are given an integer array prices where prices[i] is the price of a
 * given stock on the i^th day.
 * 
 * Design an algorithm to find the maximum profit. You may complete at most k
 * transactions.
 * 
 * Notice that you may not engage in multiple transactions simultaneously
 * (i.e., you must sell the stock before you buy again).
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: k = 2, prices = [2,4,1]
 * Output: 2
 * Explanation: Buy on day 1 (price = 2) and sell on day 2 (price = 4), profit
 * = 4-2 = 2.
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: k = 2, prices = [3,2,6,5,0,3]
 * Output: 7
 * Explanation: Buy on day 2 (price = 2) and sell on day 3 (price = 6), profit
 * = 6-2 = 4. Then buy on day 5 (price = 0) and sell on day 6 (price = 3),
 * profit = 3-0 = 3.
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 0 <= k <= 100
 * 0 <= prices.length <= 1000
 * 0 <= prices[i] <= 1000
 * 
 * 
 */

// @lc code=start
class Solution {
 public:
  int maxProfit(int max_transactions, vector<int>& prices) {
    int n = prices.size();
    if (max_transactions == 0 || n == 0) return 0;
    // [b, s) -> max profit of buying at/after `b` and selling before `s`
    auto max_profit_in_range = vector<vector<int>>(n, vector<int>(n+1));
    for (int buy = 0; buy != n; ++buy) {
      auto min_price_in_prefix = prices[buy];
      for (int sell = buy + 1; sell != n;/* all ready incremented */) {
        auto new_price = prices[sell];
        max_profit_in_range[buy][++sell] = max(
            new_price - min_price_in_prefix/* sell at `sell` */,
            max_profit_in_range[buy][sell]/* or already sold */);
        min_price_in_prefix = min(min_price_in_prefix, new_price);
      }
    }
    // k -> max profit in [0, k) with at most `t` transactions
    auto max_profit_in_prefix = max_profit_in_range.front();
    for (int t = min(max_transactions, n/2); 2 <= t; --t) {
      for (int last_sell = n; 0 < last_sell; --last_sell) {
        auto max_profit = max_profit_in_prefix[last_sell];
        for (int last_buy = last_sell - 1; 0 <= last_buy; --last_buy) {
          auto one_more_transaction_max_profit =
              max_profit_in_range[last_buy][last_sell] +
              max_profit_in_prefix[last_buy];
          max_profit = max(max_profit, one_more_transaction_max_profit);
        }
        max_profit_in_prefix[last_sell] = max_profit;
      }
    }
    return max_profit_in_prefix.back();
  }
};
// @lc code=end

