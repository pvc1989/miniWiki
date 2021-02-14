/*
 * @lc app=leetcode id=309 lang=cpp
 *
 * [309] Best Time to Buy and Sell Stock with Cooldown
 *
 * https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/description/
 *
 * algorithms
 * Medium (48.10%)
 * Likes:    3398
 * Dislikes: 106
 * Total Accepted:    183.4K
 * Total Submissions: 381.2K
 * Testcase Example:  '[1,2,3,0,2]'
 *
 * Say you have an array for which the i^th element is the price of a given
 * stock on day i.
 * 
 * Design an algorithm to find the maximum profit. You may complete as many
 * transactions as you like (ie, buy one and sell one share of the stock
 * multiple times) with the following restrictions:
 * 
 * 
 * You may not engage in multiple transactions at the same time (ie, you must
 * sell the stock before you buy again).
 * After you sell your stock, you cannot buy stock on next day. (ie, cooldown 1
 * day)
 * 
 * 
 * Example:
 * 
 * 
 * Input: [1,2,3,0,2]
 * Output: 3 
 * Explanation: transactions = [buy, sell, cooldown, buy, sell]
 * 
 */

// @lc code=start
class Solution {
 public:
  int maxProfit(vector<int>& prices) {
    int size = prices.size();
    if (size < 2) return 0;
    // curr == 1
    // sell := 0 stock is being holding.
    int max_profit_if_sell_curr = max(0, prices[1] - prices[0]);
    int max_profit_if_sell_prev = 0;
    // hold := 1 stock is being holding.
    int max_profit_if_hold_curr = -min(prices[0], prices[1]);
    int max_profit_if_hold_prev = -prices[0];
    auto next = prices.begin() + 1;
    while (++next != prices.end()) {
      auto price_next = *next;
      int max_profit_if_sell_next = max(
          max_profit_if_sell_curr,
          max_profit_if_hold_curr + price_next);
      int max_profit_if_hold_next = max(
          max_profit_if_hold_curr,
          max_profit_if_sell_prev - price_next);
      // shift one day off
      max_profit_if_sell_prev = max_profit_if_sell_curr;
      max_profit_if_hold_prev = max_profit_if_hold_curr;
      max_profit_if_sell_curr = max_profit_if_sell_next;
      max_profit_if_hold_curr = max_profit_if_hold_next;
    }
    return max_profit_if_sell_curr;
  }
};
// @lc code=end

