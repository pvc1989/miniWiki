/*
 * @lc app=leetcode id=123 lang=cpp
 *
 * [123] Best Time to Buy and Sell Stock III
 *
 * https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/description/
 *
 * algorithms
 * Hard (39.85%)
 * Likes:    3219
 * Dislikes: 83
 * Total Accepted:    275.2K
 * Total Submissions: 690.2K
 * Testcase Example:  '[3,3,5,0,0,3,1,4]'
 *
 * Say you have an array for which the i^th element is the price of a given
 * stock on day i.
 * 
 * Design an algorithm to find the maximum profit. You may complete at most two
 * transactions.
 * 
 * Note: You may not engage in multiple transactions at the same time (i.e.,
 * you must sell the stock before you buy again).
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: prices = [3,3,5,0,0,3,1,4]
 * Output: 6
 * Explanation: Buy on day 4 (price = 0) and sell on day 6 (price = 3), profit
 * = 3-0 = 3.
 * Then buy on day 7 (price = 1) and sell on day 8 (price = 4), profit = 4-1 =
 * 3.
 * 
 * Example 2:
 * 
 * 
 * Input: prices = [1,2,3,4,5]
 * Output: 4
 * Explanation: Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit
 * = 5-1 = 4.
 * Note that you cannot buy on day 1, buy on day 2 and sell them later, as you
 * are engaging multiple transactions at the same time. You must sell before
 * buying again.
 * 
 * 
 * Example 3:
 * 
 * 
 * Input: prices = [7,6,4,3,1]
 * Output: 0
 * Explanation: In this case, no transaction is done, i.e. max profit = 0.
 * 
 * 
 * Example 4:
 * 
 * 
 * Input: prices = [1]
 * Output: 0
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= prices.length <= 10^5
 * 0 <= prices[i] <= 10^5
 * 
 * 
 */

// @lc code=start
class Solution {
 public:
  int maxProfit(vector<int>& prices) {
    auto n = prices.size();
    auto max_profit_in_prefix = vector<int>(n);  // k -> max profit in [0, k)
    auto max_profit_in_suffix = vector<int>(n);  // k -> max profit in [k, n)
    auto min_price_in_prefix = prices.front();
    auto max_price_in_suffix = prices.back();
    for (int k = 1; k != n; ++k) {
      auto profit = prices[k] - min_price_in_prefix;
      max_profit_in_prefix[k] = max(max_profit_in_prefix[k-1], profit);
      min_price_in_prefix = min(min_price_in_prefix, prices[k]);
    }
    for (int k = n-2; k >= 0; --k) {
      auto profit = max_price_in_suffix - prices[k];
      max_profit_in_suffix[k] = max(max_profit_in_suffix[k+1], profit);
      max_price_in_suffix = max(max_price_in_suffix, prices[k]);
    }
    int max_profit = 0;
    for (int k = 0; k != n; ++k) {
      auto single_transaction_max_profix = max_profit_in_prefix[k];
      auto double_transaction_max_profit =
           single_transaction_max_profix + max_profit_in_suffix[k];
      max_profit = max(max_profit, 
                       max(single_transaction_max_profix,
                           double_transaction_max_profit));
    }
    return max_profit;
  }
};
// @lc code=end

