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
  int maxProfit(int n_transactions, vector<int>& prices) {
    int n_days = prices.size();
    if (n_transactions == 0 || n_days == 0) return 0;
    auto profit_if_sell = vector<int>(n_days);
    auto profit_if_hold = vector<int>(n_days);
    profit_if_hold[0] = -prices[0];
    for (int transaction = 0; transaction < n_transactions; ++transaction) {
      for (int day = 1; day < n_days; ++day) {
        profit_if_hold[day] =
            max(profit_if_hold[day - 1], profit_if_sell[day - 1] - prices[day]);
      }
      for (int day = 1; day < n_days; ++day) {
        profit_if_sell[day] =
            max(profit_if_sell[day - 1], profit_if_hold[day - 1] + prices[day]);
      }
    }
    return profit_if_sell.back();
  }
};
// @lc code=end

