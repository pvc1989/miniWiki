/*
 * @lc app=leetcode id=204 lang=cpp
 *
 * [204] Count Primes
 *
 * https://leetcode.com/problems/count-primes/description/
 *
 * algorithms
 * Easy (32.30%)
 * Likes:    3015
 * Dislikes: 772
 * Total Accepted:    467.7K
 * Total Submissions: 1.4M
 * Testcase Example:  '10'
 *
 * Count the number of prime numbers less than a non-negative number, n.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: n = 10
 * Output: 4
 * Explanation: There are 4 prime numbers less than 10, they are 2, 3, 5, 7.
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: n = 0
 * Output: 0
 * 
 * 
 * Example 3:
 * 
 * 
 * Input: n = 1
 * Output: 0
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 0 <= n <= 5 * 10^6
 * 
 * 
 */

// @lc code=start
class Solution {
 public:
  int countPrimes(int n) {
    if (n <= 2)
      return 0;
    auto primes = vector<int>{2};
    for (int x = 3; x < n; ++x) {
      bool x_is_prime = true;
      for (auto p : primes) {
        if (x % p == 0) {
          x_is_prime = false;
          break;
        }
        if (p * p > x) {
          break;
        }
      }
      // printf("%d is %s prime.\n", x, x_is_prime ? "a" : "not a");
      if (x_is_prime)
        primes.push_back(x);
    }
    return primes.size();
  }
};
// @lc code=end

