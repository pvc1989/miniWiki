/*
 * @lc app=leetcode id=923 lang=cpp
 *
 * [923] 3Sum With Multiplicity
 *
 * https://leetcode.com/problems/3sum-with-multiplicity/description/
 *
 * algorithms
 * Medium (36.28%)
 * Likes:    521
 * Dislikes: 96
 * Total Accepted:    26.6K
 * Total Submissions: 70.2K
 * Testcase Example:  '[1,1,2,2,3,3,4,4,5,5]\n8'
 *
 * Given an integer array arr, and an integer target, return the number of
 * tuples i, j, k such that i < j < k and arr[i] + arr[j] + arr[k] == target.
 * 
 * As the answer can be very large, return it modulo 10^9 + 7.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: arr = [1,1,2,2,3,3,4,4,5,5], target = 8
 * Output: 20
 * Explanation: 
 * Enumerating by the values (arr[i], arr[j], arr[k]):
 * (1, 2, 5) occurs 8 times;
 * (1, 3, 4) occurs 8 times;
 * (2, 2, 4) occurs 2 times;
 * (2, 3, 3) occurs 2 times.
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: arr = [1,1,2,2,2,2], target = 5
 * Output: 12
 * Explanation: 
 * arr[i] = 1, arr[j] = arr[k] = 2 occurs 12 times:
 * We choose one 1 from [1,1] in 2 ways,
 * and two 2s from [2,2,2,2] in 6 ways.
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 3 <= arr.length <= 3000
 * 0 <= arr[i] <= 100
 * 0 <= target <= 300
 * 
 * 
 */

// @lc code=start
class Solution {
  static int64_t choose3(int64_t n) {
    return n * (n - 1) * (n - 2) / 6;
  }
  static int64_t choose2(int64_t n) {
    return n * (n - 1) / 2;
  }
 public:
  int threeSumMulti(vector<int>& a, int target) {
    int64_t total_count = 0;
    auto value_to_count = array<int64_t, 101>();
    for (auto value : a) {
      ++value_to_count[value];
    }
    for (int a_i = 0; a_i < 101; ++a_i) {
      auto a_i_count = value_to_count[a_i];
      if (a_i_count == 0) continue;
      if (a_i * 3 == target && a_i_count >= 3) {
        total_count += choose3(a_i_count);
      }
      for (int a_j = a_i + 1; a_j < 101; ++a_j) {
        auto a_j_count = value_to_count[a_j];
        if (a_j_count == 0) continue;
        auto a_i_plus_a_j = a_i + a_j;
        if (a_i + a_i_plus_a_j == target) {
          total_count += choose2(a_i_count) * a_j_count;
        }
        if (a_i_plus_a_j + a_j == target && a_j_count >= 2) {
          total_count += a_i_count * choose2(a_j_count);
        }
        auto a_k = target - a_i_plus_a_j;
        if (a_j < a_k && a_k < 101) {
          auto a_k_count = value_to_count[a_k];
          total_count += a_i_count * a_j_count * a_k_count;
        }
      }
    }
    return total_count % 1000000007;
  }
};
// @lc code=end

