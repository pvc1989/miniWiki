/*
 * @lc app=leetcode id=1383 lang=cpp
 *
 * [1383] Maximum Performance of a Team
 *
 * https://leetcode.com/problems/maximum-performance-of-a-team/description/
 *
 * algorithms
 * Hard (35.82%)
 * Likes:    901
 * Dislikes: 36
 * Total Accepted:    30.9K
 * Total Submissions: 74.8K
 * Testcase Example:  '6\n[2,10,3,1,5,8]\n[5,4,3,9,7,2]\n2'
 *
 * You are given two integers n and k and two integer arrays speed and
 * efficiency both of length n. There are n engineers numbered from 1 to n.
 * speed[i] and efficiency[i] represent the speed and efficiency of the i^th
 * engineer respectively.
 * 
 * Choose at most k different engineers out of the n engineers to form a team
 * with the maximum performance.
 * 
 * The performance of a team is the sum of their engineers' speeds multiplied
 * by the minimum efficiency among their engineers.
 * 
 * Return the maximum performance of this team. Since the answer can be a huge
 * number, return it modulo 10^9 + 7.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: n = 6, speed = [2,10,3,1,5,8], efficiency = [5,4,3,9,7,2], k = 2
 * Output: 60
 * Explanation: 
 * We have the maximum performance of the team by selecting engineer 2 (with
 * speed=10 and efficiency=4) and engineer 5 (with speed=5 and efficiency=7).
 * That is, performance = (10 + 5) * min(4, 7) = 60.
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: n = 6, speed = [2,10,3,1,5,8], efficiency = [5,4,3,9,7,2], k = 3
 * Output: 68
 * Explanation:
 * This is the same example as the first but k = 3. We can select engineer 1,
 * engineer 2 and engineer 5 to get the maximum performance of the team. That
 * is, performance = (2 + 10 + 5) * min(5, 4, 7) = 68.
 * 
 * 
 * Example 3:
 * 
 * 
 * Input: n = 6, speed = [2,10,3,1,5,8], efficiency = [5,4,3,9,7,2], k = 4
 * Output: 72
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= <= k <= n <= 10^5
 * speed.length == n
 * efficiency.length == n
 * 1 <= speed[i] <= 10^5
 * 1 <= efficiency[i] <= 10^8
 * 
 * 
 */

// @lc code=start
class Solution {
 public:
  int maxPerformance(int n, vector<int>& speed, vector<int>& efficiency, int k) {
    auto pairs = vector<pair<int, int>>(n);
    for (int i = 0; i != n; ++i) {
      pairs[i] = { speed[i], efficiency[i] };
    }
    sort(pairs.begin(), pairs.end(), [](auto& left, auto& right){
      return left.second > right.second;
    });
    auto speed_minpq = priority_queue<int, vector<int>, greater<int>>();
    speed_minpq.emplace(pairs[0].first);
    int64_t curr_sum = speed_minpq.top();
    auto curr_perf = curr_sum * pairs[0].second;
    auto max_perf = curr_perf;
    for (int i = 1; i < n; ++i) {
      auto [curr_speed, curr_effi] = pairs[i];
      if (speed_minpq.size() == k) {
        curr_sum -= speed_minpq.top();
        speed_minpq.pop();
      }
      speed_minpq.emplace(curr_speed);
      curr_sum += curr_speed;
      curr_perf = curr_sum * curr_effi;
      max_perf = max(max_perf, curr_perf);
    }
    return max_perf % 1000000007L;
  }
};
// @lc code=end

