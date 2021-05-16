/*
 * @lc app=leetcode id=1354 lang=cpp
 *
 * [1354] Construct Target Array With Multiple Sums
 *
 * https://leetcode.com/problems/construct-target-array-with-multiple-sums/description/
 *
 * algorithms
 * Hard (31.26%)
 * Likes:    505
 * Dislikes: 65
 * Total Accepted:    21.2K
 * Total Submissions: 66.2K
 * Testcase Example:  '[9,3,5]'
 *
 * Given an array of integers target. From a starting array, A consisting of
 * all 1's, you may perform the following procedure :
 * 
 * 
 * let x be the sum of all elements currently in your array.
 * choose index i, such that 0 <= i < target.size and set the value of A at
 * index i to x.
 * You may repeat this procedure as many times as needed.
 * 
 * 
 * Return True if it is possible to construct the target array from A otherwise
 * return False.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: target = [9,3,5]
 * Output: true
 * Explanation: Start with [1, 1, 1] 
 * [1, 1, 1], sum = 3 choose index 1
 * [1, 3, 1], sum = 5 choose index 2
 * [1, 3, 5], sum = 9 choose index 0
 * [9, 3, 5] Done
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: target = [1,1,1,2]
 * Output: false
 * Explanation: Impossible to create target array from [1,1,1,1].
 * 
 * 
 * Example 3:
 * 
 * 
 * Input: target = [8,5]
 * Output: true
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * N == target.length
 * 1 <= target.length <= 5 * 10^4
 * 1 <= target[i] <= 10^9
 * 
 * 
 */


// @lc code=start
#ifdef NDEBUG
#define debug_printf(...) 
#else
#define debug_printf(...) printf(__VA_ARGS__)
#endif
class Solution {
 public:
  bool isPossible(vector<int>& target) {
    int n = target.size();
    if (n == 1)  // corner case
      return target[0] == 1;
    sort(target.begin(), target.end());
    auto after_1 = upper_bound(target.begin(), target.end(), 1);
    if (after_1 == target.end())
      return true;   // initial state
    auto first_n = lower_bound(target.begin(), target.end(), n);
    if (first_n != after_1)
      return false;  // there are numbers in (1, n), which cannot be sums
    assert(first_n != target.end());
    auto sums = priority_queue<int>(first_n, target.end());
    assert(!sums.empty());  // corollary of (first_n != target.end())
    // main part
    auto curr_max = sums.top();
    auto prev_sum = curr_max;
    auto curr_sum = accumulate(target.begin(), target.end(), (int64_t)0);
    auto tail_cut_sum = curr_sum - curr_max;
    assert(tail_cut_sum > 0);
    if (tail_cut_sum == 1)
      return true;
    if (tail_cut_sum >= prev_sum)  // prev_sum - tail_cut_sum == prev_max > 0
      return false;
    // if (tail_cut_sum < prev_max) then (prev_max is still the max)
    auto prev_max = prev_sum % tail_cut_sum;
    while (1 != sums.top()) {
      debug_printf("curr_sum - curr_max = %d = %d - %d\n", tail_cut_sum, curr_sum, curr_max);
      debug_printf("prev_sum - prev_max = %d = %d - %d\n", tail_cut_sum, prev_sum, prev_max);
      debug_printf("%d -> %d\n", curr_max, prev_max);
      sums.pop(); sums.emplace(prev_max); // replace curr_max with prev_max
      // update curr's and prev's
      curr_max = prev_max;
      curr_sum = curr_max + tail_cut_sum;
      prev_sum = curr_max = sums.top();
      tail_cut_sum = curr_sum - curr_max;
      if (tail_cut_sum == 0) {
        return false;
      }
      if (tail_cut_sum == 1) {
        debug_printf("tail_cut_sum == 1\n");
        return true;
      }
      if (tail_cut_sum >= prev_sum) {
        debug_printf("tail_cut_sum = %d, prev_sum = %d\n", tail_cut_sum, prev_sum);
        return prev_sum == 1;
      }
      prev_max = prev_sum % tail_cut_sum;
    }
    debug_printf("prev_max = %d, sums.top = %d\n", prev_max, sums.top());
    return 1 == sums.top();
  }
};
// @lc code=end

