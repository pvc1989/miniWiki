/*
 * @lc app=leetcode id=823 lang=cpp
 *
 * [823] Binary Trees With Factors
 *
 * https://leetcode.com/problems/binary-trees-with-factors/description/
 *
 * algorithms
 * Medium (36.58%)
 * Likes:    632
 * Dislikes: 77
 * Total Accepted:    29.5K
 * Total Submissions: 67.2K
 * Testcase Example:  '[2,4]'
 *
 * Given an array of unique integers, arr, where each integer arr[i] is
 * strictly greater than 1.
 * 
 * We make a binary tree using these integers, and each number may be used for
 * any number of times. Each non-leaf node's value should be equal to the
 * product of the values of its children.
 * 
 * Return the number of binary trees we can make. The answer may be too large
 * so return the answer modulo 10^9 + 7.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: arr = [2,4]
 * Output: 3
 * Explanation: We can make these trees: [2], [4], [4, 2, 2]
 * 
 * Example 2:
 * 
 * 
 * Input: arr = [2,4,5,10]
 * Output: 7
 * Explanation: We can make these trees: [2], [4], [5], [10], [4, 2, 2], [10,
 * 2, 5], [10, 5, 2].
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= arr.length <= 1000
 * 2 <= arr[i] <= 10^9
 * 
 * 
 */

// @lc code=start
class Solution {
  using Int = int64_t;
 public:
  int numFactoredBinaryTrees(vector<int>& values) {
    Int total_count = 0, size = values.size();
    sort(values.begin(), values.end());
    constexpr Int kBase = 1000000007;
    auto counts = vector<Int>(size);
    for (Int index = 0; index != size; ++index) {
      Int curr_count = 1, head = 0, last = index;
      Int curr_value = values[index];
      while (head <= last) {
        auto diff = Int(values[head]) * values[last] - curr_value;
        if (diff == 0) {
          auto scale = (head == last ? 1 : 2);
          auto n_new_trees = scale * counts[head] * counts[last];
          curr_count = (curr_count + n_new_trees) % kBase;
        }
        if (diff >= 0) { --last; }
        if (diff <= 0) { ++head; }
      }
      counts[index] = curr_count;
      total_count = (total_count + curr_count) % kBase;
    }
    return total_count;
  }
};
// @lc code=end

