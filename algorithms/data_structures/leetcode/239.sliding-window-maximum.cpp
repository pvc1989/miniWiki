/*
 * @lc app=leetcode id=239 lang=cpp
 *
 * [239] Sliding Window Maximum
 *
 * https://leetcode.com/problems/sliding-window-maximum/description/
 *
 * algorithms
 * Hard (44.60%)
 * Likes:    5213
 * Dislikes: 215
 * Total Accepted:    368.3K
 * Total Submissions: 825.4K
 * Testcase Example:  '[1,3,-1,-3,5,3,6,7]\n3'
 *
 * You are given an array of integers nums, there is a sliding window of size k
 * which is moving from the very left of the array to the very right. You can
 * only see the k numbers in the window. Each time the sliding window moves
 * right by one position.
 * 
 * Return the max sliding window.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
 * Output: [3,3,5,5,6,7]
 * Explanation: 
 * Window position                Max
 * ---------------               -----
 * [1  3  -1] -3  5  3  6  7       3
 * ⁠1 [3  -1  -3] 5  3  6  7       3
 * ⁠1  3 [-1  -3  5] 3  6  7       5
 * ⁠1  3  -1 [-3  5  3] 6  7       5
 * ⁠1  3  -1  -3 [5  3  6] 7       6
 * ⁠1  3  -1  -3  5 [3  6  7]      7
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: nums = [1], k = 1
 * Output: [1]
 * 
 * 
 * Example 3:
 * 
 * 
 * Input: nums = [1,-1], k = 1
 * Output: [1,-1]
 * 
 * 
 * Example 4:
 * 
 * 
 * Input: nums = [9,11], k = 2
 * Output: [11]
 * 
 * 
 * Example 5:
 * 
 * 
 * Input: nums = [4,-2], k = 2
 * Output: [4]
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= nums.length <= 10^5
 * -10^4 <= nums[i] <= 10^4
 * 1 <= k <= nums.length
 * 
 * 
 */

// @lc code=start
class Solution {
  struct Record {
    int score, index;
    Record(int s, int i) : score(s), index(i) {}
    bool operator<(Record const &rhs) const { return score < rhs.score; }
  };
 public:
  vector<int> maxSlidingWindow(vector<int> &index_to_value, int width) {
    auto max_values = vector<int>();
    auto max_pq = priority_queue<Record>();
    for (int index = 0; index < width - 1; ++index) {
      max_pq.emplace(index_to_value[index], index);
    }
    int last = index_to_value.size() - 1;
    for (int tail = width - 1; tail <= last; ++tail) {
      max_pq.emplace(index_to_value[tail], tail);
      while (max_pq.top().index <= tail - width) {
        max_pq.pop();
      }
      max_values.push_back(max_pq.top().score);
    }
    return max_values;
  }
};
// @lc code=end

