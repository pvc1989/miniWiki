/*
 * @lc app=leetcode id=128 lang=cpp
 *
 * [128] Longest Consecutive Sequence
 *
 * https://leetcode.com/problems/longest-consecutive-sequence/description/
 *
 * algorithms
 * Hard (46.20%)
 * Likes:    4686
 * Dislikes: 228
 * Total Accepted:    375K
 * Total Submissions: 811K
 * Testcase Example:  '[100,4,200,1,3,2]'
 *
 * Given an unsorted array of integers nums, return the length of the longest
 * consecutive elements sequence.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: nums = [100,4,200,1,3,2]
 * Output: 4
 * Explanation: The longest consecutive elements sequence is [1, 2, 3, 4].
 * Therefore its length is 4.
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: nums = [0,3,7,2,5,8,4,6,0,1]
 * Output: 9
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 0 <= nums.length <= 10^4
 * -10^9 <= nums[i] <= 10^9
 * 
 * 
 * 
 * Follow up: Could you implement the O(n) solution?
 */

// @lc code=start
class Solution {
 public:
  int longestConsecutive(vector<int>& nums) {
    if (nums.size() == 0) { return 0; }
    decltype(nums.end() - nums.begin()) max_length = 1;
    sort(nums.begin(), nums.end());
    auto slow = nums.begin();
    auto fast = slow + 1;
    while (fast != nums.end()) {
      switch (*fast - *(fast - 1)) {
      case 0:
        ++slow;
        break;
      case 1:
        break;
      default:
        max_length = max(max_length, fast - slow);
        slow = fast;
      };
      ++fast;
    }
    return max(max_length, fast - slow);
  }
};
// @lc code=end

