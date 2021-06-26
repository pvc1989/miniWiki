/*
 * @lc app=leetcode id=315 lang=cpp
 *
 * [315] Count of Smaller Numbers After Self
 *
 * https://leetcode.com/problems/count-of-smaller-numbers-after-self/description/
 *
 * algorithms
 * Hard (42.32%)
 * Likes:    3958
 * Dislikes: 119
 * Total Accepted:    184.5K
 * Total Submissions: 438.4K
 * Testcase Example:  '[5,2,6,1]'
 *
 * You are given an integer array nums and you have to return a new counts
 * array. The counts array has the property where counts[i] is the number of
 * smaller elements to the right of nums[i].
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: nums = [5,2,6,1]
 * Output: [2,1,1,0]
 * Explanation:
 * To the right of 5 there are 2 smaller elements (2 and 1).
 * To the right of 2 there is only 1 smaller element (1).
 * To the right of 6 there is 1 smaller element (1).
 * To the right of 1 there is 0 smaller element.
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: nums = [-1]
 * Output: [0]
 * 
 * 
 * Example 3:
 * 
 * 
 * Input: nums = [-1,-1]
 * Output: [0,0]
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= nums.length <= 10^5
 * -10^4 <= nums[i] <= 10^4
 * 
 * 
 */

// @lc code=start
class Solution {
  static constexpr int kOffset = 10001;
  static constexpr int kMaxIndex = 20001;
  array<int, kMaxIndex+1> binary_index_tree_{0};

  int lastBit(int index) {
    return index & -index;
  }

  void add(int index) {
    while (index <= kMaxIndex) {
      ++binary_index_tree_[index];
      index += lastBit(index);
    }
  }

  int get(int index) {
    int sum = 0;
    while (index) {
      sum += binary_index_tree_[index];
      index -= lastBit(index);
    }
    return sum;
  }

 public:  
  vector<int> countSmaller(vector<int>& nums) {
    int n = nums.size();
    auto counts = vector<int>(n);
    for (int i = n-1; i >= 0; --i) {
      auto index = nums[i] + kOffset;
      counts[i] = get(index-1);
      add(index);
    }
    return counts;
  }
};
// @lc code=end

