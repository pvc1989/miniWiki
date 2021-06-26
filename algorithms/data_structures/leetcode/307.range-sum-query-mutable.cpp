/*
 * @lc app=leetcode id=307 lang=cpp
 *
 * [307] Range Sum Query - Mutable
 *
 * https://leetcode.com/problems/range-sum-query-mutable/description/
 *
 * algorithms
 * Medium (37.68%)
 * Likes:    2124
 * Dislikes: 117
 * Total Accepted:    152.8K
 * Total Submissions: 405.3K
 * Testcase Example:  '["NumArray","sumRange","update","sumRange"]\n[[[1,3,5]],[0,2],[1,2],[0,2]]'
 *
 * Given an integer array nums, handle multiple queries of the following
 * types:
 * 
 * 
 * Update the value of an element in nums.
 * Calculate the sum of the elements of nums between indices left and right
 * inclusive where left <= right.
 * 
 * 
 * Implement the NumArray class:
 * 
 * 
 * NumArray(int[] nums) Initializes the object with the integer array nums.
 * void update(int index, int val) Updates the value of nums[index] to be
 * val.
 * int sumRange(int left, int right) Returns the sum of the elements of nums
 * between indices left and right inclusive (i.e. nums[left] + nums[left + 1] +
 * ... + nums[right]).
 * 
 * 
 * 
 * Example 1:
 * 
 * 
 * Input
 * ["NumArray", "sumRange", "update", "sumRange"]
 * [[[1, 3, 5]], [0, 2], [1, 2], [0, 2]]
 * Output
 * [null, 9, null, 8]
 * 
 * Explanation
 * NumArray numArray = new NumArray([1, 3, 5]);
 * numArray.sumRange(0, 2); // return 1 + 3 + 5 = 9
 * numArray.update(1, 2);   // nums = [1, 2, 5]
 * numArray.sumRange(0, 2); // return 1 + 2 + 5 = 8
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= nums.length <= 3 * 10^4
 * -100 <= nums[i] <= 100
 * 0 <= index < nums.length
 * -100 <= val <= 100
 * 0 <= left <= right < nums.length
 * At most 3 * 10^4 calls will be made to update and sumRange.
 * 
 * 
 */

// @lc code=start
class NumArray {
  int max_index_;
  vector<int> binary_index_tree_;

  static int lastBit(int index) {
    return index & -index;
  }
  void add(int index, int val) {
    while (index <= max_index_) {
      binary_index_tree_[index] += val;
      index += lastBit(index);
    }
  }
  int get(int index) const {
    int sum = 0;
    while (index) {
      sum += binary_index_tree_[index];
      index -= lastBit(index);
    }
    return sum;
  }

 public:
  NumArray(vector<int>& nums) {
    max_index_ = nums.size();
    binary_index_tree_.resize(max_index_+1);
    for (int i = nums.size()-1; i >= 0; --i) {
      add(i+1, nums[i]);
    }
  }
  void update(int index, int val) {
    ++index;
    val -= get(index) - get(index-1);
    add(index, val);
  }
  int sumRange(int left, int right) {
    return get(right+1) - get(left);
  }
};

/**
 * Your NumArray object will be instantiated and called as such:
 * NumArray* obj = new NumArray(nums);
 * obj->update(index,val);
 * int param_2 = obj->sumRange(left,right);
 */
// @lc code=end

