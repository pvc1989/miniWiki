/*
 * @lc app=leetcode id=341 lang=cpp
 *
 * [341] Flatten Nested List Iterator
 *
 * https://leetcode.com/problems/flatten-nested-list-iterator/description/
 *
 * algorithms
 * Medium (54.74%)
 * Likes:    2156
 * Dislikes: 814
 * Total Accepted:    212.9K
 * Total Submissions: 385.7K
 * Testcase Example:  '[[1,1],2,[1,1]]'
 *
 * You are given a nested list of integers nestedList. Each element is either
 * an integer or a list whose elements may also be integers or other lists.
 * Implement an iterator to flatten it.
 * 
 * Implement the NestedIterator class:
 * 
 * 
 * NestedIterator(List<NestedInteger> nestedList) Initializes the iterator with
 * the nested list nestedList.
 * int next() Returns the next integer in the nested list.
 * boolean hasNext() Returns true if there are still some integers in the
 * nested list and false otherwise.
 * 
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: nestedList = [[1,1],2,[1,1]]
 * Output: [1,1,2,1,1]
 * Explanation: By calling next repeatedly until hasNext returns false, the
 * order of elements returned by next should be: [1,1,2,1,1].
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: nestedList = [1,[4,[6]]]
 * Output: [1,4,6]
 * Explanation: By calling next repeatedly until hasNext returns false, the
 * order of elements returned by next should be: [1,4,6].
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= nestedList.length <= 500
 * The values of the integers in the nested list is in the range [-10^6,
 * 10^6].
 * 
 * 
 */

// @lc code=start
/**
 * // This is the interface that allows for creating nested lists.
 * // You should not implement it, or speculate about its implementation
 * class NestedInteger {
 *   public:
 *     // Return true if this NestedInteger holds a single integer, rather than a nested list.
 *     bool isInteger() const;
 *
 *     // Return the single integer that this NestedInteger holds, if it holds a single integer
 *     // The result is undefined if this NestedInteger holds a nested list
 *     int getInteger() const;
 *
 *     // Return the nested list that this NestedInteger holds, if it holds a nested list
 *     // The result is undefined if this NestedInteger holds a single integer
 *     const vector<NestedInteger> &getList() const;
 * };
 */

class NestedIterator {
  vector<int> values_;
  int next_;

  void emplace(const vector<NestedInteger> &nested_list) {
    for (auto const& nested_integer : nested_list) {
      if (nested_integer.isInteger())
        values_.emplace_back(nested_integer.getInteger());
      else
        emplace(nested_integer.getList());
    }
  }

 public:
  NestedIterator(vector<NestedInteger> &nested_list) {
    emplace(nested_list);
    next_ = 0;
  }

  int next() {
    return values_.at(next_++);
  }

  bool hasNext() {
    return next_ != values_.size();
  }
};

/**
 * Your NestedIterator object will be instantiated and called as such:
 * NestedIterator i(nestedList);
 * while (i.hasNext()) cout << i.next();
 */
// @lc code=end

