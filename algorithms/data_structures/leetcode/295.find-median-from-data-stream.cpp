/*
 * @lc app=leetcode id=295 lang=cpp
 *
 * [295] Find Median from Data Stream
 *
 * https://leetcode.com/problems/find-median-from-data-stream/description/
 *
 * algorithms
 * Hard (46.76%)
 * Likes:    3782
 * Dislikes: 71
 * Total Accepted:    273.1K
 * Total Submissions: 582.6K
 * Testcase Example:  '["MedianFinder","addNum","addNum","findMedian","addNum","findMedian"]\n' +
  '[[],[1],[2],[],[3],[]]'
 *
 * Median is the middle value in an ordered integer list. If the size of the
 * list is even, there is no middle value. So the median is the mean of the two
 * middle value.
 * For example,
 * 
 * [2,3,4], the median is 3
 * 
 * [2,3], the median is (2 + 3) / 2 = 2.5
 * 
 * Design a data structure that supports the following two operations:
 * 
 * 
 * void addNum(int num) - Add a integer number from the data stream to the data
 * structure.
 * double findMedian() - Return the median of all elements so far.
 * 
 * 
 * 
 * 
 * Example:
 * 
 * 
 * addNum(1)
 * addNum(2)
 * findMedian() -> 1.5
 * addNum(3) 
 * findMedian() -> 2
 * 
 * 
 * 
 * 
 * Follow up:
 * 
 * 
 * If all integer numbers from the stream are between 0 and 100, how would you
 * optimize it?
 * If 99% of all integer numbers from the stream are between 0 and 100, how
 * would you optimize it?
 * 
 * 
 */

// @lc code=start
class MedianFinder {
  priority_queue<int> maxpq_of_lepht_;  // ordered[0, n/2)
  priority_queue<int, vector<int>,
        greater<int>> minpq_of_right_;  // ordered[n/2, n)
 public:
  /** initialize your data structure here. */
  MedianFinder() {
  }
  void addNum(int num) {
    if (minpq_of_right_.empty() || minpq_of_right_.top() <= num) {
      minpq_of_right_.emplace(num);
    } else {
      assert(num < minpq_of_right_.top());
      maxpq_of_lepht_.emplace(num);
    }
    if (maxpq_of_lepht_.size() > minpq_of_right_.size()) {
      minpq_of_right_.emplace(maxpq_of_lepht_.top());
      maxpq_of_lepht_.pop();
    } else if (maxpq_of_lepht_.size() < minpq_of_right_.size() - 1) {
      maxpq_of_lepht_.emplace(minpq_of_right_.top());
      minpq_of_right_.pop();
    }
    assert(maxpq_of_lepht_.size() <= minpq_of_right_.size());
    assert(maxpq_of_lepht_.size() >= minpq_of_right_.size() - 1);
  }
  
  double findMedian() {
    return maxpq_of_lepht_.size() == minpq_of_right_.size()
        ? (maxpq_of_lepht_.top() + minpq_of_right_.top()) / 2.0
        : minpq_of_right_.top();
  }
};

/**
 * Your MedianFinder object will be instantiated and called as such:
 * MedianFinder* obj = new MedianFinder();
 * obj->addNum(num);
 * double param_2 = obj->findMedian();
 */
// @lc code=end

