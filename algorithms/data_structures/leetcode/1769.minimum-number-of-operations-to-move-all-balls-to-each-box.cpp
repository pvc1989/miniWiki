/*
 * @lc app=leetcode id=1769 lang=cpp
 *
 * [1769] Minimum Number of Operations to Move All Balls to Each Box
 *
 * https://leetcode.com/problems/minimum-number-of-operations-to-move-all-balls-to-each-box/description/
 *
 * algorithms
 * Medium (89.16%)
 * Likes:    138
 * Dislikes: 12
 * Total Accepted:    13.4K
 * Total Submissions: 15K
 * Testcase Example:  '"110"'
 *
 * You have n boxes. You are given a binary string boxes of length n, where
 * boxes[i] is '0' if the i^th box is empty, and '1' if it contains one ball.
 * 
 * In one operation, you can move one ball from a box to an adjacent box. Box i
 * is adjacent to box j if abs(i - j) == 1. Note that after doing so, there may
 * be more than one ball in some boxes.
 * 
 * Return an array answer of size n, where answer[i] is the minimum number of
 * operations needed to move all the balls to the i^th box.
 * 
 * Each answer[i] is calculated considering the initial state of the boxes.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: boxes = "110"
 * Output: [1,1,3]
 * Explanation: The answer for each box is as follows:
 * 1) First box: you will have to move one ball from the second box to the
 * first box in one operation.
 * 2) Second box: you will have to move one ball from the first box to the
 * second box in one operation.
 * 3) Third box: you will have to move one ball from the first box to the third
 * box in two operations, and move one ball from the second box to the third
 * box in one operation.
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: boxes = "001011"
 * Output: [11,8,5,4,3,4]
 * 
 * 
 * Constraints:
 * 
 * 
 * n == boxes.length
 * 1 <= n <= 2000
 * boxes[i] is either '0' or '1'.
 * 
 * 
 */

// @lc code=start
class Solution {
 public:
  vector<int> minOperations(string boxes) {
    auto n = boxes.size();
    auto prefix_count = vector<int>(n + 1);  // count `1` in [0, k)
    auto suffix_count = vector<int>(n + 1);  // count `1` in [k, n)
    int prefix_ops = 0;  // count ops to move `1` in [0, k) to k
    int suffix_ops = 0;  // count ops to move `1` in [k, n) to k
    for (int k = 0; k != n; ++k) {
      prefix_count[k + 1] = prefix_count[k] + (boxes[k] == '1' ? 1 : 0);
    }
    for (int k = n - 1; k >= 0; --k) {
      suffix_count[k] = suffix_count[k + 1] + (boxes[k] == '1' ? 1 : 0);
      suffix_ops += (boxes[k] == '1') ? k : 0;
    }
    auto operations = vector<int>(n);
    int k = 0;
    operations[k] = suffix_ops;
    while (++k != n) {
      // cout << prefix_ops << " += " << prefix_count[k] << ", ";
      // cout << suffix_ops << " -= " << suffix_count[k] << endl;
      prefix_ops += prefix_count[k];
      suffix_ops -= suffix_count[k];
      operations[k] = prefix_ops + suffix_ops;
    }
    // assert(suffix_ops == 0);
    return operations;
  }
};
// @lc code=end

