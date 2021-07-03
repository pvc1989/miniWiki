/*
 * @lc app=leetcode id=363 lang=cpp
 *
 * [363] Max Sum of Rectangle No Larger Than K
 *
 * https://leetcode.com/problems/max-sum-of-rectangle-no-larger-than-k/description/
 *
 * algorithms
 * Hard (38.77%)
 * Likes:    1509
 * Dislikes: 95
 * Total Accepted:    68.2K
 * Total Submissions: 170.9K
 * Testcase Example:  '[[1,0,1],[0,-2,3]]\n2'
 *
 * Given an m x n matrix matrix and an integer k, return the max sum of a
 * rectangle in the matrix such that its sum is no larger than k.
 * 
 * It is guaranteed that there will be a rectangle with a sum no larger than
 * k.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: matrix = [[1,0,1],[0,-2,3]], k = 2
 * Output: 2
 * Explanation: Because the sum of the blue rectangle [[0, 1], [-2, 3]] is 2,
 * and 2 is the max number no larger than k (k = 2).
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: matrix = [[2,2,-1]], k = 3
 * Output: 3
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * m == matrix.length
 * n == matrix[i].length
 * 1 <= m, n <= 100
 * -100 <= matrix[i][j] <= 100
 * -10^5 <= k <= 10^5
 * 
 * 
 * 
 * Follow up: What if the number of rows is much larger than the number of
 * columns?
 * 
 */

// @lc code=start
class Solution {
  int n_rows_, n_cols_;
  array<array<int, 100>, 100> pos_to_sum_;

  void accumulate(vector<vector<int>> const& matrix) {
    pos_to_sum_[0][0] = matrix[0][0];
    for (int r = 1; r < n_rows_; ++r) {
      pos_to_sum_[r][0] = matrix[r][0] + pos_to_sum_[r-1][0];
    }
    for (int c = 1; c < n_cols_; ++c) {
      pos_to_sum_[0][c] = matrix[0][c] + pos_to_sum_[0][c-1];
    }
    for (int r = 1; r < n_rows_; ++r) {
      for (int c = 1; c < n_cols_; ++c) {
        pos_to_sum_[r][c] = matrix[r][c] + pos_to_sum_[r-1][c]
            + pos_to_sum_[r][c-1] - pos_to_sum_[r-1][c-1];
      }
    }
  }

  int getSum(int r1, int c1, int r2, int c2) {
    /* all indices are inclusive */
    return pos_to_sum_[r2][c2]
        - (r1 ? pos_to_sum_[r1-1][c2] : 0)
        - (c1 ? pos_to_sum_[r2][c1-1] : 0)
        + (r1 && c1 ? pos_to_sum_[r1][c1] : 0);
  }

 public:
  int maxSumSubmatrix(vector<vector<int>>& matrix, int k) {
    n_rows_ = matrix.size();
    n_cols_ = matrix[0].size();
    accumulate(matrix);
    int max_sum = INT_MIN;
    for (int height = 1; height <= n_rows_; ++height) {
      for (int r1 = 0; r1 + height <= n_rows_; ++r1) {
        auto r2 = r1 + height - 1;
        auto prefix_sums = set<int>{ 0 };
        for (int c2 = 0; c2 != n_cols_; ++c2) {
          auto curr_sum = getSum(r1, 0, r2, c2);
          auto prev_sum = curr_sum - k;
          // printf("curr_sum[%d][0][%d][%d] = %d, prev_sum = %d\n", r1, r2, c2, curr_sum, prev_sum);
          auto iter = prefix_sums.lower_bound(prev_sum);
          if (iter != prefix_sums.end()) {
            prev_sum = *iter;
            assert(curr_sum - k <= prev_sum);
            max_sum = max(max_sum, curr_sum - prev_sum);
            if (max_sum == k)
              return k;
          }
          prefix_sums.emplace(curr_sum);
        }
      }
    }
    assert(max_sum <= k);
    return max_sum;
  }
};
// @lc code=end

