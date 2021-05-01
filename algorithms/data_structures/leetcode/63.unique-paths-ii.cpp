/*
 * @lc app=leetcode id=63 lang=cpp
 *
 * [63] Unique Paths II
 *
 * https://leetcode.com/problems/unique-paths-ii/description/
 *
 * algorithms
 * Medium (35.34%)
 * Likes:    2687
 * Dislikes: 295
 * Total Accepted:    367.7K
 * Total Submissions: 1M
 * Testcase Example:  '[[0,0,0],[0,1,0],[0,0,0]]'
 *
 * A robot is located at the top-left corner of a m x n grid (marked 'Start' in
 * the diagram below).
 * 
 * The robot can only move either down or right at any point in time. The robot
 * is trying to reach the bottom-right corner of the grid (marked 'Finish' in
 * the diagram below).
 * 
 * Now consider if some obstacles are added to the grids. How many unique paths
 * would there be?
 * 
 * An obstacle and space is marked as 1 and 0 respectively in the grid.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
 * Output: 2
 * Explanation: There is one obstacle in the middle of the 3x3 grid above.
 * There are two ways to reach the bottom-right corner:
 * 1. Right -> Right -> Down -> Down
 * 2. Down -> Down -> Right -> Right
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: obstacleGrid = [[0,1],[0,0]]
 * Output: 1
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * m == obstacleGrid.length
 * n == obstacleGrid[i].length
 * 1 <= m, n <= 100
 * obstacleGrid[i][j] is 0 or 1.
 * 
 * 
 */

// @lc code=start
class Solution {
 public:
  int uniquePathsWithObstacles(vector<vector<int>>& grid) {
    auto dp = array<array<int64_t, 100>, 100>();
    int row_max = grid.size() - 1, col_max = grid[0].size() - 1;
    dp[row_max][col_max] = !grid[row_max][col_max];
    for (int row = row_max - 1; row >= 0; --row)
      dp[row][col_max] = grid[row][col_max] ? 0 : dp[row + 1][col_max];
    for (int col = col_max - 1; col >= 0; --col)
      dp[row_max][col] = grid[row_max][col] ? 0 : dp[row_max][col + 1];
    for (int row = row_max - 1; row >= 0; --row)
      for (int col = col_max - 1; col >= 0; --col)
        dp[row][col] = grid[row][col] ? 0 : dp[row + 1][col] + dp[row][col + 1];
    return dp[0][0];
  }
};
// @lc code=end

