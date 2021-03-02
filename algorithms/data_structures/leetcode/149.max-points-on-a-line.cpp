/*
 * @lc app=leetcode id=149 lang=cpp
 *
 * [149] Max Points on a Line
 *
 * https://leetcode.com/problems/max-points-on-a-line/description/
 *
 * algorithms
 * Hard (17.39%)
 * Likes:    12
 * Dislikes: 2
 * Total Accepted:    172.7K
 * Total Submissions: 992.5K
 * Testcase Example:  '[[1,1],[2,2],[3,3]]'
 *
 * Given an array of points where points[i] = [xi, yi] represents a point on
 * the X-Y plane, return the maximum number of points that lie on the same
 * straight line.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: points = [[1,1],[2,2],[3,3]]
 * Output: 3
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: points = [[1,1],[3,2],[5,3],[4,1],[2,3],[1,4]]
 * Output: 4
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= points.length <= 300
 * points[i].length == 2
 * -10^4 <= xi, yi <= 10^4
 * All the points are unique.
 * 
 * 
 */

// @lc code=start
class Solution {
 public:
  int maxPoints(vector<vector<int>>& points) {
    int n_points = points.size();
    if (n_points < 3) {
      return n_points;
    }
    int max_n_collinear = 0;
    sort(points.begin(), points.end());
    for (int i = 0; i != n_points; ++i) {
      for (int j = i + 1; j != n_points; ++j) {
        auto edge_ij = array<int, 2>{
            points[j][0] - points[i][0], points[j][1] - points[i][1]
        };
        int n_collinear = 2;
        for (int k = j + 1; k != n_points; ++k) {
          auto edge_jk = array<int, 2>{
              points[k][0] - points[j][0], points[k][1] - points[j][1]
          };
          if (edge_ij[0] * edge_jk[1] == edge_jk[0] * edge_ij[1]) {
            ++n_collinear;
          }
        }
        max_n_collinear = max(max_n_collinear, n_collinear);
      }
    }
    return max_n_collinear;
  }
};
// @lc code=end

