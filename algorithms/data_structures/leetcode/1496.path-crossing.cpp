/*
 * @lc app=leetcode id=1496 lang=cpp
 *
 * [1496] Path Crossing
 *
 * https://leetcode.com/problems/path-crossing/description/
 *
 * algorithms
 * Easy (55.27%)
 * Likes:    262
 * Dislikes: 6
 * Total Accepted:    23.2K
 * Total Submissions: 42K
 * Testcase Example:  '"NES"'
 *
 * Given a string path, where path[i] = 'N', 'S', 'E' or 'W', each representing
 * moving one unit north, south, east, or west, respectively. You start at the
 * origin (0, 0) on a 2D plane and walk on the path specified by path.
 * 
 * Return True if the path crosses itself at any point, that is, if at any time
 * you are on a location you've previously visited. Return False otherwise.
 * 
 * 
 * Example 1:
 * 
 * 
 * 
 * 
 * Input: path = "NES"
 * Output: false 
 * Explanation: Notice that the path doesn't cross any point more than once.
 * 
 * 
 * Example 2:
 * 
 * 
 * 
 * 
 * Input: path = "NESWW"
 * Output: true
 * Explanation: Notice that the path visits the origin twice.
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= path.length <= 10^4
 * path will only consist of characters in {'N', 'S', 'E', 'W}
 * 
 * 
 */

// @lc code=start
class Solution {
  static u_int64_t makePoint(u_int64_t x, u_int64_t y) {
    return (x << 32) | (y & 0xFFFFFFFF);
  }
 public:
  bool isPathCrossing(string path) {
    auto points = unordered_set<u_int32_t>();
    u_int32_t x = 0, y = 0;
    points.emplace(0);
    for (char c : path) {
      switch (c) {
      case 'E': ++x; break;
      case 'N': ++y; break;
      case 'S': --y; break;
      case 'W': --x; break;
      }
      auto p = (x << 16) | (y & 0xFFFF);
      auto iter = points.find(p);
      if (iter != points.end()) {
        return true;
      } else {
        points.emplace_hint(iter, p);
      }
    }
    return false;
  }
};
// @lc code=end

