/*
 * @lc app=leetcode id=207 lang=cpp
 *
 * [207] Course Schedule
 *
 * https://leetcode.com/problems/course-schedule/description/
 *
 * algorithms
 * Medium (44.31%)
 * Likes:    5715
 * Dislikes: 238
 * Total Accepted:    580K
 * Total Submissions: 1.3M
 * Testcase Example:  '2\n[[1,0]]'
 *
 * There are a total of numCourses courses you have to take, labeled from 0 to
 * numCourses - 1. You are given an array prerequisites where prerequisites[i]
 * = [ai, bi] indicates that you must take course bi first if you want to take
 * course ai.
 * 
 * 
 * For example, the pair [0, 1], indicates that to take course 0 you have to
 * first take course 1.
 * 
 * 
 * Return true if you can finish all courses. Otherwise, return false.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: numCourses = 2, prerequisites = [[1,0]]
 * Output: true
 * Explanation: There are a total of 2 courses to take. 
 * To take course 1 you should have finished course 0. So it is possible.
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: numCourses = 2, prerequisites = [[1,0],[0,1]]
 * Output: false
 * Explanation: There are a total of 2 courses to take. 
 * To take course 1 you should have finished course 0, and to take course 0 you
 * should also have finished course 1. So it is impossible.
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= numCourses <= 10^5
 * 0 <= prerequisites.length <= 5000
 * prerequisites[i].length == 2
 * 0 <= ai, bi < numCourses
 * All the pairs prerequisites[i] are unique.
 * 
 * 
 */

// @lc code=start
class Solution {
  bool hasCycle(vector<vector<int>> const& adj, int u,
      vector<bool> *marked, vector<bool> *finished) {
    if (finished->at(u))
      return false;
    if (marked->at(u))
      return true;
    marked->at(u) = true;
    for (int v : adj[u]) {
      if (hasCycle(adj, v, marked, finished))
        return true;
    }
    marked->at(u) = false;
    finished->at(u) = true;
    return false;
  }

 public:
  bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
    auto adj = vector<vector<int>>(numCourses);
    for (auto& edge : prerequisites) {
      adj[edge[0]].emplace_back(edge[1]);
    }
    auto marked = vector<bool>(numCourses);
    auto finished = vector<bool>(numCourses);
    for (int i = 0; i != numCourses; ++i) {
      if (finished[i])
        continue;
      if (hasCycle(adj, i, &marked, &finished))
        return false;
    }
    return true;
  }
};
// @lc code=end

