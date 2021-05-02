/*
 * @lc app=leetcode id=1192 lang=cpp
 *
 * [1192] Critical Connections in a Network
 *
 * https://leetcode.com/problems/critical-connections-in-a-network/description/
 *
 * algorithms
 * Hard (50.23%)
 * Likes:    2234
 * Dislikes: 112
 * Total Accepted:    104.1K
 * Total Submissions: 206.4K
 * Testcase Example:  '4\n[[0,1],[1,2],[2,0],[1,3]]'
 *
 * There are n servers numbered from 0 to n-1 connected by undirected
 * server-to-server connections forming a network where connections[i] = [a, b]
 * represents a connection between servers a and b. Any server can reach any
 * other server directly or indirectly through the network.
 * 
 * A critical connection is a connection that, if removed, will make some
 * server unable to reach some other server.
 * 
 * Return all critical connections in the network in any order.
 * 
 * 
 * Example 1:
 * 
 * 
 * 
 * 
 * Input: n = 4, connections = [[0,1],[1,2],[2,0],[1,3]]
 * Output: [[1,3]]
 * Explanation: [[3,1]] is also accepted.
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= n <= 10^5
 * n-1 <= connections.length <= 10^5
 * connections[i][0] != connections[i][1]
 * There are no repeated connections.
 * 
 * 
 */

// @lc code=start
class Solution {
  vector<vector<int>> v_to_neighbors_;
  vector<int> v_to_new_rank_;  // the value of `rank` when `v` was found
  vector<int> v_to_min_rank_;  // the min `rank` in all vertices that `v` can reach
  int rank_;

 private:
  void dfs(int prev, int curr, vector<vector<int>>* bridges) {
    assert(v_to_new_rank_[curr] == -1);
    v_to_new_rank_[curr] = v_to_min_rank_[curr] = rank_++;
    for (auto next : v_to_neighbors_[curr]) {
      if (next == prev)
        continue;  // ignore back edge
      if (v_to_new_rank_[next] == -1) {
        dfs(curr, next, bridges);
        if (v_to_new_rank_[next] == v_to_min_rank_[next]) {
          bridges->emplace_back(2);
          bridges->back()[0] = curr;
          bridges->back()[1] = next;
        }
      }
      v_to_min_rank_[curr] = min(v_to_min_rank_[curr], v_to_min_rank_[next]);
    }
  }
 public:
  vector<vector<int>> criticalConnections(int n, vector<vector<int>>& connections) {
    v_to_neighbors_.resize(n);
    v_to_new_rank_.resize(n, -1);
    v_to_min_rank_.resize(n, -1);
    rank_ = 0;
    // build adj lists
    for (auto& edge : connections) {
      int u = edge[0], v = edge[1];
      v_to_neighbors_[u].emplace_back(v);
      v_to_neighbors_[v].emplace_back(u);
    }
    // find bridges
    vector<vector<int>> bridges;
    for (int v = 0; v != n; ++v)
      if (v_to_new_rank_[v] == -1)
        dfs(v, v, &bridges);
    return bridges;
  }
};
// @lc code=end

