/*
 * @lc app=leetcode id=354 lang=cpp
 *
 * [354] Russian Doll Envelopes
 *
 * https://leetcode.com/problems/russian-doll-envelopes/description/
 *
 * algorithms
 * Hard (36.51%)
 * Likes:    1765
 * Dislikes: 53
 * Total Accepted:    90.5K
 * Total Submissions: 245.3K
 * Testcase Example:  '[[5,4],[6,4],[6,7],[2,3]]'
 *
 * You are given a 2D array of integers envelopes where envelopes[i] = [wi, hi]
 * represents the width and the height of an envelope.
 * 
 * One envelope can fit into another if and only if both the width and height
 * of one envelope is greater than the width and height of the other envelope.
 * 
 * Return the maximum number of envelopes can you Russian doll (i.e., put one
 * inside the other).
 * 
 * Note: You cannot rotate an envelope.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: envelopes = [[5,4],[6,4],[6,7],[2,3]]
 * Output: 3
 * Explanation: The maximum number of envelopes you can Russian doll is 3
 * ([2,3] => [5,4] => [6,7]).
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: envelopes = [[1,1],[1,1],[1,1]]
 * Output: 1
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= envelopes.length <= 5000
 * envelopes[i].length == 2
 * 1 <= wi, hi <= 10^4
 * 
 * 
 */

// @lc code=start
class Solution {
  int visit(const vector<vector<int>>& adj, int i, vector<int>* depths) {
    auto depth_i = depths->at(i);
    if (depth_i == -1) {
      depth_i = 0;
      for (int j : adj[i]) {
        depth_i = max(depth_i, visit(adj, j, depths));
      }
      depths->at(i) = ++depth_i;
    }
    return depth_i;
  }
  int maxEnvelopesByDP(vector<vector<int>>& envelopes) {
    auto n = envelopes.size();
    // build the DAG
    auto adj = vector<vector<int>>(n);
    for (int i = 0; i != n; ++i) {
      auto w_i = envelopes[i][0];
      auto h_i = envelopes[i][1];
      for (int j = 0; j != n; ++j) {
        if (w_i < envelopes[j][0] && h_i < envelopes[j][1]) {
          adj[i].push_back(j);
        }
      }
    }
    // find max_depth by DP
    auto depths = vector<int>(n, -1);
    auto max_depth = 1;
    for (int i = 0; i != n; ++i) {
      max_depth = max(max_depth, visit(adj, i, &depths));
    }
    return max_depth;
  }

 public:
  int maxEnvelopes(vector<vector<int>>& envelopes) {
    auto cmp = [](auto& e_i, auto& e_j) {
      return e_i[0] < e_j[0] || e_i[0] == e_j[0] && e_i[1] > e_j[1];
    };
    sort(envelopes.begin(), envelopes.end(), cmp);
    auto heights = vector<int>();
    for (auto& e : envelopes) {
      auto h = e[1];
      auto iter = lower_bound(heights.begin(), heights.end(), h);
      if (iter != heights.end()) {
        *iter = h;
      } else {
        heights.emplace_back(h);
      }
    }
    return heights.size();
  }
};
// @lc code=end

