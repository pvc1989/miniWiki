/*
 * @lc app=leetcode id=943 lang=cpp
 *
 * [943] Find the Shortest Superstring
 *
 * https://leetcode.com/problems/find-the-shortest-superstring/description/
 *
 * algorithms
 * Hard (43.43%)
 * Likes:    600
 * Dislikes: 85
 * Total Accepted:    14.6K
 * Total Submissions: 33.1K
 * Testcase Example:  '["alex","loves","leetcode"]'
 *
 * Given an array of strings words, return the smallest string that contains
 * each string in words as a substring. If there are multiple valid strings of
 * the smallest length, return any of them.
 * 
 * You may assume that no string in words is a substring of another string in
 * words.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: words = ["alex","loves","leetcode"]
 * Output: "alexlovesleetcode"
 * Explanation: All permutations of "alex","loves","leetcode" would also be
 * accepted.
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: words = ["catg","ctaagt","gcta","ttca","atgcatc"]
 * Output: "gctaagttcatgcatc"
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= words.length <= 12
 * 1 <= words[i].length <= 20
 * words[i] consists of lowercase English letters.
 * All the strings of words are unique.
 * 
 * 
 */

// @lc code=start
struct Record {
  int dist{-1};
  int next{-1};
};
class Solution {
  int size_;
  int mask_all_;
  array<array<int, 12>, 12> dist_;
  array<array<Record, 12>, 1<<12> dp_;

  int dp(int mask, int curr) {
    assert(0 <= curr && curr < size_);
    assert(mask & (1<<curr));
    assert(0 <= mask && mask <= mask_all_);
    auto opt_dist = dp_[mask][curr].dist;
    if (opt_dist == -1) {
      int opt_next;
      for (int next = 0; next < size_; ++next) {
        if (mask & (1<<next))
          continue;
        auto dist = dist_[curr][next] + dp(mask ^ (1<<next), next);
        if (opt_dist < dist) {
          opt_dist = dist;
          opt_next = next;
        }
      }
      dp_[mask][curr].dist = opt_dist;
      dp_[mask][curr].next = opt_next;
    }
    return opt_dist;
  }
 public:
  string shortestSuperstring(vector<string>& words) {
    size_ = words.size();
    mask_all_ = (1<<size_) - 1;
    // Populate dist_ and dp_
    for (int i = 0; i < size_; ++i) {
      for (int j = 0; j < size_; ++j) {
        if (i != j) {
          for (int k = min(words[i].size(), words[j].size()); k >= 0; --k) {
            if (words[i].substr(words[i].size()-k, k) == words[j].substr(0, k)) {
              dist_[i][j] = k;
              break;
            }
          }
        }
      }
      dp_[mask_all_][i].dist = 0;
    }    
    
    // try each word as head
    int opt_head;
    int max_dist = -1;
    for (int head = 0; head < size_; ++head) {
      auto dist = dp(1<<head, head);
      if (max_dist < dist) {
        max_dist = dist;
        opt_head = head;
      }
    }
    
    // build the superstring
    auto curr = opt_head;
    auto result = words[curr];
    int mask = 1<<curr;
    while (mask < mask_all_) {
      auto next = dp_[mask][curr].next;
      assert(0 <= curr && curr < size_);
      assert(0 <= next && next < size_);
      assert(curr != next);
      auto dist = dist_[curr][next];
      result.append(words[next].substr(dist));
      mask ^= (1<<next);
      curr = next;
    }
    return result;
  }
};
// @lc code=end

