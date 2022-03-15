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
class Solution {
  struct Suffix {
    int next{-1}, overlap{-1};
  };

  array<array<Suffix, 12>, (1 << 12)> dp_;  // dp[suffix_set][head]
  array<array<int, 12>, 12> overlap_;
  int n_, all_;

  void init(const vector<string> &words) {
    n_ = words.size();
    all_ = (1 << n_) - 1;
    for (int i = 0; i < n_; ++i) {
      dp_[1 << i][i].overlap = 0;
      for (int j = 0; j < n_; ++j) {
        if (i == j) {
          overlap_[i][j] = words[i].size();
        } else {
          int l = min(words[i].size(), words[j].size());
          while (l) {
            auto suffix = words[i].substr(words[i].size() - l);
            auto prefix = words[j].substr(0, l);
            if (suffix == prefix) {
              break;
            }
            --l;
          }
          overlap_[i][j] = l;
          // cout << words[i] << overlap_[i][j] << words[j] << endl;
        }
      }
    }
  }
  
  int dp(int suffix_set, int head) {
    auto my_overlap = dp_[suffix_set][head].overlap;
    if (my_overlap == -1) {
      int my_next{-1};
      for (int next = 0; next < n_; ++next) {
        auto next_suffix = suffix_set ^ (1 << head);
        if (next_suffix & (1 << next)) {
          // next \in next_suffix
          auto overlap = overlap_[head][next];
          overlap += dp(next_suffix, next);
          if (my_overlap < overlap) {
            my_overlap = overlap;
            my_next = next;
          }
        }
      }
      assert(0 <= my_next && my_next < n_);
      dp_[suffix_set][head].next = my_next;
      dp_[suffix_set][head].overlap = my_overlap;
    }
    return my_overlap;
  }

 public:
  string shortestSuperstring(vector<string>& words) {
    init(words);
    int max_overlap{-1}, max_head{-1};
    // try each word as head
    for (int head = 0; head < n_; ++head) {
      if (max_overlap < dp(all_, head)) {
        max_overlap = dp(all_, head);
        max_head = head;
      }
    }
    assert(0 <= max_head && max_head < n_);
    // build the superstring
    auto supstr = words[max_head];
    auto head = max_head;
    auto suffix_set = all_;
    while (true) {
      auto next = dp_[suffix_set][head].next;
      if (next == -1) {
        break;
      }
      auto overlap = overlap_[head][next];
      supstr += words[next].substr(overlap);
      suffix_set ^= (1 << head);
      head = next;
    }
    return supstr;
  }
};
// @lc code=end

