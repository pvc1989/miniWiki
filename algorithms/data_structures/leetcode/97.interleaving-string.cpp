/*
 * @lc app=leetcode id=97 lang=cpp
 *
 * [97] Interleaving String
 *
 * https://leetcode.com/problems/interleaving-string/description/
 *
 * algorithms
 * Medium (32.69%)
 * Likes:    2526
 * Dislikes: 129
 * Total Accepted:    203.9K
 * Total Submissions: 607.8K
 * Testcase Example:  '"aabcc"\n"dbbca"\n"aadbbcbcac"'
 *
 * Given strings s1, s2, and s3, find whether s3 is formed by an interleaving
 * of s1 and s2.
 * 
 * An interleaving of two strings s and t is a configuration where they are
 * divided into non-empty substrings such that:
 * 
 * 
 * s = s1 + s2 + ... + sn
 * t = t1 + t2 + ... + tm
 * |n - m| <= 1
 * The interleaving is s1 + t1 + s2 + t2 + s3 + t3 + ... or t1 + s1 + t2 + s2 +
 * t3 + s3 + ...
 * 
 * 
 * Note: a + b is the concatenation of strings a and b.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbcbcac"
 * Output: true
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbbaccc"
 * Output: false
 * 
 * 
 * Example 3:
 * 
 * 
 * Input: s1 = "", s2 = "", s3 = ""
 * Output: true
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 0 <= s1.length, s2.length <= 100
 * 0 <= s3.length <= 200
 * s1, s2, and s3 consist of lowercase English letters.
 * 
 * 
 * 
 * Follow up: Could you solve it using only O(s2.length) additional memory
 * space?
 * 
 */

// @lc code=start
#define NDEBUG
class Solution {
  array<array<array<char, 201>, 101>, 101> dp1_;
  array<array<array<char, 201>, 101>, 101> dp2_;
  int n1_, n2_, n3_;
  
  static bool prefix_eq(string const& s1, int i1,
                        string const& s2, int i2,
                        int prefix_len) {
    if (i1 + prefix_len > s1.size() || i2 + prefix_len > s2.size())
      return false;
    for (int k = 0; k != prefix_len; ++k)
      if (s1[i1 + k] != s2[i2 + k])
        return false;
    return true;
  }
  char dp1(string const& s1, int i1,
           string const& s2, int i2,
           string const& s3, int i3) {
    auto v = dp1_[i1][i2][i3];
    if (v == -1) {
      v = 0;
      assert(i1 + i2 == i3);
      if (i2 == n2_) {
        assert(n1_-i1 == n3_-i3);
        v = prefix_eq(s1, i1, s3, i3, n3_-i3);
      }
      else {
        int l = 0;
        while (i1 + l < n1_ && s1[i1+l] == s3[i3+l]) {
          ++l;
          if (dp2(s1, i1+l, s2, i2, s3, i3+l)) {
            v = 1;
            break;
          }
        }
      }
      dp1_[i1][i2][i3] = v;
    }
    return v;
  }
  char dp2(string const& s1, int i1,
           string const& s2, int i2,
           string const& s3, int i3) {
    auto v = dp2_[i1][i2][i3];
    if (v == -1) {
      v = 0;
      assert(i1 + i2 == i3);
      if (i1 == n1_) {
        assert(n2_-i2 == n3_-i3);
        v = prefix_eq(s2, i2, s3, i3, n3_-i3);
      }
      else {
        int l = 0;
        while (i2+l < n2_ && s2[i2+l] == s3[i3+l]) {
          ++l;
          if (dp1(s1, i1, s2, i2+l, s3, i3+l)) {
            v = 1;
            break;
          }
        }
      }
      dp2_[i1][i2][i3] = v;
    }
    return v;
  }
 public:
  bool isInterleave(string s1, string s2, string s3) {
    n1_ = s1.size(); n2_ = s2.size(); n3_ = s3.size();
    if (n1_ + n2_ != n3_)
      return false;
    for (auto& x : dp1_)
      for (auto& y : x)
        y.fill(-1);
    for (auto& x : dp2_)
      for (auto& y : x)
        y.fill(-1);
    return dp1(s1, 0, s2, 0, s3, 0) || dp2(s1, 0, s2, 0, s3, 0);
  }
};
// @lc code=end

