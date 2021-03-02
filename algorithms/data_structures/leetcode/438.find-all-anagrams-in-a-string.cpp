/*
 * @lc app=leetcode id=438 lang=cpp
 *
 * [438] Find All Anagrams in a String
 *
 * https://leetcode.com/problems/find-all-anagrams-in-a-string/description/
 *
 * algorithms
 * Medium (44.96%)
 * Likes:    3931
 * Dislikes: 198
 * Total Accepted:    339.8K
 * Total Submissions: 755.6K
 * Testcase Example:  '"cbaebabacd"\n"abc"'
 *
 * Given a string s and a non-empty string p, find all the start indices of p's
 * anagrams in s.
 * 
 * Strings consists of lowercase English letters only and the length of both
 * strings s and p will not be larger than 20,100.
 * 
 * The order of output does not matter.
 * 
 * Example 1:
 * 
 * Input:
 * s: "cbaebabacd" p: "abc"
 * 
 * Output:
 * [0, 6]
 * 
 * Explanation:
 * The substring with start index = 0 is "cba", which is an anagram of "abc".
 * The substring with start index = 6 is "bac", which is an anagram of
 * "abc".
 * 
 * 
 * 
 * Example 2:
 * 
 * Input:
 * s: "abab" p: "ab"
 * 
 * Output:
 * [0, 1, 2]
 * 
 * Explanation:
 * The substring with start index = 0 is "ab", which is an anagram of "ab".
 * The substring with start index = 1 is "ba", which is an anagram of "ab".
 * The substring with start index = 2 is "ab", which is an anagram of "ab".
 * 
 * 
 */

// @lc code=start
class Solution {
 public:
  vector<int> findAnagrams(string s, string p) {
    auto heads = vector<int>();
    if (s.size() >= p.size()) {
      auto p_counter = array<int, 26>();
      auto s_counter = p_counter;
      int n_s = s.size(), n_p = p.size();
      for (auto c : p) { ++p_counter[c - 'a']; }
      for (int i = 0; i != n_p; ++i) {
        ++s_counter[s[i] - 'a'];
      }
      int head = 0, tail = n_p;
      while (tail < n_s) {
        if (s_counter == p_counter) { heads.push_back(head); }
        --s_counter[s[head++] - 'a'];
        ++s_counter[s[tail++] - 'a'];
      }
      if (s_counter == p_counter) { heads.push_back(head); }
    }
    return heads;
  }
};
// @lc code=end

