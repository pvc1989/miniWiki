/*
 * @lc app=leetcode id=916 lang=cpp
 *
 * [916] Word Subsets
 *
 * https://leetcode.com/problems/word-subsets/description/
 *
 * algorithms
 * Medium (48.69%)
 * Likes:    573
 * Dislikes: 93
 * Total Accepted:    32.1K
 * Total Submissions: 63.7K
 * Testcase Example:  '["amazon","apple","facebook","google","leetcode"]\n["e","o"]'
 *
 * We are given two arrays A and B of words.  Each word is a string of
 * lowercase letters.
 * 
 * Now, say that word b is a subset of word a if every letter in b occurs in a,
 * including multiplicity.  For example, "wrr" is a subset of "warrior", but is
 * not a subset of "world".
 * 
 * Now say a word a from A is universal if for every b in B, b is a subset of
 * a. 
 * 
 * Return a list of all universal words in A.  You can return the words in any
 * order.
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: A = ["amazon","apple","facebook","google","leetcode"], B = ["e","o"]
 * Output: ["facebook","google","leetcode"]
 * 
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: A = ["amazon","apple","facebook","google","leetcode"], B = ["l","e"]
 * Output: ["apple","google","leetcode"]
 * 
 * 
 * 
 * Example 3:
 * 
 * 
 * Input: A = ["amazon","apple","facebook","google","leetcode"], B = ["e","oo"]
 * Output: ["facebook","google"]
 * 
 * 
 * 
 * Example 4:
 * 
 * 
 * Input: A = ["amazon","apple","facebook","google","leetcode"], B =
 * ["lo","eo"]
 * Output: ["google","leetcode"]
 * 
 * 
 * 
 * Example 5:
 * 
 * 
 * Input: A = ["amazon","apple","facebook","google","leetcode"], B =
 * ["ec","oc","ceo"]
 * Output: ["facebook","leetcode"]
 * 
 * 
 * 
 * 
 * Note:
 * 
 * 
 * 1 <= A.length, B.length <= 10000
 * 1 <= A[i].length, B[i].length <= 10
 * A[i] and B[i] consist only of lowercase letters.
 * All words in A[i] are unique: there isn't i != j with A[i] == A[j].
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 */

// @lc code=start
class Solution {
 public:
  vector<string> wordSubsets(vector<string>& A, vector<string>& B) {
    // build counter for B
    auto B_counter = array<int, 26>();
    for (auto& b : B) {
      auto b_counter = array<int, 26>();
      for (auto c : b) {
        auto i = c - 'a';
        B_counter[i] = max(B_counter[i], ++b_counter[i]);
      }
    }
    // check each a in A
    auto result = vector<string>();
    for (auto& a : A) {
      auto a_counter = array<int, 26>();
      for (auto c : a) {
        ++a_counter[c - 'a'];
      }
      auto a_is_universal = true;
      for (int i = 0; i != 26; ++i) {
        if (B_counter[i] > a_counter[i]) {
          a_is_universal = false;
          break;
        }
      }
      if (a_is_universal) {
        result.emplace_back(a);
      }
    }
    return result;
  }
};
// @lc code=end

