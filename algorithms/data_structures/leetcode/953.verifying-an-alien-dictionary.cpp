/*
 * @lc app=leetcode id=953 lang=cpp
 *
 * [953] Verifying an Alien Dictionary
 *
 * https://leetcode.com/problems/verifying-an-alien-dictionary/description/
 *
 * algorithms
 * Easy (51.85%)
 * Likes:    1542
 * Dislikes: 643
 * Total Accepted:    212.5K
 * Total Submissions: 408.8K
 * Testcase Example:  '["hello","leetcode"]\n"hlabcdefgijkmnopqrstuvwxyz"'
 *
 * In an alien language, surprisingly they also use english lowercase letters,
 * but possibly in a different order. The order of the alphabet is some
 * permutation of lowercase letters.
 * 
 * Given a sequence of words written in the alien language, and the order of
 * the alphabet, return true if and only if the given words are sorted
 * lexicographicaly in this alien language.
 * 
 * Example 1:
 * 
 * 
 * Input: words = ["hello","leetcode"], order = "hlabcdefgijkmnopqrstuvwxyz"
 * Output: true
 * Explanation: As 'h' comes before 'l' in this language, then the sequence is
 * sorted.
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: words = ["word","world","row"], order = "worldabcefghijkmnpqstuvxyz"
 * Output: false
 * Explanation: As 'd' comes after 'l' in this language, then words[0] >
 * words[1], hence the sequence is unsorted.
 * 
 * 
 * Example 3:
 * 
 * 
 * Input: words = ["apple","app"], order = "abcdefghijklmnopqrstuvwxyz"
 * Output: false
 * Explanation: The first three characters "app" match, and the second string
 * is shorter (in size.) According to lexicographical rules "apple" > "app",
 * because 'l' > '∅', where '∅' is defined as the blank character which is less
 * than any other character (More info).
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= words.length <= 100
 * 1 <= words[i].length <= 20
 * order.length == 26
 * All characters in words[i] and order are English lowercase letters.
 * 
 * 
 */

// @lc code=start
class Solution {
 public:
  bool isAlienSorted(vector<string>& words, string order) {
    assert(order.size() == 26);
    auto ctoi = array<int, 26>();
    for (int i = 0; i != 26; ++i) {
      ctoi[order[i] - 'a'] = i;
    }
    auto cmp = [&](const string& x, const string& y){
      int i = 0;
      for (i = 0; i < x.size() && i < y.size(); ++i) {
        int d = ctoi[x[i] - 'a'] - ctoi[y[i] - 'a'];
        if (d < 0) {
          return true;
        } else if (d > 0) {
          return false;
        } else {
          continue;
        }
      }
      return x.size() < y.size();
    };
    return is_sorted(words.begin(), words.end(), cmp);
  }
};
// @lc code=end

