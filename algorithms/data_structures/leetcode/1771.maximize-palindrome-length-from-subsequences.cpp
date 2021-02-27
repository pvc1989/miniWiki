/*
 * @lc app=leetcode id=1771 lang=cpp
 *
 * [1771] Maximize Palindrome Length From Subsequences
 *
 * https://leetcode.com/problems/maximize-palindrome-length-from-subsequences/description/
 *
 * algorithms
 * Hard (33.33%)
 * Likes:    138
 * Dislikes: 5
 * Total Accepted:    3.6K
 * Total Submissions: 10.9K
 * Testcase Example:  '"cacb"\n"cbba"'
 *
 * You are given two strings, word1 and word2. You want to construct a string
 * in the following manner:
 * 
 * 
 * Choose some non-empty subsequence subsequence1 from word1.
 * Choose some non-empty subsequence subsequence2 from word2.
 * Concatenate the subsequences: subsequence1 + subsequence2, to make the
 * string.
 * 
 * 
 * Return the length of the longest palindrome that can be constructed in the
 * described manner. If no palindromes can be constructed, return 0.
 * 
 * A subsequence of a string s is a string that can be made by deleting some
 * (possibly none) characters from s without changing the order of the
 * remaining characters.
 * 
 * A palindrome is a string that reads the same forward as well as backward.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: word1 = "cacb", word2 = "cbba"
 * Output: 5
 * Explanation: Choose "ab" from word1 and "cba" from word2 to make "abcba",
 * which is a palindrome.
 * 
 * Example 2:
 * 
 * 
 * Input: word1 = "ab", word2 = "ab"
 * Output: 3
 * Explanation: Choose "ab" from word1 and "a" from word2 to make "aba", which
 * is a palindrome.
 * 
 * Example 3:
 * 
 * 
 * Input: word1 = "aa", word2 = "bb"
 * Output: 0
 * Explanation: You cannot construct a palindrome from the described method, so
 * return 0.
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= word1.length, word2.length <= 1000
 * word1 and word2 consist of lowercase English letters.
 * 
 */

// @lc code=start
class Solution {
 public:
  int longestPalindrome(string x, string y) {
    auto wall = x.size();
    x += y;
    auto tail = x.size();
    // dp[s][h] := size of any longest palindrome in x[h, h + s)
    auto dp = vector<vector<int>>(tail + 1, vector<int>(tail));
    dp[1] = vector<int>(tail, 1);
    int max_palindrome_size = 0;
    for (int size = 2; size <= tail; ++size) {
      for (int head = 0, last = head + size - 1; last < tail; ++head, ++last) {
        if (x[head] == x[last]) {
          dp[size][head] = dp[size - 2][head + 1] + 2;
          if (head < wall && wall <= last) {
            max_palindrome_size = max(max_palindrome_size, dp[size][head]);
            // printf("\ndp(x[%d, %d]) == %d", head, last, dp[size][head]);
          }
        } else {
          dp[size][head] = max(dp[size - 1][head], dp[size - 1][head + 1]);
          assert(dp[size - 1][head + 1] <= dp[size][head]);
        }
      }
    }
    return max_palindrome_size;
  }
};
// @lc code=end

