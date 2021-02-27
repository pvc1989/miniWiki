/*
 * @lc app=leetcode id=1768 lang=cpp
 *
 * [1768] Merge Strings Alternately
 *
 * https://leetcode.com/problems/merge-strings-alternately/description/
 *
 * algorithms
 * Easy (79.79%)
 * Likes:    83
 * Dislikes: 2
 * Total Accepted:    13.3K
 * Total Submissions: 16.7K
 * Testcase Example:  '"abc"\n"pqr"'
 *
 * You are given two strings word1 and word2. Merge the strings by adding
 * letters in alternating order, starting with word1. If a string is longer
 * than the other, append the additional letters onto the end of the merged
 * string.
 * 
 * Return the merged string.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: word1 = "abc", word2 = "pqr"
 * Output: "apbqcr"
 * Explanation: The merged string will be merged as so:
 * word1:  a   b   c
 * word2:    p   q   r
 * merged: a p b q c r
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: word1 = "ab", word2 = "pqrs"
 * Output: "apbqrs"
 * Explanation: Notice that as word2 is longer, "rs" is appended to the end.
 * word1:  a   b 
 * word2:    p   q   r   s
 * merged: a p b q   r   s
 * 
 * 
 * Example 3:
 * 
 * 
 * Input: word1 = "abcd", word2 = "pq"
 * Output: "apbqcd"
 * Explanation: Notice that as word1 is longer, "cd" is appended to the end.
 * word1:  a   b   c   d
 * word2:    p   q 
 * merged: a p b q c   d
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= word1.length, word2.length <= 100
 * word1 and word2 consist of lowercase English letters.
 * 
 */

// @lc code=start
class Solution {
 public:
  string mergeAlternately(string word1, string word2) {
    auto merged = string();
    auto iter1 = word1.begin(), tail1 = word1.end();
    auto iter2 = word2.begin(), tail2 = word2.end();
    bool read_iter1 = true;
    while (iter1 != tail1 && iter2 != tail2) {
      merged += (read_iter1) ? *iter1++ : *iter2++;
      read_iter1 = !read_iter1;
    }
    while (iter1 != tail1) {
      merged += *iter1++;
    }
    while (iter2 != tail2) {
      merged += *iter2++;
    }
    return merged;
  }
};
// @lc code=end

