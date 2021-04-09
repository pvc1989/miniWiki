/*
 * @lc app=leetcode id=17 lang=cpp
 *
 * [17] Letter Combinations of a Phone Number
 *
 * https://leetcode.com/problems/letter-combinations-of-a-phone-number/description/
 *
 * algorithms
 * Medium (49.40%)
 * Likes:    5720
 * Dislikes: 511
 * Total Accepted:    800.6K
 * Total Submissions: 1.6M
 * Testcase Example:  '"23"'
 *
 * Given a string containing digits from 2-9 inclusive, return all possible
 * letter combinations that the number could represent. Return the answer in
 * any order.
 * 
 * A mapping of digit to letters (just like on the telephone buttons) is given
 * below. Note that 1 does not map to any letters.
 * 
 * 
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: digits = "23"
 * Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: digits = ""
 * Output: []
 * 
 * 
 * Example 3:
 * 
 * 
 * Input: digits = "2"
 * Output: ["a","b","c"]
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 0 <= digits.length <= 4
 * digits[i] is a digit in the range ['2', '9'].
 * 
 * 
 */

// @lc code=start
class Solution {
  array<string, 10> digit_to_letters_ = {
    "", "", "abc", "def",
    "ghi", "jkl", "mno",
    "pqrs", "tuv", "wxyz"
  };
 public:
  vector<string> letterCombinations(string digits) {
    auto curr = vector<string>{ };
    auto prev = vector<string>{ };
    if (digits.size()) {
      curr.emplace_back("");
    }
    for (char d : digits) {
      swap(curr, prev);
      curr.clear();
      for (char c : digit_to_letters_[d - '0']) {
        for (auto& pr : prev) {
          curr.emplace_back(pr);
          curr.back().push_back(c);
        }
      }
    }
    return curr;
  }
};
// @lc code=end

