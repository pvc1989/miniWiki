/*
 * @lc app=leetcode id=166 lang=cpp
 *
 * [166] Fraction to Recurring Decimal
 *
 * https://leetcode.com/problems/fraction-to-recurring-decimal/description/
 *
 * algorithms
 * Medium (22.29%)
 * Likes:    1077
 * Dislikes: 2188
 * Total Accepted:    145.9K
 * Total Submissions: 654K
 * Testcase Example:  '1\n2'
 *
 * Given two integers representing the numerator and denominator of a fraction,
 * return the fraction in string format.
 * 
 * If the fractional part is repeating, enclose the repeating part in
 * parentheses.
 * 
 * If multiple answers are possible, return any of them.
 * 
 * It is guaranteed that the length of the answer string is less than 10^4 for
 * all the given inputs.
 * 
 * 
 * Example 1:
 * Input: numerator = 1, denominator = 2
 * Output: "0.5"
 * Example 2:
 * Input: numerator = 2, denominator = 1
 * Output: "2"
 * Example 3:
 * Input: numerator = 2, denominator = 3
 * Output: "0.(6)"
 * Example 4:
 * Input: numerator = 4, denominator = 333
 * Output: "0.(012)"
 * Example 5:
 * Input: numerator = 1, denominator = 5
 * Output: "0.2"
 * 
 * 
 * Constraints:
 * 
 * 
 * -2^31 <= numerator, denominator <= 2^31 - 1
 * denominator != 0
 * 
 * 
 */

// @lc code=start
class Solution {
  static string getQuotientString(int64_t numerator, int64_t denominator) {
    auto quotient_string = string();
    bool negative = false;
    if (numerator < 0) {
      numerator = -numerator;
      negative = !negative;
    }
    if (denominator < 0) { 
      denominator = -denominator;
      negative = !negative;
    }
    if (negative && numerator) {
      quotient_string.push_back('-');
    }
    auto remainder = numerator % denominator;
    quotient_string.append(to_string(numerator / denominator));
    if (remainder != 0) {
      quotient_string.push_back('.');
      auto numerator_to_index = unordered_map<int64_t, size_t>();
      auto index = quotient_string.size();
      while (remainder) {
        numerator = remainder * 10;
        auto iter = numerator_to_index.find(numerator);
        if (iter == numerator_to_index.end()) {
          numerator_to_index.emplace_hint(iter, numerator, index++);
          quotient_string.append(to_string(numerator / denominator));
          remainder = numerator % denominator;
        } else {
          quotient_string.insert(iter->second, 1, '(');
          quotient_string.push_back(')');
          break;
        }
      }
    }
    return quotient_string;
  }
 public:
  string fractionToDecimal(int numerator, int denominator) {
    return getQuotientString(numerator, denominator);
  }
};
// @lc code=end

