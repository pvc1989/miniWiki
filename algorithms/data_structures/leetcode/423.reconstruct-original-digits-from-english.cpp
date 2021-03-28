/*
 * @lc app=leetcode id=423 lang=cpp
 *
 * [423] Reconstruct Original Digits from English
 *
 * https://leetcode.com/problems/reconstruct-original-digits-from-english/description/
 *
 * algorithms
 * Medium (47.51%)
 * Likes:    242
 * Dislikes: 735
 * Total Accepted:    30K
 * Total Submissions: 62K
 * Testcase Example:  '"owoztneoer"'
 *
 * Given a non-empty string containing an out-of-order English representation
 * of digits 0-9, output the digits in ascending order.
 * 
 * Note:
 * 
 * Input contains only lowercase English letters.
 * Input is guaranteed to be valid and can be transformed to its original
 * digits. That means invalid inputs such as "abc" or "zerone" are not
 * permitted.
 * Input length is less than 50,000.
 * 
 * 
 * 
 * Example 1:
 * 
 * Input: "owoztneoer"
 * 
 * Output: "012"
 * 
 * 
 * 
 * Example 2:
 * 
 * Input: "fviefuro"
 * 
 * Output: "45"
 * 
 * 
 */

// @lc code=start
class Solution {
 public:
  string originalDigits(string s) {
    auto char_counter = array<int, 26>();
    for (char c : s) {
      ++char_counter[c - 'a'];
    }
    auto int_counter = array<int, 10>();
    int curr, curr_count;
    // 0 : zero
    curr = 0;
    curr_count = char_counter['z' - 'a'];
    char_counter['e' - 'a'] -= curr_count;
    char_counter['r' - 'a'] -= curr_count;
    char_counter['o' - 'a'] -= curr_count;
    int_counter[curr] = curr_count;
    // 6 : six
    curr = 6;
    curr_count = char_counter['x' - 'a'];
    char_counter['i' - 'a'] -= curr_count;
    char_counter['s' - 'a'] -= curr_count;
    int_counter[curr] = curr_count;
    // 4 : four
    curr = 4;
    curr_count = char_counter['u' - 'a'];
    char_counter['f' - 'a'] -= curr_count;
    char_counter['o' - 'a'] -= curr_count;
    char_counter['r' - 'a'] -= curr_count;
    int_counter[curr] = curr_count;
    // 5 : five
    curr = 5;
    curr_count = char_counter['f' - 'a'];
    char_counter['i' - 'a'] -= curr_count;
    char_counter['v' - 'a'] -= curr_count;
    char_counter['e' - 'a'] -= curr_count;
    int_counter[curr] = curr_count;
    // 7 : seven
    curr = 7;
    curr_count = char_counter['s' - 'a'];
    char_counter['e' - 'a'] -= curr_count + curr_count;
    char_counter['v' - 'a'] -= curr_count;
    char_counter['n' - 'a'] -= curr_count;
    int_counter[curr] = curr_count;
    // 2 : two
    curr = 2;
    curr_count = char_counter['w' - 'a'];
    char_counter['t' - 'a'] -= curr_count;
    char_counter['o' - 'a'] -= curr_count;
    int_counter[curr] = curr_count;
    // 3 : three
    curr = 3;
    curr_count = char_counter['r' - 'a'];
    char_counter['t' - 'a'] -= curr_count;
    char_counter['h' - 'a'] -= curr_count;
    char_counter['e' - 'a'] -= curr_count + curr_count;
    int_counter[curr] = curr_count;
    // 8 : eight
    curr = 8;
    curr_count = char_counter['t' - 'a'];
    char_counter['e' - 'a'] -= curr_count;
    char_counter['i' - 'a'] -= curr_count;
    char_counter['g' - 'a'] -= curr_count;
    char_counter['h' - 'a'] -= curr_count;
    int_counter[curr] = curr_count;
    // 9 : nine
    curr = 9;
    curr_count = char_counter['i' - 'a'];
    char_counter['n' - 'a'] -= curr_count + curr_count;
    char_counter['e' - 'a'] -= curr_count;
    int_counter[curr] = curr_count;
    // 1 : one
    curr = 1;
    curr_count = char_counter['o' - 'a'];
    char_counter['n' - 'a'] -= curr_count;
    char_counter['e' - 'a'] -= curr_count;
    int_counter[curr] = curr_count;
    // construct result
    auto result = string();
    for (int i = 0; i != 10; ++i) {
      result += string(int_counter[i], '0' + i);
    }
    return result;
  }
};
// @lc code=end

