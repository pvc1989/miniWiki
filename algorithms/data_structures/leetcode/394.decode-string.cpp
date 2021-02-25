/*
 * @lc app=leetcode id=394 lang=cpp
 *
 * [394] Decode String
 *
 * https://leetcode.com/problems/decode-string/description/
 *
 * algorithms
 * Medium (52.58%)
 * Likes:    4622
 * Dislikes: 226
 * Total Accepted:    302.8K
 * Total Submissions: 575.1K
 * Testcase Example:  '"3[a]2[bc]"'
 *
 * Given an encoded string, return its decoded string.
 * 
 * The encoding rule is: k[encoded_string], where the encoded_string inside the
 * square brackets is being repeated exactly k times. Note that k is guaranteed
 * to be a positive integer.
 * 
 * You may assume that the input string is always valid; No extra white spaces,
 * square brackets are well-formed, etc.
 * 
 * Furthermore, you may assume that the original data does not contain any
 * digits and that digits are only for those repeat numbers, k. For example,
 * there won't be input like 3a or 2[4].
 * 
 * 
 * Example 1:
 * Input: s = "3[a]2[bc]"
 * Output: "aaabcbc"
 * Example 2:
 * Input: s = "3[a2[c]]"
 * Output: "accaccacc"
 * Example 3:
 * Input: s = "2[abc]3[cd]ef"
 * Output: "abcabccdcdcdef"
 * Example 4:
 * Input: s = "abc3[cd]xyz"
 * Output: "abccdcdcdxyz"
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= s.length <= 30
 * s consists of lowercase English letters, digits, and square brackets
 * '[]'.
 * s is guaranteed to be a valid input.
 * All the integers in s are in the range [1, 300].
 * 
 * 
 */

// @lc code=start
class Solution {
  string decode(string const &encoded_string, size_t head, size_t tail) {
    auto decoded_string = string();
    int count = 0;
    while (head != tail) {
      auto c = encoded_string[head++];
      if (c == '[') {
        auto left_bracket_count = 1;
        auto slow = head, fast = head;
        while (fast != tail) {
          switch (encoded_string[fast++]) {
          case '[':
            ++left_bracket_count;
            break;
          case ']':
            --left_bracket_count;
          default:
            break;
          }
          if (left_bracket_count == 0) {
            break;
          }
        }
        auto decoded_substring = decode(encoded_string, slow, fast);
        for (int i = 0; i != count; ++i) {
          decoded_string += decoded_substring;
        }
        count = 0;
        assert(encoded_string[fast - 1] == ']');
        head = fast;
      } else if ('a' <= c && c <= 'z') {
        decoded_string += c;
      } else if ('0' <= c && c <= '9') {
        count *= 10;
        count += c - '0';
      }
    }
    return decoded_string;
  }
 public:
  string decodeString(string encoded_string) {
    return decode(encoded_string, 0, encoded_string.size());
  }
};
// @lc code=end

