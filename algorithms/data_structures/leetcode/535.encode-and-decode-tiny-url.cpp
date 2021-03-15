/*
 * @lc app=leetcode id=535 lang=cpp
 *
 * [535] Encode and Decode TinyURL
 *
 * https://leetcode.com/problems/encode-and-decode-tinyurl/description/
 *
 * algorithms
 * Medium (80.95%)
 * Likes:    815
 * Dislikes: 1694
 * Total Accepted:    126.1K
 * Total Submissions: 155K
 * Testcase Example:  '"https://leetcode.com/problems/design-tinyurl"'
 *
 * Note: This is a companion problem to the System Design problem: Design
 * TinyURL.
 * 
 * TinyURL is a URL shortening service where you enter a URL such as
 * https://leetcode.com/problems/design-tinyurl and it returns a short URL such
 * as http://tinyurl.com/4e9iAk.
 * 
 * Design the encode and decode methods for the TinyURL service. There is no
 * restriction on how your encode/decode algorithm should work. You just need
 * to ensure that a URL can be encoded to a tiny URL and the tiny URL can be
 * decoded to the original URL.
 * 
 */

// @lc code=start
class Solution {
  unordered_map<string, string> long_to_short_;
  unordered_map<string, string> short_to_long_;

  string GetRandomString() {
    auto rand_str = string();
    for (int i = 0; i != 6; ++i) {
      char c = rand() % 62;
      if (c < 10) {
        c += '0';
      } else {
        c -= 10;
        if (c < 26) {
          c += 'A';
        } else {
          c -= 26;
          c += 'a';
        }
      }
      rand_str.push_back(c);
    }
    return rand_str;
  }

 public:
  // Encodes a URL to a shortened URL.
  string encode(string long_url) {
    auto iter = long_to_short_.find(long_url);
    if (iter == long_to_short_.end()) {
      auto short_url = "http://tinyurl.com/" + GetRandomString();
      while (short_to_long_.find(short_url) != short_to_long_.end()) {
        short_url = "http://tinyurl.com/" + GetRandomString();
      }
      short_to_long_.emplace(short_url, long_url);
      iter = long_to_short_.emplace_hint(iter, long_url, short_url);
    }
    return iter->second;
  }

  // Decodes a shortened URL to its original URL.
  string decode(string short_url) {
    auto iter = short_to_long_.find(short_url);
    assert(iter != short_to_long_.end());
    return iter->second;
  }
};

// Your Solution object will be instantiated and called as such:
// Solution solution;
// solution.decode(solution.encode(url));

// Your Solution object will be instantiated and called as such:
// Solution solution;
// solution.decode(solution.encode(url));
// @lc code=end

