/*
 * @lc app=leetcode id=30 lang=cpp
 *
 * [30] Substring with Concatenation of All Words
 *
 * https://leetcode.com/problems/substring-with-concatenation-of-all-words/description/
 *
 * algorithms
 * Hard (26.15%)
 * Likes:    1154
 * Dislikes: 1406
 * Total Accepted:    198.1K
 * Total Submissions: 757.2K
 * Testcase Example:  '"barfoothefoobarman"\n["foo","bar"]'
 *
 * You are given a string s and an array of strings words of the same length.
 * Return all starting indices of substring(s) in s that is a concatenation of
 * each word in words exactly once, in any order, and without any intervening
 * characters.
 * 
 * You can return the answer in any order.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: s = "barfoothefoobarman", words = ["foo","bar"]
 * Output: [0,9]
 * Explanation: Substrings starting at index 0 and 9 are "barfoo" and "foobar"
 * respectively.
 * The output order does not matter, returning [9,0] is fine too.
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: s = "wordgoodgoodgoodbestword", words = ["word","good","best","word"]
 * Output: []
 * 
 * 
 * Example 3:
 * 
 * 
 * Input: s = "barfoofoobarthefoobarman", words = ["bar","foo","the"]
 * Output: [6,9,12]
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= s.length <= 10^4
 * s consists of lower-case English letters.
 * 1 <= words.length <= 5000
 * 1 <= words[i].length <= 30
 * words[i] consists of lower-case English letters.
 * 
 * 
 */

// @lc code=start
class Solution {
  class WordBank {
    unordered_map<string, int> word_to_count_;
    int total_count_;
   public:
    explicit WordBank(vector<string> const &words) : total_count_(0) {
      for (auto &word : words) {
        ++word_to_count_[word];
        ++total_count_;
      }
    }
    int Count() const { return total_count_; }
    int Count(string const &word) const {
      auto iter = word_to_count_.find(word);
      return iter != word_to_count_.end() ? iter->second : 0;
    }
    bool TakeAway(string const &word) {
      auto iter = word_to_count_.find(word);
      if (iter != word_to_count_.end() && iter->second) {
        // cout << "TakeAway(" << word << ")" << endl;
        --(iter->second);
        --total_count_;
        return true;
      } else {
        return false;
      }
    }
    void GiveBack(string const &word) {
      auto iter = word_to_count_.find(word);
      if (iter != word_to_count_.end()) {
        // cout << "GiveBack(" << word << ")" << endl;
        ++(iter->second);
        ++total_count_;
      }
    }
  };
 public:
  vector<int> findSubstring(string s, vector<string>& words) {
    auto head_indices = vector<int>();
    auto word_bank = WordBank(words);
    int word_size = words.front().size();
    int last_index = s.size() - word_size;
    for (int mod_index = 0; mod_index != word_size; ++mod_index) {
      int head_index = mod_index;  // subs.begin() - s.begin()
      int next_index = mod_index;  // word.begin() - s.begin()
      auto word = string();
      while (head_index <= last_index) {
        while (next_index <= last_index) {  // ignore mismatching words
          word = s.substr(next_index, word_size);
          if (word_bank.Count(word) > 0) break;
          next_index += word_size;
          head_index = next_index;
        }
        while (next_index <= last_index) {  // take away matching words
          word = s.substr(next_index, word_size);
          if (!word_bank.TakeAway(word)) break;
          next_index += word_size;
        }
        if (word_bank.Count() == 0) {  // find a new substring
          head_indices.emplace_back(head_index);
        }
        if (next_index > last_index) word = "";
        while (head_index < next_index && word_bank.Count(word) == 0) {
          word_bank.GiveBack(s.substr(head_index, word_size));
          head_index += word_size;
        }
      }
    }
    return head_indices;
  }
};
// @lc code=end

