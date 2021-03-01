/*
 * @lc app=leetcode id=140 lang=cpp
 *
 * [140] Word Break II
 *
 * https://leetcode.com/problems/word-break-ii/description/
 *
 * algorithms
 * Hard (34.56%)
 * Likes:    2885
 * Dislikes: 441
 * Total Accepted:    302.3K
 * Total Submissions: 874.7K
 * Testcase Example:  '"catsanddog"\n["cat","cats","and","sand","dog"]'
 *
 * Given a non-empty string s and a dictionary wordDict containing a list of
 * non-empty words, add spaces in s to construct a sentence where each word is
 * a valid dictionary word. Return all such possible sentences.
 * 
 * Note:
 * 
 * 
 * The same word in the dictionary may be reused multiple times in the
 * segmentation.
 * You may assume the dictionary does not contain duplicate words.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input:
 * s = "catsanddog"
 * wordDict = ["cat", "cats", "and", "sand", "dog"]
 * Output:
 * [
 * "cats and dog",
 * "cat sand dog"
 * ]
 * 
 * 
 * Example 2:
 * 
 * 
 * Input:
 * s = "pineapplepenapple"
 * wordDict = ["apple", "pen", "applepen", "pine", "pineapple"]
 * Output:
 * [
 * "pine apple pen apple",
 * "pineapple pen apple",
 * "pine applepen apple"
 * ]
 * Explanation: Note that you are allowed to reuse a dictionary word.
 * 
 * 
 * Example 3:
 * 
 * 
 * Input:
 * s = "catsandog"
 * wordDict = ["cats", "dog", "sand", "and", "cat"]
 * Output:
 * []
 * 
 */

// @lc code=start
class Solution {
 private:  // bottom-up dp works, but solves too many wasted subproblems
  vector<string> bottomUp(string s, vector<string>& word_list) {
    // initialization
    auto word_set = unordered_set<string>(word_list.size() * 2);
    int min_size = INT_MAX, max_size = INT_MIN;
    for (auto& word : word_list) {
      word_set.emplace(word);
      min_size = min(min_size, int(word.size()));
      max_size = max(max_size, int(word.size()));
    }
    // dp[head] := solution on s[head, s.size())
    auto head_to_sentences = vector<vector<string>>(s.size());
    for (int head = s.size() - 1; head >= 0; --head) {
      auto& new_suffixes = head_to_sentences[head];
      for (int size = min_size; size <= max_size; ++size) {
        auto substr = s.substr(head, size);
        if (word_set.count(substr)) {
          auto tail = head + size;
          if (tail < s.size()) {
            for (auto& suffix : head_to_sentences[tail]) {
              new_suffixes.emplace_back(substr);
              new_suffixes.back().append(" ").append(suffix);
            }
          } else {
            new_suffixes.emplace_back(substr);
            break;
          }
        }
      }
    }
    return head_to_sentences.front();
  }

 private:  // top-down dp only solve necessary subproblems
  unordered_set<string> word_set_;
  vector<unique_ptr<vector<string>>> head_to_sentences_;
  int min_size_ = INT_MAX, max_size_ = INT_MIN;

  void topDownInit(const string& s, vector<string>& word_list) {
    for (auto& word : word_list) {
      word_set_.emplace(word);
      min_size_ = min(min_size_, int(word.size()));
      max_size_ = max(max_size_, int(word.size()));
    }
    head_to_sentences_.resize(s.size());
  }
  vector<string>& topDownSolve(const string& s, int head) {
    if (head_to_sentences_[head] == nullptr) {
      head_to_sentences_[head].reset(new vector<string>());
      auto& new_suffixes = *(head_to_sentences_[head]);
      for (int size = min_size_; size <= max_size_; ++size) {
        auto substr = s.substr(head, size);
        if (word_set_.count(substr)) {
          auto tail = head + size;
          if (tail < s.size()) {
            auto& old_suffixes = topDownSolve(s, tail);
            for (auto& suffix : old_suffixes) {
              new_suffixes.emplace_back(substr);
              new_suffixes.back().append(" ").append(suffix);
            }
          } else {
            new_suffixes.emplace_back(substr);
            break;
          }
        }
      }
    }
    return *(head_to_sentences_[head]);
  }
 public:
  vector<string> wordBreak(string s, vector<string>& word_list) {
    topDownInit(s, word_list);
    return topDownSolve(s, 0);
  }
};
// @lc code=end

