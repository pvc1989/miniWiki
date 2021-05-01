/*
 * @lc app=leetcode id=745 lang=cpp
 *
 * [745] Prefix and Suffix Search
 *
 * https://leetcode.com/problems/prefix-and-suffix-search/description/
 *
 * algorithms
 * Hard (35.45%)
 * Likes:    673
 * Dislikes: 277
 * Total Accepted:    35.5K
 * Total Submissions: 95.2K
 * Testcase Example:  '["WordFilter","f"]\n[[["apple"]],["a","e"]]'
 *
 * Design a special dictionary which has some words and allows you to search
 * the words in it by a prefix and a suffix.
 * 
 * Implement the WordFilter class:
 * 
 * 
 * WordFilter(string[] words) Initializes the object with the words in the
 * dictionary.
 * f(string prefix, string suffix) Returns the index of the word in the
 * dictionary which has the prefix prefix and the suffix suffix. If there is
 * more than one valid index, return the largest of them. If there is no such
 * word in the dictionary, return -1.
 * 
 * 
 * 
 * Example 1:
 * 
 * 
 * Input
 * ["WordFilter", "f"]
 * [[["apple"]], ["a", "e"]]
 * Output
 * [null, 0]
 * 
 * Explanation
 * WordFilter wordFilter = new WordFilter(["apple"]);
 * wordFilter.f("a", "e"); // return 0, because the word at index 0 has prefix
 * = "a" and suffix = 'e".
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= words.length <= 15000
 * 1 <= words[i].length <= 10
 * 1 <= prefix.length, suffix.length <= 10
 * words[i], prefix and suffix consist of lower-case English letters only.
 * At most 15000 calls will be made to the function f.
 * 
 * 
 */

// @lc code=start
class WordFilter {
  unordered_map<string, unordered_set<int>> prefix_to_indices_;
  unordered_map<string,        vector<int>> suffix_to_indices_;
 public:
  WordFilter(vector<string>& words) {
    for (int index = words.size() - 1; index >= 0; --index) {
      auto& word = words[index];
      auto  size = word.size();
      for (int length = size; length >= 0; --length) {
        auto head = 0;
        prefix_to_indices_[word.substr(head, length)].emplace(index);
        head = size - length;
        suffix_to_indices_[word.substr(head, length)].emplace_back(index);
      }
    }
  }

  int f(string prefix, string suffix) {
    auto prefix_iter = prefix_to_indices_.find(prefix);
    auto suffix_iter = suffix_to_indices_.find(suffix);
    if (prefix_iter == prefix_to_indices_.end() ||
        suffix_iter == suffix_to_indices_.end()) {
      return -1;
    }
    auto& prefix_indices = prefix_iter->second;
    auto& suffix_indices = suffix_iter->second;
    for (auto index : suffix_indices) {
      auto iter = prefix_indices.find(index);
      if (iter != prefix_indices.end())
        return index;
    }
    return -1;
  }
};

/**
 * Your WordFilter object will be instantiated and called as such:
 * WordFilter* obj = new WordFilter(words);
 * int param_1 = obj->f(prefix,suffix);
 */
// @lc code=end

