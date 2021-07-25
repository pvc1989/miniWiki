/*
 * @lc app=leetcode id=126 lang=cpp
 *
 * [126] Word Ladder II
 *
 * https://leetcode.com/problems/word-ladder-ii/description/
 *
 * algorithms
 * Hard (24.24%)
 * Likes:    2831
 * Dislikes: 308
 * Total Accepted:    245.7K
 * Total Submissions: 985.7K
 * Testcase Example:  '"hit"\n"cog"\n["hot","dot","dog","lot","log","cog"]'
 *
 * A transformation sequence from word beginWord to word endWord using a
 * dictionary wordList is a sequence of words beginWord -> s1 -> s2 -> ... ->
 * sk such that:
 * 
 * 
 * Every adjacent pair of words differs by a single letter.
 * Every si for 1 <= i <= k is in wordList. Note that beginWord does not need
 * to be in wordList.
 * sk == endWord
 * 
 * 
 * Given two words, beginWord and endWord, and a dictionary wordList, return
 * all the shortest transformation sequences from beginWord to endWord, or an
 * empty list if no such sequence exists. Each sequence should be returned as a
 * list of the words [beginWord, s1, s2, ..., sk].
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: beginWord = "hit", endWord = "cog", wordList =
 * ["hot","dot","dog","lot","log","cog"]
 * Output: [["hit","hot","dot","dog","cog"],["hit","hot","lot","log","cog"]]
 * Explanation: There are 2 shortest transformation sequences:
 * "hit" -> "hot" -> "dot" -> "dog" -> "cog"
 * "hit" -> "hot" -> "lot" -> "log" -> "cog"
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: beginWord = "hit", endWord = "cog", wordList =
 * ["hot","dot","dog","lot","log"]
 * Output: []
 * Explanation: The endWord "cog" is not in wordList, therefore there is no
 * valid transformation sequence.
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= beginWord.length <= 5
 * endWord.length == beginWord.length
 * 1 <= wordList.length <= 1000
 * wordList[i].length == beginWord.length
 * beginWord, endWord, and wordList[i] consist of lowercase English
 * letters.
 * beginWord != endWord
 * All the words in wordList are unique.
 * 
 * 
 */

// @lc code=start
class Solution {
  vector<vector<string>> result_;
  vector<vector<int>> i_to_adjlist_;
  vector<int> i_to_level_;
  int n_, i_begin_, i_end_;
  
  static bool adjacent(const string& x, const string& y) {
    int n_diff = 0;
    for (int i = x.size() - 1; i >= 0; --i) {
      n_diff += (x[i] != y[i] ? 1 : 0);
      if (n_diff > 1)
        return false;
    }
    return n_diff == 1;
  }

  void build_i_to_adjlist(vector<string> const &i_to_word) {
    i_to_adjlist_.resize(n_, vector<int>());
    for (int i = 0; i < n_; ++i) {
      for (int j = i + 1; j < n_; ++j) {
        if (adjacent(i_to_word[i], i_to_word[j])) {
          i_to_adjlist_[i].push_back(j);
          i_to_adjlist_[j].push_back(i);
        }
      }
    }
  }
  
  void build_i_to_level() {
    int level = 0;
    i_to_level_.resize(n_, INT_MAX);
    vector<int> curr_level, next_level{i_begin_};
    i_to_level_[i_begin_] = 0;
    while (!next_level.empty()) {
      ++level;
      swap(curr_level, next_level);
      next_level.clear();
      for (int i : curr_level) {
        for (int j : i_to_adjlist_[i]) {
          if (INT_MAX == i_to_level_[j]) {
            i_to_level_[j] = level;
            if (j == i_end_)
              return;
            next_level.emplace_back(j);
          }
        }
      }
    }
  }
  
  void build_result(vector<string> const& i_to_word, int i, vector<string>* prefix) {
    prefix->emplace_back(i_to_word[i]);
    if (i == i_end_) {
      result_.emplace_back(*prefix);
    }
    else {
      int j_level = i_to_level_[i] + 1;
      for (int j : i_to_adjlist_[i]) {
        if (j_level == i_to_level_[j]) {
          build_result(i_to_word, j, prefix);
        }
      }
    }
    prefix->pop_back();
  }
  
public:
  vector<vector<string>> findLadders(string begin_word, string end_word, vector<string>& i_to_word) {
    unordered_map<string, int> word_to_i;
    n_ = i_to_word.size();
    for (int i = 0; i != n_; ++i)
      word_to_i[i_to_word[i]] = i;
    if (word_to_i.count(end_word) == 0)
      return result_;
    // remove difference between `begin_word` and others
    if (word_to_i.count(begin_word) == 0) {
      i_to_word.emplace_back(begin_word);
      word_to_i[begin_word] = n_++;
    }
    i_begin_ = word_to_i[begin_word];
    i_end_ = word_to_i[end_word];
    // build
    build_i_to_adjlist(i_to_word);
    build_i_to_level();
    vector<string> prefix;
    build_result(i_to_word, i_begin_, &prefix);
    return result_;
  }
};
// @lc code=end

