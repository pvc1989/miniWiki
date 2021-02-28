/*
 * @lc app=leetcode id=127 lang=cpp
 *
 * [127] Word Ladder
 *
 * https://leetcode.com/problems/word-ladder/description/
 *
 * algorithms
 * Hard (31.81%)
 * Likes:    4710
 * Dislikes: 1399
 * Total Accepted:    549.1K
 * Total Submissions: 1.7M
 * Testcase Example:  '"hit"\n"cog"\n["hot","dot","dog","lot","log","cog"]'
 *
 * A transformation sequence from word beginWord to word endWord using a
 * dictionary wordList is a sequence of words such that:
 * 
 * 
 * The first word in the sequence is beginWord.
 * The last word in the sequence is endWord.
 * Only one letter is different between each adjacent pair of words in the
 * sequence.
 * Every word in the sequence is in wordList.
 * 
 * 
 * Given two words, beginWord and endWord, and a dictionary wordList, return
 * the number of words in the shortest transformation sequence from beginWord
 * to endWord, or 0 if no such sequence exists.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: beginWord = "hit", endWord = "cog", wordList =
 * ["hot","dot","dog","lot","log","cog"]
 * Output: 5
 * Explanation: One shortest transformation is "hit" -> "hot" -> "dot" -> "dog"
 * -> "cog" with 5 words.
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: beginWord = "hit", endWord = "cog", wordList =
 * ["hot","dot","dog","lot","log"]
 * Output: 0
 * Explanation: The endWord "cog" is not in wordList, therefore there is no
 * possible transformation.
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= beginWord.length <= 10
 * endWord.length == beginWord.length
 * 1 <= wordList.length <= 5000
 * wordList[i].length == beginWord.length
 * beginWord, endWord, and wordList[i] consist of lowercase English
 * letters.
 * beginWord != endWord
 * All the strings in wordList are unique.
 * 
 * 
 */

// @lc code=start
class Solution {
  static bool adjacent(const string& x, const string& y) {
    int n_diff = 0;
    for (int i = x.size() - 1; i >= 0; --i) {
      n_diff += (x[i] != y[i] ? 1 : 0);
      if (n_diff > 1) { return false; }
    }
    return n_diff == 1;
  }
 public:
  int ladderLength(string begin_word, string end_word, vector<string>& i_to_word) {
    auto size = i_to_word.size();
    auto word_to_i = unordered_map<string, int>();
    for (int i = 0; i != size; ++i) {
      word_to_i[i_to_word[i]] = i;
    }
    if (word_to_i.count(end_word) == 0) { return 0; }
    // build the graph represented by adjacency lists
    auto graph = vector<vector<int>>(size, vector<int>());
    for (int i_1 = 0; i_1 != size; ++i_1) {
      for (int i_2 = i_1 + 1; i_2 != size; ++i_2) {
        if (adjacent(i_to_word[i_1], i_to_word[i_2])) {
          graph[i_1].push_back(i_2);
          graph[i_2].push_back(i_1);
        }
      }
    }
    // run breadth-first search on the graph
    auto marked = vector<bool>(size);
    int i_end = word_to_i[end_word];
    int level = (word_to_i.count(begin_word) ? 0 : 1);
    auto curr_level = vector<int>();
    auto next_level = vector<int>();
    next_level.push_back(word_to_i[begin_word]);
    while (!next_level.empty()) {
      swap(next_level, curr_level);
      ++level;
      while (!curr_level.empty()) {
        auto i_1 = curr_level.back(); curr_level.pop_back();
        marked[i_1] = true;
        if (i_1 == i_end) { return level; }
        for (int i_2 : graph[i_1]) {
          if (!marked[i_2]) { next_level.push_back(i_2); }
        }
      }
    }
    return 0;  // not found
  }
};
// @lc code=end

