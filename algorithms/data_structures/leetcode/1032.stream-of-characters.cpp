/*
 * @lc app=leetcode id=1032 lang=cpp
 *
 * [1032] Stream of Characters
 *
 * https://leetcode.com/problems/stream-of-characters/description/
 *
 * algorithms
 * Hard (48.77%)
 * Likes:    1418
 * Dislikes: 163
 * Total Accepted:    69.4K
 * Total Submissions: 135.6K
 * Testcase Example:  '["StreamChecker","query","query","query","query","query","query","query","query","query","query","query","query"]\n' +
  '[[["cd","f","kl"]],["a"],["b"],["c"],["d"],["e"],["f"],["g"],["h"],["i"],["j"],["k"],["l"]]'
 *
 * Design an algorithm that accepts a stream of characters and checks if a
 * suffix of these characters is a string of a given array of strings words.
 * 
 * For example, if words = ["abc", "xyz"] and the stream added the four
 * characters (one by one) 'a', 'x', 'y', and 'z', your algorithm should detect
 * that the suffix "xyz" of the characters "axyz" matches "xyz" from words.
 * 
 * Implement the StreamChecker class:
 * 
 * 
 * StreamChecker(String[] words) Initializes the object with the strings array
 * words.
 * boolean query(char letter) Accepts a new character from the stream and
 * returns true if any non-empty suffix from the stream forms a word that is in
 * words.
 * 
 * 
 * 
 * Example 1:
 * 
 * 
 * Input
 * ["StreamChecker", "query", "query", "query", "query", "query", "query",
 * "query", "query", "query", "query", "query", "query"]
 * [[["cd", "f", "kl"]], ["a"], ["b"], ["c"], ["d"], ["e"], ["f"], ["g"],
 * ["h"], ["i"], ["j"], ["k"], ["l"]]
 * Output
 * [null, false, false, false, true, false, true, false, false, false, false,
 * false, true]
 * 
 * Explanation
 * StreamChecker streamChecker = new StreamChecker(["cd", "f", "kl"]);
 * streamChecker.query("a"); // return False
 * streamChecker.query("b"); // return False
 * streamChecker.query("c"); // return False
 * streamChecker.query("d"); // return True, because 'cd' is in the wordlist
 * streamChecker.query("e"); // return False
 * streamChecker.query("f"); // return True, because 'f' is in the wordlist
 * streamChecker.query("g"); // return False
 * streamChecker.query("h"); // return False
 * streamChecker.query("i"); // return False
 * streamChecker.query("j"); // return False
 * streamChecker.query("k"); // return False
 * streamChecker.query("l"); // return True, because 'kl' is in the
 * wordlist
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= words.length <= 2000
 * 1 <= words[i].length <= 2000
 * words[i] consists of lowercase English letters.
 * letter is a lowercase English letter.
 * At most 4 * 10^4 calls will be made to query.
 * 
 * 
 */

// @lc code=start
class StreamChecker {
  struct TrieNode {
    array<unique_ptr<TrieNode>, 26> children;
    bool is_word = false;
  };

  string stream_;
  TrieNode root_;
  size_t max_size_ = 0;

 public:
  StreamChecker(vector<string>& words) {
    for (auto &w : words) {
      auto *node = &root_;
      for (auto riter = w.rbegin(); riter != w.rend(); ++riter) {
        int i = *riter - 'a';
        if (!node->children[i]) {
          node->children[i] = make_unique<TrieNode>();
        }
        node = node->children[i].get();
        if (node->is_word)
          break;
        if (riter + 1 == w.rend()) {
          node->is_word = true;
          max_size_ = max(max_size_, w.size());
        }
      }
    }
  }

  bool query(char letter) {
    bool found = false;
    stream_.push_back(letter);
    auto riter = stream_.rbegin();
    auto rend = stream_.rbegin() + min(max_size_, stream_.size());
    auto *node = &root_;
    while (riter != rend) {
      int i = *riter - 'a';
      if (!node->children[i])
        break;
      node = node->children[i].get();
      if (node->is_word) {
        found = true;
        break;
      }
      ++riter;
    }
    return found;
  }
};
class StreamCheckerBySuffixArray {
  vector<string> words_;
  string stream_;
  
  static bool cmp(const string &x, const string &y) {
    int n = min(x.size(), y.size());
    auto rx = x.rbegin(), ry = y.rbegin();
    for (int i = 0; i < n; ++i) {
      if (rx[i] < ry[i]) {
        return true;
      } else if (rx[i] > ry[i]) {
        return false;
      } else {
        continue;
      }
    }
    return x.size() < y.size();
  }
  
  static bool ends_with(const string &x, const string &y) {
    int n = min(x.size(), y.size());
    if (n < y.size())
      return false;
    auto rx = x.rbegin(), ry = y.rbegin();
    for (int i = 0; i < n; ++i) {
      if (rx[i] != ry[i]) {
        return false;
      }
    }
    return true;
  }

 public:
  StreamChecker(vector<string>& words) {
    sort(words.begin(), words.end(), cmp);
    words_.emplace_back(move(words.front()));
    for (int i = 1; i < words.size(); ++i) {
      if (!ends_with(words[i], words_.back())) {
        words_.emplace_back(move(words[i]));
      }
    }
  }

  bool query(char letter) {
    stream_.push_back(letter);
    auto iter = lower_bound(words_.begin(), words_.end(), stream_, cmp);
    if (iter != words_.begin() && ends_with(stream_, iter[-1])) {
      return true;
    }
    if (iter != words_.end() && ends_with(stream_, *iter)) {
      return true;
    }
    return false;
  }
};
/**
 * Your StreamChecker object will be instantiated and called as such:
 * StreamChecker* obj = new StreamChecker(words);
 * bool param_1 = obj->query(letter);
 */
// @lc code=end

