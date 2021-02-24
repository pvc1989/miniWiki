/*
 * @lc app=leetcode id=208 lang=cpp
 *
 * [208] Implement Trie (Prefix Tree)
 *
 * https://leetcode.com/problems/implement-trie-prefix-tree/description/
 *
 * algorithms
 * Medium (51.90%)
 * Likes:    4211
 * Dislikes: 65
 * Total Accepted:    392K
 * Total Submissions: 753.7K
 * Testcase Example:  '["Trie","insert","search","search","startsWith","insert","search"]\n' +
  '[[],["apple"],["apple"],["app"],["app"],["app"],["app"]]'
 *
 * Implement a trie with insert, search, and startsWith methods.
 * 
 * Example:
 * 
 * 
 * Trie trie = new Trie();
 * 
 * trie.insert("apple");
 * trie.search("apple");   // returns true
 * trie.search("app");     // returns false
 * trie.startsWith("app"); // returns true
 * trie.insert("app");   
 * trie.search("app");     // returns true
 * 
 * 
 * Note:
 * 
 * 
 * You may assume that all inputs are consist of lowercase letters a-z.
 * All inputs are guaranteed to be non-empty strings.
 * 
 * 
 */

// @lc code=start
class Node {
  array<unique_ptr<Node>, 26> children_;
  bool end_of_word_{false};
 public:
  Node() {
  }
  void insert(string const &word, size_t head) {
    if (head == word.size()) {
      end_of_word_ = true;
    } else {
      size_t index = word[head] - 'a';
      if (children_[index] == nullptr) {
        children_[index].reset(new Node());
      }
      children_[index]->insert(word, ++head);
    }
  }
  bool search(string const &word, size_t head) {
    if (head == word.size()) {
      return end_of_word_;
    } else {
      size_t index = word[head] - 'a';
      return children_[index] && children_[index]->search(word, ++head);
    }
  }
  bool startsWith(string const &prefix, size_t head) {
    if (head == prefix.size()) {
      return true;
    } else {
      size_t index = prefix[head] - 'a';
      return children_[index] && children_[index]->startsWith(prefix, ++head);
    }
  }
};
class Trie {
  Node root_;
 public:
  /** Initialize your data structure here. */
  Trie() {
  }
  
  /** Inserts a word into the trie. */
  void insert(string word) {
    root_.insert(word, 0);
  }
  
  /** Returns if the word is in the trie. */
  bool search(string word) {
    return root_.search(word, 0);
  }
  
  /** Returns if there is any word in the trie that starts with the given prefix. */
  bool startsWith(string prefix) {
    return root_.startsWith(prefix, 0);
  }
};

/**
 * Your Trie object will be instantiated and called as such:
 * Trie* obj = new Trie();
 * obj->insert(word);
 * bool param_2 = obj->search(word);
 * bool param_3 = obj->startsWith(prefix);
 */
// @lc code=end

