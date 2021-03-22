/*
 * @lc app=leetcode id=966 lang=cpp
 *
 * [966] Vowel Spellchecker
 *
 * https://leetcode.com/problems/vowel-spellchecker/description/
 *
 * algorithms
 * Medium (47.89%)
 * Likes:    212
 * Dislikes: 415
 * Total Accepted:    17.8K
 * Total Submissions: 36K
 * Testcase Example:  '["KiTe","kite","hare","Hare"]\n' +
  '["kite","Kite","KiTe","Hare","HARE","Hear","hear","keti","keet","keto"]'
 *
 * Given a wordlist, we want to implement a spellchecker that converts a query
 * word into a correct word.
 * 
 * For a given query word, the spell checker handles two categories of spelling
 * mistakes:
 * 
 * 
 * Capitalization: If the query matches a word in the wordlist
 * (case-insensitive), then the query word is returned with the same case as
 * the case in the wordlist.
 * 
 * 
 * Example: wordlist = ["yellow"], query = "YellOw": correct = "yellow"
 * Example: wordlist = ["Yellow"], query = "yellow": correct = "Yellow"
 * Example: wordlist = ["yellow"], query = "yellow": correct =
 * "yellow"
 * 
 * 
 * Vowel Errors: If after replacing the vowels ('a', 'e', 'i', 'o', 'u') of the
 * query word with any vowel individually, it matches a word in the wordlist
 * (case-insensitive), then the query word is returned with the same case as
 * the match in the wordlist.
 * 
 * Example: wordlist = ["YellOw"], query = "yollow": correct = "YellOw"
 * Example: wordlist = ["YellOw"], query = "yeellow": correct = "" (no
 * match)
 * Example: wordlist = ["YellOw"], query = "yllw": correct = "" (no
 * match)
 * 
 * 
 * 
 * 
 * In addition, the spell checker operates under the following precedence
 * rules:
 * 
 * 
 * When the query exactly matches a word in the wordlist (case-sensitive), you
 * should return the same word back.
 * When the query matches a word up to capitlization, you should return the
 * first such match in the wordlist.
 * When the query matches a word up to vowel errors, you should return the
 * first such match in the wordlist.
 * If the query has no matches in the wordlist, you should return the empty
 * string.
 * 
 * 
 * Given some queries, return a list of words answer, where answer[i] is the
 * correct word for query = queries[i].
 * 
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: wordlist = ["KiTe","kite","hare","Hare"], queries =
 * ["kite","Kite","KiTe","Hare","HARE","Hear","hear","keti","keet","keto"]
 * Output: ["kite","KiTe","KiTe","Hare","hare","","","KiTe","","KiTe"]
 * 
 * 
 * 
 * Note:
 * 
 * 
 * 1 <= wordlist.length <= 5000
 * 1 <= queries.length <= 5000
 * 1 <= wordlist[i].length <= 7
 * 1 <= queries[i].length <= 7
 * All strings in wordlist and queries consist only of english letters.
 * 
 * 
 */

// @lc code=start
class Solution {
  static void set_to_lower(string *s) {
    transform(s->begin(), s->end(), s->begin(),
        [](unsigned char c){ return tolower(c); });
  }
  static void filter_vowel(string *s) {
    for (auto& c : *s) {
      switch (c) {
        case 'a':
        case 'e':
        case 'i':
        case 'o':
        case 'u':
          c = 'a';
          break;
        default:
          break;
      }
    }
  }
 public:
  vector<string> spellchecker(vector<string>& wordlist, vector<string>& queries) {
    auto answer = vector<string>();
    // build dictionaries
    auto exact_match = unordered_set<string>();
    auto lower_match = unordered_map<string, int>();
    auto vowel_match = unordered_map<string, int>();
    for (int i = wordlist.size() - 1; i >= 0; --i) {
      auto word = wordlist[i];
      exact_match.emplace(word);
      set_to_lower(&word);
      lower_match[word] = i;
      filter_vowel(&word);
      vowel_match[word] = i;
    }
    // look up
    for (auto word : queries) {
      if (exact_match.find(word) != exact_match.end()) {
        answer.emplace_back(word);
      } else {
        set_to_lower(&word);
        auto iter = lower_match.find(word);
        if (iter != lower_match.end()) {
          answer.emplace_back(wordlist[iter->second]); 
        } else {
          filter_vowel(&word);
          iter = vowel_match.find(word);
          if (iter != vowel_match.end()) {
            answer.emplace_back(wordlist[iter->second]);
          } else {
            answer.emplace_back();
          }
        }
      }
    }
    return answer;
  }
};
// @lc code=end

