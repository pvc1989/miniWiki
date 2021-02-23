/*
 * @lc app=leetcode id=49 lang=cpp
 *
 * [49] Group Anagrams
 *
 * https://leetcode.com/problems/group-anagrams/description/
 *
 * algorithms
 * Medium (59.16%)
 * Likes:    4931
 * Dislikes: 222
 * Total Accepted:    859.2K
 * Total Submissions: 1.5M
 * Testcase Example:  '["eat","tea","tan","ate","nat","bat"]'
 *
 * Given an array of strings strs, group the anagrams together. You can return
 * the answer in any order.
 * 
 * An Anagram is a word or phrase formed by rearranging the letters of a
 * different word or phrase, typically using all the original letters exactly
 * once.
 * 
 * 
 * Example 1:
 * Input: strs = ["eat","tea","tan","ate","nat","bat"]
 * Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
 * Example 2:
 * Input: strs = [""]
 * Output: [[""]]
 * Example 3:
 * Input: strs = ["a"]
 * Output: [["a"]]
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= strs.length <= 10^4
 * 0 <= strs[i].length <= 100
 * strs[i] consists of lower-case English letters.
 * 
 * 
 */

// @lc code=start
class Solution {
  struct Record {
    string anagram;
    string const* address;
    explicit Record(string const& str) : anagram(str), address(&str) {
      sort(anagram.begin(), anagram.end());
    }
    bool operator<(Record const& that) const {
      return anagram <  that.anagram ||
             anagram == that.anagram && address < that.address;
    }
  };
 public:
  vector<vector<string>> groupAnagrams(vector<string>& strs) {
    auto records = vector<Record>();
    for (auto& str : strs) {
      records.emplace_back(str);
    }
    sort(records.begin(), records.end());
    auto groups = vector<vector<string>>();
    groups.emplace_back();
    groups.back().emplace_back(*(records[0].address));
    for (int i = 1; i < strs.size(); ++i) {
      if (records[i].anagram != records[i - 1].anagram) {
        groups.emplace_back();
      }
      groups.back().emplace_back(*(records[i].address));
    }
    return groups;
  }
};
// @lc code=end

