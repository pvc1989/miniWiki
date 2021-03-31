/*
 * @lc app=leetcode id=936 lang=cpp
 *
 * [936] Stamping The Sequence
 *
 * https://leetcode.com/problems/stamping-the-sequence/description/
 *
 * algorithms
 * Hard (47.50%)
 * Likes:    400
 * Dislikes: 105
 * Total Accepted:    20.4K
 * Total Submissions: 38.4K
 * Testcase Example:  '"abc"\n"ababc"'
 *
 * You want to form a target string of lowercase letters.
 * 
 * At the beginning, your sequence is target.length '?' marks.  You also have a
 * stamp of lowercase letters.
 * 
 * On each turn, you may place the stamp over the sequence, and replace every
 * letter in the sequence with the corresponding letter from the stamp.  You
 * can make up to 10 * target.length turns.
 * 
 * For example, if the initial sequence is "?????", and your stamp is "abc",
 * then you may make "abc??", "?abc?", "??abc" in the first turn.  (Note that
 * the stamp must be fully contained in the boundaries of the sequence in order
 * to stamp.)
 * 
 * If the sequence is possible to stamp, then return an array of the index of
 * the left-most letter being stamped at each turn.  If the sequence is not
 * possible to stamp, return an empty array.
 * 
 * For example, if the sequence is "ababc", and the stamp is "abc", then we
 * could return the answer [0, 2], corresponding to the moves "?????" ->
 * "abc??" -> "ababc".
 * 
 * Also, if the sequence is possible to stamp, it is guaranteed it is possible
 * to stamp within 10 * target.length moves.  Any answers specifying more than
 * this number of moves will not be accepted.
 * 
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: stamp = "abc", target = "ababc"
 * Output: [0,2]
 * ([1,0,2] would also be accepted as an answer, as well as some other
 * answers.)
 * 
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: stamp = "abca", target = "aabcaca"
 * Output: [3,0,1]
 * 
 * 
 * 
 * 
 * 
 * Note:
 * 
 * 
 * 
 * 
 * 1 <= stamp.length <= target.length <= 1000
 * stamp and target only contain lowercase letters.
 * 
 */

// @lc code=start
class Solution {
 public:
  vector<int> movesToStamp(string stamp, string target) {
    if (stamp == target) { return {0}; }
    auto result = vector<int>();
    for (bool t_changed = true; t_changed;/**/) {
      t_changed = false;
      for (int t = target.size() - stamp.size(); t >= 0; --t) {
        bool s_changed = false, s_scanned = true;
        for (int s = stamp.size() - 1; s >= 0; --s) {
          auto c = target[t + s];
          if (c == '*') {
            continue;
          } else if (c == stamp[s]) {
            s_changed = true;
          } else {
            s_scanned = false;
            break;
          }
        }
        if (s_changed && s_scanned) {
          t_changed = true;
          result.push_back(t);
          fill_n(target.begin() + t, stamp.size(), '*');
        }
      }
    }
    if (count(target.begin(), target.end(), '*') == target.size()) {
      reverse(result.begin(), result.end());
    } else {
      result.clear();
    }
    return result;
  }
};
// @lc code=end

