/*
 * @lc app=leetcode id=60 lang=cpp
 *
 * [60] Permutation Sequence
 *
 * https://leetcode.com/problems/permutation-sequence/description/
 *
 * algorithms
 * Hard (39.37%)
 * Likes:    2096
 * Dislikes: 361
 * Total Accepted:    220.9K
 * Total Submissions: 560.5K
 * Testcase Example:  '3\n3'
 *
 * The set [1, 2, 3, ..., n] contains a total of n! unique permutations.
 * 
 * By listing and labeling all of the permutations in order, we get the
 * following sequence for n = 3:
 * 
 * 
 * "123"
 * "132"
 * "213"
 * "231"
 * "312"
 * "321"
 * 
 * 
 * Given n and k, return the k^th permutation sequence.
 * 
 * 
 * Example 1:
 * Input: n = 3, k = 3
 * Output: "213"
 * Example 2:
 * Input: n = 4, k = 9
 * Output: "2314"
 * Example 3:
 * Input: n = 3, k = 1
 * Output: "123"
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= n <= 9
 * 1 <= k <= n!
 * 
 * 
 */

// @lc code=start
class Solution {
 public:
  string getPermutation(int n, int k) {
    auto factorials = array<int, 10>();
    factorials[0] = 1;
    for (int i = 1; i != 10; ++i) {
      factorials[i] = factorials[i - 1] * i;
    }
    auto digits = vector<char>(n);
    iota(digits.begin(), digits.end(), '1');
    --k;  // division is easier when using 0-based index
    auto kth_permutation = string();
    while (!digits.empty()) {
      auto n_permutations = factorials.at(digits.size() - 1);
      auto iter = digits.begin() + k / n_permutations;
      k = k % n_permutations;
      kth_permutation.push_back(*iter);
      digits.erase(iter);
    }
    return kth_permutation;
  }
};
// @lc code=end

