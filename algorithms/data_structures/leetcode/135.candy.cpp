/*
 * @lc app=leetcode id=135 lang=cpp
 *
 * [135] Candy
 *
 * https://leetcode.com/problems/candy/description/
 *
 * algorithms
 * Hard (33.05%)
 * Likes:    1338
 * Dislikes: 188
 * Total Accepted:    150.7K
 * Total Submissions: 455.7K
 * Testcase Example:  '[1,0,2]'
 *
 * There are n children standing in a line. Each child is assigned a rating
 * value given in the integer array ratings.
 * 
 * You are giving candies to these children subjected to the following
 * requirements:
 * 
 * 
 * Each child must have at least one candy.
 * Children with a higher rating get more candies than their neighbors.
 * 
 * 
 * Return the minimum number of candies you need to have to distribute the
 * candies to the children.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: ratings = [1,0,2]
 * Output: 5
 * Explanation: You can allocate to the first, second and third child with 2,
 * 1, 2 candies respectively.
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: ratings = [1,2,2]
 * Output: 4
 * Explanation: You can allocate to the first, second and third child with 1,
 * 2, 1 candies respectively.
 * The third child gets 1 candy because it satisfies the above two
 * conditions.
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * n == ratings.length
 * 1 <= n <= 2 * 10^4
 * 1 <= ratings[i] <= 2 * 10^4
 * 
 * 
 */

// @lc code=start
class Solution {
 public:
  int candy(vector<int>& ratings) {
    int n = ratings.size();
    auto candies = vector<int>(n, 1);
    auto indices = vector<int>(n);
    iota(indices.begin(), indices.end(), 0);
    auto cmp = [&](int i, int j) { return ratings[i] < ratings[j]; };
    sort(indices.begin(), indices.end(), cmp);
    for (auto i : indices) {
      if (i - 1 >= 0 && ratings[i - 1] < ratings[i]) {
        candies[i] = max(candies[i], candies[i - 1] + 1);
      }
      if (i + 1 != n && ratings[i + 1] < ratings[i]) {
        candies[i] = max(candies[i], candies[i + 1] + 1);
      }
    }
    return accumulate(candies.begin(), candies.end(), 0);
  }
};
// @lc code=end

