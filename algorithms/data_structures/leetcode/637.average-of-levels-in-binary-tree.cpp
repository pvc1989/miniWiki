/*
 * @lc app=leetcode id=637 lang=cpp
 *
 * [637] Average of Levels in Binary Tree
 *
 * https://leetcode.com/problems/average-of-levels-in-binary-tree/description/
 *
 * algorithms
 * Easy (64.83%)
 * Likes:    1751
 * Dislikes: 188
 * Total Accepted:    164.5K
 * Total Submissions: 252.3K
 * Testcase Example:  '[3,9,20,15,7]'
 *
 * Given a non-empty binary tree, return the average value of the nodes on each
 * level in the form of an array.
 * 
 * Example 1:
 * 
 * Input:
 * ⁠   3
 * ⁠  / \
 * ⁠ 9  20
 * ⁠   /  \
 * ⁠  15   7
 * Output: [3, 14.5, 11]
 * Explanation:
 * The average value of nodes on level 0 is 3,  on level 1 is 14.5, and on
 * level 2 is 11. Hence return [3, 14.5, 11].
 * 
 * 
 * 
 * Note:
 * 
 * The range of node's value is in the range of 32-bit signed integer.
 * 
 * 
 */

// @lc code=start
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
 public:
  vector<double> averageOfLevels(TreeNode* root) {
    auto averages = vector<double>();
    auto curr_level = vector<TreeNode*>();
    auto next_level = vector<TreeNode*>();
    if (root) next_level.emplace_back(root);
    while (!next_level.empty()) {
      swap(next_level, curr_level);
      next_level.clear();
      double sum = 0.0;
      for (auto* node : curr_level) {
        sum += node->val;
        if (node->left ) next_level.emplace_back(node->left );
        if (node->right) next_level.emplace_back(node->right);
      }
      averages.emplace_back(sum / curr_level.size());
    }
    return averages;
  }
};
// @lc code=end

