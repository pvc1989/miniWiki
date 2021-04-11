/*
 * @lc app=leetcode id=1302 lang=cpp
 *
 * [1302] Deepest Leaves Sum
 *
 * https://leetcode.com/problems/deepest-leaves-sum/description/
 *
 * algorithms
 * Medium (84.20%)
 * Likes:    1257
 * Dislikes: 56
 * Total Accepted:    100.1K
 * Total Submissions: 117.1K
 * Testcase Example:  '[1,2,3,4,5,null,6,7,null,null,null,null,8]'
 *
 * Given the root of a binary tree, return the sum of values of its deepest
 * leaves.
 * 
 * Example 1:
 * 
 * 
 * Input: root = [1,2,3,4,5,null,6,7,null,null,null,null,8]
 * Output: 15
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: root = [6,7,8,2,7,1,3,9,null,1,4,null,null,null,5]
 * Output: 19
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * The number of nodes in the tree is in the range [1, 10^4].
 * 1 <= Node.val <= 100
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
  int deepestLeavesSum(TreeNode* root) {
    auto curr_level = vector<TreeNode*>();
    auto next_level = vector<TreeNode*>();
    if (root) next_level.emplace_back(root);
    while (!next_level.empty()) {
      swap(curr_level, next_level);
      next_level.clear();
      for (auto* node : curr_level) {
        if (node->left ) next_level.emplace_back(node->left );
        if (node->right) next_level.emplace_back(node->right);
      }
    }
    return accumulate(curr_level.begin(), curr_level.end(), 0,
        [](int x, TreeNode *y){ return x + y->val; });
  }
};
// @lc code=end

