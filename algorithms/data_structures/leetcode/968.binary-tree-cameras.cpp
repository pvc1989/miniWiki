/*
 * @lc app=leetcode id=968 lang=cpp
 *
 * [968] Binary Tree Cameras
 *
 * https://leetcode.com/problems/binary-tree-cameras/description/
 *
 * algorithms
 * Hard (38.84%)
 * Likes:    1414
 * Dislikes: 21
 * Total Accepted:    35.4K
 * Total Submissions: 90.3K
 * Testcase Example:  '[0,0,null,0,0]'
 *
 * Given a binary tree, we install cameras on the nodes of the tree. 
 * 
 * Each camera at a node can monitor its parent, itself, and its immediate
 * children.
 * 
 * Calculate the minimum number of cameras needed to monitor all nodes of the
 * tree.
 * 
 * 
 * 
 * Example 1:
 * 
 * 
 * 
 * Input: [0,0,null,0,0]
 * Output: 1
 * Explanation: One camera is enough to monitor all nodes if placed as
 * shown.
 * 
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: [0,0,null,0,null,0,null,null,0]
 * Output: 2
 * Explanation: At least two cameras are needed to monitor all nodes of the
 * tree. The above image shows one of the valid configurations of camera
 * placement.
 * 
 * 
 * 
 * Note:
 * 
 * 
 * The number of nodes in the given tree will be in the range [1, 1000].
 * Every node has value 0.
 * 
 * 
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
  vector<array<int, 3>> min_cameras_;

  static int count(TreeNode* node, int i) {
    int n = 0;
    if (node) {
      node->val = i;
      n = 1;
      n += count(node->left , i + n);
      n += count(node->right, i + n);
    }
    return n;
  }

  int lookup(TreeNode* node, int parent) {
    /* parent = 0 : parent is uncovered
              = 1 : parent is covered by other nodes
              = 2 : parent is a camera
     */
    assert(node);
    int min_cameras = min_cameras_[node->val][parent];
    if (min_cameras == INT_MAX) {
      if (node->left) {
        if (node->right) {
          switch (parent) {
          case 2:
            min_cameras = min(min_cameras, 0 + lookup(node->left, 1) + lookup(node->right, 1));
          case 1:
            min_cameras = min(min_cameras, 0 + lookup(node->left, 0) + lookup(node->right, 1));
            min_cameras = min(min_cameras, 0 + lookup(node->left, 1) + lookup(node->right, 0));
          case 0:
            min_cameras = min(min_cameras, 1 + lookup(node->left, 2) + lookup(node->right, 2));
          default:
            break;
          }
        }
        else {
          switch (parent) {
          case 2:
            min_cameras = min(min_cameras, 0 + lookup(node->left, 1));
          case 1:
            min_cameras = min(min_cameras, 0 + lookup(node->left, 0));
          case 0:
            min_cameras = min(min_cameras, 1 + lookup(node->left, 2));
          default:
            break;
          }
        }
      }
      else {
        if (node->right) {
          switch (parent) {
          case 2:
            min_cameras = min(min_cameras, 0 + lookup(node->right, 1));
          case 1:
            min_cameras = min(min_cameras, 0 + lookup(node->right, 0));
          case 0:
            min_cameras = min(min_cameras, 1 + lookup(node->right, 2));
          default:
            break;
          }
        }
        else {
          assert(false);
        }
      }
      min_cameras_[node->val][parent] = min_cameras;
    }
    return min_cameras;
  }

  void leaves(TreeNode* node) {
    assert(node);
    if (node->left)
      leaves(node->left);
    if (node->right)
      leaves(node->right);
    else if (node->left  == nullptr) {
      assert(node->right == nullptr);
      auto v = node->val;
      min_cameras_[v][0] = 1;
      min_cameras_[v][1] = 1;
      min_cameras_[v][2] = 0;
    }
  }

 public:
  int minCameraCover(TreeNode* root) {
    int n = count(root, 0);
    min_cameras_.resize(n);
    for (auto& row : min_cameras_)
      row.fill(INT_MAX);
    leaves(root);
    return lookup(root, 1);
  }
};
// @lc code=end

