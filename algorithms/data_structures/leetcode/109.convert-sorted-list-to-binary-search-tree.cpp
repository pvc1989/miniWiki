/*
 * @lc app=leetcode id=109 lang=cpp
 *
 * [109] Convert Sorted List to Binary Search Tree
 *
 * https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/description/
 *
 * algorithms
 * Medium (50.48%)
 * Likes:    2995
 * Dislikes: 97
 * Total Accepted:    295.9K
 * Total Submissions: 578.3K
 * Testcase Example:  '[-10,-3,0,5,9]'
 *
 * Given the head of a singly linked list where elements are sorted in
 * ascending order, convert it to a height balanced BST.
 * 
 * For this problem, a height-balanced binary tree is defined as a binary tree
 * in which the depth of the two subtrees of every node never differ by more
 * than 1.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: head = [-10,-3,0,5,9]
 * Output: [0,-3,9,-10,null,5]
 * Explanation: One possible answer is [0,-3,9,-10,null,5], which represents
 * the shown height balanced BST.
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: head = []
 * Output: []
 * 
 * 
 * Example 3:
 * 
 * 
 * Input: head = [0]
 * Output: [0]
 * 
 * 
 * Example 4:
 * 
 * 
 * Input: head = [1,3]
 * Output: [3,1]
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * The number of nodes in head is in the range [0, 2 * 10^4].
 * -10^5 <= Node.val <= 10^5
 * 
 * 
 */

// @lc code=start
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
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
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
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
  TreeNode* build(ListNode* head, ListNode* tail) {
    TreeNode* root = nullptr;
    if (head) {
      auto fast = head, slow = head;
      while (fast != tail) {
        assert(fast);
        fast = fast->next;
        if (fast != tail) {
          assert(fast);
          fast = fast->next;
        }
        else {
          break;
        }
        slow = slow->next;
      }
      assert(fast == tail);
      // printf("head = %p, slow = %p, tail = %p\n", head, slow, tail);
      if (slow == head) {
        // only 1 ListNode in [head, slow)
        root = new TreeNode(head->val);
      }
      else {
        assert(slow != tail);
        // printf("head->val = %d, slow->val = %d, tail->val = %d\n", head->val, slow->val, tail ? tail->val : INT_MAX);
        root = new TreeNode(slow->val, build(head, slow), nullptr);
        if (slow->next != tail)
          root->right = build(slow->next, tail);
      }
    }
    return root;
  }
 public:
  TreeNode* sortedListToBST(ListNode* head) {
    return build(head, nullptr);
  }
};
// @lc code=end

