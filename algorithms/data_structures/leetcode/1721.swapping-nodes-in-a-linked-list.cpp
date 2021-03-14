/*
 * @lc app=leetcode id=1721 lang=cpp
 *
 * [1721] Swapping Nodes in a Linked List
 *
 * https://leetcode.com/problems/swapping-nodes-in-a-linked-list/description/
 *
 * algorithms
 * Medium (64.29%)
 * Likes:    265
 * Dislikes: 18
 * Total Accepted:    20K
 * Total Submissions: 30.8K
 * Testcase Example:  '[1,2,3,4,5]\n2'
 *
 * You are given the head of a linked list, and an integer k.
 * 
 * Return the head of the linked list after swapping the values of the k^th
 * node from the beginning and the k^th node from the end (the list is
 * 1-indexed).
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: head = [1,2,3,4,5], k = 2
 * Output: [1,4,3,2,5]
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: head = [7,9,6,6,7,8,3,0,9,5], k = 5
 * Output: [7,9,6,6,8,7,3,0,9,5]
 * 
 * 
 * Example 3:
 * 
 * 
 * Input: head = [1], k = 1
 * Output: [1]
 * 
 * 
 * Example 4:
 * 
 * 
 * Input: head = [1,2], k = 1
 * Output: [2,1]
 * 
 * 
 * Example 5:
 * 
 * 
 * Input: head = [1,2,3], k = 2
 * Output: [1,2,3]
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * The number of nodes in the list is n.
 * 1 <= k <= n <= 10^5
 * 0 <= Node.val <= 100
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
class Solution {
 public:
  ListNode* swapNodes(ListNode* head, int k) {
    int n = 0;
    auto* node = head;
    while (node) {
      ++n;
      node = node->next;
    }
    int k_reverse = n - k + 1;
    auto* kth_from_head = head;
    auto* kth_from_tail = head;
    node = head;
    for (int i = 1; i <= n; ++i) {
      if (i == k) {
        kth_from_head = node;
      }
      if (i == k_reverse) {
        kth_from_tail = node;
      }
      node = node->next;
    }
    swap(kth_from_head->val, kth_from_tail->val);
    return head;
  }
};
// @lc code=end

