/*
 * @lc app=leetcode id=25 lang=cpp
 *
 * [25] Reverse Nodes in k-Group
 *
 * https://leetcode.com/problems/reverse-nodes-in-k-group/description/
 *
 * algorithms
 * Hard (46.27%)
 * Likes:    4270
 * Dislikes: 419
 * Total Accepted:    376.5K
 * Total Submissions: 804.3K
 * Testcase Example:  '[1,2,3,4,5]\n2'
 *
 * Given a linked list, reverse the nodes of a linked list k at a time and
 * return its modified list.
 * 
 * k is a positive integer and is less than or equal to the length of the
 * linked list. If the number of nodes is not a multiple of k then left-out
 * nodes, in the end, should remain as it is.
 * 
 * You may not alter the values in the list's nodes, only nodes themselves may
 * be changed.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: head = [1,2,3,4,5], k = 2
 * Output: [2,1,4,3,5]
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: head = [1,2,3,4,5], k = 3
 * Output: [3,2,1,4,5]
 * 
 * 
 * Example 3:
 * 
 * 
 * Input: head = [1,2,3,4,5], k = 1
 * Output: [1,2,3,4,5]
 * 
 * 
 * Example 4:
 * 
 * 
 * Input: head = [1], k = 1
 * Output: [1]
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * The number of nodes in the list is in the range sz.
 * 1 <= sz <= 5000
 * 0 <= Node.val <= 1000
 * 1 <= k <= sz
 * 
 * 
 * 
 * Follow-up: Can you solve the problem in O(1) extra memory space?
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
  ListNode* jump(ListNode* head, int k) {
    int i = 0;
    while (head && ++i < k) {
      head = head->next;
    }
    return head;
  }
  void reverse(ListNode* head, ListNode* tail) {
    assert(head && tail);
    auto curr = head;
    auto next = curr->next;
    head = tail->next;
    //   reversed list = [head, tail->next)
    // unreversed list = [curr, tail->next)
    while (curr != tail) {
      next = curr->next;
      curr->next = head;
      head = curr;
      curr = next;
    }
    tail->next = head;
  }
 public:
  ListNode* reverseKGroup(ListNode* head, int k) {
    if (k == 1)
      return head;
    auto curr = head;
    auto next = jump(curr, k);
    auto prev_tail = curr;
    if (next) {
      head = next;
      // curr+0 -> ... -> curr+(k-1) == next
      reverse(curr, next);
      // next+0 -> ... -> next+(k-1) == curr
      curr = curr->next;
      next = jump(curr, k);
    }
    while (next) {
      // curr+0 -> ... -> curr+(k-1) == next
      reverse(curr, next);
      // next+0 -> ... -> next+(k-1) == curr
      prev_tail->next = next;
      prev_tail = curr;
      curr = curr->next;
      next = jump(curr, k);
    }
    return head;
  }
};
// @lc code=end

