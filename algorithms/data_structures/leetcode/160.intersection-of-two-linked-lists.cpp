/*
 * @lc app=leetcode id=160 lang=cpp
 *
 * [160] Intersection of Two Linked Lists
 *
 * https://leetcode.com/problems/intersection-of-two-linked-lists/description/
 *
 * algorithms
 * Easy (43.23%)
 * Likes:    5152
 * Dislikes: 583
 * Total Accepted:    619K
 * Total Submissions: 1.4M
 * Testcase Example:  '8\n[4,1,8,4,5]\n[5,6,1,8,4,5]\n2\n3'
 *
 * Write a program to find the node at which the intersection of two singly
 * linked lists begins.
 * 
 * For example, the following two linked lists:
 * 
 * 
 * begin to intersect at node c1.
 * 
 * 
 * 
 * Example 1:
 * 
 * 
 * 
 * Input: intersectVal = 8, listA = [4,1,8,4,5], listB = [5,6,1,8,4,5], skipA =
 * 2, skipB = 3
 * Output: Reference of the node with value = 8
 * Input Explanation: The intersected node's value is 8 (note that this must
 * not be 0 if the two lists intersect). From the head of A, it reads as
 * [4,1,8,4,5]. From the head of B, it reads as [5,6,1,8,4,5]. There are 2
 * nodes before the intersected node in A; There are 3 nodes before the
 * intersected node in B.
 * 
 * 
 * 
 * Example 2:
 * 
 * 
 * 
 * Input: intersectVal = 2, listA = [1,9,1,2,4], listB = [3,2,4], skipA = 3,
 * skipB = 1
 * Output: Reference of the node with value = 2
 * Input Explanation: The intersected node's value is 2 (note that this must
 * not be 0 if the two lists intersect). From the head of A, it reads as
 * [1,9,1,2,4]. From the head of B, it reads as [3,2,4]. There are 3 nodes
 * before the intersected node in A; There are 1 node before the intersected
 * node in B.
 * 
 * 
 * 
 * 
 * Example 3:
 * 
 * 
 * 
 * Input: intersectVal = 0, listA = [2,6,4], listB = [1,5], skipA = 3, skipB =
 * 2
 * Output: null
 * Input Explanation: From the head of A, it reads as [2,6,4]. From the head of
 * B, it reads as [1,5]. Since the two lists do not intersect, intersectVal
 * must be 0, while skipA and skipB can be arbitrary values.
 * Explanation: The two lists do not intersect, so return null.
 * 
 * 
 * 
 * 
 * Notes:
 * 
 * 
 * If the two linked lists have no intersection at all, return null.
 * The linked lists must retain their original structure after the function
 * returns.
 * You may assume there are no cycles anywhere in the entire linked
 * structure.
 * Each value on each linked list is in the range [1, 10^9].
 * Your code should preferably run in O(n) time and use only O(1) memory.
 * 
 * 
 */

// @lc code=start
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
  int GetLength(ListNode* head) {
    int length = -1;
    while (head) {
      ++length;
      head = head->next;
    }
    return length;
  }
  ListNode* Reverse(ListNode* head) {
    ListNode* new_head = nullptr;
    auto* curr = head;
    while (curr) {
      auto* next = new_head;
      new_head = curr;
      curr = curr->next;
      new_head->next = next;
    }
    return new_head;
  }
  ListNode* GetLastNode(ListNode* head) {
    while (head->next) {
      head = head->next;
    }
    return head;
  }
 public:
  ListNode* getIntersectionNode(ListNode *head_a, ListNode *head_b) {
    if (!head_a || !head_b) { return nullptr; }
    if (head_a == head_b) { return head_a; }
    ListNode* intersection_node = nullptr;
    auto length_a = GetLength(head_a);
    auto length_b = GetLength(head_b);
    auto* last_a = Reverse(head_a);
    auto* last_b = GetLastNode(head_b);
    if (last_b == head_a) {
      auto length_b_new = GetLength(head_b);
      auto length_c = (length_a + length_b - length_b_new) / 2;
      for (int i = length_b - length_c; i != 0; --i) {
        head_b = head_b->next;
      }
      intersection_node = head_b;
    }
    Reverse(last_a);
    return intersection_node;
  }
};
// @lc code=end

