/*
 * @lc app=leetcode id=86 lang=cpp
 *
 * [86] Partition List
 *
 * https://leetcode.com/problems/partition-list/description/
 *
 * algorithms
 * Medium (43.63%)
 * Likes:    2029
 * Dislikes: 386
 * Total Accepted:    258.9K
 * Total Submissions: 589.1K
 * Testcase Example:  '[1,4,3,2,5,2]\n3'
 *
 * Given the head of a linked list and a value x, partition it such that all
 * nodes less than x come before nodes greater than or equal to x.
 * 
 * You should preserve the original relative order of the nodes in each of the
 * two partitions.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: head = [1,4,3,2,5,2], x = 3
 * Output: [1,2,2,4,3,5]
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: head = [2,1], x = 2
 * Output: [1,2]
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * The number of nodes in the list is in the range [0, 200].
 * -100 <= Node.val <= 100
 * -200 <= x <= 200
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
  ListNode* partition(ListNode* head, int x) {
    ListNode *head_lt = nullptr, *last_lt = nullptr;
    ListNode *head_ge = nullptr, *last_ge = nullptr;
    while (head) {
      if (head->val < x) {
        if (last_lt) {
          assert(head_lt);
          last_lt->next = head;
        } else {
          head_lt = head;
        }
        last_lt = head;
      } else {
        if (last_ge) {
          assert(head_ge);
          last_ge->next = head;
        } else {
          head_ge = head;
        }
        last_ge = head;
      }
      head = head->next;
    }
    if (last_lt)
      last_lt->next = head_ge;
    else
      head_lt = head_ge;
    if (last_ge)
      last_ge->next = nullptr;
    return head_lt;
  }
};
// @lc code=end

