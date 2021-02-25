/*
 * @lc app=leetcode id=621 lang=cpp
 *
 * [621] Task Scheduler
 *
 * https://leetcode.com/problems/task-scheduler/description/
 *
 * algorithms
 * Medium (51.69%)
 * Likes:    4415
 * Dislikes: 847
 * Total Accepted:    243.8K
 * Total Submissions: 470.8K
 * Testcase Example:  '["A","A","A","B","B","B"]\n2'
 *
 * Given a characters array tasks, representing the tasks a CPU needs to do,
 * where each letter represents a different task. Tasks could be done in any
 * order. Each task is done in one unit of time. For each unit of time, the CPU
 * could complete either one task or just be idle.
 * 
 * However, there is a non-negative integer n that represents the cooldown
 * period between two same tasks (the same letter in the array), that is that
 * there must be at least n units of time between any two same tasks.
 * 
 * Return the least number of units of times that the CPU will take to finish
 * all the given tasks.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: tasks = ["A","A","A","B","B","B"], n = 2
 * Output: 8
 * Explanation: 
 * A -> B -> idle -> A -> B -> idle -> A -> B
 * There is at least 2 units of time between any two same tasks.
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: tasks = ["A","A","A","B","B","B"], n = 0
 * Output: 6
 * Explanation: On this case any permutation of size 6 would work since n = 0.
 * ["A","A","A","B","B","B"]
 * ["A","B","A","B","A","B"]
 * ["B","B","B","A","A","A"]
 * ...
 * And so on.
 * 
 * 
 * Example 3:
 * 
 * 
 * Input: tasks = ["A","A","A","A","A","A","B","C","D","E","F","G"], n = 2
 * Output: 16
 * Explanation: 
 * One possible solution is
 * A -> B -> C -> A -> D -> E -> A -> F -> G -> A -> idle -> idle -> A -> idle
 * -> idle -> A
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= task.length <= 10^4
 * tasks[i] is upper-case English letter.
 * The integer n is in the range [0, 100].
 * 
 * 
 */

// @lc code=start
class Solution {
 public:
  int leastInterval(vector<char>& tasks, int min_gap) {
    auto task_to_count = array<int, 26>();
    for (auto t : tasks) { ++task_to_count[t - 'A']; }
    sort(task_to_count.begin(), task_to_count.end(), greater<int>());
    auto first_less_freq_task = upper_bound(
        task_to_count.begin(), task_to_count.end(),
        task_to_count.front(), greater<int>());
    int count_most_freq_tasks = first_less_freq_task - task_to_count.begin();
    int sum_less_freq_tasks =
        tasks.size() - count_most_freq_tasks * task_to_count.front();
    int sum_idle_time_units =
        (min_gap - count_most_freq_tasks + 1) * (task_to_count.front() - 1);
    return sum_idle_time_units < max(0, sum_less_freq_tasks)
        ? tasks.size()  // no gaps to fill
        : count_most_freq_tasks * task_to_count.front() + sum_idle_time_units;
  }
};
// @lc code=end

