/*
 * @lc app=leetcode id=630 lang=cpp
 *
 * [630] Course Schedule III
 *
 * https://leetcode.com/problems/course-schedule-iii/description/
 *
 * algorithms
 * Hard (33.78%)
 * Likes:    1047
 * Dislikes: 42
 * Total Accepted:    29.1K
 * Total Submissions: 85.7K
 * Testcase Example:  '[[100,200],[200,1300],[1000,1250],[2000,3200]]'
 *
 * There are n different online courses numbered from 1 to n. You are given an
 * array courses where courses[i] = [durationi, lastDayi] indicate that the
 * i^th course should be taken continuously for durationi days and must be
 * finished before or on lastDayi.
 * 
 * You will start on the 1^st day and you cannot take two or more courses
 * simultaneously.
 * 
 * Return the maximum number of courses that you can take.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: courses = [[100,200],[200,1300],[1000,1250],[2000,3200]]
 * Output: 3
 * Explanation: 
 * There are totally 4 courses, but you can take 3 courses at most:
 * First, take the 1^st course, it costs 100 days so you will finish it on the
 * 100^th day, and ready to take the next course on the 101^st day.
 * Second, take the 3^rd course, it costs 1000 days so you will finish it on
 * the 1100^th day, and ready to take the next course on the 1101^st day. 
 * Third, take the 2^nd course, it costs 200 days so you will finish it on the
 * 1300^th day. 
 * The 4^th course cannot be taken now, since you will finish it on the 3300^th
 * day, which exceeds the closed date.
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: courses = [[1,2]]
 * Output: 1
 * 
 * 
 * Example 3:
 * 
 * 
 * Input: courses = [[3,2],[4,3]]
 * Output: 0
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= courses.length <= 10^4
 * 1 <= durationi, lastDayi <= 10^4
 * 
 * 
 */

// @lc code=start
class Solution {
 public:
  int scheduleCourse(vector<vector<int>>& courses) {
    // Sort all courses by their deadlines:
    sort(courses.begin(), courses.end(), [](auto const& left, auto const& right) {
      return left[1] < right[1];
    });
    // Maintain a MaxPQ of courses, using their durations as Keys:
    auto cmp = [&courses](int left, int right) {
      return courses[left][0] < courses[right][0];
    };
    auto pq = priority_queue<int, vector<int>, decltype(cmp)>(cmp);
    int sum{0};  // sum of durations of all taken courses
    for (int i{0}; i != courses.size(); ++i) {
      auto& curr = courses[i];
      if (sum + curr[0] <= curr[1]) {
        // Add this course without replacing any taken course:
        pq.emplace(i);
        sum += curr[0];
      } else if (!pq.empty()) {
        // Try to decrease sum by replacing the longest course:
        auto& top = courses[pq.top()];
        auto new_sum = sum - top[0] + curr[0];
        if (new_sum < sum && new_sum <= curr[1]) {
          sum = new_sum;
          pq.pop();
          pq.emplace(i);
        }
      }
    }
    return pq.size();
  }
};
// @lc code=end

