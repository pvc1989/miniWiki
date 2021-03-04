/*
 * @lc app=leetcode id=1507 lang=cpp
 *
 * [1507] Reformat Date
 *
 * https://leetcode.com/problems/reformat-date/description/
 *
 * algorithms
 * Easy (60.10%)
 * Likes:    103
 * Dislikes: 199
 * Total Accepted:    18.5K
 * Total Submissions: 30.7K
 * Testcase Example:  '"20th Oct 2052"'
 *
 * Given a date string in the form Day Month Year, where:
 * 
 * 
 * Day is in the set {"1st", "2nd", "3rd", "4th", ..., "30th", "31st"}.
 * Month is in the set {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug",
 * "Sep", "Oct", "Nov", "Dec"}.
 * Year is in the range [1900, 2100].
 * 
 * 
 * Convert the date string to the format YYYY-MM-DD, where:
 * 
 * 
 * YYYY denotes the 4 digit year.
 * MM denotes the 2 digit month.
 * DD denotes the 2 digit day.
 * 
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: date = "20th Oct 2052"
 * Output: "2052-10-20"
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: date = "6th Jun 1933"
 * Output: "1933-06-06"
 * 
 * 
 * Example 3:
 * 
 * 
 * Input: date = "26th May 1960"
 * Output: "1960-05-26"
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * The given dates are guaranteed to be valid, so no error handling is
 * necessary.
 * 
 * 
 */

// @lc code=start
class Solution {
  unordered_map<string, string> months_ {
    {"Jan", "-01-"}, {"Feb", "-02-"}, {"Mar", "-03-"}, {"Apr", "-04-"},
    {"May", "-05-"}, {"Jun", "-06-"}, {"Jul", "-07-"}, {"Aug", "-08-"},
    {"Sep", "-09-"}, {"Oct", "-10-"}, {"Nov", "-11-"}, {"Dec", "-12-"}
  };
 public:
  string reformatDate(string date) {
    int head = 0, tail = date.find_first_of("snrt");
    auto day = date.substr(head, tail - head);
    if (day.size() == 1) day.insert(0, "0");
    head = tail + 3;
    auto month = months_[date.substr(head, 3)];
    head += 4;
    return date.substr(head, 4) + month + day;
  }
};
// @lc code=end

