/*
 * @lc app=leetcode id=841 lang=cpp
 *
 * [841] Keys and Rooms
 *
 * https://leetcode.com/problems/keys-and-rooms/description/
 *
 * algorithms
 * Medium (65.44%)
 * Likes:    1554
 * Dislikes: 111
 * Total Accepted:    106.8K
 * Total Submissions: 162.5K
 * Testcase Example:  '[[1],[2],[3],[]]'
 *
 * There are N rooms and you start in room 0.  Each room has a distinct number
 * in 0, 1, 2, ..., N-1, and each room may have some keys to access the next
 * room. 
 * 
 * Formally, each room i has a list of keys rooms[i], and each key rooms[i][j]
 * is an integer in [0, 1, ..., N-1] where N = rooms.length.  A key rooms[i][j]
 * = v opens the room with number v.
 * 
 * Initially, all the rooms start locked (except for room 0). 
 * 
 * You can walk back and forth between rooms freely.
 * 
 * Return true if and only if you can enter every room.
 * 
 * 
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: [[1],[2],[3],[]]
 * Output: true
 * Explanation:  
 * We start in room 0, and pick up key 1.
 * We then go to room 1, and pick up key 2.
 * We then go to room 2, and pick up key 3.
 * We then go to room 3.  Since we were able to go to every room, we return
 * true.
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: [[1,3],[3,0,1],[2],[0]]
 * Output: false
 * Explanation: We can't enter the room with number 2.
 * 
 * 
 * Note:
 * 
 * 
 * 1 <= rooms.length <= 1000
 * 0 <= rooms[i].length <= 1000
 * The number of keys in all rooms combined is at most 3000.
 * 
 * 
 */

// @lc code=start
class Solution {
 public:
  bool canVisitAllRooms(vector<vector<int>>& rooms) {
    auto visited = vector<bool>(rooms.size());
    auto curr_level = vector<int>{ };
    auto next_level = vector<int>{0};
    int n_visited = 0;
    while (!next_level.empty()) {
      swap(curr_level, next_level);
      next_level.clear();
      for (auto u : curr_level) {
        if (visited[u] == false) {
          visited[u] = true;
          ++n_visited;
          for (auto v : rooms[u]) {
            next_level.push_back(v);
          }
        }
      }
    }
    return n_visited == rooms.size(); 
  }
};
// @lc code=end

