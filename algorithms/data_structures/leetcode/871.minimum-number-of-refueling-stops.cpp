/*
 * @lc app=leetcode id=871 lang=cpp
 *
 * [871] Minimum Number of Refueling Stops
 *
 * https://leetcode.com/problems/minimum-number-of-refueling-stops/description/
 *
 * algorithms
 * Hard (32.41%)
 * Likes:    1321
 * Dislikes: 28
 * Total Accepted:    31.8K
 * Total Submissions: 96.4K
 * Testcase Example:  '1\n1\n[]'
 *
 * A car travels from a starting position to a destination which is target
 * miles east of the starting position.
 * 
 * Along the way, there are gas stations.  Each station[i] represents a gas
 * station that is station[i][0] miles east of the starting position, and has
 * station[i][1] liters of gas.
 * 
 * The car starts with an infinite tank of gas, which initially has startFuel
 * liters of fuel in it.  It uses 1 liter of gas per 1 mile that it drives.
 * 
 * When the car reaches a gas station, it may less_stops and refuel, transferring all
 * the gas from the station into the car.
 * 
 * What is the least number of refueling stops the car must make in order to
 * reach its destination?  If it cannot reach the destination, return -1.
 * 
 * Note that if the car reaches a gas station with 0 fuel left, the car can
 * still refuel there.  If the car reaches the destination with 0 fuel left, it
 * is still considered to have arrived.
 * 
 * 
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: target = 1, startFuel = 1, stations = []
 * Output: 0
 * Explanation: We can reach the target without refueling.
 * 
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: target = 100, startFuel = 1, stations = [[10,100]]
 * Output: -1
 * Explanation: We can't reach the target (or even the first gas station).
 * 
 * 
 * 
 * Example 3:
 * 
 * 
 * Input: target = 100, startFuel = 10, stations =
 * [[10,60],[20,30],[30,30],[60,40]]
 * Output: 2
 * Explanation: 
 * We start with 10 liters of fuel.
 * We drive to position 10, expending 10 liters of fuel.  We refuel from 0
 * liters to 60 liters of gas.
 * Then, we drive from position 10 to position 60 (expending 50 liters of
 * fuel),
 * and refuel from 10 liters to 50 liters of gas.  We then drive to and reach
 * the target.
 * We made 2 refueling stops along the way, so we return 2.
 * 
 * 
 * 
 * 
 * Note:
 * 
 * 
 * 1 <= target, startFuel, stations[i][1] <= 10^9
 * 0 <= stations.length <= 500
 * 0 < stations[0][0] < stations[1][0] < ... < stations[stations.length-1][0] <
 * target
 * 
 * 
 * 
 * 
 * 
 */

// @lc code=start
class Solution {
 public:
  int minRefuelStops(int target, int start_fuel, vector<vector<int>>& stations) {
    if (target <= start_fuel) {
      return 0;
    }
    array<int64_t, 501> stops_to_reachable;
    stops_to_reachable[0] = start_fuel;
    int size = stations.size();
    int curr_stops = 0;
    int min_stops = INT_MAX;
    for (int next = 0; next != size; ++next) {
      auto next_position = stations[next][0];
      auto next_enlength = stations[next][1];
      if (stops_to_reachable[curr_stops] < next_position) {
        // cannot reach stations[next]
        break;
      }
      auto next_reachable = stops_to_reachable[curr_stops] + next_enlength;
      stops_to_reachable[++curr_stops] = next_reachable;
      if (target <= next_reachable) {
        min_stops = min(min_stops, curr_stops);
      }
      int less_stops = curr_stops-1;
      while (less_stops > 0) {
        if (next_position <= stops_to_reachable[less_stops-1]) {
          // stops_to_reachable[less_stops] might be improved by stations[next]
          auto reachable = stops_to_reachable[less_stops-1] + next_enlength;
          stops_to_reachable[less_stops] = max(stops_to_reachable[less_stops], reachable);
          if (target <= reachable) {
            min_stops = min(min_stops, less_stops);
          }
        }
        else {
          assert(stops_to_reachable[less_stops-1] < next_position);
          break;
        }
        --less_stops;
      }
      if (min_stops == 1 || min_stops == less_stops+1) {
        break;
      }
    }
    return min_stops == INT_MAX ? -1 : min_stops;
  }
};
// @lc code=end
