/*
 * @lc app=leetcode id=150 lang=cpp
 *
 * [150] Evaluate Reverse Polish Notation
 *
 * https://leetcode.com/problems/evaluate-reverse-polish-notation/description/
 *
 * algorithms
 * Medium (37.91%)
 * Likes:    1453
 * Dislikes: 512
 * Total Accepted:    270.4K
 * Total Submissions: 712.8K
 * Testcase Example:  '["2","1","+","3","*"]'
 *
 * Evaluate the value of an arithmetic expression in Reverse Polish Notation.
 * 
 * Valid operators are +, -, *, /. Each operand may be an integer or another
 * expression.
 * 
 * Note:
 * 
 * 
 * Division between two integers should truncate toward zero.
 * The given RPN expression is always valid. That means the expression would
 * always evaluate to a result and there won't be any divide by zero
 * operation.
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: ["2", "1", "+", "3", "*"]
 * Output: 9
 * Explanation: ((2 + 1) * 3) = 9
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: ["4", "13", "5", "/", "+"]
 * Output: 6
 * Explanation: (4 + (13 / 5)) = 6
 * 
 * 
 * Example 3:
 * 
 * 
 * Input: ["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]
 * Output: 22
 * Explanation: 
 * ⁠ ((10 * (6 / ((9 + 3) * -11))) + 17) + 5
 * = ((10 * (6 / (12 * -11))) + 17) + 5
 * = ((10 * (6 / -132)) + 17) + 5
 * = ((10 * 0) + 17) + 5
 * = (0 + 17) + 5
 * = 17 + 5
 * = 22
 * 
 * 
 */

// @lc code=start
class Solution {
 public:
  int evalRPN(vector<string>& tokens) {
    auto operands = stack<int>();
    for (auto& str : tokens) {
      auto c = str[0];
      if ('0' <= c && c <= '9' || c == '-' && str.size() > 1) {
        operands.push(stoi(str));
      } else {
        assert(operands.size() > 1);
        auto rhs = operands.top(); operands.pop();
        auto lhs = operands.top(); operands.pop();
        switch (c) {
        case '+':
          operands.push(lhs + rhs);
          break;
        case '-':
          operands.push(lhs - rhs);
          break;
        case '*':
          operands.push(lhs * rhs);
          break;
        case '/':
          operands.push(lhs / rhs);
          break;
        default:
          assert(false);
          break;
        }
      }
    }
    assert(operands.size() == 1);
    return operands.top();
  }
};
// @lc code=end

