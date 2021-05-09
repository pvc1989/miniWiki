/*
 * @lc app=leetcode id=906 lang=cpp
 *
 * [906] Super Palindromes
 *
 * https://leetcode.com/problems/super-palindromes/description/
 *
 * algorithms
 * Hard (32.74%)
 * Likes:    141
 * Dislikes: 218
 * Total Accepted:    8.7K
 * Total Submissions: 25.2K
 * Testcase Example:  '"4"\n"1000"'
 *
 * Let's say a positive integer is a super-palindrome if it is a palindrome,
 * and it is also the square of a palindrome.
 * 
 * Given two positive integers left and right represented as strings, return
 * the number of super-palindromes integers in the inclusive range [left,
 * right].
 * 
 * 
 * Example 1:
 * 
 * 
 * Input: left = "4", right = "1000"
 * Output: 4
 * Explanation: 4, 9, 121, and 484 are superpalindromes.
 * Note that 676 is not a superpalindrome: 26 * 26 = 676, but 26 is not a
 * palindrome.
 * 
 * 
 * Example 2:
 * 
 * 
 * Input: left = "1", right = "2"
 * Output: 1
 * 
 * 
 * 
 * Constraints:
 * 
 * 
 * 1 <= left.length, right.length <= 18
 * left and right consist of only digits.
 * left and right cannot have leading zeros.
 * left and right represent integers in the range [1, 10^18].
 * left is less than or equal to right.
 * 
 * 
 */

// @lc code=start
class Solution {
  const vector<int64_t> super_palindromes_{
      0, 1, 4, 9, 121, 484, 10201, 12321, 14641, 40804, 44944, 1002001, 1234321, 4008004, 100020001, 102030201, 104060401, 121242121, 123454321, 125686521, 400080004, 404090404, 10000200001, 10221412201, 12102420121, 12345654321, 40000800004, 1000002000001, 1002003002001, 1004006004001, 1020304030201, 1022325232201, 1024348434201, 1210024200121, 1212225222121, 1214428244121, 1232346432321, 1234567654321, 4000008000004, 4004009004004, 100000020000001, 100220141022001, 102012040210201, 102234363432201, 121000242000121, 121242363242121, 123212464212321, 123456787654321, 400000080000004, 10000000200000001, 10002000300020001, 10004000600040001, 10020210401202001, 10022212521222001, 10024214841242001, 10201020402010201, 10203040504030201, 10205060806050201, 10221432623412201, 10223454745432201, 12100002420000121, 12102202520220121, 12104402820440121, 12122232623222121, 12124434743442121, 12321024642012321, 12323244744232321, 12343456865434321, 12345678987654321, 40000000800000004, 40004000900040004
  };
  static vector<int64_t> build() {
    auto super_palindromes = vector<int64_t>();
    for (int64_t x = 0; x <= int64_t(1e9); ++x) {
      auto x_sq = x * x;
      if (isPalindrome(to_string(x)) && isPalindrome(to_string(x_sq)))
        super_palindromes.emplace_back(x_sq);
    }
    return super_palindromes;
  }
  
  unordered_set<int64_t> palindromes_;
  static bool isPalindrome(string const& x) {
    auto rx = x.rbegin();
    for (int i = x.size() / 2; i >= 0; --i) {
      if (x[i] != rx[i])
        return false;
    }
    return true;
  }
  bool isPalindrome(int64_t x) {
    auto iter = palindromes_.find(x);
    if (iter != palindromes_.end())
      return true;
    else if (isPalindrome(to_string(x))) {
      palindromes_.emplace(x);
      return true;
    }
    else
      return false;
  }

  int searchSuperPalindromesInRange(string left, string right) {
    int count = 0;
    int64_t left_ll = stoll(left), right_ll = stoll(right);
    int64_t left_sqrt = sqrt(left_ll), right_sqrt = sqrt(right_ll);
    while (left_sqrt * left_sqrt >= left_ll)
      left_sqrt--;
    left_sqrt++;
    while (right_sqrt * right_sqrt <= right_ll)
      right_sqrt++;
    right_sqrt--;
    printf("%lld <= x <= %lld\n", left_sqrt, right_sqrt);
    for (auto x = left_sqrt; x <= right_sqrt; ++x) {
      auto x_sq = x * x;
      assert(left_ll <= x_sq && x_sq <= right_ll);
      if (isPalindrome(x) && isPalindrome(x_sq)) {
        count++;
        printf("%lld ^ 2 == %lld\n", x, x_sq);
      }
    }
    return count;
  }

 public:
  int superpalindromesInRange(string left, string right) {
    auto head = super_palindromes_.begin();
    auto tail = super_palindromes_.end();
    return upper_bound(head, tail, stoll(right))
         - lower_bound(head, tail, stoll(left));
  }
};
// @lc code=end

