#include <vector>
#include <string>
#include <iostream>
#include <cassert>

using namespace std;

class Solution {
  static bool OneCharMatch(char p, char t) {
    return p != '*' && (p == '.' || p == t);
  }
 public:
  bool isMatch(string text, string pattern) {
    // `dp_table[i][j] == true` means
    // `pattern[i, pattern.size())` covers `text[j, text.size())`
    auto dp_table = vector<vector<bool>>(pattern.size()+1);
    for (auto& row : dp_table) { row.resize(text.size()+1, false); }
    dp_table.back().back() = true;  // empty pattern only covers empty text
    // build the dp table (bottom-up)
    for (int i = pattern.size()-1; i != -1; --i) {
      for (int j = text.size(); j != -1; --j) {
        if (pattern[i] == '*') {
          dp_table[i][j] = false;
        } else if (j == text.size()) {
          dp_table[i][j] = (i+2 <= pattern.size() && dp_table[i+2][j]
              && pattern[i+1] == '*'/* `` can only be covered by `[a-z]*` */);
        } else if (OneCharMatch(pattern[i], text[j])) {
          assert(j+1 <= text.size());
          if (i+1 == pattern.size()) {
            dp_table[i][j] = (j+1 == text.size());
          } else if (pattern[i+1] != '*') {
            dp_table[i][j] = dp_table[i+1][j+1];
          } else {
            assert(i+2 <= pattern.size());
            dp_table[i][j] = dp_table[i+2][j] || dp_table[i][j+1];
          }
        } else if (i+1 == pattern.size()) {
          assert(false == dp_table[i][j]);
        } else {
          assert(i+2 <= pattern.size());
          dp_table[i][j] = pattern[i+1] == '*' && dp_table[i+2][j];
        }
      }
    }
    // `dp_table[0][0] == true` means `pattern` covers `text`
    return dp_table[0][0];
  }
};

int main(int argc, char* argv[]) {
  auto sol = Solution();
  assert(argc == 3);
  auto text = string(argv[1]);
  auto pattern = string(argv[2]);
  cout << '`' << text << "` is";
  cout << (sol.isMatch(text, pattern) ? " " : " NOT ");
  cout << "covered by `" << pattern << '`' << endl;
}
