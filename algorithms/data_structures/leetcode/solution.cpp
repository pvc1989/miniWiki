#include <algorithm>
#include <array>
#include <cassert>
#include <climits>
#include <cstdio>
#include <deque>
#include <forward_list>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <new>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace std;

#include "30.substring-with-concatenation-of-all-words.cpp"

int main(int argc, char* argv[]) {
  auto solution = Solution();
  auto s = string("barfoofoobarthefoobarman");
  auto words = vector<string>{"bar","foo","foo","the"};
  cout << solution.findSubstring(s, words).size() << endl;
}
