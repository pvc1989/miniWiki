---
title: 文件读写
---

# IO Streams

## `getline()`

假设有如下文本文件：

```txt
q4 365  34   427    14
t3, 1 ,   45     , 65
t3, 101 ,   45     , 6
q4     13    23   54  94
```

需要逐行读取该文件，并将分隔符统一为单个空格，即获得以下输出：

```txt
q4 365 34 427 14
t3 1 45 65
t3 101 45 6
q4 13 23 54 94
```

以下代码可实现该需求：

```cpp
/*
build:
  g++ -std=c++14 -O2 -o parse parse.cpp
run:
  ./parse input.txt
 */
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
 
int main(int argc, char** argv) {
  if (argc < 2) {
    return -1;
  }
  auto filename = std::string(argv[1]);
  auto filestrm = std::ifstream(filename);
  auto line = std::string();
  while (std::getline(filestrm, line)) {
    auto linestrm = std::istringstream(line);
    std::string type; int a, b, c, d;
    if (line[0] == 'q') {
      // split by space
      linestrm >> type >> a >> b >> c >> d;
      std::cout << type << ' ' << a << ' ' << b << ' ' << c << ' ' << d << '\n';
    } else if (line[0] == 't') {
      // split by comma
      std::getline(linestrm, type, ',');
      std::string str_buf;
      std::getline(linestrm, str_buf, ',');
      a = std::stoi(str_buf);
      char raw_str_buf[1024];
      linestrm.getline(raw_str_buf, sizeof(raw_str_buf), ',');
      b = std::stoi(raw_str_buf);
      std::getline(linestrm, str_buf, ',');
      c = std::stoi(str_buf);
      std::cout << type << ' ' << a << ' ' << b << ' ' << c << '\n';
    }
  }
}
```

# C-style IO
