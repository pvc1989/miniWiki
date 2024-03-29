---
title: Vim
---

Vi 是所有 Linux 系统都自带的命令行界面文本编辑器，而 [Vim](https://www.vim.org/) (**V**i **IM**proved) 则是它的增强版。

# 基本功能

本节列举 Vim 的基本功能，所有内容同时适用于 Vi 和 Vim。

## 三种模式

Vim 有三种模式：

| 类型 | 功能 |
| ---- | ---- |
| COMMAND 模式 (默认) | 移动光标，删除或复制/粘贴字符或整行字符串 |
| COMMAND-LINE 模式 | 搜索/替换字符串，读取/保存文件 |
| INSERT 模式 | 移动光标，输入/删除字符 |

这几种模式通过以下方式来切换：

| From | To | By |
| ---- | -- | -- |
| COMMAND | COMMAND-LINE | 输入 `:/?` 中的任意一个字符 |
| COMMAND | INSERT | 输入 `ioar` 或 `IOAR` 中的任意一个字符 |
| INSERT | COMMAND | 按下 `[Esc]` 键 |
| COMMAND-LINE | COMMAND | 按下 `[Esc]` 键 |

## 典型步骤

### 打开

在 Shell 中输入

```shell
vim name
```

以新建或打开名为 `name` 的文件。也可以不带文件名，直接进入 Vim。

### 编辑

文件打开后，默认进入 COMMAND 模式。按下 `IOARioar` 中的任意一个字母，进入 INSERT 模式。此时，可以像在其他文本编辑器中一样，对文件进行编辑。

### 保存

编辑过程中或完成编辑后，如果需要保存当前修改，则按下 `[Esc]` 键回到 COMMAND 模式。然后再按下 `:` 进入 COMMAND-LINE 模式，在 `:` 后输入 `w + [Enter]`，

- 若有写入权限，则修改内容将被写入 (**w**rite) 文件。
- 若没有写入权限，则报错。

### 退出

完成编辑后，按下 `[Esc]` 键回到 COMMAND 模式。然后再按下 `:` 进入 COMMAND-LINE 模式，在 `:` 后输入 `q + [Enter]`，

- 若文件内容没有被修改，则直接退出 (**q**uit)。
- 若文件内容被修改过且没有被写入，则报错。

# 常用命令

## COMMAND 模式

### 移动光标

| 命令 | 功能 |
| ---- | ---- |
| `H` 或 `8H` | 向左移动 1 或 8 个字符 |
| `J` 或 `8J` | 向下移动 1 或 8 个字符 |
| `J` 或 `8K` | 向上移动 1 或 8 个字符 |
| `L` 或 `8L` | 向右移动 1 或 8 个字符 |
| `8 + [Space]` | 向右移动 8 个字符 |
| `8 + [Enter]` | 向下移动 8 行 |
| `[Ctrl] + F` | 向下移动一页，相当于 `[PageDown]` |
| `[Ctrl] + B` | 向上移动一页，相当于 `[PageUp]` |
| `0` | 跳到光标所在行首，相当于 `[Home]` |
| `$` | 跳到光标所在行尾，相当于 `[End]` |
| `H` | 跳到当前页面的第一行 |
| `L` | 跳到当前页面的最后一行 |
| `G` | 跳到当前文件的最后一行 |
| `8G` | 跳到当前文件的第 8 行 |
| `gg` | 跳到当前文件的第一行，相当于 `1G` |

### 搜索

| 命令 | 功能 |
| ---- | ---- |
| `/word` | 向后查找 `word` |
| `?word` | 向前查找 `word` |
| `n` | 重复前一次查找 |
| `N` |  反向进行前一次查找 |

### 替换

| 命令 | 功能 |
| ---- | ---- |
| `:1,8s/old/new/g` | 在 `[1,8]` 行之间，将 `old` 替换为 `new` |
| `:1,$s/old/new/g` | 在全文中，将 `old` 替换为 `new` |
| `:1,$s/old/new/gc` | 同上，替换前要求确认 |

### 删除/复制

删除光标所在行内的字符，其中 `c` 表示光标所在的列（从 1 开始）：

| 命令 | 功能 |
| ---- | ---- |
| `x` | 删除 `c`，相当于 `[Del]` |
| `8x` | 删除 `[c, c+8)`，相当于连按 8 次 `[Del]` |
| `X` | 删除 `c-1`，相当于 `[Backspace]` |
| `8X` | 删除 `[c-8, c)`，相当于连按 8 次 `[Backspace]` |
| `d0` | 删除 `[first, c)`，`first` 表示光标所在行的第 1 个字符 |
| `d$` | 删除 `[c, last)`，`last` 表示换行符 |

删除整行，其中 `r` 表示表示光标所在的行（从 1 开始）：

| 命令 | 功能 |
| ---- | ---- |
| `dd` | 删除 `r` |
| `8dd` | 删除 `[r, r+8)` |
| `dG` | 删除 `[r, last]` |
| `d1G` | 删除 `[first, r]` |
| `d8G` | 删除 `[min(r,8), max(r,8)]` |

以上各删除命令中的 `d` 替换为 `y` 则为对应的复制命令。

### 粘贴

假设已经复制了 8 个字符或行，`c` 表示当前光标所在字符或行。

| 命令 | 功能 |
| ---- | ---- |
| `p` | 粘贴在 `[c+1, c+8]`，原 `c+1` 顺延到 `c+9` |
| `P` | 粘贴在 `[c, c+8)`，原 `c` 顺延到 `c+8` |
| `J` | 将光标所在行尾的换行符替换为空格 |

### 撤销/重做

| 命令 | 功能 |
| ---- | ---- |
| `U` | 撤销前一个动作 |
| `[Ctrl] + R` | 重做被撤销的动作 |
| `.` | 重做前一个动作 |

## COMMAND-LINE 模式

| 命令 | 功能 |
| ---- | ---- |
| `:w` | **w**rite |
| `:w!` | 强行 **w**rite |
| `:q` | **q**uit |
| `:q!` | 强行 **q**uit |
| `:wq` | 先 **w**rite 再 **q**uit |
| `ZZ` | 若没有修改过，则相当于 `:q`，否则相当于 `:wq` |
| `:w name` | 将当前数据写入名为 `name` 的文件，类似于"另存为" |
| `:r name` | 从名为 `name` 的文件读取数据，插入到光标所在行的下一行 |
| `:1,8 w name` | 将 `[1,8]` 行的内容写入名为 `name` 的文件 |
| `:! cmd` | 暂时离开 Vim，执行名为 `cmd` 的 Shell 命令 |

## INSERT 模式

从 COMMAND 模式进入 INSERT 模式有以下 4 种方式：

| 命令 | 功能 |
| ---- | ---- |
| `i` | 从光标所在字符前开始插入 |
| `I` | 从光标所在行的第一个非空白字符前开始插入 |
| `a` | 从光标所在字符后开始插入 |
| `A` | 从光标所在行的最后一个字符后开始插入 |
| `o` | 在光标所在行后面插入新的一行 |
| `O` | 在光标所在行前面插入新的一行 |
| `r` | 将光标所在字符替换为随后输入的一个字符，然后回到 COMMAND 模式 |
| `R` | 将光标所在字符逐个替换为随后输入的字符，直到按下 `[Esc]` 回到 COMMAND 模式 |

- 通过 `iIaAoO` 进入 INSERT 模式后，左下角会出现 `-- INSERT --`。
- 通过 `R` 进入 *R*eplace  模式后，左下角会出现 `-- REPLACE --`。

# 设置选项

## 通过命令设置

如果已经进入 Vim，并且需要临时修改 Vim 设置，可以利用 `:set` 命令。这种方式只在当前 Vim 进程中有效。例如：

| 命令 | 功能 |
| ---- | ---- |
| `:set all` | 查看当前设置参数 |
| `:set` | 显示非默认参数 |
| `:set number` | 左侧显示行号 |
| `:set hlsearch` | 高亮搜索 |
| `:set autoindent` | 自动缩进 |
| `:set backup` | 自动备份 |
| `:set ruler` | 右下角显示光标位置 |
| `:set showmode` | 左下角显示模式信息 |
| `:set backspace=0/1/2` | `2` 表示可以删除任意字符，否则只能删除最近输入的字符 |
| `:set bg=dark/light` | 暗色/亮色背景 |
| `:syntax on/off` | 开启/关闭语法高亮 |

完整列表可以在 Vim 中通过以下命令查到：

```shell
:help option-list
```

## 通过文件设置

Vim 的设置文件包括：

- 当前主机设置文件 `/etc/vimrc`。
- 当前用户设置文件 `~/.vimrc`，优先级高于前者。

具体语法与 COMMAND-LINE 模式中的命令一样。每一行代表一条命令，并且行首的 `:` 可以省略，行尾可以用双引号 `"` 开启注释。

# 高级功能

## 区块选择

如果有以下文件：

```
192.168.1.1  host1.school.edu
192.168.1.2  host2.school.edu
192.168.1.3  host3.school.edu
```

现在要将 `host1`，`host2`，`host3` 复制并粘贴到各行最后，即改为以下形式：

```
192.168.1.1  host1.school.edu  host1
192.168.1.2  host2.school.edu  host2
192.168.1.3  host3.school.edu  host3
```

这种需求在 Vim 中可以很容易地实现。在 COMMAND 模式下：

| 命令 | 功能 |
| ---- | ---- |
| `v` | 选择字符，光标扫过的字符反色 |
| `V` | 选择整行，光标扫过的整行反色 |
| `[Ctrl] + v` | 选择区块，光标扫过的矩形区块反色 |
| `y` | cop**y** 选中的反色区域 |
| `d` | **d**elete 选中的反色区域 |
| `p` | **p**aste 选中的反色区域 |

## 多文件编辑

在 Shell 中输入以下命令，可以打开多个文件：

```shell
vim file1 file2
```

用以下命令可以查看或切换 Vim 打开的文件：

| 命令 | 功能 |
| ---- | ---- |
| `:files` | 列出当前 Vim 打开的所有文件 |
| `:n` | 编辑下一个文件 |
| `:N` | 编辑上一个文件 |

典型的操作是：从 `file1.txt` 中复制一些内容，然后切换到 `file2.txt`，将复制的内容粘贴到 `file2.txt` 中。

`:q` 或 `:wq` 命令的作用是退出当前 Vim 进程。因此，如果 Vim 打开了多个文件，那么所有文件都将被退出。

## 多窗口编辑

前面的**多文件模式**只会显示一个文件的内容，而这里的**多窗口模式**则会将窗口分割 (**sp**lit) 为多块，用以同时显示一个或多个文件的内容。

| 命令  | 功能                       |
| ----- | -------------------------- |
| `:sp` | 在分出的窗口中打开当前文件 |
| `:sp filename` | 在分出的窗口中新建或打开指定文件 |

先按住 `[Ctrl]`，再按住 `W`，再按第三个键，可以在不同窗口间切换：

| 组合键  | 功能                       |
| ----- | -------------------------- |
| `[Ctrl] + W + J` | 光标移到下一个窗口 |
| `[Ctrl] + W + K` | 光标移到上一个窗口 |
| `[Ctrl] + W + Q` | 关闭当前窗口 |

> 以上组合键可能会与其他快捷键冲突。

## 自动补全

### 一般搜索

在 INSERT 模式中，按以下组合键执行搜索：

| 组合键  | 补全依据 |
| ------ | ---- |
| `[Ctrl] + P` | 向前 (**P**revious) 搜索 |
| `[Ctrl] + N` | 向后 (**N**ext) 搜索 |

如果只有一个匹配项，则自动执行补全; 如果有多个匹配项，则弹出候选列表，可以用方向键进行选择。

### 设置大小写模式

| 选项 | 功能 |
| ------ | ---- |
| `:set ignorecase` | 搜索时忽略大小写 |
| `:set infercase` | 搜索时忽略大小写，提示时推测大小写 |

### 设置搜索范围

Vim 根据以下设置来确定关键词搜索范围：

```shell
:set complete=key,key,key
```

其中 `key` 可以为下表中的一个或多个(相互之间用 `,` 隔开)选项：

| 选项 | 功能 |
| ------ | ---- |
| `.` | 当前文件 |
| `d` | 当前文件及被其`#include`的文件 |
| `i` | 被当前文件`#include`的文件 |
| `b` | 被载入 **b**uffer 中的文件 |
| `u` | 未被载入 buffer 的文件 |
| `kfile` | 名为`file`的文件 |
| `k` | 被`dictionary`选项定义的文件 |
| `t` | `tags`文件 |
| `w` | 其他窗口中的文件 |

其中`#include`指定的搜索路径由 Vim 的`path`选项决定。

### 设置字典文件

如果有一些常用词需要经常输入，可以定义一组字典文件` /path/math.txt`，`/path/physics.txt`，这样`[Ctrl] + P/N`会在指定的字典文件中进行搜索。字典文件可以通过如下方式设定：

```shell
:set dictionary=/path/math.txt,/path/physics.txt
:set complete=k/path/math.txt,k/path/physics.txt
```

### 精细搜索

若要进行更为精细的搜索，则需要先按下组合键 `[Ctrl] + X`，然后再按下下表中任意一个组合键：

| 组合键  | 搜索对象 |
| ------ | -------- |
| `[Ctrl] + D` | 当前文件及被其 `#include` 的文件中由 `#define` 定义的宏 |
| `[Ctrl] + F` | 当前目录中的文件名 |
| `[Ctrl] + K` | 字典文件中的词 |
| `[Ctrl] + I` | 当前文件及被其 `#include` 的文件中的词 |
| `[Ctrl] + L` | 整行 |

在 `[Ctrl] + X` 模式下，`[Ctrl] + P/N` 的功能类似于方向键。

### 搜索标签

一个**标签 (tag)** 代表一个 C 函数 **原型 (prototype)**。利用程序 ctags 可以生成一个标签列表，并保存到标签文件 `tags` 中。例如：

```shell
ctags *.c *.h
```

默认情况下，Vim 只显示函数名称。通过以下设置：

```shell
:set showfulltag
```

可以改为显示完整的函数原型。
