---
title: Linux
---

# [安装](./install/README.md)

# 文本编辑器

## [Vim](./vim.md)

Vim 是 Linux 系统下常用的一款 CLI 文本编辑器。
打开终端 (`[Ctrl] + [Alt] + T`) 并尝试以下命令：

```shell
vim --version
```

如果提示没有安装，可以通过以下命令下载并安装：

```shell
sudo apt install vim
```

其中 `apt` 是 Debian/Ubuntu 下的软件包管理命令。
在 CentOS/Fedora 下用 `yum` 替换。

# Shell

## `sh`

`sh` is a POSIX-compliant command interpreter (shell).
It is implemented by re-execing as either `bash`, `dash`, or `zsh` as determined by the symbolic link located at `/private/var/select/sh`.
If `/private/var/select/sh` does not exist or does not point to a valid shell, `sh` will use one of the supported shells.

## `bash`

### 启动方式

根据启动方式，可以将 shell 分为

- **login shell**：启动时*需要*输入用户名、密码（如：登入远程主机），并依次运行
  - 全局脚本 `/etc/profile`
  - 当前用户脚本 `~/.bash_profile`
- **non-login shell**：启动时*无需*输入用户名、密码（如：在 GUI 中打开命令行终端），从主进程继承环境变量，并依次运行
  - 全局脚本 `/etc/bash.bashrc`
  - 当前用户脚本 `~/.bashrc`

### 样式微调

Ubuntu 默认的终端（`/bin/bash`）提示格式为

```
username@hostname:~$ 
```

其中 `username@hostname` 是同一种颜色。
为了更方便地区分 `username` 与 `hostname`，可以在终端设置文件 (如 `~/.bashrc`) 中修改颜色及格式设置。

在所有 Linux 发行版下，都可以用 Vim 来修改：

```shell
vim ~/.bashrc
```

对于不熟悉 Vim 的新手，可以用任何一款 GUI 文本编辑器进行修改。
例如在 Ubuntu 中，可以用 [gedit](http://www.gedit.org) 代替 Vim：

```shell
gedit ~/.bashrc
```

找到以下内容
```shell
if [ "$color_prompt" = yes ]; then
    PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
else
    PS1='${debian_chroot:+($debian_chroot)}\u@\h:\w\$ '
fi
```

将其修改为

```shell
if [ "$color_prompt" = yes ]; then
    PS1='\[\033[34m\]\u\[\033[00m\]@\[\033[31m\]\h\[\033[00m\]:\[\033[32m\]\w\[\033[00m\]\n\$ '
else
    PS1='${debian_chroot:+($debian_chroot)}\u@\h:\w\$ '
fi
```

在 `~/.bashrc` 文件中，另一个建议修改的地方是：

```shell
# -a 选项会显示隐藏文件的信息，日常使用中是多余的：
alias ll='ls -alF'
# 建议修改为：
alias ll='ls -lF'
```

保存并退出，然后用 `source` 命令使其生效 (或重启终端)：

```shell
source ~/.bashrc
```

## `zsh`

安装及配置：

```shell
sudo apt install zsh
# 下载好看的样式：
sh -c "$(wget https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O -)"
# 或手动下载 install.sh，再执行该脚本：
mv install.sh ~/
chmod 755 install.sh
sh -c ./install.sh
# 修改配置：
vim ~/.zshrc
```

设为默认 shell：

```shell
echo $SHELL           # 查看 login shell
chsh -s $(which zsh)  # 更换 login shell
# 退出当前 shell 并重新打开
```

## [`ssh`](./ssh.md)

## 文件系统操作

### `locate`

*Locate* files by name.

### `find`

*Find* files in a directory hierarchy.

```shell
# Find files by extension:
    find root_path -name '*.ext'

# Find files matching multiple path/name patterns:
    find root_path -path '**/path/**/*.ext' -or -name '*pattern*'

# Find directories matching a given name, in case-insensitive mode:
    find root_path -type d -iname '*lib*'

# Find files matching a given pattern, excluding specific paths:
    find root_path -name '*.py' -not -path '*/site-packages/*'

# Find files matching a given size range:
    find root_path -size +500k -size -10M

# Run a command for each file (use `{}` within the command to access the filename):
    find root_path -name '*.ext' -exec wc -l {} \;

# Find files modified in the last 7 days and delete them:
    find root_path -daystart -mtime -7 -delete

# Find empty (0 byte) files and delete them:
    find root_path -type f -empty -delete
```

### `touch`

Change file timestamps, i.e. *touch* a file.

### `stat`

- GNU: Display file or file system *stat*us.
- BSD: Display file *stat*us.

```shell
# Show file properties such as size, permissions, creation and access dates among others:
    stat file

# Same as above but verbose (more similar to Linux's `stat`):
    stat -x file

# Show only octal file permissions:
    stat -f %Mp%Lp file

# Show owner and group of the file:
    stat -f "%Su %Sg" file

# Show the size of the file in bytes:
    stat -f "%z %N" file
```

### `du`

Show *d*isk *u*sage.

```shell
# Show disk usage of current folder:
du -sh
# Show disk usage of Videos:
du -sh Videos/
# Show disk usage of level-1 subfolders in current folder:
du -h --max-depth=1
# Show disk usage of level-1 subfolders in Pictures:
du -h --max-depth=1 Pictures/
```

## 数据流操作

### `cat`

Con*cat*enate files and print on the standard output.

### `head`

Output the *head* (i.e. the first part) of files.

### `tail`

Output the *tail* (i.e. the last part) of files.

### `sort`

*Sort* lines of text files.

### `uniq`

Report or omit repeated lines.

### `sed`

*S*tream *ed*itor for filtering and transforming text.

```shell
# Replace the first occurrence of a string in a file, and print the result:
    sed 's/find/replace/' filename
# Replace all occurrences of an extended regular expression in a file:
    sed -E 's/regular_expression/replace/g' filename
# Replace all occurrences of a string [i]n a file, overwriting the file (i.e. in-place):
    sed -i '' 's/find/replace/g' filename
# Replace only on lines matching the line pattern:
    sed '/line_pattern/s/find/replace/' filename
# Print only text between n-th line till the next empty line:
    sed -n 'line_number,/^$/p' filename
# Apply multiple find-replace expressions to a file:
    sed -e 's/find/replace/' -e 's/find/replace/' filename
# Replace separator `/` by any other character not used in the find or replace patterns, e.g. `#`:
    sed 's#find#replace#' filename
# [d]elete the line at the specific line number [i]n a file, overwriting the file:
    sed -i '' 'line_numberd' filename
```

### `awk`

A versatile programming language for working on files.

```shell
# Print the fifth column (a.k.a. field) in a space-separated file:
    awk '{print $5}' filename
# Print the second column of the lines containing "foo" in a space-separated file:
    awk '/foo/ {print $2}' filename
# Print the last column of each line in a file, using a comma (instead of space) as a field separator:
    awk -F ',' '{print $NF}' filename
# Sum the values in the first column of a file and print the total:
    awk '{s+=$1} END {print s}' filename
# Print every third line starting from the first line:
    awk 'NR%3==1' filename
# Print different values based on conditions:
    awk '{if ($1 == "foo") print "Exact match foo"; else if ($1 ~ "bar") print "Partial match bar"; else print "Baz"}' filename
# Print all lines where the 10th column value equals the specified value :
    awk '($10 == value)'
# Print all the lines which the 10th column value is between a min and a max :
    awk '($10 >= min_value && $10 <= max_value)'
```

### `xargs`

E*x*ecute a command with piped *arg*ument*s* coming from another command, a file, etc. 
The input is treated as a single block of text and split into separate pieces on spaces, tabs, newlines and end-of-file.

```shell
# Run a command using the input data as arguments:
    arguments_source | xargs command
# Run multiple chained commands on the input data:
    arguments_source | xargs sh -c "command1 && command2 | command3"
# Delete all files with a `.backup` extension (`-print0` uses a null character to split file names, and `-0` uses it as delimiter):
    find . -name '*.backup' -print0 | xargs -0 rm -v
# Execute the command once for each input line, replacing any occurrences of the placeholder (here marked as `_`) with the input line:
    arguments_source | xargs -I _ command _ optional_extra_arguments
# Parallel runs of up to `max-procs` processes at a time; the default is 1. If `max-procs` is 0, xargs will run as many processes as possible at a time:
    arguments_source | xargs -P max-procs command
```

### `grep`

Print lines that match patterns.

```shell
# Search for a pattern within a file:
    grep "search_pattern" path/to/file
# Search for an exact string (disables regular expressions):
    grep --fixed-strings "exact_string" path/to/file
# Search for a pattern in all files recursively in a directory, showing line numbers of matches, ignoring binary files:
    grep --recursive --line-number --binary-files=without-match "search_pattern" path/to/directory
# Use extended regular expressions (supports `?`, `+`, `{}`, `()` and `|`), in case-insensitive mode:
    grep --extended-regexp --ignore-case "search_pattern" path/to/file
# Print 3 lines of context around, before, or after each match:
    grep --context|before-context|after-context=3 "search_pattern" path/to/file
# Print file name and line number for each match:
    grep --with-filename --line-number "search_pattern" path/to/file
# Search for lines matching a pattern, printing only the matched text:
    grep --only-matching "search_pattern" path/to/file
# Search stdin for lines that do not match a pattern:
    cat path/to/file | grep --invert-match "search_pattern"
```

### `wc`

Print newline, *w*ord, and byte *c*ounts for each file.

```shell
# Count [l]ines in file:
    wc -l file
# Count [w]ords in file:
    wc -w file
# Count [c]haracters (bytes) in file:
    wc -c file
# Count characters in file (taking [m]ulti-byte character sets into account):
    wc -m file
# Use standard input to count lines, words and characters (bytes) in that order:
    find . | wc
```

### `tee` 

Read from standard input and write to standard output and files.

```shell
# Copy standard input to each file, and also to standard output:
    echo "example" | tee path/to/file
# Append to the given files, do not overwrite:
    echo "example" | tee -a path/to/file
# Print standard input to the terminal, and also pipe it into another program for further processing:
    echo "example" | tee /dev/tty | xargs printf "[%s]"
# Create a directory called "example", count the number of characters in "example" and write "example" to the terminal:
    echo "example" | tee >(xargs mkdir) >(wc -c)
```

### `>`, `>>`

|   名称   |     覆盖      |      追加      |
| :------: | :-----------: | :------------: |
| `stdout` | `1> filename` | `1>> filename` |
| `stderr` | `2> filename` | `2>> filename` |

```shell
# 正常信息、错误信息 均输出到 屏幕
$ find ~/.. -name .bash_history
# 正常信息 输出到 stdout.txt 文件，错误信息 输出到 屏幕
$ find ~/.. -name .bash_history > stdout.txt
# 正常信息 输出到 stdout.txt 文件，错误信息 输出到 stderr.txt 文件
$ find ~/.. -name .bash_history > stdout.txt 2> stderr.txt
# 正常信息、错误信息 均输出到 stdout_stderr.txt 文件
$ find ~/.. -name .bash_history 2>&1 stdout_stderr.txt
```

# 系统管理

## 设备管理

```shell
# 查看 CPU 信息
cat /proc/cpuinfo | grep "model name"
# 查看 RAM 信息
cat /proc/meminfo | grep Mem
```

## 权限管理

### `id` - display user *id*entity

### `chmod` - *ch*ange a file's *mod*e

### `umask` - set file mode creation *mask*

### `su` - run a command with *s*ubstitute *u*ser

### `sudo` - execute a command as another user

### `chown` - *ch*ange a file's *own*er

### `chgrp` - *ch*ange a file's *gr*ou*p*

### `passwd` - change user *passw*or*d*


## 进程管理

### 基本概念
本节的**程序 (program)** 特指（存储在磁盘中的）可执行文件，而**进程 (process)** 则是（被操作系统加载到内存中的）某个程序的运行实例。
操作系统在加载一个程序使其成为一个进程时，会为其分配一个 **Process ID (PID)** 并附上进程触发者的 **User ID (UID)** 及 **Group ID (GID)**。

进程之间可能存在依赖关系，即一个进程由另一个进程触发。
- 被依赖者（即触发者）被称为**亲进程 (parent process)**。
- 依赖者（即被触发者）被称为**子进程 (child process)**。
- 子进程将亲进程的 UID、GID 继承下来，并以*亲进程的 PID* 作为自己的 **Parent Process ID (PPID)**。

### 查看进程

|          命令           |                     功能                     |
| :---------------------: | :------------------------------------------: |
|         `ps -l`         |            只显示与自己相关的进程            |
|        `ps aux`         |                 显示所有进程                 |
|        `pstree`         |               显示进程依赖关系               |
|    `pstree -p 12345`    | 只显示 PID 为 `12345` 的那个进程的亲代及后代 |
|    `pstree -u user`     | 只显示用户名为 `user` 的那些进程的亲代及后代 |
|          `top`          |     动态显示所有进程状态（按 `q` 退出）      |
| `top -o +cpu -s 3 -n 4` |  CPU 升序、采样周期 `3` 秒、最多 `4` 个进程  |

### 管理进程

|          命令           |                     功能                     |
| :---------------------: | :------------------------------------------: |
|     `kill -9 12345`     |      强制关闭 PID 为 `12345` 的那个进程      |
|    `kill -15 12345`     |      正常关闭 PID 为 `12345` 的那个进程      |
|  `killall -9 process`   |      强制关闭名为 `process` 的那些进程       |
| `killall -s -9 process` |               显示但不执行操作               |


### 管理任务

由同一个 shell 进程触发的子进程称为**任务 (job)**。
若系统只提供了一个 shell 进程，则用户通常需要将*任务*在**前台 (foreground)** 与**后台 (background)** 之间来回切换，以便让多个任务同时运行。

⚠️ 这里的*前后台*是相对于*当前 shell 进程*而言的；若这个 shell 进程是远程主机提供的，则*退出 shell* 意味着退出所有后台任务（仍在后台的任务会阻断 shell 的退出）。
《[利用 SSH 访问远程 Linux 主机](./ssh.md)》介绍了如何访问远程主机以及在*远程主机的后台*运行任务。

|     命令      |             功能              |
| :-----------: | :---------------------------: |
|    `top &`    |    在后台启动并运行 `top`     |
|  `Ctrl + Z`   |   将当前任务暂停并移入后台    |
|    `jobs`     |      显示当前的后台任务       |
|    `fg %2`    | 将任务 `2` 从后台移到前台运行 |
|    `bg %2`    |     令任务 `2` 在后台运行     |
|   `kill -l`   |        列出可用的信号         |
| `kill -2 %2`  |  相当于在后台按下 `Ctrl + C`  |
| `kill -9 %2`  |    强制结束并删除任务 `2`     |
| `kill -15 %2` |    按正常步骤结束任务 `2`     |
| `kill -19 %2` |  相当于在后台按下 `Ctrl + Z`  |

以下示例演示了表中主要命令的用法：
1. 在后台启动并运行一个任务：
   ```shell
   $ top &
   [1] 60137
   ```
    返回值 `[1]` 为其**任务编号 (job number)**，`60137` 为 PID。
1. 启动一个需要运行很长时间的任务，按 `Ctrl + Z` 将其暂停并移入后台：
   ```shell
   $ find / > temp.txt
   # 需要运行很长时间（除非该系统几乎没有被使用过）
   # 按 `Ctrl + Z` 将进程暂停并移入后台
   [2]+  Stopped                 find / > temp.txt
   ```
1. 此时已有两个后台任务，可用 `jobs` 查看：
   ```shell
   $ jobs
   [1]-  Stopped                 vim
   [2]+  Stopped                 find / > temp.txt
   $ jobs -l
   [1]- 60137 Stopped (tty output): 22top
   [2]+ 60141 Suspended: 18           find / > temp.txt
   ```
   其中 `+` 与 `-` 分别表示*最后一个*与*倒数第二个*被移入后台的任务。
1. 令（停止的）任务在前台或后台运行：
   ```shell
   $ fg %2  # `find ~ > temp.txt` 在前台恢复运行
   # 输出几乎不会停止，按 `Ctrl + Z` 移入后台
   $ bg %2  # `find ~ > temp.txt` 在后台恢复运行
   $ jobs
   [1]+  Stopped                 top
   [2]-  Running                 find ~ > temp.txt &
   ```
1. 向后台任务发送信号：
   ```shell
   $ kill %1  # 正常结束 任务[1]
   $ jobs  # 短时间内 任务[1] 的状态变为 Terminated
   [1]-  Terminated: 15          top
   [2]+  Stopped                 find / > temp.txt
   $ jobs  # 较长时间后，只剩下 任务[2]
   [2]+  Stopped                 find / > temp.txt
   $ kill -2 %2  # 在后台 Ctrl + C 任务[2]
   $ jobs  # 短时间内 任务[2] 的状态变为 Interrupt
   [2]+  Interrupt: 2            find / > temp.txt
   ```

## 网络管理

### `ifconfig` - *config*ure a network *i*nter*f*ace

```shell
$ ifconfig  # 查询所有网卡
eno1: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 192.168.1.10  netmask 255.255.255.0  broadcast 192.168.1.255
        inet6 2002:a86:4beb:0:e41:f0c6:4354:e13  prefixlen 64  scopeid 0x0<global>
        inet6 2002:a86:4beb:0:42fa:7a9f:24a8:eee2  prefixlen 64  scopeid 0x0<global>
        inet6 fe80::126e:97a8:b5b9:4395  prefixlen 64  scopeid 0x20<link>
        ether 44:37:e6:bb:ec:a3  txqueuelen 1000  (Ethernet)
        RX packets 235305  bytes 332897613 (332.8 MB)
        RX errors 0  dropped 108  overruns 0  frame 0
        TX packets 59627  bytes 4690785 (4.6 MB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0
        device interrupt 20  memory 0xf7100000-f7120000  

lo: flags=73<UP,LOOPBACK,RUNNING>  mtu 65536
        inet 127.0.0.1  netmask 255.0.0.0
        inet6 ::1  prefixlen 128  scopeid 0x10<host>
        loop  txqueuelen 1000  (Local Loopback)
        RX packets 6633  bytes 576493 (576.4 KB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 6633  bytes 576493 (576.4 KB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0


$ ifconfig eno1  # 查询指定网卡
eno1: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 192.168.1.10  netmask 255.255.255.0  broadcast 192.168.1.255
        inet6 2002:a86:4beb:0:e41:f0c6:4354:e13  prefixlen 64  scopeid 0x0<global>
        inet6 2002:a86:4beb:0:42fa:7a9f:24a8:eee2  prefixlen 64  scopeid 0x0<global>
        inet6 fe80::126e:97a8:b5b9:4395  prefixlen 64  scopeid 0x20<link>
        ether 44:37:e6:bb:ec:a3  txqueuelen 1000  (Ethernet)
        RX packets 235397  bytes 332910768 (332.9 MB)
        RX errors 0  dropped 109  overruns 0  frame 0
        TX packets 59668  bytes 4698974 (4.6 MB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0
        device interrupt 20  memory 0xf7100000-f7120000  


$ sudo ifconfig eno1 192.168.1.111  # 设置指定网卡（除 IPv4 地址外，采用默认值）
$ sudo ifconfig eno1 192.168.1.111 netmask 255.255.255.0 mtu 8000  # 更精细的设置
```

### `ip` - show / manipulate routing, network devices, interfaces and tunnels

```shell
$ ip -s link show  # 显示所有网卡的信息及统计（`-s`）
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN mode DEFAULT group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
    RX: bytes  packets  errors  dropped overrun mcast   
    670613     7532     0       0       0       0       
    TX: bytes  packets  errors  dropped carrier collsns 
    670613     7532     0       0       0       0       
2: eno1: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 2000 qdisc fq_codel state UP mode DEFAULT group default qlen 1000
    link/ether 44:37:e6:bb:ec:a3 brd ff:ff:ff:ff:ff:ff
    RX: bytes  packets  errors  dropped overrun mcast   
    334900636  240457   0       127     0       6420    
    TX: bytes  packets  errors  dropped carrier collsns 
    4964034    61586    0       0       0       0       
    altname enp0s25

$ ip addr show eno1  # 显示指定网卡的地址（`inet` 及 `inet6`）
2: eno1: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc fq_codel state UP group default qlen 1000
    link/ether 44:37:e6:bb:ec:a3 brd ff:ff:ff:ff:ff:ff
    altname enp0s25
    inet 192.168.1.10/24 brd 192.168.1.255 scope global dynamic noprefixroute eno1
       valid_lft 85979sec preferred_lft 85979sec
    inet6 2002:a86:4beb:0:7aa1:88c7:e85:b3db/64 scope global temporary dynamic 
       valid_lft 2370sec preferred_lft 1770sec
    inet6 2002:a86:4beb:0:e41:f0c6:4354:e13/64 scope global dynamic mngtmpaddr noprefixroute 
       valid_lft 2370sec preferred_lft 1770sec
    inet6 fe80::126e:97a8:b5b9:4395/64 scope link noprefixroute 
       valid_lft forever preferred_lft forever

$ sudo ip link set eno1 up|down   # 开关指定网卡
$ sudo ip link set eno1 mtu 8000  # 调整个别参数
```

### `ping` - send ICMP ECHO_REQUEST to network hosts

```shell
$ ping bing.com  # 持续测试，直到 Ctrl + C 停止
$ ping -c 4 bing.com  # 测试 4 次
PING china.bing123.com (202.89.233.100) 56(84) bytes of data.
64 bytes from 202.89.233.100 (202.89.233.100): icmp_seq=1 ttl=115 time=3.59 ms
64 bytes from 202.89.233.100 (202.89.233.100): icmp_seq=2 ttl=115 time=3.27 ms
64 bytes from 202.89.233.100 (202.89.233.100): icmp_seq=3 ttl=115 time=3.50 ms
64 bytes from 202.89.233.100 (202.89.233.100): icmp_seq=4 ttl=115 time=3.25 ms

--- china.bing123.com ping statistics ---
4 packets transmitted, 4 received, 0% packet loss, time 3004ms
rtt min/avg/max/mdev = 3.252/3.402/3.589/0.144 ms

```

### `traceroute` - print the *route* packets *trace* to network host

```shell
$ traceroute bing.com
$ traceroute -w 1 -n -T bing.com  # 等待 1s、不解析域名、用 TCP 检测
traceroute to bing.com (204.79.197.200), 30 hops max, 60 byte packets
 1  192.168.1.1  0.231 ms  17.699 ms  17.682 ms
 2  10.134.74.1  139.197 ms  139.177 ms  139.686 ms
 3  10.254.1.1  139.660 ms  139.638 ms  139.622 ms
 4  211.71.0.29  139.597 ms  139.581 ms  139.567 ms
 5  106.120.221.209  139.592 ms  139.576 ms  139.512 ms
 6  * * *
 7  * * *
 8  * * *
 9  * * *
10  * * *
11  * * *
12  * * *
13  * * *
14  204.79.197.200  260.581 ms *  260.553 ms

```

### `host` - DNS lookup utility

```shell
$ host bing.com
bing.com has address 13.107.21.200
bing.com has address 204.79.197.200
bing.com has IPv6 address 2620:1ec:c11::200
bing.com mail is handled by 10 bing-com.mail.protection.outlook.com.
```

### `nslookup` - query Internet *n*ame *s*ervers interactively

```shell
$ nslookup bing.com
Server:         127.0.0.53
Address:        127.0.0.53#53

Non-authoritative answer:
Name:   bing.com
Address: 204.79.197.200
Name:   bing.com
Address: 13.107.21.200
Name:   bing.com
Address: 2620:1ec:c11::200

```

### `netstat` - show *net*work *stat*us
