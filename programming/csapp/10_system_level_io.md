---
title: 系统级读写
---

【**读写 (Input/Output, I/O)**】在*主存储器*与*外部设备*之间转移数据。

- [Unix I/O](#unix-io)
- [Standard I/O](#standard-io)
- [Robust I/O](#robust-io)
- [Signal I/O](./8_exceptional_control_flow.md#signal-io)

# 1. Unix I/O<a href id="unix-io"></a>

【**文件 (file)**】在 Linux 系统中所有读写设备（网卡、硬盘、终端）被统称为文件。

- 【**描述符 (descriptor)**】内核分配的非负（小）整数，其中前三个为系统预留：
  - `0 == STDIN_FILENO` 
  - `1 == STDOUT_FILENO`
  - `2 == STDERR_FILENO`
- 【**文件位置 (file position)**】内核维护的非负整数（字节数）
  - 【**查找 (seek)**】跳至指定位置，详见 [`lseek`](https://www.man7.org/linux/man-pages/man2/lseek.2.html)
  - 【**文件末尾 (end-of-file, EOF)**】*读取字节数*大于等于*剩余字节数*所触发的*事件*
- 【**关闭文件 (close file)**】释放数据结构、返还描述符

# 2. 文件

## 常规文件 (regular file)

- 【**文本文件 (text file)**】只含 ASCII 或 Unicode 字符，可视为文本行序列，以换行符表示**行末尾 (end-of-line, EOL)**：
  - 【`\n`】即 LF (line feed)，用于 Linux 及 macOS 系统
  - 【`\r\n`】其中 `\r` 即 CR (carriage return)，用于 Windows 系统及网络
- 【**二进制文件 (binary file)**】其他任意类型文件

## 目录 (directory)

一种特殊的文件，其数据为一数组，数组元素为指向其他文件的链接。

- 特殊目录
  - 【`.`】
  - 【`..`】
  - 【`/`】
- 常用命令
  - 【`pwd`】**p**rint **w**orking **d**irectory
  - 【`cd`】**c**hange **d**irectory
  - 【`mkdir`】**m**a**k**e **dir**ectory
  - 【`rmdir`】**r**e**m**ove **dir**ectory
- 路径
  - 绝对路径
  - 相对路径

## 套接字 (socket)

见《[网络编程](./11_network_programming.md)》

# 3. 开关文件

## [`open()`](https://www.man7.org/linux/man-pages/man2/open.2.html)

```c
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
int open(char *filename, int flags, mode_t mode);
    // Returns: new file descriptor if OK, −1 on error
```

### `flags`

- `O_RDONLY`
- `O_WRONLY`
- `O_RDWR`
- `O_CREAT`
- `O_TRUNC`
- `O_APPEND`

### `mode`

只在 `flags` 含 `O_CREAT | O_TMPFILE` 时起作用。

- `S_IRUSR, S_IWUSR, S_IXUSR` can be **r**ead/**w**rite/e**x**ecute by current **us**e**r**
- `S_IRGRP, S_IWGRP, S_IXGRP` can be **r**ead/**w**rite/e**x**ecute by current **gr**ou**p**
- `S_IROTH, S_IWOTH, S_IXOTH` can be **r**ead/**w**rite/e**x**ecute by **oth**ers

每个进程有一个 `umask` 值，可由 `umask()` 设置。

```c
#define DEF_MODE  S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH/* rw-rw-rw- */
#define DEF_UMASK S_IWGRP|S_IWOTH                  /* ~DEF_UMASK == rwxr-xr-x */
umask(DEF_UMASK); /* umask = DEF_UMASK */
fd = Open("foo.txt", O_CREAT|O_TRUNC|O_WRONLY, DEF_MODE);
        /* set access permission bits to (DEF_MODE & ~DEF_UMASK) 即 rw-r--r-- */
```

## [`close()`](https://www.man7.org/linux/man-pages/man2/close.2.html)

```c
#include <unistd.h>
int close(int fd);
```

# 4. 读写文件

## [`read()`](https://www.man7.org/linux/man-pages/man2/read.2.html)<a href id="unix-read"></a>

```c
#include <unistd.h>
ssize_t  read(int fd,       void *buf, size_t n);
    // Returns: number of bytes read    if OK, −1 on error, 0 on EOF
```

## [`write()`](https://www.man7.org/linux/man-pages/man2/write.2.html)<a href id="unix-write"></a>

```c
#include <unistd.h>
ssize_t write(int fd, const void *buf, size_t n);
    // Returns: number of bytes written if OK, −1 on error
```

## Short Count

`read()` 及 `write()` 的返回值（实际读写的字节数），可能小于传入的 `n`（请求读写的字节数）。

不会发生于读（遇到 EOF 除外）写硬盘，但可能发生于

1. 读到文件末尾，即检测到 EOF：
   - 假设从 `fd` 的当前位置起还有 `20` 字节未读，则调用 `read(fd, buf, 50)` 返回 `20`，再调用 `read(fd, buf += 20, 50)` 返回 `0`，这两次的返回值都属于 short count。
1. 从终端读入文本行：
   - `read` 每次读入一行，返回该行的字节数。
1. 读写[网络套接字](./11_network_programming.md#socket)：
   - 小缓存或长延迟，使得单次调用 `read()` 或 `write()` 只能读写部分数据，因此 *robust* (network) applications 需多次调用 `read()` 或 `write()`。
1. 读写 Linux 管道。

# 5. Robust I/O<a href id="robust-io"></a>

## 5.1. 无缓冲读写

接口与 [Unix I/O](#unix-io) 的 [`read()`](#unix-read) 及 [`write()`](#unix-write) 相同。

适用于从[网络套接字](./11_network_programming.md#socket)读写二进制数据。

```c
#include "csapp.h"
ssize_t rio_readn (int fd, void *usrbuf, size_t n);
ssize_t rio_writen(int fd, void *usrbuf, size_t n);
     /* Returns: number of bytes transferred if OK, −1 on error
                        , 0 on EOF or `n == 0` (rio_readn only) */
```

### `rio_readn()`

多次调用 [Unix I/O](#unix-io) 的 [`read()`](#unix-read)，直到请求的字节数都被读入，或遇到 EOF。

Short count 只会出现在遇到 EOF 时。

```c
ssize_t rio_readn (int fd, void *head, size_t n) {
  size_t nleft = n;
  ssize_t nread;
  char *pos = head;

  while (nleft > 0) {
    if ((nread = read(fd, pos, nleft)) < 0) {
      if (errno == EINTR) /* Interrupted by sig handler return */
        nread = 0;        /* and call `read()` again */
      else
        return -1;        /* `errno` set by `read()` */ 
    }
    else if (nread == 0)
      break;              /* EOF */
    nleft -= nread; pos += nread;
  }
  return (n - nleft);     /* short count only on EOF */
}
```

### `rio_writen()`

多次调用 [Unix I/O](#unix-io) 的 [`write()`](#unix-write)，直到请求的字节数都被写出。

Short count 不会出现。

```c
ssize_t rio_writen(int fd, void *head, size_t n) {
  size_t nleft = n;
  ssize_t nwritten;
  char *pos = head;

  while (nleft > 0) {
    if ((nwritten = write(fd, pos, nleft)) <= 0) {
      if (errno == EINTR)  /* Interrupted by sig handler return */
        nwritten = 0;      /* and call `write()` again */
      else
        return -1;         /* `errno` set by `write()` */
    }
    nleft -= nwritten; pos += nwritten;
  }
  return n;  /* never returns a short count */
}
```

## 5.2. 带缓冲读入

【需求】[线程安全](./12_concurrent_programming.md#thread-safe)；支持交替读取文本行与二进制数据。

```c
#include "csapp.h"
ssize_t rio_readlineb(rio_t *rp, void *usrbuf, size_t maxlen); /* read 1 line */
ssize_t rio_readnb   (rio_t *rp, void *usrbuf, size_t n);      /* read n bytes */
    // Returns: number of bytes read if OK, 0 on EOF, −1 on error
void    rio_readinitb(rio_t *rp, int fd); /* once per fd */
```

⚠️ 后缀 `b` 表示带缓冲的，不要与无缓冲读写混用。

### `rio_readinitb()`

初始化用户 buffer（每次读一个 file 对应一个 buffer）：

```c
#define RIO_BUFSIZE 8192

typedef struct {
  int rio_fd;            /* Descriptor for this internal buf */
  int rio_cnt;          /* Unread bytes in this internal buf */
  char *rio_bufpos; /* Next unread byte in this internal buf */
  char rio_buf[RIO_BUFSIZE]; /* Internal buffer */
} rio_t;

void rio_readinitb(rio_t *rp, int fd) {
  rp->rio_fd = fd;
  rp->rio_cnt = 0;  
  rp->rio_bufpos = rp->rio_buf;
}
```

### `rio_read()`

与 [Unix I/O](#unix-io) 的 [`read()`](#unix-read) 接口相同，但先（尽量多地）读入缓冲区，再复制（指定片段）给用户。

💡 用 `rio_read()` 可以减少直接调用 [`read()`](#unix-read) 的次数，后者开销巨大。

```c
static ssize_t rio_read(rio_t *rp, char *usrbuf/* user buffer */,
                        size_t n/* number of bytes requested by the user */) {
  int cnt;

  while (rp->rio_cnt <= 0) {  /* Refill if buf is empty */
    rp->rio_cnt = read(rp->rio_fd, rp->rio_buf, RIO_BUFSIZE);
    if (rp->rio_cnt < 0) {
      if (errno != EINTR/* Interrupted by sig handler return */)
        return -1;
    }
    else if (rp->rio_cnt == 0)  /* EOF */
      return 0;
    else
      rp->rio_bufpos = rp->rio_buf; /* Reset buffer ptr */
  }

  /* Copy  bytes from `rp->rio_bufpos` to `usrbuf` */
  cnt = min(n, rp->rio_cnt);
  memcpy(usrbuf, rp->rio_bufpos, cnt);
  rp->rio_bufpos += cnt; rp->rio_cnt -= cnt;
  return cnt;
}
```

### `rio_readlineb()`

读取一行字符，满足以下条件之一时停止：

1. 已读入 `maxlen` 字节
1. 遇到 EOF
1. 遇到 `\n`

```c
ssize_t rio_readlineb(rio_t *rp, void *head, size_t maxlen) {
  int n/* 当前字符串长度 */, rc/* 单次读取字节数 */;
  char c, *pos = head;

  for (n = 1/* 字符串总是以 `\0` 结尾，故长度至少为 1 */; n < maxlen; n++) { 
    if ((rc = rio_read(rp, &c, 1)) == 1) {
      *(pos++) = c;
      if (c == '\n') {
        n++;
        break;
      }
    }
    else if (rc == 0) {
      if (n == 1)
        return 0; /* EOF, no data read */
      else
        break;    /* EOF, some data was read */
    }
    else
      return -1;  /* Error */
  }
  *pos = 0;       /* end of string */
  return n - 1;   /* 不计 `\0` */
}
```

### `rio_readnb()`

读取若干字节，满足以下条件之一时停止：

1. 已读入 `n` 字节
1. 遇到 EOF

```c
ssize_t rio_readnb(rio_t *rp, void *head, size_t n) {
  size_t nleft = n;
  ssize_t nread;
  char *pos = head;

  while (nleft > 0) {
    if ((nread = rio_read(rp, pos, nleft)) < 0) 
      return -1;
    else if (nread == 0)
      break;              /* EOF */
    nleft -= nread; pos += nread;
  }
  return (n - nleft);     /* return >= 0 */
}
```

# 6. 读取文件元数据<a href id="meta"></a>

## `stat()`

```c
#include <unistd.h>
#include <sys/stat.h>
int stat(const char *filename, struct stat *buf);
int fstat(int fd, struct stat *buf);
```

## `struct stat`

```c
/* Metadata returned by the stat and fstat functions */
/* included by sys/stat.h */
struct stat {
  dev_t st_dev; /* Device */
  ino_t st_ino; /* inode */
  mode_t st_mode; /* Protection and file type */
  nlink_t st_nlink; /* Number of hard links */
  uid_t st_uid; /* User ID of owner */
  gid_t st_gid; /* Group ID of owner */
  dev_t st_rdev; /* Device type (if inode device) */
  off_t st_size; /* Total size, in bytes */
  unsigned long st_blksize; /* Block size for filesystem I/O */
  unsigned long st_blocks; /* Number of blocks allocated */
  time_t st_atime; /* Time of last access */
  time_t st_mtime; /* Time of last modification */
  time_t st_ctime; /* Time of last change */
};
```

## `statcheck.c`

```c
#include "csapp.h"

int main (int argc, char **argv) {
  struct stat stat;
  char *type, *readok;

  if (argc != 2) {
    fprintf(stderr, "usage: %s <filename>\n", argv[0]);
    exit(0);
  }
  Stat(argv[1], &stat);
  if (S_ISREG(stat.st_mode))     /* Determine file type */
    type = "regular";
  else if (S_ISDIR(stat.st_mode))
    type = "directory";
  else 
    type = "other";
  if ((stat.st_mode & S_IRUSR))  /* Check read access */
    readok = "yes";
  else
    readok = "no";

  printf("type: %s, read: %s\n", type, readok);
  exit(0);
}
```

# 7. 读取目录内容

## 开关目录


```c
#include <sys/types.h>
#include <dirent.h>
DIR *opendir(const char *name);
    /* Returns: pointer to handle if OK, NULL on error */
int closedir(DIR *dirp);
    /* Returns: 0 on success, −1 on error */
```

## `readdir()`

```c
#include <dirent.h>
struct dirent {
  ino_t d_ino;   /* inode number */
  char  d_name[256]; /* Filename */
};
struct dirent *readdir(DIR *dirp);
    /* Returns: pointer to next directory entry if OK,
                NULL if no more entries or error */
```

⚠️ 只能通过检查 `errno` 是否被修改，来判断是出错，还是到达列表末尾。

## `readdir.c`

```c
#include "csapp.h"

int main(int argc, char **argv) {
  DIR *streamp; 
  struct dirent *dep; 

  if (argc != 2) {
    printf("usage: %s <pathname>\n", argv[0]);
    exit(1);
  }
  streamp = Opendir(argv[1]);
  errno = 0;
  while ((dep = readdir(streamp)) != NULL) { 
    printf("Found file: %s\n", dep->d_name); 
  } 
  if (errno != 0)
    unix_error("readdir error");

  Closedir(streamp); 
  exit(0);
}
```

# 8. 共享文件

![](./ics3/io/filesharing.svg)

## [`fork()`](./8_exceptional_control_flow.md#fork) 再探

子进程继承其 parent's open file table，表中每一项的引用计数加一：

![](./ics3/io/afterfork.svg)

# 9. 读写重定向<a href id="dup2"></a>

```c
#include <unistd.h>
int dup2(int oldfd, int newfd/* close if already open */);
    /* Returns: nonnegative descriptor if OK, −1 on error */
```

`dup2(4, 1)` **dup**licate `fd[4]` **to** `fd[1]`，结果如下：

![](./ics3/io/dupafter.svg)

# 10. [Standard I/O](https://en.cppreference.com/w/c/io)<a href id="standard-io"></a>

C 标准库提供，将 file 及其对应的 buffer 抽象为 [FILE stream](https://en.cppreference.com/w/c/io/FILE)。

```c
#include <stdio.h>
extern FILE *stdin;  /* Standard input  (descriptor 0) */
extern FILE *stdout; /* Standard output (descriptor 1) */
extern FILE *stderr; /* Standard error  (descriptor 2) */
```

## 开关文件

```c
FILE *fopen(const char *filename, const char *mode);
int  fclose(FILE *stream);
```

## 读写字符串

```c
char *fgets(      char *dst, int count, FILE *stream);
int   fputs(const char *src,            FILE *stream);
```

## 读写对象

```c
size_t fread (      void *dst, size_t size, size_t count, FILE *stream);
size_t fwrite(const void *src, size_t size, size_t count, FILE *stream );
```

## 格式化读写

```c
int   scanf(              const char *format, ...);
int  fscanf(FILE *stream, const char *format, ...);
int  sscanf(char *buffer, const char *format, ...);
int  printf(              const char *format, ...);
int fprintf(FILE *stream, const char *format, ...);
int sprintf(char *buffer, const char *format, ...);
int  fflush(FILE *output); /* undefined behavior for input */
```

# 11. I/O 库的选择

![](./ics3/io/iofunctions.svg)

|            I/O 库            |                           适用场景                           |                             缺点                             |
| :--------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|     [Unix I/O](#unix-io)     | [读取文件元数据](#meta)、[信号处置器](./8_exceptional_control_flow.md#signal)内部 | 难以处理 short count、[系统调用](./8_exceptional_control_flow.md#syscall)开销大 |
| [Standard I/O](#standard-io) |                        终端、硬盘文件                        | 无法获取元数据、非[线程安全](./12_concurrent_programming.md#thread-safe)、不能读写[网络套接字](./11_network_programming.md#socket) |
|   [Robust I/O](#robust-io)   |       [网络套接字](./11_network_programming.md#socket)       | 不支持格式化读写（需借助 [Standard I/O](#standard-io) 中的 `sscanf()` 及 `sprintf()` 完成） |

⚠️ 不要用 `fgets()`、`scanf()` 或 `rio_readlineb()` 等读二进制文件。

# [全书目录](../csapp.md#全书目录)
