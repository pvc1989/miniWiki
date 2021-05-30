---
title: 网络编程
---

# 1. 客户端-服务器模型

『客户端-服务器模型 (client-server model)』是所有网络应用的基础。

- 【服务器 (server)】管理资源、提供服务的进程。
- 【客户端 (client)】请求资源、使用服务的进程。

『交易 (transaction)』是客户端与服务器交互的基本操作，含以下四步：

1. 客户端向服务器发送『请求 (request)』。
2. 服务器接收该请求，并根据需要操纵其管理的资源。
3. 服务器向客户端发送『响应 (response)』，并等待下一个请求。
4. 客户端接收该响应，并根据需要处理之。

![](https://csapp.cs.cmu.edu/3e/ics3/netp/cliservsteps.pdf)

# 2. 网络

对一台主机而言，『网络 (newwork)』相当于另一种读写设备。

## 局域网

【局域网 (local area network, LAN)】目前最流行的局域网技术为『以太网 (Ethernet)』

### 以太网段

【以太网段 (Ethernet segment)】由多台主机通过以太网线（双绞线）连接到一台集线器所形成的小型网络，可覆盖一间或一层房屋。

- 【以太网线 (Ethernet wire)】一端连接到主机上的『以太网适配器 (Ethernet adapter)』，另一端连接到集线器上的『端口 (port)』。
- 【集线器 (hub)】只是将每个端口上的数据被动地复制到其他所有端口。因此连接到同一台集线器上的所有主机，都能看到相同的数据。
- 每个以太网适配器都拥有一个长度为 48-bit 的物理地址，以区别于其他适配器。
- 一台主机向其他主机发送的最小数据块称作『帧 (frame)』，其内容除『有效载荷 (payload)』外，还包括来源地址、目标地址、帧长度的『头标  (header)』。

### 桥接的以太网

【桥接的以太网 (bridged Ethernet)】由多个以太网段连接到多个网桥所形成的中型网络，可覆盖整栋建筑或整个校园。

- 【网桥 (bridge)】能够自动感知哪些主机可以连接到哪些端口，从而有选择地转发数据。
- 网桥与网桥之间的带宽可达 1 Gb/s，网桥与集线器之间的带宽通常为 100 Mb/s。

![](https://csapp.cs.cmu.edu/3e/ics3/netp/bridge.pdf)

## 广域网

【广域网 (wide area network, WAN)】

## 互联网

【互联网 (interconnected network, internet)】由多个局域网及广域网连接到多个路由器所形成的大型网络，可覆盖全球。

![](https://csapp.cs.cmu.edu/3e/ics3/netp/internet.pdf)

- 【路由器 (router)】在多个局域网及广域网之间转发数据的设备。
- 【协议 (protocol)】运行在主机及路由器上的程序，用于协调不同局域网及广域网技术。
  - 【命名格式 (naming scheme)】定义格式统一的主机地址。
  - 【发送机制 (delivery mechanism)】定义数据打包的方式。
    - 局域网帧 = 局域网头标 + 有效载荷（互联网包）
    - 互联网包 = 互联网头标 + 有效载荷（用户数据）

![](https://csapp.cs.cmu.edu/3e/ics3/netp/intertrans.pdf)

# 3. 因特网

『全局 IP 因特网 (global IP Internet)』，简称『因特网 (Internet)』是所有『互联网 (internet)』中最著名、最成功的一个。

因特网中的每台主机都运行着实现了『TCP/IP 协议』的软件。

- 【IP = Internet Protocol】提供最基本的命名格式，以及主机到主机的『不可靠 (unreliable)』传输机制。
- 【UDP = Unreliable Datagram Protocol】基于 IP 的简单扩展，提供进程到进程的不可靠传输。
- 【TCP = Transmission Control Protocol】基于 IP 的复杂扩展，提供进程到进程的双向可靠传输。

对于普通程序员，因特网可以理解为由世界上所有具备以下性质的主机所构成的互联网：

- 主机之集被映射到『IP 地址 (IP address)』之集。
- IP 地址之集被映射到『因特网域名 (Internet domain name)』之集。
- 一台主机上的进程可通过『连接 (connection)』与另一台主机上的进程『通信 (communicate)』。

## 3.1. IP 地址

IPv4 地址可以用 32-bit 无符号整数表示。由于历史的原因，它被定义为结构体：

```c
/* IP address structure */
struct in_addr {
    uint32_t s_addr; /* in network byte order (big-endian) */
};
```

TCP/IP 规定所有整数都采用『大端 (big-endian)』字节顺序。以下函数用于主机与网络字节顺序的转换：

```c
#include <arpa/inet.h>
/* Host byte order TO Network byte order */
uint32_t htonl(uint32_t hostlong);
uint16_t htons(uint16_t hostshort);
/* Network byte order TO Host byte order */
uint32_t ntohl(uint32_t netlong);
uint16_t ntohs(unit16_t netshort);
```

【带点的十进制记法 (dotted decimal notation)】用 `.` 隔开的一串十进制整数，例如：地址 `0x8002c2f2` 可以用 `128.2.194.242` 表示。以下函数用于网络地址 (Network) 与其字符串表示 (Presentation) 之间的转换：

```c
#include <arpa/inet.h>
int inet_pton(AF_INET, const char *src, void *dst); /* returns
    `1` if OK, 0 if `src` is invalid, `−1` on error and sets `errno` */
const char *inet_ntop(AF_INET, const void *src, char *dst, socklen_t size); /* returns
    pointer to a dotted-decimal string if OK, `NULL` on error
```

其中 `AF_INET` 表示 `void *` 指向 32-bit 的 IPv4 地址（ `AF_INET6` 表示 `void *` 指向 128-bit 的 IPv6 地址）。

## 3.2. 因特网域名

【域名 (domain name)】用 `.` 隔开的一组单词（字符、数字、下划线），例如：`csapp.cs.cmu.edu` 为本书网站的域名，从右向左依次为

- 顶级域名 `edu`，同级的域名还有 `gov`, `com` 等。
- 二级域名 `cmu`，同级的域名还有 `mit`, `princeton` 等。
- 三级域名 `cs`，同级的域名还有 `ee`, `art` 等。

【域名系统 (domain name system, DNS)】
- 管理域名与 IP 地址之间映射关系到分布式数据库。
- 已知域名，可以用 `nslookup` 查询相应的 IP 地址（可能有零到多个）。

## 3.3. 因特网连接

【连接 (connection)】一对可以收发字节流的进程。
- 【点到点 (point-to-point)】建立在一对进程之间。
- 【全双工 (full duplex)】可以在同一时间双向传输。
- 【可靠 (reliable)】字节流的接收顺序与发送顺序一致。

![](https://csapp.cs.cmu.edu/3e/ics3/netp/connection.pdf)

【套接字 (socket)】连接的一端，用形如 `ip_address:port` 的『套接字地址』表示，其中 `port` 为 16-bit 的『端口号』。

- 客户端的端口号不固定，由操作系统内核自动分配。
- 常用服务的端口号相对固定，列于 `/etc/services` 中。

# 4. 套接字接口<a href id="socket"></a>

![](https://csapp.cs.cmu.edu/3e/ics3/netp/sockoverview.pdf)

## 4.1. 套接字地址结构

- 对于系统内核，套接字是通信的一个端点。
- 对于应用程序，套接字是一个打开的文件。

```c
/* IPv4 套接字地址结构 */
struct sockaddr_in {
    uint16_t        sin_family;  /* 协议族，总是取 `AF_INET` */
    uint16_t        sin_port;    /* 端口号，大端字节顺序 */
    struct in_addr  sin_addr;    /* IPv4 地址，大端字节顺序 */
    unsigned char   sin_zero[8]; /* 对齐至 sizeof(struct sockaddr) */
};

/* 范型套接字地址结构，用于 connect, bind, accept 等函数 */
struct sockaddr {
    uint16_t  sa_family;    /* Protocol family */
    char      sa_data[14];  /* Address data  */
};

typedef struct sockaddr SA;
```

## 4.2. `socket()`

客户端及服务器用此函数获得（部分打开的）套接字。若成功则返回『套接字描述符 (socket descriptor)』，否则返回 `-1`。

```c
#include <sys/types.h>
#include <sys/socket.h>
int socket(int domain, int type, int protocol);

/* 建议用 getaddrinfo() 获得实参 */
socket_fd = Socket(AF_INET/* IPv4 */, SOCK_STREAM/* 为连接一端 */, 0);
```

## 4.3. `connect()`

客户端用此函数向服务器发送连接请求并等待。若成功则返回 `0`，否则返回 `-1`。

```c
#include <sys/socket.h>
int connect(int client_fd, const SA *server_addr,
            socklen_t addr_len/* sizeof(sockaddr_in) */);
```

至此，客户端可通过在 `client_fd` 上读写数据，实现与服务器通信。

## 4.4. `bind()`

服务器用此函数将套接字描述符 `server_fd` 与套接字地址 `server_addr` 关联。
若成功则返回 `0`，否则返回 `-1`。

```c
#include <sys/socket.h>
int bind(int server_fd, const SA *server_addr,
         socklen_t server_addr_len/* sizeof(sockaddr_in) */);
```

## 4.5. `listen()`

服务器用此函数将『活跃套接字 (active socket)』转变为『监听套接字 (listening socket)』。若成功则返回 `0`，否则返回 `-1`。<a href id="listen"></a>
- 【活跃套接字】`socket()` 返回的默认是这种，供客户端所使用。
- 【监听套接字】供服务器接收连接请求，记作 `listen_fd`。

```c
#include <sys/socket.h>
int listen(int active_fd, int backlog/* 队列大小（请求个数）提示，通常为 1024 */);
```

## 4.6. `accept()`

服务器用此函数等待客户端发来的连接请求。
若成功，则返回异于 `listen_fd` 的 `connect_fd`，并获取客户端地址；否则返回 `-1`。

```c
#include <sys/socket.h>
int accept(int listen_fd, SA *client_addr, int *client_addr);
```

至此，服务器可通过在 `connect_fd` 上读写数据，实现与客户端通信。

## 4.7. 信息提取

### `getaddrinfo()`

```c
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>

struct addrinfo {
    int ai_flags; /* AI_ADDRCONFIG | AI_CANONNAME | AI_NUMERICSERV | AI_PASSIVE */
    int ai_family; /* AF_INET or AF_INET6 */
    int ai_socktype; /* SOCK_STREAM */
    int ai_protocol; /* Third arg to socket function */
    char *ai_canonname; /* Canonical hostname */
    size_t ai_addrlen; /* Size of ai_addr struct */
    struct sockaddr *ai_addr; /* Ptr to socket address structure */
    struct addrinfo *ai_next; /* Ptr to next item in linked list */
};

int getaddrinfo(
    const char *host/* 域名 或 十进制地址*/,
    const char *service/* 服务名 或 端口号 */,
    const struct addrinfo *hints/* NULL 或 只含前四项的 addrinfo */,
    struct addrinfo **result/* 输出链表 */
);
void freeaddrinfo(struct addrinfo *result);
const char *gai_strerror(int errcode);
```

![](https://csapp.cs.cmu.edu/3e/ics3/netp/addrinfolist.pdf)

`getaddrinfo()` 返回一个链表（需用 `freeaddrinfo()` 释放），其中每个结点为 `struct addrinfo` 类型。
- 客户端依次用每个结点提供的信息尝试 `socket()` 及 `connect()`，直到成功返回。
- 服务器依次用每个结点提供的信息尝试 `socket()` 及 `bind()`，直到成功返回。

### `getnameinfo()`

`getnameinfo()` 与 `getaddrinfo()` 的功能相反。

```c
#include <sys/socket.h>
#include <netdb.h>
int getnameinfo(const struct sockaddr *sa, socklen_t salen,
                char *host, size_t hostlen,    /* 可以为空，即 NULL, 0 */
                char *service, size_t servlen, /* 同上，但至多一行为空 */
                int flags/* NI_NUMERICHOST | NI_NUMERICSERV */);
```

### 示例：`hostname.c`

```c
#include "csapp.h"

int main(int argc, char **argv) {
  struct addrinfo *p, *listp, hints;
  char buf[MAXLINE];
  int rc, flags;

  if (argc != 2) {
    fprintf(stderr, "usage: %s <domain name>\n", argv[0]);
    exit(0);
  }

  memset(&hints, 0, sizeof(struct addrinfo));                         
  hints.ai_family = AF_INET;       /* IPv4 only */
  hints.ai_socktype = SOCK_STREAM; /* 只关注连接 */
  if ((rc = getaddrinfo(argv[1], NULL, &hints, &listp)) != 0) {
    fprintf(stderr, "getaddrinfo error: %s\n", gai_strerror(rc));
    exit(1);
  }

  flags = NI_NUMERICHOST; /* 以十进制 IP 地址表示 host */
  for (p = listp; p; p = p->ai_next) { /* 遍历链表 */
    Getnameinfo(p->ai_addr, p->ai_addrlen, buf, MAXLINE, NULL, 0, flags);
    printf("%s\n", buf);
  }

  Freeaddrinfo(listp);
  exit(0);
}
```

## 4.8. 辅助函数

### `open_clientfd()`

此函数提供了对客户端调用 `getaddrinfo()`、`socket()`、`connect()` 的封装。

```c
#include "csapp.h"
int open_clientfd(char *hostname, char *port/* 端口号 */) {
  int clientfd, rc;
  struct addrinfo hints, *listp, *p;

  /* Get a list of potential server addresses */
  memset(&hints, 0, sizeof(struct addrinfo));
  hints.ai_socktype = SOCK_STREAM;  /* Open a connection */
  hints.ai_flags = AI_NUMERICSERV;  /* ... using a numeric port arg. */
  hints.ai_flags |= AI_ADDRCONFIG;  /* Recommended for connections */
  if ((rc = getaddrinfo(hostname, port, &hints, &listp)) != 0) {
    fprintf(stderr, "getaddrinfo failed (%s:%s): %s\n", hostname, port, gai_strerror(rc));
    return -2;
  }

  for (p = listp; p; p = p->ai_next) {
    if ((clientfd = socket(p->ai_family, p->ai_socktype, p->ai_protocol)) < 0) 
      continue;
    if (connect(clientfd, p->ai_addr, p->ai_addrlen) != -1) 
      break; /* 连接成功，则终止遍历 */
    if (close(clientfd) < 0) { /* 连接失败，则关闭文件，再尝试下一个 */
      fprintf(stderr, "open_clientfd: close failed: %s\n", strerror(errno));
      return -1;
    }
  }

  freeaddrinfo(listp);
  return p ? clientfd : -1;
}
```

### `open_listenfd()`

此函数提供了对服务器调用 `getaddrinfo()`、`socket()`、`bind()`、`listen()` 的封装。

```c
#include "csapp.h"
int open_listenfd(char *port) {
  struct addrinfo hints, *listp, *p;
  int listenfd, rc, optval=1;

  memset(&hints, 0, sizeof(struct addrinfo));
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = AI_ADDRCONFIG | AI_NUMERICSERV
      | AI_PASSIVE/* host=NULL, all ai_addr=*.*.*.* */;
  if ((rc = getaddrinfo(NULL, port, &hints, &listp)) != 0) {
    fprintf(stderr, "getaddrinfo failed (port %s): %s\n", port, gai_strerror(rc));
    return -2;
  }

  for (p = listp; p; p = p->ai_next) {
    if ((listenfd = socket(p->ai_family, p->ai_socktype, p->ai_protocol)) < 0) 
      continue;

    /* Eliminates "Address already in use" error from bind() */
    setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR,
               (const void *)&optval , sizeof(int));

    if (bind(listenfd, p->ai_addr, p->ai_addrlen) == 0)
      break;
    if (close(listenfd) < 0) {
      fprintf(stderr, "open_listenfd close failed: %s\n", strerror(errno));
      return -1;
    }
  }

  freeaddrinfo(listp);
  if (!p)
    return -1;

  if (listen(listenfd, LISTENQ) < 0) {
    close(listenfd);
    return -1;
  }
  return listenfd;
}
```

## 4.9. 示例：回音系统

### `echoclient.c`

```c
#include "csapp.h"

int main(int argc, char **argv) {
  int clientfd;
  char *host, *port, buf[MAXLINE];
  rio_t rio;

  if (argc != 3) {
    fprintf(stderr, "usage: %s <host> <port>\n", argv[0]);
    exit(0);
  }
  host = argv[1];
  port = argv[2];

  clientfd = Open_clientfd(host, port);
  Rio_readinitb(&rio, clientfd);

  while (Fgets(buf, MAXLINE, stdin) != NULL) {
    Rio_writen(clientfd, buf, strlen(buf)); // 向服务器发送
    Rio_readlineb(&rio, buf, MAXLINE);      // 从服务器读取
    Fputs(buf, stdout);                     // 在客户端打印
  }
  Close(clientfd);  // 客户端关闭套接字描述符，服务器会检测到 EOF
  exit(0);
}
```

### `echoserveri.c`

```c
#include "csapp.h"

void echo(int connfd);

int main(int argc, char **argv) {
  int listenfd, connect_fd;
  socklen_t client_len;
  struct sockaddr_storage client_addr;  /* Enough space for any address */
  char client_hostname[MAXLINE], client_port[MAXLINE];

  if (argc != 2) {
    fprintf(stderr, "usage: %s <port>\n", argv[0]);
    exit(0);
  }

  listenfd = Open_listenfd(argv[1]);
  while (1) {
    client_len = sizeof(struct sockaddr_storage); 
    connect_fd = Accept(listenfd, (SA *)&client_addr, &client_len);
    Getnameinfo((SA *)&client_addr, client_len,
                client_hostname, MAXLINE, 
                client_port, MAXLINE, 0);
    printf("Connected to (%s, %s)\n", client_hostname, client_port);
    echo(connect_fd);
    Close(connect_fd);
  }
  exit(0);
}

void echo(int connect_fd) {
  size_t n; 
  char buf[MAXLINE]; 
  rio_t rio;

  Rio_readinitb(&rio, connect_fd);
  while((n = Rio_readlineb(&rio, buf, MAXLINE)) != 0) {
    printf("server received %d bytes\n", (int)n);
    Rio_writen(connect_fd, buf, n);
  }
}
```

【迭代型服务器 (iterative server)】同一时间只能服务一个客户端，不同客户端要排成队列依次接受服务。

### 演示

```shell
# server:
$ sudo ./echoserveri 23333
[sudo] password for user:
# client:
$ ./echoclient localhost 23333
# server:
Connected to (localhost, 44250)
# client:
hello, world
# server:
server received 13 bytes
# client:
hello, world
Ctrl+D
# server:
# wait for the next request
Ctrl+C
```

# 5. 网页服务器

## 5.1. 网页基础

【超文本传输协议 (HyperText Transfer Protocol, HTTP)】『网页 (Web)』服务的协议
- 【浏览器 (browser)】HTTP 的客户端
- 【内容 (content)】HTTP 的客户端向服务器请求的数据

【[超文本标记语言 (HyperText Markup Language, HTML)](../../documenting/web/html.md)】
- 【页面 (page)】HTML 的程序
- 【标签 (tag)】HTML 的指令

## 5.2. 网页内容

MIME (Multipurpose Internet Mail Extensions)<a href id="mime"></a>

网页内容分类
- 【静态内容 (static content)】服务器传送给客户端的文件
- 【动态内容 (dynamic content)】先在服务器上计算得到、再传输给客户端的数据

URL (universal resource locator)
- `http://www.google.com:80/index.html`，其中
  - `www.google.com` 为域名，`80` 为端口号（`http` 服务的默认端口号为 `80`，可省略）。
  - `/index.html` 为服务器上的静态内容文件，其中 `/` 表示服务器上用于存放静态内容的目录（如 `/usr/httpd/html/`）。
- `http://bluefish.ics.cs.cmu.edu:8000/cgi-bin/adder?15000&213`，其中
  - `bluefish.ics.cs.cmu.edu` 为域名，`8000` 为端口号。
  - `/cgi-bin/adder` 为服务器上的可执行文件（功能为加法），其中 `/cgi-bin/` 表示服务器上用于存放动态内容的目录（如 `/usr/httpd/cgi-bin/`）。
  - `?` 之后为 `adder` 的实参列表，`&` 用于分隔实参。

## 5.3. HTTP 交易

### `telnet`

在客户端（命令行终端）内输入以下内容，以发起连接：

```shell
$ telnet csapp.cs.cmu.edu 80
```

服务器返回以下三行，显示在客户端：

```
Trying 128.2.100.230...
Connected to csapptest.cs.cmu.edu.
Escape character is '^]'.
```

### HTTP 请求

在客户端输入以下内容（含空行）：

```
GET / HTTP/1.1
Host: csapp.cs.cmu.edu

```

其中

- 第一行称作『请求行 (request line)』，格式为 `method URI version`
  - 若客户端为浏览器，则 URI 为 URL 的后缀（位于域名、端口号之后的部分）。
  - 若客户端为代理服务器，则 URI 为整个 URL。
- 第二行开始为『请求页眉 (request header)』，格式为 `header_name: header_data`
- 最后的空行表示页眉结尾。

### HTTP 响应

服务器返回以下内容，打印在客户端：

```
HTTP/1.1 200 OK
Date: Sat, 22 May 2021 16:43:27 GMT
Server: Apache/2.2.15 (Red Hat)
Last-Modified: Mon, 20 Jan 2020 21:53:54 GMT
ETag: "8440768-1882-59c9953d5c080"
Accept-Ranges: bytes
Content-Length: 6274
Connection: close
Content-Type: text/html; charset=UTF-8

```

其中

- 第一行称作『响应行 (response line)』，格式为 `version status_code status_message`
- 第二行开始为『响应页眉 (response header)』
- 最后的空行表示页眉结尾。

接下来，客户端显示网页的 HTML 源代码：

```html
<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd">

<html>
<head>
<title>CS:APP3e, Bryant and O'Hallaron</title>
<link href="http://csapp.cs.cmu.edu/3e/css/csapp.css" rel="stylesheet" type="text/css">
</head>
...
</body>
</html>

```

几秒钟后，客户端显示以下内容，并退出 `telnet`：
```
Connection closed by foreign host.
```

## 5.4. 提供动态内容

CGI (common gateway interface)

### 客户端向服务器传递实参

`GET` 将 URI 中的实参传送给服务器，其中

- `?` 分隔文件名与实参。
- `&` 分割实参列表（实参列表中不能含空格）。
- 实参本身含空格的，用特殊字符 `%20` 表示。

### 服务器向子进程传递实参

服务器收到 `GET /cgi-bin/adder?15000&213 HTTP/1.1` 后，会依次

- 用 `fork` 创建一个子进程。
- 将 CGI 环境变量 `QUERY_STRING` 设为 `15000&213`。 
- 用 `execve` 加载 `/cgi-bin/adder` 程序。

详见 [`serve_dynamic()`](#`serve_dynamic()`)。

### 服务器向子进程传递其他信息

CGI 标准定义了一些环境变量，用于传递信息：

- `QUERY_STRING` 程序实参
- `SERVER_PORT` 服务器的[监听端口](#listen)
- `REQUEST_METHOD` 可以为 `GET` 或 `POST`
- `REMOTE_HOST` 客户端域名
- `REMOTE_ADDR` 客户端 IP 地址
- `CONTENT_TYPE` 仅供 `POST`，表示所请求对象的 [MIME](#mime) 类型
- `CONTENT_LENGTH` 仅供 `POST`，表示所请求对象的字节数

### 子进程向客户端发送结果

加载 CGI 程序前，子进程会用 `dup2` 将 `stdout` 重定向到 `connect_fd`，从而传递给客户端。

详见 [`serve_dynamic()`](#`serve_dynamic()`)。


### `adder.c`

```c
#include "csapp.h"

int main(void) {
  char *buf, *p;
  char arg1[MAXLINE], arg2[MAXLINE], content[MAXLINE];
  int n1=0, n2=0;

  /* Extract the two arguments */
  if ((buf = getenv("QUERY_STRING")) != NULL) {
    p = strchr(buf, '&');
    *p = '\0';
    strcpy(arg1, buf);
    strcpy(arg2, p+1);
    n1 = atoi(arg1);
    n2 = atoi(arg2);
  }

  /* Make the response body */
  sprintf(content, "Welcome to add.com: ");
  sprintf(content, "%sTHE Internet addition portal.\r\n<p>", content);
  sprintf(content, "%sThe answer is: %d + %d = %d\r\n<p>", 
          content, n1, n2, n1 + n2);
  sprintf(content, "%sThanks for visiting!\r\n", content);

  /* Generate the HTTP response */
  printf("Connection: close\r\n");
  printf("Content-length: %d\r\n", (int)strlen(content));
  printf("Content-type: text/html\r\n\r\n");
  printf("%s", content);
  fflush(stdout);

  exit(0);
}
```

# 6. 示例：`tiny.c`

## `main()`

```c
int main(int argc, char **argv) {
  int listen_fd, connect_fd;
  char hostname[MAXLINE], port[MAXLINE];
  socklen_t client_len;
  struct sockaddr_storage client_addr;

  if (argc != 2) {
    fprintf(stderr, "usage: %s <port>\n", argv[0]);
    exit(1);
  }

  listen_fd = Open_listenfd(argv[1]);
  while (1) { // 迭代型服务器
    client_len = sizeof(client_addr);
    connect_fd = Accept(listen_fd, (SA *)&client_addr, &client_len);
    Getnameinfo((SA *)&client_addr, client_len,
                hostname, MAXLINE,
                port, MAXLINE, 0);
    printf("Accepted connection from (%s, %s)\n", hostname, port);
    doit(connect_fd); // 处置请求
    Close(connect_fd);
  }
}
```

## `doit()`

```c
void doit(int fd) {
  int is_static;
  struct stat sbuf;
  char buf[MAXLINE], method[MAXLINE], uri[MAXLINE], version[MAXLINE];
  char filename[MAXLINE], cgiargs[MAXLINE];
  rio_t rio;

  Rio_readinitb(&rio, fd);
  if (!Rio_readlineb(&rio, buf, MAXLINE))
    return;
  printf("%s", buf);
  sscanf(buf, "%s %s %s", method, uri, version);
  if (strcasecmp(method, "GET")) { // 只支持 GET
    clienterror(fd, method, "501", "Not Implemented",
                "Tiny does not implement this method");
    return;
  }
  read_requesthdrs(&rio); // 忽略请求页眉

  is_static = parse_uri(uri, filename, cgiargs);
  if (stat(filename, &sbuf) < 0) {
    clienterror(fd, filename, "404", "Not found",
                "Tiny couldn't find this file");
    return;
  }

  if (is_static) {
    if (!(S_ISREG(sbuf.st_mode)) || !(S_IRUSR & sbuf.st_mode)) {
      clienterror(fd, filename, "403", "Forbidden",
                  "Tiny couldn't read the file");
      return;
    }
    serve_static(fd, filename, sbuf.st_size);
  }
  else {
    if (!(S_ISREG(sbuf.st_mode)) || !(S_IXUSR & sbuf.st_mode)) {
      clienterror(fd, filename, "403", "Forbidden",
                  "Tiny couldn't run the CGI program");
      return;
    }
    serve_dynamic(fd, filename, cgiargs);
  }
}
```

## `clienterror()`

```c
void clienterror(int fd, char *cause, char *errnum, 
                 char *shortmsg, char *longmsg) {
  char buf[MAXLINE];

  /* Print the HTTP response headers */
  sprintf(buf, "HTTP/1.0 %s %s\r\n", errnum, shortmsg);
  Rio_writen(fd, buf, strlen(buf));
  sprintf(buf, "Content-type: text/html\r\n\r\n");
  Rio_writen(fd, buf, strlen(buf));

  /* Print the HTTP response body */
  sprintf(buf, "<html><title>Tiny Error</title>");
  Rio_writen(fd, buf, strlen(buf));
  sprintf(buf, "<body bgcolor=""ffffff"">\r\n");
  Rio_writen(fd, buf, strlen(buf));
  sprintf(buf, "%s: %s\r\n", errnum, shortmsg);
  Rio_writen(fd, buf, strlen(buf));
  sprintf(buf, "<p>%s: %s\r\n", longmsg, cause);
  Rio_writen(fd, buf, strlen(buf));
  sprintf(buf, "<hr><em>The Tiny Web server</em>\r\n");
  Rio_writen(fd, buf, strlen(buf));
}
```

## `read_requesthdrs()`

```c
void read_requesthdrs(rio_t *rp) {
  char buf[MAXLINE];

  Rio_readlineb(rp, buf, MAXLINE);
  printf("%s", buf);
  while(strcmp(buf, "\r\n")) {
    Rio_readlineb(rp, buf, MAXLINE);
    printf("%s", buf);
  }
  return;
}
```

## `parse_uri()`

```c
int parse_uri(char *uri, char *filename, char *cgiargs) {
  char *ptr;

  if (!strstr(uri, "cgi-bin")) {  /* Static content */
    strcpy(cgiargs, "");
    strcpy(filename, "."); strcat(filename, uri);
    if (uri[strlen(uri)-1] == '/')   /* use the default filename */
      strcat(filename, "home.html"); /* which is `./home.html`   */
    return 1;
  }
  else {  /* Dynamic content */
    ptr = index(uri, '?');
    if (ptr) {
      strcpy(cgiargs, ptr+1);
      *ptr = '\0';
    }
    else 
      strcpy(cgiargs, "");
    strcpy(filename, "."); strcat(filename, uri);
    return 0;
  }
}
```

## `serve_static()`

```c
void serve_static(int fd, char *filename, int filesize) {
  int srcfd;
  char *srcp, filetype[MAXLINE], buf[MAXBUF];

  /* Send response headers to client */
  get_filetype(filename, filetype);
  sprintf(buf, "HTTP/1.0 200 OK\r\n");
  Rio_writen(fd, buf, strlen(buf));
  sprintf(buf, "Server: Tiny Web Server\r\n");
  Rio_writen(fd, buf, strlen(buf));
  sprintf(buf, "Content-length: %d\r\n", filesize);
  Rio_writen(fd, buf, strlen(buf));
  sprintf(buf, "Content-type: %s\r\n\r\n", filetype);
  Rio_writen(fd, buf, strlen(buf));

  /* Send response body to client */
  srcfd = Open(filename, O_RDONLY, 0);
  srcp = Mmap(0, filesize, PROT_READ, MAP_PRIVATE, srcfd, 0); // Section 9.8
  Close(srcfd);
  Rio_writen(fd, srcp, filesize);
  Munmap(srcp, filesize);
}
```

## `serve_dynamic()`

```c
void serve_dynamic(int fd, char *filename, char *cgiargs) {
  char buf[MAXLINE], *emptylist[] = { NULL };

  /* Return first part of HTTP response */
  sprintf(buf, "HTTP/1.0 200 OK\r\n"); 
  Rio_writen(fd, buf, strlen(buf));
  sprintf(buf, "Server: Tiny Web Server\r\n");
  Rio_writen(fd, buf, strlen(buf));

  if (Fork() == 0) { /* Child */
    /* Real server would set all CGI vars here */
    setenv("QUERY_STRING", cgiargs, 1);
    Dup2(fd, STDOUT_FILENO);         /* Redirect stdout to client */
    Execve(filename, emptylist, environ); /* Run CGI program */
  }
  Wait(NULL); /* Parent waits for and reaps child */
}
```

