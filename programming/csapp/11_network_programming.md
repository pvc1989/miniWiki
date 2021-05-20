---
title: 网络编程
---

# 1. 客户端-服务器模型

“客户端-服务器模型 (client-server model)”是所有网络应用的基础。

- 【服务器 (server)】管理资源、提供服务的进程。
- 【客户端 (client)】请求资源、使用服务的进程。

“交易 (transaction)”是客户端与服务器交互的基本操作，含以下四步：

1. 客户端向服务器发送“请求 (request)”。
2. 服务器接收该请求，并根据需要操纵其管理的资源。
3. 服务器向客户端发送“响应 (response)”，并等待下一个请求。
4. 客户端接收该响应，并根据需要处理之。

# 2. 网络

对一台主机而言，“网络 (newwork)”相当于另一种读写设备。

网络分类：

- 【局域网 (local area network, LAN)】目前最流行的局域网技术为“以太网 (Ethernet)”
  - 【以太网段 (Ethernet segment)】由多台主机通过以太网线（双绞线）连接到一台集线器所形成的小型网络，可覆盖一间或一层房屋。
    - 【以太网线 (Ethernet wire)】一端连接到主机上的“以太网适配器 (Ethernet adapter)”，另一端连接到集线器上的“端口 (port)”。
    - 【集线器 (hub)】只是将每个端口上的数据被动地复制到其他所有端口。因此连接到同一台集线器上的所有主机，都能看到相同的数据。
    - 每个以太网适配器都拥有一个长度为 48-bit 的物理地址，以区别于其他适配器。
    - 一台主机向其他主机发送的最小数据块称作“帧 (frame)”，其内容除“有效载荷 (payload)”外，还包括来源地址、目标地址、帧长度的“头标  (header)”。
  - 【桥接的以太网 (bridged Ethernet)】由多个以太网段连接到多个网桥所形成的中型网络，可覆盖整栋建筑或整个校园。
    - 【网桥 (bridge)】能够自动感知哪些主机可以连接到哪些端口，从而有选择地转发数据。
    - 网桥与网桥之间的带宽可达 1 Gb/s，网桥与集线器之间的带宽通常为 100 Mb/s。
- 【广域网 (wide area network, WAN)】
- 【互联网 (interconnected network, internet)】由多个局域网及广域网连接到多个路由器所形成的大型网络，可覆盖全球。
  - 【路由器 (router)】在多个局域网及广域网之间转发数据的设备。
  - 【协议 (protocol)】运行在主机及路由器上的程序，用于协调不同局域网及广域网技术。
    - 【命名格式 (naming scheme)】定义格式统一的主机地址。
    - 【发送机制 (delivery mechanism)】定义数据打包的方式。
      - 局域网帧 = 局域网头标 + 有效载荷（互联网包）
      - 互联网包 = 互联网头标 + 有效载荷（用户数据）

# 3. 全局 IP 因特网

“全局 IP 因特网 (global IP Internet)”，简称“因特网 (Internet)”是所有“互联网 (internet)”中最著名、最成功的一个。

因特网中的每台主机都运行着实现了“TCP/IP 协议”的软件。

- 【IP = Internet Protocol】提供最基本的命名格式，以及主机到主机的“不可靠 (unreliable)”传输机制。
- 【UDP = Unreliable Datagram Protocol】基于 IP 的简单扩展，提供进程到进程的不可靠传输。
- 【TCP = Transmission Control Protocol】基于 IP 的复杂扩展，提供进程到进程的双向可靠传输。

对于普通程序员，因特网可以理解为由世界上所有具备以下性质的主机所构成的互联网：

- 主机之集被映射到“IP 地址 (IP address)”之集。
- IP 地址之集被映射到“因特网域名 (Internet domain name)”之集。
- 一台主机上的进程可通过“连接 (connection)”与另一台主机上的进程“通信 (communicate)”。

## 3.1. IP 地址

IPv4 地址可以用 32-bit 无符号整数表示。由于历史的原因，它被定义为结构体：

```c
/* IP address structure */
struct in_addr {
    uint32_t s_addr; /* in network byte order (big-endian) */
};
```

TCP/IP 规定所有整数都采用“大端 (big-endian)”字节顺序。以下函数用于主机与网络字节顺序的转换：

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

【套接字 (socket)】连接的一端，用形如 `ip_address:port` 的“套接字地址”表示，其中 `port` 为 16-bit 的“端口号”。

- 客户端的端口号不固定，由操作系统内核自动分配。
- 常用服务的端口号相对固定，列于 `/etc/services` 中。
