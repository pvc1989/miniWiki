---
title: 计算机网络层次结构
---

# 0. 层次结构模型

## 0.1 七层 ISO/OSI 参考模型

## 0.2 四层 TCP/IP 模型

## 0.3 五层模型

# 1. 物理层

# 2. 数据链路层

# 3. 网络层

## 3.1 路由算法

分类

- 静态路由算法：需管理员手工配置，适用于拓扑变化不大的小型网络。
- 动态路由算法：路由器之间彼此交换信息，自适应地算出路由表项，适用于大型网络。
  - 全局性～：所有路由器掌握完整网络拓扑和链路费用信息，e.g. 链路状态路由算法
  - 分散性～：每个路由器只掌握物理相连的邻居及链路费用，e.g. 距离向量路由算法

路由选择协议

- 内部网关协议：RIP 或 OSPF
- 外部网关协议：BGP-4

## 3.2 IPv4

### 数据报

```c
struct IpHeader {
  /* [0, 4) */
  uint4_t version, header_length/* 4-Byte */;
  uint8_t service;
  uint16_t packet_length/* 1-Byte */;
  /* [4, 8) */
  uint16_t identifier;
  uint3_t tag;
  uint13_t offset/* 8-Byte */;
  /* [8, 12) */
  uint8_t life_time/* -- at each router */;
  uint8_t protocal/* TCP = 6, UDP = 17 */;
  uint16_t header_check_sum;
  /* [12, 16) */
  uint32_t source_ip;
  /* [16, 20) */
  uint32_t target_ip;
  /* [20, ..) */
  int[] optional_and_padding;
};
```

受数据链路层 MTU 限制，较长的 IP 数据报将被分为若干片：

- 标识：同一数据报的所有分片使用同一标识
- 标志：最高位不用；中间位记作 `DF`，表示 Don't Fragment；最低位记作 `MF`，表示 More Fragments。
- 片偏移：某片在原分组中的相对位置，以 8-Byte 为单位。

### 地址

IP 地址 = 网络号 + 主机号

特殊 IP 地址：

- 【默认路由】网络号（剩余部分）全为 `0`，主机号全为 `0`，只可作源
- 【特定主机】网络号（剩余部分）全为 `0`，主机号为特定值，只可作目标
- 【网络名称】网络号为特定值，主机号全为 `0` 
- 【在本网内广播】网络号（剩余部分）全为 `1`，主机号全为 `1`，只可作目标
- 【对某网络广播】网络号为特定值，主机号全为 `1`，只可作目标
- 【回环地址】网络号为 `127`，主机号不全为 `0` 且不全为 `1`

|    类型    |     A      |      B       |      C       |      D       |      E       |
| :--------: | :--------: | :----------: | :----------: | :----------: | :----------: |
| 网络号大小 |   8-bit    |    16-bit    |    24-bit    |              |              |
|   前几位   |    `0`     |     `10`     |    `110`     |    `1110`    |    `1111`    |
| 首字节范围 | `[1, 127)` | `[128, 192)` | `[192, 224)` | `[224, 240)` | `[240, 255)` |
| 可用网段数 |  $2^7-2$   |  $2^{14}-2$  |  $2^{21}-2$  |              |              |
| 最大主机数 | $2^{24}-2$ |  $2^{16}-2$  |  $2^{8}-2$   |              |              |

私有地址

- 【A 类】`10.*.*.*`，共 1 个网段
- 【B 类】`172.16.*.*`~`172.31.*.*`，共 16 个网段
- 【C 类】`192.168.0.*`~`192.168.255.*`，共 256 个网段

### NAT

【网络地址转换 (Network Address Translation)】

- 【WAN 口】`global_ip:global_port`
- 【LAN 口】`local_ip:local_port`

### 子网

- 两级 IP 地址：网络号 + 主机号（不全为 `0` 且不全为 `1`）
  - 子网掩码：A 类 `255.0.0.0`，B 类 `255.255.0.0`，C 类 `255.255.255.0`
- 三级 IP 地址：网络号 + 子网号 + 主机号（不全为 `0` 且不全为 `1`）
  - 例：IP 地址 `141.14.72.24`，子网掩码 `255.255.192.0`，网络地址 `141.14.64.0`

## 3.3 IPv6

## 3.4 路由协议

# 4. 传输层

提供

- 进程与进程之间的逻辑通信
- 复用、分用
- 差错检测

## 4.1 UDP

特点：无连接、首部开销小、尽最大努力交付（不可靠）

### 首部格式

```c
struct UdpHeader {
  uint16_t source_port, target_port;
  uint16_t datagram_length, check_sum;
};
```

### 校验方式

```c
struct PseudoHeader {
  uint32_t source_ip, target_ip;
  uint8_t zero = 0;
  uint8_t protocal = 17;  /* UDP */
  uint16_t datagram_length;
};
```

将*伪首部 + UDP 报文段*视为 `uint16_t[]`，以（二进制）*反码之和的反码*作为**校验和**：

- 发送端：计算校验和时，以 `0` 作为 `udp_header.check_sum` 的值。
- 接收端：所得校验和各位全为 `1` 则无差错。

## 4.2 TCP

特点：有连接、一对一、可靠交付、全双工、面向字节流

### 首部格式

```c
struct TcpHeader {
  uint16_t source_port, target_port;
  uint32_t ack/* acknowledge number */;
  uint32_t seq/* sequence number */;
  uint4_t header_length/* 以 4B 为长度 */; uint6_t reserved;
  uint1_t URG/* URGent */, ACK/* ACKnowledge */, PSH/* PuSH */;
  uint1_t RST/* ReSeT */, SYN/* SYNchronize*/, FIN/* FINish */;
  uint16_t receive_window;
  uint16_t check_sum, urgent_ptr;
  uint32_t optional[];
};
```

### 连接管理

- 连接请求（三次握手）：
  1. Client 向 Server 发送*请求*报文段（`SYN=1, seq=x`）
  2. Server 向 Client 发送*确认*报文段（`SYN=1, ACK=1, seq=y, ack=x+1`）
  3. Client 向 Server 发送*确认*报文段（`ACK=1, seq=x+1, ack=y+1`），可携带数据
- 连接释放（四次挥手）：
  1. Client 向 Server 发送*释放*报文段（`FIN=1, seq=u`）
  2. Server 向 Client 发送*确认*报文段（`ACK=1, seq=v, ack=u+1`），进入半关闭状态（Server 仍可向 Client 传输数据）
  3. Server 向 Client 发送*释放*报文段（`FIN=1, ACK=1, seq=w, ack=u+1`）
  4. Client 向 Server 发送*确认*报文段 `ACK=1, seq=u+1, ack=w+1`，再等待 2 MSL（最长报文段寿命的两倍）
     - 若 Server 未收到 Client 的确认报文段 [4]，则向其重传释放报文段 [3]。
     - 若 Client 在 2 MSL 内未收到 Server 重传的报文段 [3]，则彻底释放连接。

### 可靠传输

- 校验（同 UDP）
- 序号
  - 字节序号：该字节在用户数据（字节流）中的序号
  - 报文段序号：该报文段的用户数据的首字节的序号
- 确认
  - 确认号：期望收到对方的下一个报文段的（用户数据首字节）序号
  - 累积确认：只确认数据流中截至第一个丢失字节的部分（即序号取自 `[0, ack)` 的字节）
- 重传
  - 超时重传：超过*重传时间*（略大于 RTT 的动态均值）仍未收到确认
  - 快速重传：发送端收到同一个报文段的三个*冗余确认*

### 流量控制

匹配接收端读取速率与发送端发送速率

- 接收端：接收端缓存大小、读取速率有限，通过 `tcp_header` 向发送端许诺**接收窗口 (receive window, `rwnd`)**
- 发送端：以 `min(rwnd, cwnd)` 为**发送窗口 (send window, `swnd`)** 的上限

### 拥塞控制

发送端按以下四种算法（窗口单位为 MSS，时间单位为 RTT）计算**拥塞窗口 (congestion window, `cwnd`)**：

1. 慢开始：令 `cwnd` 从 `1` 开始（试探网络性能）指数增长 (`*= 2`)；若 `cwnd >= ss_thresh`，则改用*拥塞避免 [2]*。
2. 拥塞避免：令 `cwnd` 线性增长 (`+= 1`)；若出现超时，则判定拥塞，取 `ss_thresh = cwnd / 2`，改用*慢开始 [1]*。
3. 快重传：若收到三个冗余确认（触发快速重传），则（相对于超时提前）判定拥塞。
4. 快恢复：取 `ss_thresh = cwnd / 2`，再令 `cwnd` 从 `ss_thresh` 开始线性增长。

# 5. 应用层

