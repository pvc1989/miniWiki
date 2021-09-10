---
Title: 计算机网络
---

# 0. 分层结构模型

## 0.1 七层 ISO/OSI 参考模型

## 0.2 四层 TCP/IP 模型

## 0.3 五层模型

# 1. 物理层

# 2. 数据链路层

# 3. 网络层

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
- 连接释放（四次握手）：
  1. Client 向 Server 发送*释放*报文段（`FIN=1, seq=u`）
  2. Server 向 Client 发送*确认*报文段（`ACK=1, seq=v, ack=u+1`），进入半关闭状态（Server 仍可向 Client 传输数据）
  3. Server 向 Client 发送*释放*报文段（`FIN=1, ACK=1, seq=w, ack=u+1`）
  4. Client 向 Server 发送*确认*报文段 `ACK=1, seq=u+1, ack=w+1`，再等待 2 MSL（最长报文段寿命的两倍）
     - 若 Server 未收到 Client 的确认报文段 [4]，则向其重传的释放报文段 [3]。
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

