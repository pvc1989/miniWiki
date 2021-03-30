---
title: 六：存储层次
---

# 1. 存储技术

## RAM (Random Access Memory)

### SRAM (Static RAM)

速度较快，但价格昂贵，用于“缓存 (cache)”。

SRAM 将每个位存储于具有两个稳定状态的“单元 (cell)”中，
后者由含有六个晶体管电路来实现，能够在通电时保持其所处的状态，对扰动不敏感。

### DRAM (Dynamic RAM)

速度较慢，但价格便宜，用于“主存 (main memory)”。

DRAM 将每个位以电荷形式存储于“电容 (capacitor)”中，后者对扰动很敏感。
因此，存储系统必须周期地读、写其中的信息，以避免扰动破坏 DRAM 中的信息。

### 传统 DRAMs

每块 DRAM 芯片被划分为 $d$ 个“超级单元 (supercell)”，每块超级单元含 $w$ 个存储单元，故整块 DRAM 芯片可存储 $d\times w$ 位信息。

所有超级单元被编为含 $r$ 行、$c$ 列（含 $d = r \times c$ 个元素）的二维数组。

信息通过“针脚 (pin)”传入、传出 DRAM：
- 【数据针脚】用于传输一个超级单元的数据。
- 【地址针脚】用于传输上述二维数组的地址。

读取第 $(r, c)$ 个超级单元中的数据分两步完成：
- 【RAS (row access strobe)】从二维数组读取第 $r$ 行数据，缓存于 DRAM 内部。
- 【CAS (column access strobe)】读取上述缓存区的第 $c$ 列数据。

### 存储模块

DRAM 芯片被打包为“存储模块 (memory modules)”，俗称“内存条”，后者可插入“主板 (motherboard)”的扩展槽中。

每个超级单元只能存储 1 字节信息，故每个 8 字节数据（`void *` 或 `double`）需由分布在 8 块芯片上、具有相同二维地址的 8 个超级单元来存储。

位于存储模块上的“存储控制器 (memory controller)”负责
- 将内存地址翻译为 DRAM 芯片的二维地址。
- 将分布存储的 8 个字节组合成完整的 8 字节数据。

### 增强型 DRAMs

- Fast page mode DRAM (FPM DRAM)
- Synchronous DRAM (SDRAM)
- Double Data-Rate Synchronous DRAM (DDR SDRAM)
  - DDR
  - DDR2
  - DDR3
- Video RAM (VRAM)

### 非易变存储

DRAMs 及 SRAMs 仅能在通电时保存信息，因此是“易变的 (volatile)”。
能够将信息保存到断电后的存储设备被称为“非易变的 (non-volatile)”，它们在历史上也被称为“只读存储 (read-only memory, ROM)”。

| ROM 类型 | 写入次数 |
|:-------:|:-------:|
| programmable ROM (PROM) | $1$ |
| erasable PROM (EPROM) | $10^3$ |
| electrically  EPROM (EEPROM) | $10^5$ |

### 访问主存

“总线 (bus)”是连接 CPU 与存储或读写设备的共享电路。
- “系统总线 (system bus)”连接 CPU 中的“总线接口 (bus interface)”与“读写桥 (I/O bridge)”。
- “存储总线 (memory bus)”连接“读写桥”与“主存 (main memory)”。
- “读写总线 (I/O bus)”连接“读写桥”与“读写设备 (I/O devices)”。

## 硬盘存储

“硬盘 (disk)”的存储量大（RAM 的数千倍），但读写速度慢（DRAM 的十万倍、SRAM 的百万倍）。

### 硬盘几何

- 【转轴 (spindle)】以固定速率转动，转速通常为每分钟数千至上万转。
- 【盘片 (platter)】上下“盘面 (surface)”覆盖磁性存储材料。
- 【磁道 (track)】盘面上的同心圆环。
- 【柱面 (cylinder)】同轴各盘面的所有直径相等的磁道。
- 【扇区 (sector)】磁道上的一段扇形区域，每个扇区通常存储 512 字节数据。
- 【间隔 (gap)】扇区之间用于识别扇区的区域。

### 硬盘容量

$$
容量 =
\frac{#字节}{扇区}\times
\frac{平均#扇区}{磁道}\times
\frac{#磁道}{盘面}\times
\frac{#盘面}{盘片}\times
\frac{#盘片}{硬盘}
$$


### 硬盘操作

- 【读写头 (read/write head)】
- 【作动臂 (actuator arm)】
- 【查找 (seek)】

- 【查找时间 (seek time)】移动摇臂使读写头位于所需磁道上方的时间，平均 3~9 ms。
- 【旋转延迟 (rotational latency)】所需扇区转到读写头下的时间，平均 30/RPM s，约 4 ms。
- 【传输时间 (transfer time)】远小于前两部分。

### 逻辑区块

“硬盘控制器 (disk controller)”
- 存储于硬盘固件中。
- 负责维护“逻辑区块 (logical block)”与“物理扇区 (physical sector)”之间的映射。

### 连接读写设备

读写总线被设计为独立于 CPU，并且被所有所有设备共享：
- 【USB (Universal Serial Bus)】连接到鼠标、键盘、闪存等设备。
- 【显卡 (graphics card)】连接到显示器。
- 【主机总线适配器 (host bus adapter)】连接到硬盘控制器。
- 【网络适配器 (network adapter)】连接到网络。

### 访问硬盘

硬盘读取步骤：
1. CPU 向硬盘所关联到地址写入命令、逻辑区块编号、数据存储地址，以发起硬盘读取。在硬盘执行读取时，CPU 转去执行其他指令。
1. 硬盘控制器从扇区读取数据，将数据直接写入主存。这种绕过 CPU 的内存访问方式，称为“直接内存访问 (direct memory access, DMA)”。
1. 当 DMA 完成后，硬盘控制器向 CPU 发出中断信号。CPU 收到该信号后，暂停执行其他指令。

## 固态硬盘

“固态硬盘 (solid state disk, SSD)”是一种基于“闪存 (flash memory)”技术的存储设备。

- 优点
  - 没有机械装置，更安静、更抗震。
  - 更省电。
  - 读写速度更快。
- 缺点
  - 充分长时间（约 30 年）使用后，可能“磨尽 (wear out)”。
  - 更贵。

## 发展趋势

- 不同存储技术在价格与性能之间寻求不同程度的平衡。
- 不同存储技术的价格及性能的发展速度不一致。
- DRAM 速度滞后于 CPU 速度。

# 2. 局部性

# 3. 存储层次

# 4. 缓存技术

# 5. 编写缓存友好的代码

# 6. 缓存对程序性能的影响
