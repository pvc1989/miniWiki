---
Title: OpenFOAM
---

# 构建

## `openfoam.com`

### 下载代码

首次下载：

```shell
git clone https://develop.openfoam.com/Development/openfoam.git
cd openfoam
git submodule init
```

后续更新：

```shell
git pull
git submodule update
```

### 第三方库

```shell
sudo apt update
sudo apt install build-essential autoconf autotools-dev cmake gawk gnuplot
sudo apt install flex libfl-dev libreadline-dev zlib1g-dev openmpi-bin libopenmpi-dev mpi-default-bin mpi-default-dev
sudo apt install libgmp-dev libmpfr-dev libmpc-dev
sudo apt install libscotch-dev libptscotch-dev libfftw3-dev libboost-system-dev libboost-thread-dev libcgal-dev
```

⚠️ 用上述方法安装 [Open MPI](https://www.open-mpi.org) 后，应当可以执行以下命令：<a href id="open-mpi"></a>

- `orterun --version` 应返回 Open MPI 的版本号。
- `mpicc --show` 应返回编译命令，OpenFOAM 将用它完成编译、链接。

### 配置环境

加载默认配置：

```shell
source <source_dir>/etc/bashrc  # 首次运行后，会设置 WM_PROJECT_DIR=<source_dir>
```

如果出现以下错误：

```
gcc: error: unrecognized command-line option '--showme:link'
```

或者要修改部分选项，可以在 `source $WM_PROJECT_DIR/etc/bashrc` 后面“追加 (append)”所要修改的选项（详见 `$WM_PROJECT_DIR/etc/bashrc` 中的注释），常用的有：

- 【`WM_PROJECT_USER_DIR=<user_dir>`】将默认的 `/home/<user>/OpenFOAM/<user>-<version>`  替换为用户指定的 `<user_dir>`。
- 【`WM_MPLIB=USERMPI`】用[本地搭建的 MPI 环境](../../programming/mpi/README.md)替换[系统自带的 Open MPI](#open-mpi)。根据 `$WM_PROJECT_DIR/etc/config.sh/mpi` 中的注释，用户需要在加载此项前设置好 `$WM_PROJECT_DIR/wmake/rules/General/mplibUSERMPI ` 文件。

示例：

```shell
foam  # 即 cd $WM_PROJECT_DIR
# 设置 USERMPI 的环境变量：
USERMPI_INSTALL=<usermpi_install>  # 指向 USERMPI 的安装目录
PATH=$USERMPI_INSTALL/bin:$PATH    # 指向 USERMPI 的 bin 目录
export PATH
# 设置 mplibUSERMPI 文件：
echo "PINC  = -I$USERMPI_INSTALL/include"    > wmake/rules/General/mplibUSERMPI
echo "PLIBS = -L$USERMPI_INSTALL/lib -lmpi" >> wmake/rules/General/mplibUSERMPI
# 加载修改过的配置：
source etc/bashrc WM_MPLIB=USERMPI WM_PROJECT_USER_DIR=$WM_PROJECT_DIR/../$WM_PROJECT_VERSION
```

为了在每次打开 shell 前自动设置环境变量，建议在 `~/.bashrc` 或 `~/.zshrc` 中加入以下命令：

```shell
PATH=$HOME/shared/mpich/install/bin:$PATH
export PATH
source $HOME/shared/openfoam.com/source/etc/bashrc WM_MPLIB=USERMPI
source $WM_PROJECT_DIR/etc/bashrc WM_MPLIB=USERMPI WM_PROJECT_USER_DIR=$WM_PROJECT_DIR/../$WM_PROJECT_VERSION
```

该示例假设：

- `$HOME/shared/mpich/install/` 为 MPICH 的安装目录（含 `bin`、`include`、`lib`、`share` 四个标准子目录）。
- `$HOME/shared/openfoam.com/source/` 为 OpenFOAM 的源代码目录。

### 编译运行

系统检查：

```shell
foamSystemCheck
```

应当返回：

```
Checking basic system...
-------------------------------------------------------------------------------
Shell:       zsh
[The zsh shell is generally okay to use]
Host:        host1
OS:          Linux version 5.11.0-18-generic
User:        common

System check: PASS
==================
Can continue to OpenFOAM installation.
```

编译 OpenFOAM：

```shell
./Allwmake -j -s -q  # compiles with all cores (-j), reduced output (-s, -silent), with queuing (-q, -queue)
foamInstallationTest
```

应当输出：

```
Executing foamInstallationTest

Basic setup :
-------------------------------------------------------------------------------
...

Summary
-------------------------------------------------------------------------------
Base configuration ok.
Critical systems ok.

Done
```

运行演示算例：

```shell
mkdir -p $FOAM_RUN
run  # 即 cd $FOAM_RUN
cp -r $FOAM_TUTORIALS/incompressible/simpleFoam/pitzDaily ./
cd pitzDaily
blockMesh   # 生成网格
simpleFoam  # 运行求解器
touch result.foam  # 用 ParaView 查看结果
```

## `openfoam.org`

# 并行

演示算例：[Supersonic flow over a forward-facing step](https://www.openfoam.com/documentation/tutorial-guide/3-compressible-flow/3.2-supersonic-flow-over-a-forward-facing-step)

## 单机

在 `1` 台机器上运行 `4` 个进程：

```shell
cp -r $FOAM_TUTORIALS/compressible/sonicFoam/laminar/forwardStep forwardStep_1x4
cd forwardStep_1x4
vim system decomposeDict
```



```cpp
/* system/decomposeDict */
```



```
decomposePar
mpiexec -n 4 sonicFoam
reconstructPar
```

## 分布式

在 `2` 台机器上分别运行 `20` 个进程：

```shell
cp -r $FOAM_TUTORIALS/compressible/sonicFoam/laminar/forwardStep forwardStep_2x40
cd forwardStep_2x20
vim system decomposeDict
echo "host1:20"  > hostlist
echo "host2:20" >> hostlist
mpiexec -n 40 -f hostlist sonicFoam
reconstructPar
```

