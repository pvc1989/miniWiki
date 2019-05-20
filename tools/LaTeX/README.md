# LaTeX --- 结构化文档排版系统

## 参考资料
- [慕子的知乎专栏](https://zhuanlan.zhihu.com/typography-and-latex)

## TeXLive
[TeXLive](https://tug.org/texlive/) 是一款开源、跨平台的 TeX 发行版。

### 中文支持
0. 卸载 [CTeX 套装](http://ctex.org)，原因参见[《2018年，为什么不推荐使用 CTeX 套装了》](https://zhuanlan.zhihu.com/p/45174503)。
1. 完整安装 [TeXLive](https://tug.org/texlive/)，大约需要 5 GB 硬盘空间。
2. 安装完成后，应当可以用以下命令打开《CTeX 宏集手册》：
```shell
texdoc ctex
```
3. 创建中文文档 [`hello.tex`](./hello.tex)，用 `xelatex` 命令编译：
```shell
mkdir build
cd build
xelatex ../hello.tex
```

## TeXStudio
[TeXStudio](https://texstudio.org/) 是一款开源、跨平台的 TeX 编辑器。

## LyX
[LyX](https://lyx.org) 是一款开源、支持 *所见即所思* 的 LaTeX 前端，兼具 LaTeX 排版效果优美和 Word *所及即所得* 的优势。

### 中文支持
#### 环境配置
1. 完整安装 TeXLive。
2. 安装 [LyX](https://www.lyx.org/Download)。
3. 在 LyX `首选项` 中设置环境变量 `PATH`，确保第 1 步所安装的可执行文件（例如 `xelatex` 命令）能够被 LyX 搜索到。

#### 文档设置
打开 LyX，新建一个文档（`文档类` 可任选），然后进入 `文档` → `首选项` 进行如下设置：
1. `LaTeX 导言区` → 添加 `\usepackage{ctex}`。
2. `语言` → `语言` → 选择 `汉语 (简体中文)`。
2. `语言` → `文件编码` → `其他` → 选择 `Unicode (XeTeX) (utf8)`。
3. `字体` → 勾选 `使用非 TeX 字体（通过 XeTeX/LuaTeX）`。

### 代码高亮

#### 环境配置
1. 安装 `pygments`：
```shell
pip install pygments
pip3 install pygments
```

#### 文档设置
进入 `文档` → `首选项` 进行如下设置：
1. `程序列表` → `语法高亮支持包` → 勾选 `Minted` 
2. `程序列表` → `空白处` 可以设置 `minted` 宏包参数，例如：
```latex
style=xcode
frame=leftline
baselinestretch={1.0}
breaklines=true
fontsize={\footnotesize}
```
3. `输出格式` → 勾选 `允许运行外部程序`。
4. 首次编译该文档时，会弹出警告对话框，选择 `对该文档总是允许`。
