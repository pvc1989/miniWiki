# LaTeX

## 参考链接
- [慕子的知乎专栏](https://zhuanlan.zhihu.com/typography-and-latex)

## 发行版与编辑器

### TeX Live
[TeX Live](https://tug.org/texlive/) 是一款开源、跨平台的 TeX 发行版。

1. 下载并 *完整* 安装 [TeX Live](https://tug.org/texlive/acquire-mirror.html)，大约需要 5 GB 硬盘空间。
2. 安装完成后，应当可以用以下命令打开《[CTeX 宏集手册](https://ctan.org/pkg/ctex)》：
```shell
texdoc ctex
```

### TeXstudio
[TeXstudio](https://texstudio.org/) 是一款开源、跨平台的 TeX 编辑器。

## 中文支持
0. 卸载 CTeX 套装，原因参见[《2018年，为什么不推荐使用 CTeX 套装了》](https://zhuanlan.zhihu.com/p/45174503)。
1. 完整安装 [TeX Live](#TeX-Live)。
2. 创建中文文档 [`hello.tex`](./hello.tex)，用 `xelatex` 命令编译：
```shell
mkdir build
cd build
xelatex ../hello.tex
```

## 字体设置

### `fontspec`
[`fontspec` 宏包](https://ctan.org/pkg/fontspec)用于设置文档 *正文* 字体，默认情况下会影响到 `\mathrm` 等 *数学* 字体：
```latex
\usepackage{fontspec}
\setmainfont{Courier}
```
宏包选项  `no-math`  可以消除对数学字体的影响。

### `mathspec`
[`mathspec` 宏包](https://ctan.org/pkg/fontspec)用于设置（数学环境中）*数字*、*拉丁字母*、*希腊字母* 的字体，默认情况下以 `no-math` 选项加载 [`fontspec`](#`fontspec`)：

```latex
\usepackage{mathspec}
\setmainfont{Palatino}
\setmathsfont{Courier}
\setmathrm{Optima}
```

[CTeX 文档类](https://ctan.org/pkg/ctex)会自动加载 [`fontspec`](#`fontspec`)。
如果要使用 [`mathspec`](#`mathspec`)，则应在 `\documentclass` 之前为  `fontspec` 传入 `no-math` 选项：

```latex
\PassOptionsToPackage{no-math}{fontspec}
\documentclass{ctexart}
\usepackage{mathspec}
```
或者选择其他（不自动加载 [`fontspec`](#`fontspec`) 的）文档类，而在 `mathspec` 之后加载 `ctex`：
```latex
\documentclass{article}
\usepackage{mathspec}
\usepackagep[heading]{ctex}
```

### `unicode-math`
[`unicode-math` 宏包](https://ctan.org/pkg/unicode-math)用于设置数学符号字体：
```latex
\usepackage{unicode-math}
\unimathsetup{math-style=TeX}
\setmathfont{texgyrepagella-math.otf}
\setmathfont{Neo-Euler}[range=\mathalpha]
```

## 数学物理符号
### `physics`

### `siunitx`
```latex
$ \SI[<options>]{<number>}[<preunit>]{<unit>} $
$ R = \SI{8.3144598(48)}{J.mol^{-1}.K^{-1}} $
$ R = \SI{8.3144598(48)}{\joule\per\kelvin\per\mole} $
```

## 绘图
### `tikz`

# LyX
[LyX](https://lyx.org) 是一款开源、支持「所见即所思」的 [LaTeX](./README.md) 前端，兼具 LaTeX 排版效果优美和 Word「所及即所得」的优势。

## 中文支持
### 环境配置
1. 完整安装 [TeX Live](#TeX-Live)。
2. 安装 [LyX](https://www.lyx.org/Download)。
3. 在 LyX「首选项」中设置环境变量 `PATH`，确保第 1 步所安装的可执行文件（例如 `xelatex` 命令）能够被 LyX 搜索到。

### 文档设置
打开 LyX，新建一个文档（「文档类」可任选），然后进入「文档」→「首选项」进行如下设置：
1. 「LaTeX 导言区」→ `\usepackage{ctex}`。
2. 「语言」→「语言」→「汉语 (简体中文)」。
3. 「语言」→「文件编码」→「其他」→「Unicode (XeTeX) (utf8)」。
4. 「字体」→「使用非 TeX 字体（通过 XeTeX/LuaTeX）」。

### 字体设置

[CTeX 系列文档类](https://ctan.org/pkg/ctex)以及上一节第 4 步的「使用非 TeX 字体」会自动加载 [`fontspec`](#`fontspec`)。
如果进一步勾选「数学：非 TeX 字体默认值」，则还会自动加载 [`unicode-math`](#`unicode-math`)。

如果要进行更精细的字体设置，则不应勾选「使用非 TeX 字体（通过 XeTeX/LuaTeX）」，而是在「LaTeX 导言区」中[手动加载字体设置宏包](#字体设置)。

## 代码高亮

### 环境配置
1. 安装 `pygments`：
```shell
pip install pygments
pip3 install pygments
```

### 文档设置
进入「文档」→「首选项」进行如下设置：
1. 「程序列表」→「语法高亮支持包」→「Minted」
2. 「程序列表」→「空白处」可以设置 `minted` 宏包参数，例如：
```latex
style=xcode
frame=leftline
baselinestretch={1.0}
breaklines=true
fontsize={\footnotesize}
```
3. 「输出格式」→「允许运行外部程序」。
4. 首次编译该文档时，会弹出警告对话框，选择「对该文档总是允许」。

### 插入代码文件

1. 「插入」→「文件」→「子文档」，弹出对话框。
2. 「文件」后的空白处输入源文件地址，或点击「浏览」选择源文件。
3. 「包含类别」选择「程序列表」。
4. 「更多参数」右侧的空白处添加语言，例如  `language=python`。
