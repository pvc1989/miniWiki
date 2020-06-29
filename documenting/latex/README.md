# LaTeX

## 发行版 + 编辑器

### TeX Live
[TeX Live](https://tug.org/texlive/) 是一款开源、跨平台的 TeX「发行版 (distribution)」。

1. 下载并 *完整* 安装 [TeX Live](https://tug.org/texlive/acquire-mirror.html)，大约需要 5 GB 硬盘空间。
2. 安装完成后，应当可以用以下命令打开 [`ctex` 宏包](https://ctan.org/pkg/ctex)的文档《[CTeX 宏集手册](https://ctan.org/pkg/ctex)》：
   ```shell
   texdoc ctex
   ```

### TeXstudio
[TeXstudio](https://texstudio.org/) 是一款开源、跨平台的 TeX「编辑器 (editor)」。

### LyX

[LyX](https://lyx.org) 是一款开源、支持「所见即所思」的 LaTeX 前端，兼具 LaTeX 排版效果优美和 Word「所及即所得」的优势。

先完整安装 [TeX Live](#TeX-Live)，再安装 [LyX](https://www.lyx.org/Download)。二者都安装好后，在 LyX「首选项」中设置环境变量 `PATH`，确保 `xelatex` 等命令能够被 LyX 搜索到。

- [LyX 中文支持](#LyX-中文支持)
- [LyX 字体设置](#LyX-字体设置)
- [LyX 代码高亮](#LyX-代码高亮)

## 中文支持

### `ctex`

1. 卸载 CTeX 套装，原因参见《[2018年，为什么不推荐使用 CTeX 套装了](https://zhuanlan.zhihu.com/p/45174503)》。
1. 完整安装 [TeX Live](#TeX-Live)。
1. 从以下两种方式中任选一种，创建中文文档 [`hello.tex`](./hello.tex)：
  - CTeX 文档类，例如 `ctexart`、`ctexrep`、`ctexbook`。
  - 其他文档类 +  [CTeX 宏包](https://ctan.org/pkg/ctex)。
1. 用 `xelatex` 命令编译：
   ```shell
   mkdir build
   cd build
   xelatex ../hello.tex
   ```

### LyX 中文支持

新建一个 LyX 文档（「文档类」可任选），然后进入「文档」→「首选项」进行如下设置：

1. 「LaTeX 导言区」→ `\usepackage{ctex}`。
2. 「语言」→「语言」→「汉语 (简体中文)」。
3. 「语言」→「文件编码」→「其他」→「Unicode (XeTeX) (utf8)」。
4. 「字体」→「使用非 TeX 字体（通过 XeTeX/LuaTeX）」。

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

### LyX 字体设置

[CTeX 系列文档类](https://ctan.org/pkg/ctex)以及《[LyX 中文支持](#LyX-中文支持)》一节第 4 步的「使用非 TeX 字体」会自动加载 [`fontspec`](#`fontspec`)。
如果进一步勾选「数学：非 TeX 字体默认值」，则还会自动加载 [`unicode-math`](#`unicode-math`)。

如果要进行更精细的字体设置，则不应勾选「使用非 TeX 字体（通过 XeTeX/LuaTeX）」，而是在「LaTeX 导言区」中[手动加载字体设置宏包](#字体设置)。

## 数学物理符号

### 符号类型与间距
TeX 将数学符号分为以下几类：

|    类型     |     命令     |                      示例                       |
| :---------: | :----------: | :---------------------------------------------: |
|  Ordinary   |  `\mathord`  |                       $a$                       |
|  Operator   |  `\mathop`   |                     $\sum$                      |
|   Binary    |  `\mathbin`  |                    $\times$                     |
|  Relation   |  `\mathrel`  |                      $\le$                      |
|   Opening   | `\mathopen`  |                    $\biggl($                    |
|   Closing   | `\mathclose` |                    $\biggr)$                    |
| Punctuation | `\mathpunct` |                       $,$                       |
|    Inner    | `\mathinner` | $\dfrac12$ 或 $\left(\phantom{\dfrac12}\right)$ |

并将它们之间的距离定义为（其中 * = 不可能，0 = 无间距，1 = `\thinmuskip`，2 = `\mediumskip`，3 = `\thickmuskip`，( ) = 在上下标模式中忽略）

|       | Ord  |  Op  | Bin  | Rel  | Open | Close | Punct | Inner |
| ----: | :--: | :--: | :--: | :--: | :--: | :---: | :---: | :---: |
|   Ord |  0   |  1   | (2)  | (3)  |  0   |   0   |   0   |  (1)  |
|    Op |  1   |  1   |  *   | (3)  |  0   |   0   |   0   |  (1)  |
|   Bin | (2)  | (2)  |  *   |  *   | (2)  |   *   |   *   |  (2)  |
|   Rel | (3)  | (3)  |  *   |  0   | (3)  |   0   |   0   |  (3)  |
|  Open |  0   |  0   |  *   |  0   |  0   |   0   |   0   |   0   |
| Close |  0   |  1   | (2)  | (3)  |  0   |   0   |   0   |  (1)  |
| Punct | (1)  | (1)  |  *   | (1)  | (1)  |  (1)  |  (1)  |  (1)  |
| Inner | (1)  |  1   | (2)  | (3)  | (1)  |   0   |  (1)  |  (1)  |

比较以下写法（注意括号前后的间距）：

|                       示例                       |             缺点             |
| :----------------------------------------------: | :--------------------------: |
|            $f ( \dfrac{x^2}{2} ) dx$             | 括号尺寸偏小，微分号前缺间距 |
|       $f \left( \dfrac{x^2}{2} \right) dx$       | 函数名与开括号之间有多余间距 |
|     $f \biggl( \dfrac{x^2}{2} \biggr) \, dx$     |      括号尺寸需手动调整      |
| $f \mathopen{} \left( \dfrac{x^2}{2} \right) dx$ |   需手动插入 `\mathopen{}`   |

### `physics`

### `siunitx`
```latex
$ \SI[<options>]{<number>}[<preunit>]{<unit>} $
$ R = \SI{8.3144598(48)}{J.mol^{-1}.K^{-1}} $
$ R = \SI{8.3144598(48)}{\joule\per\kelvin\per\mole} $
```

## 代码高亮

### `minted` + `pygments`

- `minted` 是 LaTeX 宏包，若完整安装了 [TeX Live](#TeX-Live)，则无需单独安装。
- `pygments` 是 Python 第三方包，一般需要单独安装：

```shell
pip install pygments
pip3 install pygments
```

### 插入外部代码文件
- `import` 宏包用于插入文件。
- `minted` 宏包用于代码高亮。
- 二者混用可能会引发路径错误，修复方案可参考《[宏包 import 对 minted 无效](https://zhuanlan.zhihu.com/p/39117864)》。

### LyX 代码高亮

进入「文档」→「首选项」进行如下设置：

1. 「程序列表」→「语法高亮支持包」→「Minted」。
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

在 LyX 文件中插入外部代码文件的步骤如下：
1. 「插入」→「文件」→「子文档」，弹出对话框。
2. 「文件」后的空白处输入源文件地址，或点击「浏览」选择源文件。
3. 「包含类别」选择「程序列表」。
4. 「更多参数」右侧的空白处添加语言，例如  `language=python`。

## 绘图

### `tikz`

## 参考资料
### 链接
- [CTAN (The Comprehensive TeX Archive Network)](https://ctan.org/)
- [TeX - LaTeX Stack Exchange](https://tex.stackexchange.com/)
- [慕子](https://www.zhihu.com/people/muzzi)的[知乎专栏](https://zhuanlan.zhihu.com/typography-and-latex)
  - [(La)TeX 世界的几个层次](https://zhuanlan.zhihu.com/p/45507356)
  - [LaTeX，一个多义词](https://zhuanlan.zhihu.com/p/45996183)
  - [2018年，为什么不推荐使用 CTeX 套装了](https://zhuanlan.zhihu.com/p/45174503)
  - [2019 年，不用 CTEX 套装的新理由](https://zhuanlan.zhihu.com/p/73304856)
- [常用参考文档](https://zhuanlan.zhihu.com/p/43938945)

### 书籍
- 《The TeXbook》Knuth 1984
- 《LaTeX: a Document Preparation System》Lamport 1994
- 《The LaTeX Companion》Mittelbach 2005
- [《More Math into LaTeX》Grätzer  2016](https://doi.org/10.1007/978-3-319-23796-1)
- 《[TeX](https://en.wikibooks.org/wiki/TeX)》and 《[LaTeX](https://en.wikibooks.org/wiki/LaTeX)》on Wikibooks

### ⚠️ 信息污染源
- 百度搜索
- 中文博客
- 祖传文档
