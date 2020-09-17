---
title: LaTeX
---

# LaTeX<a name="LaTeX"></a>
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
- 《[More Math into LaTeX](https://doi.org/10.1007/978-3-319-23796-1)》Grätzer  2016
- 《[TeX](https://en.wikibooks.org/wiki/TeX)》and 《[LaTeX](https://en.wikibooks.org/wiki/LaTeX)》on Wikibooks

### ⚠️ 信息污染源
- 百度搜索
- 中文博客
- 祖传文档

## 发行版 + 编辑器

### TeX Live<a name="TeX-Live"></a>
[TeX Live](https://tug.org/texlive/) 是一款开源、跨平台的 TeX ***发行版 (distribution)***。

1. 下载并 *完整* 安装 [TeX Live](https://tug.org/texlive/acquire-mirror.html)，大约需要 5 GB 硬盘空间。
2. 安装完成后，应当可以用 `texdoc ctex` 命令打开《[CTeX 宏集手册](https://ctan.org/pkg/ctex)》。

### TeXstudio
[TeXstudio](https://texstudio.org/) 是一款开源、跨平台的 TeX ***编辑器 (editor)***。

## 中文支持

### CTeX

1. 卸载 [CTeX 套装](#http://www.ctex.org/CTeX)，原因参见《[2018年，为什么不推荐使用 CTeX 套装了](https://zhuanlan.zhihu.com/p/45174503)》。
1. 完整安装 [TeX Live](#TeX-Live)。
1. 从以下两种方式中任选一种，创建中文文档 [`hello.tex`](./hello.tex)：
  - CTeX 文档类，例如 `ctexart`、`ctexbook` 等。
  - 其他文档类 + [CTeX 宏包](https://ctan.org/pkg/ctex)。
1. 用 `xelatex` 命令编译：
   ```shell
   mkdir build
   cd build
   xelatex ../hello.tex
   ```

## 字体设置

### fontspec
[fontspec 宏包](https://ctan.org/pkg/fontspec) 用于设置文档 *正文* 字体，默认情况下会影响到 `\mathrm` 等 *数学* 字体（宏包选项 `no-math` 可以消除对数学字体的影响）：

```latex
\usepackage[no-math 宏包]{fontspec}
\setmainfont{Courier}
```

### mathspec
[mathspec 宏包](https://ctan.org/pkg/fontspec) 用于设置（数学环境中）*阿拉伯数字*、*拉丁字母*、*希腊字母* 的字体，默认情况下以 `no-math` 选项加载 [fontspec 宏包](#fontspec)：

```latex
\usepackage{mathspec}
\setmainfont{Palatino}
\setmathsfont{Courier}
\setmathrm{Optima}
```

[CTeX 文档类](https://ctan.org/pkg/ctex) 会自动加载 [fontspec 宏包](#fontspec)。
如果要使用 [mathspec 宏包](#mathspec)，则应
- 在 `\documentclass` 之前将 `no-math` 选项传入 [fontspec 宏包](#fontspec)：
  ```latex
  \PassOptionsToPackage{no-math}{fontspec}
  \documentclass{ctexart}
  \usepackage{mathspec}
  ```
- 或者选择其他（不自动加载 [fontspec 宏包](#fontspec) 的）文档类，而在 [mathspec 宏包](#mathspec) 之后加载 ctex：
  ```latex
  \documentclass{article}
  \usepackage{mathspec}
  \usepackagep[heading]{ctex}
  ```

### unicode-math
[unicode-math 宏包](https://ctan.org/pkg/unicode-math) 用于设置 *数学符号* 字体：
```latex
\usepackage{unicode-math}
\unimathsetup{math-style=TeX}
\setmathfont{texgyrepagella-math.otf}
\setmathfont{Neo-Euler}[range=\mathalpha]
```
*完整* 安装 [TeX Live](#TeX-Live) 后，应当可以用 `texdoc unimath` 命令打开《Every symbol (most symbols) defined by `unicode-math`》。

## 数学物理符号

### 符号类型与间距
TeX 将数学符号分为以下几类：

|    类型     |     命令     |          示例           |          代码         |
| :---------: | :----------: | :---------------------: | :---------------------: |
|  Ordinary   |  `\mathord`  |           $a$           |           `a`           |
|  Operator   |  `\mathop`   |         $\sum$          |         `\sum`          |
|   Binary    |  `\mathbin`  |        $\times$         |        `\times`         |
|  Relation   |  `\mathrel`  |          $\le$          |          `\le`          |
|   Opening   | `\mathopen`  |        $\biggl($        |        `\biggl(`        |
|   Closing   | `\mathclose` |        $\biggr)$        |        `\biggr)`        |
| Punctuation | `\mathpunct` |           $,$           |           `,`           |
|    Inner    | `\mathinner` | $\left(\dfrac12\right)$ | `\left(\dfrac12\right)` |

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

|                       示例                       |                       代码                       |             缺点             |
| :----------------------------------------------: | :----------------------------------------------: | :--------------------------: |
|            $f ( \dfrac{x^2}{2} ) dx$             |            `f ( \dfrac{x^2}{2} ) dx`             | 括号尺寸偏小，微分号前缺间距 |
|       $f \left( \dfrac{x^2}{2} \right) dx$       |       `f \left( \dfrac{x^2}{2} \right) dx`       | 函数名与开括号之间有多余间距 |
|     $f \biggl( \dfrac{x^2}{2} \biggr) \, dx$     |     `f \biggl( \dfrac{x^2}{2} \biggr) \, dx`     |      括号尺寸需手动调整      |
| $f \mathopen{} \left( \dfrac{x^2}{2} \right) dx$ | `f \mathopen{} \left( \dfrac{x^2}{2} \right) dx` |   需手动插入 `\mathopen{}`   |

### physics

### siunitx
```latex
$ \SI[<options>]{<number>}[<preunit>]{<unit>} $
$ R = \SI{8.3144598(48)}{J.mol^{-1}.K^{-1}} $
$ R = \SI{8.3144598(48)}{\joule\per\kelvin\per\mole} $
```

## 代码高亮

### minted + pygments

- [minted 宏包](https://ctan.org/pkg/minted) 是 LaTeX 宏包，若完整安装了 [TeX Live](#TeX-Live)，则无需单独安装。
- pygments 是 Python 第三方包，一般需要单独安装：

```shell
$ pip  install pygments
$ pip3 install pygments
```

### 插入外部代码文件
- [import 宏包](https://ctan.org/pkg/import) 用于插入文件。
- [minted 宏包](https://ctan.org/pkg/minted) 用于代码高亮。
- 二者混用可能会引发路径错误，修复方案可参考《[宏包 import 对 minted 无效](https://zhuanlan.zhihu.com/p/39117864)》。

## 绘图

### tikz

# LyX<a name="LyX"></a>

[LyX](https://lyx.org) 是一款开源、支持“所见即所思”的 LaTeX 前端，兼具 LaTeX 排版效果优美和 Word “所及即所得”的优势。

先完整安装 [TeX Live](#TeX-Live)，再安装 [LyX](https://www.lyx.org/Download)。二者都安装好后，在 LyX【首选项】中设置环境变量 `PATH` 以确保 `xelatex` 等命令能够被 LyX 搜索到。

## 中文支持<a name="LyX-中文支持"></a>

新建一个 LyX 文档，其【文档类】可任选，然后进入【文档】→【首选项】进行如下设置：

1. 【LaTeX 导言区】中增加一行 `\usepackage{ctex}`
2. 【语言】→【语言】→【汉语（简体中文）】
3. 【语言】→【文件编码】→【其他】→【Unicode (XeTeX) (utf8)】
4. 【字体】→【使用非 TeX 字体（通过 XeTeX/LuaTeX）】

## 字体设置<a name="LyX-字体设置"></a>

[CTeX 系列文档类](https://ctan.org/pkg/ctex) 及《[LyX 中文支持](#LyX-中文支持)》一节第 4 步的【使用非 TeX 字体】都会自动加载 [fontspec 宏包](#fontspec)。
如果进一步勾选【数学：非 TeX 字体默认值】，则还会自动加载 [unicode-math 宏包](#unicode-math)。

如果要进行更精细的字体设置，则不应勾选【使用非 TeX 字体】，而是在【LaTeX 导言区】中手动加载[字体设置](#字体设置)宏包。

## 代码高亮<a name="LyX-代码高亮"></a>

进入【文档】→【首选项】进行如下设置：

1. 【程序列表】→【语法高亮支持包】→【Minted】。

2. 【程序列表】→【空白处】可以设置 minted 宏包参数，例如：

   ```latex
   style=xcode
   frame=leftline
   baselinestretch={1.0}
   breaklines=true
   fontsize={\footnotesize}
   ```

3. 【输出格式】→【允许运行外部程序】。

4. 首次编译该文档时，会弹出警告对话框，选择【对该文档总是允许】。

在 LyX 文件中插入外部代码文件的步骤如下：

1. 【插入】→【文件】→【子文档】，弹出对话框。
2. 【文件】后的空白处输入源文件地址，或点击【浏览】选择源文件。
3. 【包含类别】选择【程序列表】。
4. 【更多参数】右侧的空白处添加语言，例如  `language=python`。

# MathJax<a name="MathJax"></a>

## 常用链接

- [主页](https://www.mathjax.org/)
- [文档](https://docs.mathjax.org/en/latest/index.html)
- [源码](https://github.com/mathjax/MathJax-src)
- [组件](https://github.com/mathjax/MathJax)

## [配置、加载](https://docs.mathjax.org/en/latest/web/configuration.html)

【方法一】直接在网页（HTML 文件）内配置、加载：

```html
<script>
MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']]
  },
};
</script>
<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js">
</script>
```

【方法二】将配置写入 `mathjax-config.js` 文件，再由网页依次加载此文件及 MathJax 组件：

```js
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']]
  },
  svg: {
    fontCache: 'global'
  }
};
```

```html
<script src="mathjax-config.js" defer></script>
<script type="text/javascript" id="MathJax-script" defer
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js">
</script>
```

【方法三】将配置、加载均写入 `mathjax-load.js` 文件，再由网页加载此文件：

```js
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']]
  },
  svg: {
    fontCache: 'global'
  }
};

(function () {
  var script = document.createElement('script');
  script.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js';
  script.async = true;
  document.head.appendChild(script);
})();
```

```html
<script src="mathjax-load.js" async></script>
```

## [定义 TeX 宏](https://docs.mathjax.org/en/latest/input/tex/macros.html)

【方法一】直接在数学环境中定义，作用域为当前网页内的所有数学环境：

```tex
\(
   \def\RR{{\bf R}}
   \def\bold#1{{\bf #1}}
\)
```

【方法二】在配置 MathJax 时定义，作用域为加载该配置的所有数学环境：

```js
window.MathJax = {
  tex: {
    macros: {
      RR: "{\\bf R}",
      bold: ["{\\bf #1}", 1]
    }
  }
};
```
