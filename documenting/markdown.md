---
title: Markdown
---

# Markdown 标记语言<a href id="Markdown"></a>

## [Markdown](https://daringfireball.net/projects/markdown/syntax)

### [GFM (GitHub Flavored Markdown)](https://github.github.com/gfm/)

## [kramdown](https://kramdown.gettalong.org/)

# Typora 编辑器<a href id="Typora"></a>
[Typora](https://typora.io/) 是一款支持*所见即所得*的 GFM 编辑器。

## 数学公式渲染

Typora 的数学公式渲染功能是由 [MathJax](./latex/README.md#MathJax) 实现的，常规用法参见《[Math and Academic Functions](https://support.typora.io/Math/)》。

MathJax 通常需要 CDN 提供在线服务，但 Typora 封装了 MathJax 及一些常用的[第三方扩展](http://docs.mathjax.org/en/latest/options/ThirdParty.html)，因此离线时也可以进行数学公式渲染。

⚠️ [GFM](https://github.github.com/gfm/) 不支持数学公式渲染，因此本节所提及的效果*只在 Typora 中可见*。

### 用 `\def` 定义局部宏<a href id="local"></a>
与其他 Markdown 渲染工具一样，在数学环境内，可以用 TeX 命令 `\def` 定义一些**宏 (macro)**。
这些宏的**作用域 (scope)**为*当前文件*，例如：

```latex
$$
\def\RR{\mathbb{R}}
\def\Differential#1{\mathrm{d}#1}
\def\PartialDerivative#1#2{\frac{\partial #1}{\partial #2}}

\RR \quad
\Differential{x} \quad
\PartialDerivative{u}{t}
$$					
```
效果如下（只在 Typora 中可见）：

$$
\def\RR{\mathbb{R}}
\def\Differential#1{\mathrm{d}#1}
\def\PartialDerivative#1#2{\frac{\partial #1}{\partial #2}}

\RR \quad
\Differential{x} \quad
\PartialDerivative{u}{t}
$$

### 在配置文件中定义全局宏<a href id="global"></a>

为了让自定义宏的作用域为*所有本地 Markdown 文件*，需要将宏的定义写在 Typora 内部（引自 [masonlr](https://github.com/typora/typora-issues/issues/100#issuecomment-282169741)）。
⚠️ 更新或重新安装 Typora 后，需要重新进行配置。

下面以 macOS 10.15.5 (19F101) 上的 Typora 0.9.9.34 (4498) 为例。

用文本编辑器打开 `/Applications/Typora.app/Contents/Resources/TypeMark/index.html` 文件，找到其中的 `TeX` 字段，修改前应该是下面这个样子：
```js
TeX: {
  extensions: ["noUndefined.js", "autoload-all.js", "AMSmath.js", "AMSsymbols.js", "mediawiki-texvc.js"],
  mhchem: { legacy: false }
},
```
将全局自定义宏写入 `Macros` 中：
```js
TeX: {
  extensions: ["noUndefined.js", "autoload-all.js", "AMSmath.js", "AMSsymbols.js", "mediawiki-texvc.js"],
  Macros: {
    BlackboardBold: ["\\mathbb{#1}",1],
    Calligraphic: ["\\mathcal{#1}",1]
  },
  mhchem: { legacy: false }
},
```
设置完成后，在数学环境中可以直接使用这两个命令（效果在重启 Typora 后可见）：
$$
\BlackboardBold{A} \equiv \mathbb{A} \qquad \Calligraphic{B} \equiv \mathcal{B}
$$

### 引入第三方 `.js` 文件<a href id="js"></a>
更符合模块化原则的方案是：将全局自定义宏写入 `.js` 文件，由上述文件中的 `extensions` 对其进行调用。MathJax 提供了一些模仿同名 LaTeX 宏包的[第三方扩展文件](https://github.com/mathjax/MathJax-third-party-extensions/tree/master/legacy)。如果对其效果不满意，可以自己写一个 `mymacros.js` 文件。

下面以（自 Typora 0.11.0 (5581) 起默认被引入的）[`physics.js`](https://github.com/ickc/MathJax-third-party-extensions/tree/gh-pages/physics) 为例：

1. 将该文件放入 `/Applications/Typora.app/Contents/Resources/TypeMark/lib/MathJax3/es5/input/tex/extensions` 文件夹中。
1. 在[前一节](#global)已提到的 `/Applications/Typora.app/Contents/Resources/TypeMark/index.html` 文件中，找到 `TeX` 字段，将 `"physics.js"` 追加到 `extensions` 的尾部。

设置完成后，在数学环境中可以直接使用 `physics.js` 中定义过的命令（效果在重启 Typora 后可见）：

$$
\ket{\psi} = \sum_i\ket{e_i}\bra{e_i}\ket{\psi}
$$

## 引用锚点

自 [2018 年 8 月](https://github.com/typora/typora-issues/issues/1072#issuecomment-414101157)起，Typora 支持用 `<a href id="anchor_name"></a>` 设置**锚点 (anchor)**。
该功能可用于

- 创建对文档内（除公式、代码环境外）任意元素的引用。
- 简化对复杂标题（长度过长，或含空格、数学、代码等元素）的引用。
  - 若不用锚点，则引用本页内的《[用 `\def` 定义局部宏](#local)》《[在配置文件中定义全局宏](#global)》《[引入第三方 `.js` 文件](#js)》将较为繁琐。

## 样式修改

在配置文件 `/Applications/Typora.app/Contents/Resources/TypeMark/index.html` 中的 `</head>` 标签前添加以下内容：

```html
<link rel="stylesheet" href="path-to-your.css">
```

[`pvcstyle.css`](../assets/css/pvcstyle.css) 可作为样式文件 `path-to-your.css` 的示例。

