# Markdown --- 轻量级标记语言

## 语法
- [Markdown](https://daringfireball.net/projects/markdown/syntax)
- [GitHub Flavored Markdown (GFM)](https://github.github.com/gfm/)

## Typora
[Typora](https://typora.io/) 是一款支持 *所见即所得* 的 GFM 编辑器。

### 数学公式渲染
Typora 的数学公式渲染功能是借助于 [MathJax](https://www.mathjax.org/) 实现的，常规用法参见[官方文档](https://support.typora.io/Math/)。
Typora 将 MathJax 及一些常用的[第三方扩展](http://docs.mathjax.org/en/latest/options/ThirdParty.html)封装在软件内，因此离线时也可以进行数学公式渲染。

⚠️ [GFM](https://github.github.com/gfm/) 不支持数学公式渲染，因此本节所提及的效果 *只在 Typora 中可见*。

#### 局部自定义宏
与其他 Markdown 渲染工具一样，在数学环境内，可以用 TeX 命令 `\def` 定义一些宏。
这些宏的 scope 为当前文件，例如：
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

#### 全局自定义宏
为了让自定义宏的 scope 为所有本地 Markdown 文件，需要将宏定义写在 Typora 内部（参见 [masonlr](https://github.com/typora/typora-issues/issues/100#issuecomment-282169741)）。
以 macOS 10.13 + Typora 0.9.9.18.1 为例，具体做法有以下两种。

⚠️ 更新或重新安装 Typora 后，需要重新进行配置。

##### 写入 `Macros`
用文本编辑器打开 `/Applications/Typora.app/Contents/Resources/TypeMark/index.html`，找到其中的 `TeX:` 字段，修改前应该是下面这个样子：
```js
TeX: {
    extensions: ["noUndefined.js", "autoload-all.js", "AMSmath.js", "AMSsymbols.js", "mediawiki-texvc.js"]
},
```
将全局自定义宏写入 `Macros` 中，注意 `]` 后面的 `,` 不能省略：
```js
TeX: {
    extensions: ["noUndefined.js", "autoload-all.js", "AMSmath.js", "AMSsymbols.js", "mediawiki-texvc.js"],
    Macros: {
    	BlackboardBold: ["\\mathbb{#1}",1],
    	Calligraphic: ["\\mathcal{#1}",1]
    }
},
```
设置完成后，在数学环境中可以直接使用这两个命令，效果在重启 Typora 后可见： 
$$
\BlackboardBold{A} \Calligraphic{B}
$$

##### 调用 `.js` 文件
更符合模块化原则的方案是：将全局自定义宏写入 `.js` 文件，由上述文件中的 `extensions` 对其进行调用。
以 [`physics.js`](https://github.com/ickc/MathJax-third-party-extensions/tree/gh-pages/physics) 为例：将该文件放入 `/Applications/Typora.app/Contents/Resources/TypeMark/lib/MathJax-2.6.1/extensions/TeX/` 文件夹中，然后在 `/Applications/Typora.app/Contents/Resources/TypeMark/index.html` 文件中的 `extensions` 尾部追加 `"physics.cs"`：
```js
TeX: {
    extensions: ["noUndefined.js", "autoload-all.js", "AMSmath.js", "AMSsymbols.js", "mediawiki-texvc.js", "physics.js"]
},
```
设置完成后，在数学环境中可以直接使用 `physics.cs` 中定义过的命令，效果在重启 Typora 后可见：
$$
\ket{\psi}\bra{\phi}
$$

MathJax 提供了一些模仿同名 LaTeX 宏包的[第三方扩展文件](https://github.com/mathjax/MathJax-third-party-extensions/tree/master/legacy)。
如果对其效果不满意，可以自己写一个 `mymacros.js` 文件，调用方式和这里的 `physics.cs` 相同。