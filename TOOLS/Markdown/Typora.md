# Typora

[Typora](https://typora.io/) 是一款高颜值的 [Markdown](https://daringfireball.net/projects/markdown/) 编辑器.

## 数学渲染

Typora 的数学渲染功能是借助于 [MathJax](https://www.mathjax.org/) 实现的. 常规使用方式参见[官方文档](https://support.typora.io/Math/). Typora 将 MathJax 及一些常用的[第三方扩展](http://docs.mathjax.org/en/latest/options/ThirdParty.html)封装在软件内, 因此离线时也可以完成数学渲染.

### 局部自定义宏

与其他 Markdown 渲染工具一样, 在数学环境内, 可以用 TeX 命令`\def`定义一些宏. 这些宏的作用域为当前文件, 例如:

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
效果如下:
$$
\def\RR{\mathbb{R}}
\def\Differential#1{\mathrm{d}#1}
\def\PartialDerivative#1#2{\frac{\partial #1}{\partial #2}}

\RR \quad
\Differential{x} \quad
\PartialDerivative{u}{t}
$$

### 全局自定义宏
为了让自定义宏的作用域为本地所有 Markdown 文件, 需要将宏定义写在 Typora 内部, 参见 [masonlr](https://github.com/typora/typora-issues/issues/100#issuecomment-282169741). 以 macOS 10.13 + Typora 0.9.9.18.1 为例, 具体做法有以下两种:
#### 写入`Macros`
用文本编辑器打开`/Applications/Typora.app/Contents/Resources/TypeMark/index.html`, 找到其中的`"TeX:"`字段, 修改前应该是下面这个样子: 

```js
TeX: {
    extensions: ["noUndefined.js", "autoload-all.js", "AMSmath.js", "AMSsymbols.js", "mediawiki-texvc.js"]
},
```

将全局自定义宏写入`Macros`中, 注意`"]"`后面的`","`:

```js
TeX: {
    extensions: ["noUndefined.js", "autoload-all.js", "AMSmath.js", "AMSsymbols.js", "mediawiki-texvc.js"],
    Macros: {
    	BlackboardBold: ["\\mathbb{#1}",1],
    	Calligraphic: ["\\mathcal{#1}",1]
    }
},
```
在数学环境中可以直接使用这两个命令(效果在重启 Typora 后可见):  
$$
\BlackboardBold{A} \Calligraphic{B}
$$

#### 调用`.js`文件

更符合模块化原则的方案是: 将全局自定义宏写入`.js`文件, 由上述文件中的`extensions`对其进行调用. 这也是 MathJax 支持的第三方扩展形式. 以[`physics.js`](https://github.com/ickc/MathJax-third-party-extensions/tree/gh-pages/physics)为例: 将该文件放入`/Applications/Typora.app/Contents/Resources/TypeMark/lib/MathJax-2.6.1/extensions/TeX/`文件夹中, 然后在`/Applications/Typora.app/Contents/Resources/TypeMark/index.html`文件中的`extensions`尾部追加`"physics.cs"`:
```js
TeX: {
    extensions: ["noUndefined.js", "autoload-all.js", "AMSmath.js", "AMSsymbols.js", "mediawiki-texvc.js", "physics.js"]
},
```

在数学环境中可以直接使用`physics.cs`中定义过的命令, 效果在重启 Typora 后可见: 
$$
\ket{\psi}\bra{\phi}
$$

[这里](https://github.com/mathjax/MathJax-third-party-extensions/tree/master/legacy)有一些模仿同名 LaTeX 宏包的扩展文件. 如果对其效果不满意, 可以自己写一个`mymacros.js`文件, 调用方式和这里的`physics.cs`完全相同. 