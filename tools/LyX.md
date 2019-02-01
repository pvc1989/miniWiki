# 中文支持

## 环境配置

0. 卸载 CTeX 套装, 原因参见 [2018年，为什么不推荐使用 CTeX 套装了](https://zhuanlan.zhihu.com/p/45174503)
1. 安装 [TeX Live](http://tug.org/texlive/), 选择 `完整` 模式 (约 5 GB)
2. 安装 LyX
3. 在 LyX `首选项` 中设置环境变量 `PATH`, 确保第 1 步所安装的可执行文件 (例如 `xelatex`) 能够被搜索到

## 文档设置

打开 LyX, 新建一个文档 (文档类可任选), 然后进入 `文档` → `首选项` 进行如下设置:
1. `LaTeX 导言区` → 添加 `\usepackage{ctex}`
2. `语言` → `文件编码` → `其他` → 选择 `Unicode (XeTeX) (utf8)`
3. `字体` → 勾选 `使用非 TeX 字体 (通过 XeTeX/LuaTeX)`


# 代码高亮

## 环境配置

1. 安装 `pygments`
```shell
pip install pygments
```

## 文档设置

进入 `文档` → `首选项` 进行如下设置:
1. `程序列表` → `语法高亮支持包` → 勾选 `Minted` 
2. `程序列表` → `空白处` 可以设置 `minted` 宏包参数, 例如
```latex
style=xcode
frame=leftline
baselinestretch={1.0}
breaklines=true
fontsize={\footnotesize}
```
3. `输出格式` → 勾选 `允许运行外部程序`
4. 首次编译该文档时, 会弹出警告对话框, 选择 `对该文档总是允许`