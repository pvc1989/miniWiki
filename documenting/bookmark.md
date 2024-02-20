---
title: 书签编辑器
---

# PDF 书签编辑器
## [PDFBookmarker](https://github.com/pvc1989/pdfbookmarker)<a name="PDFBookmarker"></a>
### 用法示例
```bash
# 将 toc.txt 中的目录添加到 book.pdf，另存为 new-book.pdf
python3 pdfbookmarker.py book.pdf toc.txt new-book.pdf
# 将 toc.txt 中的目录添加到 book.pdf，另存为 book-new.pdf（默认文件名）
python3 pdfbookmarker.py book.pdf toc.txt
```
其中 `toc.txt` 为纯文本文件，每一行由三部分构成：
```
<nested level>"<bookmark title>"|<page number>
```
例如：
```
+"目录"|3
+"1  第一章标题"|5
++"1.1  第一节标题"|5
++"1.1  第二节标题"|9
```
[PVC](https://github.com/pvc1989/pdfbookmarker) 在 [RussellLuo](https://github.com/RussellLuo/pdfbookmarker) 的基础上，增加了对页码偏移的支持：用独占一行的 `#n` 表示「自下一行起，页码值 `+n`」。
因此，以上目录文件可以等价地写为：

```txt
#1
+"目录"|2
#4
+"1  第一章标题"|1
++"1.1  第一节标题"|1
++"1.1  第二节标题"|5
```

# DjVu 书签编辑器
## [DJVUSED](http://djvu.sourceforge.net/doc/man/djvused.html)<a name="DJVUSED"></a>
### 常用选项
- `-e <script>` 从命令行获取命令
- `-s` 执行完命令后保存文件
- `-u` 按 UTF-8 输出

### 用法示例
```bash
# 打印第 3 页的文字
djvused file.djvu -e 'select 3; print-pure-txt'
# 打印目录（UTF-8）
djvused file.djvu -e 'print-outline' -u
# 设置目录
djvused file.djvu -e 'set-outline toc.txt' -s
```
其中 `toc.txt` 为纯文本文件，格式如下：
```lisp
(bookmarks
  ("目录" "#2")
  ("1  第一章标题" "#5"
    ("1.1  第一节标题" "#5")
    ("1.2  第二节标题" "#7")
  )
  ("索引" "#21")
)
```
