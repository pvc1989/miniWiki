# 编辑书签

## PDF

## DjVu
### [DJVUSED](http://djvu.sourceforge.net/doc/man/djvused.html)
#### 常用选项
- `-e <script>` 从命令行获取命令
- `-s` 执行完命令后保存文件
- `-u` 按 UTF-8 输出

#### 用法实例
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
