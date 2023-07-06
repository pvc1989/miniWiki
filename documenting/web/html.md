---
title: HTML
---

# 概述

## 入门教程

[翁恺《HTML5》](https://study.163.com/course/courseMain.htm?courseId=171001)

# 启动服务

## 服务端

```shell
# 进入服务器某一目录，作为网站根目录
cd <server_folder>
# 创建网页 html 文件
cat hello > index.html
# 启动 http 服务，开启 5000 端口
ruby -run -e httpd . -p5000 &
```

## 用户端

用浏览器打开 `http://server_ip:5000` 即可看到 `hello` 字样。

若服务端 有 `<server_folder>/some_path/cat.jpg` 文件，则用户端可在  `http://server_ip:5000/some_path/cat.jpg` 看到该文件。
