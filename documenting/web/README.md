---
title : 网页
---

# HTML

# CSS

## 概述
现代网页提倡内容与样式分离：
- 内容由 ***HTML (Hyper Text Markup Language)*** 定义。
- 样式由 ***CSS (Cascading Style Sheets)*** 定义。

三种方式：
- 外部样式：在 `<head>` 中，引入外部 CSS 文件（推荐），例如
  ```html
  <link rel="stylesheet" type="text/css" href="mystyle.css">
  ```
- 内部样式：在 `<head>` 中，加入 `<style>` 字段，例如
  ```html
  <style>
    p {background-color : white;}
  </style>
  ```
- 内联样式：在任意标记对象中，加入 `style` 属性，例如
  ```html
  <p style="background-color : green;">本段背景为绿色</p>
  ```

## 背景

### 背景颜色

```html
<body style="background-color : red;">本页背景为红色
  <p style="background-color : #00FF00;">本段背景为绿色</p>
  <p style="background-color : rgb(0,0,255);">本段背景为蓝色</p>
  <p style="background-color : rgba(0,0,255,0.5);">本段背景为蓝色、透明度为 0.5</p>
</body>
```

### 背景图片

```html
<p style="background-image : url(./images/cat.jpg);">本段背景为 <code>./images/cat.jpg</code></p>
<p style="background-image : url(./images/cat.jpg);
          background-repeat : repeat-x;
          background-attachment : fixed;">本段背景为 <code>./images/cat.jpg</code>、沿 X 方向重复、固定</p>
<p style="background : url(./images/cat.jpg) center fixed;">本段背景为 <code>./images/cat.jpg</code>、居中、固定</p>
```

## 文本

### 段落

```html
<p style="color : rgb(255,0,0)">文字为红色</p>
<p style="text-align : center">居中对齐</p>
<p style="text-align : justify">两端对齐</p>
<p style="text-transform : capitalize">all men are created equal</p>
<p style="text-decoration : line-through underline">打删除线及下划线</p>
<p style="text-indent : 2em">首行缩进二字符</p>
<p style="line-height : 2">行距为二倍</p>
<p style="word-spacing : 10px">单词间相距十个像素</p>
<p style="letter-spacing : 10mm">字符间相距十毫米</p>
```

### 字体

```html
<p style="font-family : serif">有衬线 serif</p>
<p style="font-family : sans-serif">无衬线 sans-serif</p>
<p style="font-family : monospace">等宽 monospace</p>
```

```html
<p style="font-style : normal">直立 normal</p>
<p style="font-style : italic">意大利 italic</p>
<p style="font-style : oblique">倾斜 oblique</p>
```

```html
<p style="font-variant : small-caps">小型大写 small-caps</p>
<p style="font-weight : bold">加粗 bold</p>
<p style="font-size : 2em">二倍大小的字符</p>
```

### 效果

```html
<h1 style="text-shadow : 2px 3px 4px rgba(0,255,0,0.5)">文字带阴影</h1>
<h1 style="text-shadow : 0px -1px 0px #000000, 0px +1px 3px #606060;
           color : #606060;">文字带多重阴影</h1>
```

```html
<h1 style="outline-color : red;
           outline-width : 10;
           outline-style : dashed">文字带边框</h1>
```

## 列表

## 表格

# JavaScript
