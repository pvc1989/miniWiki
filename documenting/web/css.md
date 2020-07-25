# CSS

## 入门

[翁恺《CSS3》](https://study.163.com/course/courseMain.htm?courseId=190001)

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
    p {background-color: white;}
  </style>
  ```
- 内联样式：在任意标记对象中，加入 `style` 属性，例如
  ```html
  <p style="background-color: green;">本段背景为绿色</p>
  ```

## 背景

### 背景颜色

```html
<body style="background-color: red;">本页背景为红色
  <p style="background-color: #00FF00;">本段背景为绿色</p>
  <p style="background-color: rgb(0,0,255);">本段背景为蓝色</p>
  <p style="background-color: rgba(0,0,255,0.5);">本段背景为蓝色、透明度为 0.5</p>
</body>
```

### 背景图片

```html
<p style="background-image: url(./images/cat.jpg);">本段背景为 <code>./images/cat.jpg</code></p>
<p style="background-image: url(./images/cat.jpg);
          background-repeat: repeat-x;
          background-attachment: fixed;">本段背景为 <code>./images/cat.jpg</code>、沿 X 方向重复、固定</p>
<p style="background: url(./images/cat.jpg) center fixed;">本段背景为 <code>./images/cat.jpg</code>、居中、固定</p>
```

## 文本

### 段落

```html
<p style="color: rgb(255,0,0)">文字为红色</p>
<p style="text-align: center">居中对齐</p>
<p style="text-align: justify">两端对齐</p>
<p style="text-transform: capitalize">all men are created equal</p>
<p style="text-decoration: line-through underline">打删除线及下划线</p>
<p style="text-indent: 2em">首行缩进二字符</p>
<p style="line-height: 2">行距为二倍</p>
<p style="word-spacing: 10px">单词间相距十个像素</p>
<p style="letter-spacing: 10mm">字符间相距十毫米</p>
```

### 字体

```html
<p style="font-family: serif">有衬线 serif</p>
<p style="font-family: sans-serif">无衬线 sans-serif</p>
<p style="font-family: monospace">等宽 monospace</p>
```

```html
<p style="font-style: normal">直立 normal</p>
<p style="font-style: italic">意大利 italic</p>
<p style="font-style: oblique">倾斜 oblique</p>
```

```html
<p style="font-variant: small-caps">小型大写 small-caps</p>
<p style="font-weight: bold">加粗 bold</p>
<p style="font-size: 2em">二倍大小的字符</p>
```

### 效果

```html
<h1 style="text-shadow: 2px 3px 4px rgba(0,255,0,0.5)">文字带阴影</h1>
<h1 style="text-shadow: 0px -1px 0px #000000, 0px +1px 3px #606060;
           color: #606060;">文字带多重阴影</h1>
```

```html
<h1 style="outline-color: red;
           outline-width: 10;
           outline-style: dashed">文字带边框</h1>
```

## 列表

```html
<ul style="list-style-type: circle;
           list-style-position: outside">
  <li>数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学数学</li>
  <li>物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理物理</li>
</ul>
```

## 表格

```html
<table style="border: 1px solid blue; border-collapse: collapse;
              caption-side: bottom; table-layout: fixed;">
  <caption>成绩</caption>
  <tr>
    <th style="border: 2px solid blue; width: 50px">数学</th>
    <th style="border: 2px solid blue; height: 50px;">物理</th>
    <th style="border: 2px solid blue; vertical-align: top;">化学</th>
  </tr>
  <tr>
    <td style="border: 1px solid blue; text-align: right;">95</td>
    <td style="border: 1px solid blue; padding: 10pt;">87</td>
    <td style="border: 1px solid blue">94</td>
  </tr>
</table>
```

## 边距

```html
<p>
一段文字一段文字一段文字一段文字一段文字一段文字一段文字一段文字一段文字一段文字一段文字一段文字一段文字一段文字一段文字一段文字一段文字一段文字一段文字一段文字一段文字一段文字
<img src="not_found.jpg"
     alt="上、右、下、左边距分别为 50px、40px、30px、20px"
     style="border: solid;margin: 50px 40px 30px 20px;"
/>
一段文字一段文字一段文字一段文字一段文字一段文字一段文字一段文字一段文字一段文字一段文字一段文字一段文字一段文字一段文字一段文字一段文字一段文字一段文字一段文字一段文字一段文字
</p>
```

## 定位

### 相对定位

对象占据自己原来的位置，并相对于该位置作偏移。

```html
<div>
  <p>第一段</p>
  <p>第二段</p>
  <p style="position: relative; left: -20px; bottom: +20px;">第三段</p>
  <p>第四段</p>
  <p>第五段</p>
</div>
```

### 绝对定位

对象放弃自己原来的位置，并相对于外层可定位对象的位置作偏移。

```html
<div style="position: relative; left: +20px;">
  <p>第一段</p>
  <p>第二段</p>
  <p style="position: absolute; left: -20px; bottom: 20px;">第三段</p>
  <p>第四段</p>
  <p>第五段</p>
</div>
```

### 浮动定位

```html
<img src="not_found.jpg" style="float: right"/>
```

## 样式选择器

### 元素选择器

```html
<head>
  <style>
    h1 {text-decoration: underline;}
    h1, th {border: 2px solid blue;}
  </style>
</head>
```

### 类选择器

```html
<head>
  <style>
    *.warning {color: rgb(128, 0, 0);}
    p.important {border: 1px solid red;}
  </style>
</head>
<body>
  <p class="important warning">重要段落</p>
</body>
```

### 属性选择器

```html
<head>
  <style>
    *[title] {color: blue;}
  </style>
</head>
<body>
  <p title="hello">一个有 title 属性的段落</p>
</body>
```

### 后代选择器

```html
<head>
  <style>
    p {border: 1px solid blue;}
    p em {background-color: yellow;}
  </style>
</head>
<body>
  <p>一段<em>被强调的</em>文字</p>
</body>
```

在 `style` 中：

- 以 `p   em` 为首的样式作用于 `p` 内部的 `em` 对象。
- 以 `p > em` 为首的样式作用于 `p` 内部不含于其他标签的 `em` 对象。
- 以 `p + em` 为首的样式作用于 `p` 后第一个 `em` 对象。
