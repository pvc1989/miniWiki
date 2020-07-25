---
title : 网页
---

# HTML

## 入门

[翁恺《HTML5》](https://study.163.com/course/courseMain.htm?courseId=171001)

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

# JavaScript

## 概述

JavaScript 是一种脚本语言，用它写成的代码可以在解释器（浏览器）中动态运行。

### 入门教程

[翁恺《JavaScript》](https://study.163.com/course/courseMain.htm?courseId=195001)

### 在 HTML 中使用 JavaScript

JavaScript 代码可以

- 直接嵌入到 HTML 文件的 `head` 或 `body` 中：

  ```html
  <body onload="alert('hi')">
    <script>
      document.write("<h1>hello, world</h1>");
    </script>
  </body>
  ```

- 存放在 `.js` 文件中，再通过以下方式引入到 HTML 中：

  ```html
  <script src="util.js"></script>
  ```

## 变量

定义变量时不需指定变量的类型，只需将关键词 `var` 置于变量名前：

```js
var hello = "hello";
document.write(hello);  // 输出 "hello"
```

数值不分整型与浮点型：

```js
var age = 16;
age = age / 3;        // age == 5.333333333333333
age++;                // age = age + 1;
document.write(age);  // 输出 "6.333333333333333"
```

二元运算符 `+` 有两种含义：
- 两侧均为数值对象时，表示数学加号。
- 任意一侧为字符串时，表示字符串连接。

```js
var hello = "hello ";
hello = hello + "world ";
var age = 16;
document.write(hello + age);  // 输出 "hello world 16"
```

## 分支

熟悉 C 语言的读者可跳过本节。

### `?:` 运算符

```js
var score = 59;
(score >= 60) ? alert("及格了！") : alert("不及格！");
alert("得分：" + score);
```

### `if`-`else` 语句

```js
var score = 59;
if (score >= 60) {
  alert("及格了！");
}
else {
  alert("不及格！");
}
alert("得分：" + score);
```

### `switch` 语句

```js
var score = 75;
score = score - score % 10;  // score == 70
switch (score / 10) {
case 10:
case 9:
  alert("A");
  break
case 8:
  alert("B");
  break
case 7:
  alert("C");
  break
case 6:
  alert("D");
  break
default:
  alert("F");
}
```

## 循环

### `while` 语句

```js
var count = 0;
while (count < 3) {
  alert(++count);
}
```

### `do`-`while` 语句

```js
var count = 5;
do {
  alert(count--);
} while (count != 0);
alert("点火！");
```

### `for` 语句

```js
for (i = 5; i != 0; i--) {
  alert(i);
}
alert("点火！");
```

## 函数

### 普通函数

```js
function print(s) { document.write(s); }
print("hello");
```

```js
function factorial(n) {
  if (n < 0) {
    alert("Illegal input!");
  }
  else {
    return (n <= 1) ? 1 : n * factorial(n-1);
  }
}
document.write(factorial(5));
```

### 函数对象

```js
var f = new Function("x", "y", "return x * y");
function g(x, y) { return x * y; }
document.write(f(4, 5), " == ", g(4, 5));
```

### 变量作用域

- 定义在所有函数外部的变量，对整个网页可见。
- 定义在某个函数内部的变量，仅对当前函数可见，并覆盖同名的全局变量。
- 一个局部变量在一个函数内部只能定义一次（即使其中一个位于 `{}` 内部）。

## 数组

### 列表初始化

```js
var scores = ["red", "green", "blue"];
for (i = 0; i != scores.length; ++i) {
  document.write(scores[i], "<br>");
}
```

### 直接初始化

```js
var scores = Array("red", "green", "blue");
for (i = 0; i != scores.length; ++i) {
  document.write(scores[i], "<br>");
}
```

### 默认初始化

```js
var colors = new Array();
colors[0] = "red";
colors[1] = "green";
colors[colors.length] = "blue";
for (i = 0; i != colors.length; ++i) {
  document.write(colors[i], "<br>");
}
```

### 长度变化

向数组的 `length` 成员赋值可用来显式指定数组长度。

若在任意位置插入元素，则数组长度会根据需要自动调整，中间空出的元素是未定义的：

```js
var colors = new Array();
colors[0] = "red";
colors[1] = "green";
// colors[2] = undefined;
colors[3] = "blue";
for (i = 0; i != colors.length; ++i) {
  document.write(colors[i], "<br>");
}
```

### 数据结构

```js
var colors = new Array();
colors.push("red", "green");
colors.push("blue");
document.write(colors.toString(), "<br>");  // red,green,blue
colors.pop();
document.write(colors.toString(), "<br>");  // red,green
colors.shift();
document.write(colors.toString(), "<br>");  // green
```

## 对象

### 创建对象

```js
var object = new Object();
var circle = {x: 0, y: 0, r: 2};
```

### 增加成员

```js
var book = new Object();
book.title = "ABC";
book.price = 30.5;
```

### 删除成员

```js
delete book.title;
book.price = null;
```

### 遍历成员

```js
var book = {title: "ABC", price: 30.5};
for (var member in book) {
  document.write(member, " == ", book[member], "<br>");
}
```

### 构造函数

```js
function Circle(x, y, r) {
  this.x = x;
  this.y = y;
  this.r = r;
  this.area = function() {
    return 3.1415926535897932384626 * this.r * this.r;
  };
}
var circle = new Circle(0, 0, 10);
document.write(circle.area());
```

### 对象原型

```js
function Book(title, price) {
  this.title = title;
  this.price = price;
  this.discount = 1.0;
}
Book.prototype = {
  constructor: Book,
  setDiscount : function(discount) { return this.discount = discount; },
  getPrice : function() { return this.price * this.discount; },
};
var a = new Book("A", 10);
var b = new Book("B", 10);
b.setDiscount(0.8);
document.write(a.getPrice(), "<br>");
document.write(b.getPrice(), "<br>");
```

## `window`

浏览器可以看作一个名为 `window` 对象。
所有全局变量实际上是 `window` 的成员。

```js
var answer = 12;
alert(window.answer);
```

### 事件处理器

```html
<p onmouseover="alert('hi');" onmouseout="alert('bye');">一个段落</p>
```

### 简单对话框

```html
<script>
  if (confirm("继续？")) {
    alert("好！");
  } else {
    alert("再见");
  }
  var name = prompt("姓名：");
  alert(name);
</script>
```

### 打开新窗口

```html
<body onload="setInterval('update()', 2000);">
  <script>
    var w = open("https://bing.com", "bing", "resizable=yes", "width=400", "height=300");
  </script>
</body>
```

### `location`

```html
<head>
  <script>
    function jump() {
      location = "https://bing.com";
    }
  </script>
</head>
<body onload="setInterval('jump()', 2000);">
  <script>
    document.write(location);
  </script>
</body>
```

## `document`

`window.document` 表示当前 HTML 页面。

```js
for (x in document) {
  document.write(x, "<br>");
}
```

### 成员容器

```html
<body>
  <img name="cat" src="cat.jpg"/>
  <p name="math">数学</p>
  <script>
    alert(document.cat.src);
    alert(document.images[0].src);
    alert(document.getElementsByName("math"));
  </script>
</body>
```

名为 `document.images` 的成员是一个容器，用于存储当前页面内的所有图片。
与之类似的还有 `document.forms` 及 `document.anchors` 这两个成员，
但不存在名为 `document.paragraphs` 的成员。
