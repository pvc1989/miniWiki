---
title: JavaScript
---

# 概述

JavaScript 是一种脚本语言，用它写成的代码可以在解释器（浏览器）中动态运行。

## 入门教程

[翁恺《JavaScript》](https://study.163.com/course/courseMain.htm?courseId=195001)

## 在 HTML 中使用 JavaScript

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

# 变量

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

# 分支

熟悉 C 语言的读者可跳过本节。

## `?:` 运算符

```js
var score = 59;
(score >= 60) ? alert("及格了！") : alert("不及格！");
alert("得分：" + score);
```

## `if`-`else` 语句

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

## `switch` 语句

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

# 循环

## `while` 语句

```js
var count = 0;
while (count < 3) {
  alert(++count);
}
```

## `do`-`while` 语句

```js
var count = 5;
do {
  alert(count--);
} while (count != 0);
alert("点火！");
```

## `for` 语句

```js
for (i = 5; i != 0; i--) {
  alert(i);
}
alert("点火！");
```

# 函数

## 普通函数

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

## 函数对象

```js
var f = new Function("x", "y", "return x * y");
function g(x, y) { return x * y; }
document.write(f(4, 5), " == ", g(4, 5));
```

## 变量作用域

- 定义在所有函数外部的变量，对整个网页可见。
- 定义在某个函数内部的变量，仅对当前函数可见，并覆盖同名的全局变量。
- 一个局部变量在一个函数内部只能定义一次（即使其中一个位于 `{}` 内部）。

# 数组

## 列表初始化

```js
var scores = ["red", "green", "blue"];
for (i = 0; i != scores.length; ++i) {
  document.write(scores[i], "<br>");
}
```

## 直接初始化

```js
var scores = Array("red", "green", "blue");
for (i = 0; i != scores.length; ++i) {
  document.write(scores[i], "<br>");
}
```

## 默认初始化

```js
var colors = new Array();
colors[0] = "red";
colors[1] = "green";
colors[colors.length] = "blue";
for (i = 0; i != colors.length; ++i) {
  document.write(colors[i], "<br>");
}
```

## 长度变化

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

## 数据结构

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

# 对象

## 创建对象

```js
var object = new Object();
var circle = {x: 0, y: 0, r: 2};
```

## 增加成员

```js
var book = new Object();
book.title = "ABC";
book.price = 30.5;
```

## 删除成员

```js
delete book.title;
book.price = null;
```

## 遍历成员

```js
var book = {title: "ABC", price: 30.5};
for (var member in book) {
  document.write(member, " == ", book[member], "<br>");
}
```

## 构造函数

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

## 对象原型

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

# `window`

浏览器可以看作一个名为 `window` 对象。
所有全局变量实际上是 `window` 的成员。

```js
var answer = 12;
alert(window.answer);
```

## 事件处理器

```html
<p onmouseover="alert('hi');" onmouseout="alert('bye');">一个段落</p>
```

## 简单对话框

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

## 打开新窗口

```html
<body onload="setInterval('update()', 2000);">
  <script>
    var w = open("https://bing.com", "bing", "resizable=yes", "width=400", "height=300");
  </script>
</body>
```

## `location`

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

# `document`

`window.document` 表示当前 HTML 页面。

```js
for (x in document) {
  document.write(x, "<br>");
}
```

## 成员容器

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
