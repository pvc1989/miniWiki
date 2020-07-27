---
title: Unified Modeling Language (UML)
---

# Diagrams
## Sequence Diagram

## Class Diagram

# PlantUML
[PlantUML](http://plantuml.com/) 是一个开源的 Java 组件，支持快速绘制 UML 图，并导出为 PNG、SVG 等格式的图片。

## Web Service
如果有互联网连接，则无需进行安装，通过浏览器即可直接使用[在线服务](http://www.plantuml.com/plantuml)。

## Installation
如果要在本地运行 PlantUML，则必须先安装 [Java](https://www.java.com/) 和 [Graphviz](http://graphviz.org/)。
接下来，下载 [`plantuml.jar`](http://sourceforge.net/projects/plantuml/files/plantuml.jar/download) 文件，存放到合适的本地目录。

## GUI
双击 `plantuml.jar` 文件以启动 GUI，PlantUML 会扫描工作目录，并自动生成或更新 PNG 格式的 UML 图。

## CLI
创建一个纯文本文件 [`Inheritance.txt`](./UML/Inheritance.txt)，内容为一条简单的类继承关系：
```txt
@startuml
A <|-- B
A <|-- C
@enduml
```
如果 `plantuml.jar` 位于当前目录下，通过以下命令即可生成 PNG 格式的 UML 类图：
```shell
java -jar plantuml.jar Inheritance.txt
```
如果要生成 SVG 格式的矢量图，只需加上 `-tsvg` 选项：
```shell
java -jar plantuml.jar -tsvg Inheritance.txt
```
如果 `plantuml.jar` 位于其他目录下，则需为其提供路径（假设该路径已存放在环境变量 `PLANTUML_PATH` 中）：
```shell
java -jar ${PLANTUML_PATH}/plantuml.jar -tsvg Inheritance.txt
```
效果如下：

| PNG | SVG |
| :-: | :-: |
| ![](./Inheritance.png) | ![](./Inheritance.svg)|
