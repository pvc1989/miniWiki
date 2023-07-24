---
Title: Data Models
---

# Relational Model

|               概念               |         类比         |                           定义                            |
| :------------------------------: | :------------------: | :-------------------------------------------------------: |
|             relation             |        table         |                     一组 tuples 之集                      |
|              tuple               |         row          |                  一组 attributes 的取值                   |
|            attribute             |        column        |                                                           |
|              domain              |                      |                某个 attribute 的许可值之集                |
|         relation schema          | definition of a type |               一组 attributes 及其 domains                |
|        relation instance         | value of a variable  |                                                           |

## Keys

|               概念               |                           定义                            |
| :------------------------------: | :-------------------------------------------------------: |
|             superkey             |          tuples 不同则取值不同的一组 attributes           |
|          candidate key           |             没有真子集为 superkey 的 superkey             |
|           primary key            |         设计者选择用于区分 tuple 的 candidate key         |
|      foreign-key constraint      | 关系 $r_1$ 的属性 $A$ 的值 $=$ 关系 $r_2$ 的主键 $B$ 的值 |
| referential-integrity constraint | 关系 $r_1$ 的属性 $A$ 的值 $=$ 关系 $r_2$ 的属性 $B$ 的值 |

## Schema Diagrams

|               概念               |                    符号                     |
| :------------------------------: | :-----------------------------------------: |
|          relation name           |              title of the box               |
|            attributes            |            listed inside the box            |
|           primary key            |            underlined attributes            |
|      foreign-key constraint      | a single-head arrow from $r_1.A$ to $r_2.B$ |
| referential-integrity constraint | a double-head arrow from $r_1.A$ to $r_2.B$ |



# Entity-Relationship Model

# Semi-structured Data Model

## XML

## JSON

# Object-Based Data Model
