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

## Query Languages

|    类型     |                特点                |               代表               |
| :---------: | :--------------------------------: | :------------------------------: |
| imperative  | 用户直接给出操作序列；改变状态变量 |                                  |
| functional  |     函数式操作；不改变状态变量     |        relational algebra        |
| declarative |       用户不直接给出操作步骤       | tuple/domain relational calculus |

### Relational Algebra

若两个关系 $r_1,r_2$ 的属性相同（或更一般的可比），则有以下集合运算：

|       名称        |                             符号                             |                             示例                             |
| :---------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|       union       |  $r_1\cup r_2\coloneqq\qty{t \mid t\in r_1 \lor t\in r_2 }$  | $\Pi_{courseID}(\sigma_{year=2017}(section))\cup\Pi_{courseID}(\sigma_{year=2018}(section))$ |
|     intersect     | $r_1\cap r_2\coloneqq\qty{t \mid t\in r_1 \land t\in r_2 }$  | $\Pi_{courseID}(\sigma_{year=2017}(section))\cap\Pi_{courseID}(\sigma_{year=2018}(section))$ |
|  set-difference   | $r_1 - r_2\coloneqq\qty{t \mid t\in r_1 \land t\notin r_2 }$ | $\Pi_{courseID}(\sigma_{year=2017}(section))-\Pi_{courseID}(\sigma_{year=2018}(section))$ |

更一般的关系运算：

|       名称        |                             符号                             |                          示例或备注                          |
| :---------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|      select       |     $\sigma_P(r)\coloneqq\qty{t \mid t\in r \land P(t)}$     |             $\sigma_{salary>10000}(instructor)$              |
|      project      | $\Pi_{A_{i_1},\dots,A_{i_k}}(r)$$\coloneqq\qty{(t.A_{i_1},\dots,t.A_{i_k})\mid (t.A_1,\dots,t.A_n)\equiv t\in r}$ |               $\Pi_{name,salary}(instructor)$                |
| Cartesian product | $r_1\times r_2\coloneqq\qty{(t_1,t_2) \mid t_1\in r_1 \land t_2\in r_2 }$ |                 $instructor \times teaches$                  |
|       join        | $r_1\bowtie_\theta r_2\coloneqq \sigma_\theta(r_1\times r_2)$ |   $instructor \bowtie_{instructor.ID=teaches.ID} teaches$    |
|   natural join    |                       $r_1\bowtie r_2$                       | 相当于 $r_1\bowtie_\theta r_2$ 中的 $\theta$ 为 $r_1$ 与 $r_2$ 的同名属性值相同 |
|    assignment     |                     $r_2 \leftarrow r_1$                     |          $r_2$ 只因为临时关系，否则将改变数据库状态          |
|      rename       |         $\rho_x(r)$ or $\rho_{x(A_1,\dots,A_n)}(r)$          |  将 关系 $r$ 重命名为 $x$ 并将属性重命名为 $A_1,\dots,A_n$   |

### Tuple Relational Calculus

### Domain Relational Calculus

# Entity-Relationship Model

# Semi-structured Data Model

## JSON (JavaScript Object Notation)

基本数据类型：

- int
- real
- string
- array：以 `[]` 表示，成员为基本数据类型
- object：以 `{}` 表示，成员为 `key : value` 对，其中 `key` 为 attribute name，`value` 为对应的值。

```json
{
  "ID": 22222,
  "name": { "firstname": "Albert", "lastname": "Einstein" },
  "deptname": "Physics",
  "children": [
    {"firstname": "Hans", "lastname": "Einstein" },
    {"firstname": "Eduard", "lastname": "Einstein" }
  ]
}
```

## XML (eXtensible Markup Language)

以成对的标签表示数据，支持嵌套。

```xml
<course>
  <course_id> CS-101 </course_id>
  <title> Intro. to Computer Science </title>
  <dept_name> Comp. Sci. </dept_name>
  <credits> 4 </credits>
</course>
```

## RDF (Resource Description Framework)

以 triples 表示数据，每个 triple 是形如 `(subject, predicate, object)` 的三元组，具体地可以是以下两种形式之一：

- `(id, attribute_name, value)`
- `(id1, relationship_name, id2)`

Knowledge graph：以 `subject/object` 为 node，`predicate` 为 edge。

# Object-Based Data Model
