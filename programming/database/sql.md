---
Title: SQL (Structured Query Language)
---

# Data Definition

## Basic Types

|                内置类型                |                    含义                     |
| :------------------------------------: | :-----------------------------------------: |
|      `char(n)` or `character(n)`       |                定长度字符串                 |
| `varchar(n)` or `character varying(n)` |       变长度字符串（最大长度为 `n`）        |
|           `int` or `integer`           |                                             |
|               `smallint`               |                                             |
|            `numeric(p, d)`             | 定点数（`p` 位十进制小数，小数点后 `d` 位） |
|       `real`, `double precision`       |                   浮点数                    |
|               `float(n)`               |       浮点数（至少 `n` 位十进制小数）       |

## Basic Schema Definition

创建 relation：

```sql
create table r (
	Attribute_1 Domain_1 <not null>, ..., Attribute_n Domain_n <not null>,
  <integrity_constraint_1>, ..., <integrity_constraint_1>
);
```

其中 `not null` 规定该 attribute 不能取空值，`integrity_constraint_i` 可以是

```sql
primary key (A_{j_1}, ..., A_{j_m}) -- 规定 r 的 m 个 attributes 为 r 的主键，其值唯一且不能为空
foreign key (A_{k_1}, ..., A_{k_n}) references s -- 规定 r 的 n 个 attributes 值必须为 s 的主键值
```

删除 relation：

```sql
drop table r; -- 删除 r 及其 schema
delete from r; -- 只删除 r 中的 tuples
```

增删 attributes：

```sql
alter table r add Attribute Domain; -- 增加一列，各 tuple 的该属性值为 `null
alter table r drop Attribute; -- 删除一列
```

# Basic Structure of SQL Queries

## Queries on a Single Relation

查询单一 attribute：

```sql
select dept_name from instructor; -- 结果可能含重复 tuples
select distinct dept_name from instructor; -- 从上述结果中去除重复
```

查询多个 attributes 并做算术运算：

```sql
select ID, name, salary * 1.1 from instructor;
```

带条件（可用 `and`, `or`, `not` 相连）查询：

```sql
select name from instructor where dept_name = 'Comp. Sci.' and salary > 70000;
```

## Queries on Multiple Relations

不同 relations 的同名 attributes 以 `relation.attribute` 的方式区分：

```sql
select name, instructor.dept_name, building
from instructor, department
where instructor.dept_name = department.dept_name;
```

一般形式的查询由三个 clauses 构成：

```sql
select attribute_1, ..., attribute_n
from relation_1, ..., relation_m
where predicate;
```

逻辑上分三步：

- 构造 `from`-clause 中的 `relation`s 的 Cartesian product
- 利用 `where`-clause 中的 `predicate` 筛选上述 Cartesian product 的 tuples
- 输出 `select`-clause 中的 `attribute`s of 上述 tuples（可以用 `*` 表示所有 attributes）

# Basic Operations

## `as` --- 重命名

重命名 attribute：

```sql
select name as instructor_name, course_id
from instructor, teaches
where instructor.ID = teaches.ID;
```

重命名 relations：

```sql
select T.name, S.course_id
from instructor as T, teaches as S
where T.ID = S.ID;
```

## `like` --- 字符串匹配

字符串用单引号界定，字符串内的单引号用双引号代替。

模式匹配：

- `%` 匹配任意子字符串
- `_` 匹配任意字符
- `\` 表示转义字符

例如

```sql
select dept_name from department
where building like '%Watson%'; -- 含 Watson 的 building

select dept_name from department
where building not like '%Watson%'; -- 不含 Watson 的 building
```

## `order by` --- 输出排序

按某个 attribute 升序排列：

```sql
select name from instructor where dept_name = 'Physics' order by name;
```

按多个 attributes 依次排列：

```sql
-- 先按 salary 降序排列，再对相同 salary 的 tuples 按 name 升序排列
select * from instructor order by salary desc, name asc;
```

## `between`

```sql
select name from instructor where salary between 90000 and 100000;
-- 等价于
select name from instructor where salary <= 100000 and salary >= 90000;
```

## Row Constructor

```sql
select name, course id from instructor, teaches
where instructor.ID = teaches.ID and dept_name = 'Biology';
-- 等价于
select name, course id from instructor, teaches
where (instructor.ID, dept_name) = (teaches.ID, 'Biology');
```

# Set Operations

## `union`

```sql
(select ...) union (select ...);  -- 集合并运算，结果不含重复的 tuples
(select ...) union all (select ...);  -- 结果保留重复的 tuples，重复次数 = sum(各 queries 中的重复次数)
```

## `intersect`

```sql
(select ...) intersect (select ...);  -- 集合交运算，结果不含重复的 tuples
(select ...) union all (select ...);  -- 结果保留重复的 tuples，重复次数 = min(各 queries 中的重复次数)
```

⚠️ MySQL 不支持 `intersect`。

## `except`

```sql
(select ...) except (select ...);  -- 集合差运算，结果不含重复的 tuples
(select ...) except all (select ...);  -- 结果保留重复的 tuples，重复次数 = max(0, (query1 中的重复次数) - (query2 中的重复次数))
```

⚠️ MySQL 不支持 `except`；Oracle 用 `minus` 代替 `except`；Oracle-12c 用 `multiset except` 代替 `except all`。

# Null Values

`where`-clause 中

- 含 `null` 的算术运算，结果为 `null`；
- 含 `null` 的比较运算，结果为 `unknown`。

```sql
true  and unknown  -- 结果为 unknown
false and unknown  -- 结果为 false
true   or unknown  -- 结果为 true
false  or unknown  -- 结果为 unknown
      not unknown  -- 结果为 unknown
```

`select`-clause 中的 `distinct` 将两个 `null` 视为相同的值。

# Aggregate Functions

SQL 提供 5 个聚合函数，它们以集合为输入，输出单值（的集合）。

- `avg`, `sum` 的输入必须是数值的集合
- `min`, `max`, `count` 的输入可以是其他类型数据的集合

除 `count (*)` 外，均忽略 `null`；作用于空集时，`count` 返回 `0`，其余返回 `null`。

## Basis Aggregation

```sql
select avg (salary) as avg_salary
from instructor where dept_name = 'Comp. Sci.';

select count (distinct ID)
from teaches where semester = 'Spring' and year = 2018;

select count (*) from course;
```

## `group by`

按 `dept_name` 分组，计算各组的 `avg (salary)`：

```sql
select dept_name, avg (salary) as avg_salary from instructor group by dept_name;
```

⚠️ 未出现在 `group by`-clause 里的 attributes，在 `select`-clause 中只能作为聚合函数的输入，不能作为输出的 attributes。

## `having`

对 groups 施加条件：

```sql
select dept name, avg (salary) as avg_salary
from instructor
group by dept_name having avg (salary) > 42000;
```

## 小结

逻辑顺序：

- 先由 `from`-clause 构造 Cartesian product
- 利用 `where`-clause 筛选 tuples
- 利用 `group by`-clause 分组（默认为一组）
- 利用 `having`-clause 对各 groups 进行筛选
- 输出 `select`-clause 规定的 attributes (of groups)

# Nested Subqueries

## `in` --- 集合成员

“集合”可以是 `select` 的结果，或 `(v_1, ..., v_n)`。

与 `intersect` 等价：

```sql
select distinct course_id from section
where semester = 'Fall' and year = 2017 and
	course_id in (select course_id from section where semester = 'Spring' and year = 2018);
```

与 `except` 等价：

```sql
select distinct course_id from section
where semester = 'Fall' and year = 2017 and
	course_id not in (select course_id from section where semester = 'Spring' and year = 2018);
```

## Set Comparison





# Modification of Database