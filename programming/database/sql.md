---
Title: SQL (Structured Query Language)
---

# Softwares

## MySQL

- [Tips on using MySQL](https://www.db-book.com/university-lab-dir/mysql-tips.html)

## PostgreSQL

- [Tips on using PostgreSQL](https://www.db-book.com/university-lab-dir/postgresql-tips.html)

## SQLite

### `sql.js`

- [Online SQL interpreter on db-book.com](https://www.db-book.com/university-lab-dir/sqljs.html)
- [Try SQL at w3schools.com](https://www.w3schools.com/sql/trysql.asp?filename=trysql_select_all)

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
alter table r add Attribute Domain; -- 增加一列，各 tuples 的该属性值为 null
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

## `between` --- 数值范围

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
(select ...) intersect all (select ...);  -- 结果保留重复的 tuples，重复次数 = min(各 queries 中的重复次数)
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

SQL 提供 5 个聚合函数，它们以集合为输入，以单值（的集合）为输出。

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

## `group by` --- 分组

按 `dept_name` 分组，计算各组的 `avg (salary)`：

```sql
select dept_name, avg (salary) as avg_salary from instructor group by dept_name;
```

⚠️ 未出现在 `group by`-clause 里的 attributes，在 `select`-clause 中只能作为聚合函数的输入，不能作为输出的 attributes。

## `having` --- 组条件

平均工资大于 42000 的系：<a href id="having"></a>

```sql
select dept_name, avg (salary) as avg_salary
from instructor
group by dept_name
having avg (salary) > 42000;
```

逻辑顺序：

- 先由 `from`-clause 构造 Cartesian product
- 利用 `where`-clause 筛选 tuples
- 利用 `group by`-clause 分组（默认为一组）
- 利用 `having`-clause 对各 groups 进行筛选
- 输出 `select`-clause 指定的 attributes (of groups)

# Nested Subqueries

## `in` --- $\in$

这里的“集合”可以是形如 `(select ...)` 的子查询结果，或形如 `(v_1, ..., v_n)` 的枚举集。

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

## `some` --- $\exist$

```sql
-- salary 大于子查询结果中的某个 salary
select name from instructor
where salary > some (select salary from instructor where dept name = 'Biology');
```

⚠️ 与 `any` 为同义词，早期版本的 SQL 只支持 `any`。

## `all` --- $\forall$

```sql
-- salary 大于子查询结果中的所有 salary
select name from instructor
where salary > all (select salary from instructor where dept name = 'Biology');
```

## `exists` --- 集合非空

```sql
select course_id from section as S
where semester = 'Fall' and year = 2017 and
	exists (select * from section as T
          where semester = 'Spring' and year = 2018 and S.course_id = T.course_id);
```

其中 `S` 在外层查询定义，可以在内层子查询中使用。作用域规则与高级编程语言类似。

$A\supset B$ 可以表示为

```sql
not exists (B except A)
```

上过生物系所有课程的学生：

```sql
select S.ID, S.name from student as S
where not exists (
  (select course_id from course where dept_name = 'Biology')  -- Biology 的所有课程
  except
  (select T.course_id from takes as T where S.ID = T.ID) -- 学号为 S.ID 的学生上过的课程
);
```

## `unique` --- 无重复

2017 年至多开过一次的课程：

```sql
select T.course_id from course as T
where unique (select R.course_id from section as R
              where T.course_id = R.course_id and R.year = 2017);
```

等价于

```sql
select T.course_id from course as T
where 1 >= (select count(R.course id) from section as R
            where T.course_id = R.course_id and R.year = 2017);
```

⚠️ 若 $t_1$ 与 $t_2$ 至少有一个同名 attribute 的值均为 `null`，其余同名 attributes 的值均非空且相等，则 $t_1=t_2$ 返回 `unknown`；而 `unique` 当且仅当存在 $t_1=t_2$ 为 `true` 时才返回 `false`；故在此情形下，`unique` 依然返回 `true`。

## `from`-clause 中的子查询

与 [`having`](#having) 等价的写法：

```sql
select dept_name, avg_salary
from (select dept_name, avg (salary) as avg_salary
      from instructor group by dept_name)
where avg_salary > 42000;
```

子查询结果是一个 relation，可将其命名为 `dept_avg`，它含有 `dept_name`, `avg_salary` 这两个 attributes：

```sql
select dept_name, avg_salary
from (select dept_name, avg (salary) from instructor group by dept_name)
	as dept_avg (dept_name, avg_salary)
where avg_salary > 42000;
```

⚠️ MySQL 及 PostgreSQL 规定 `from`-clause 中的子查询结果必须被命名。

自 SQL-2003 起，支持用 `lateral` 访问 `from`-clause 中已出现过的 relation：

```sql
select name, salary, avg_salary
from instructor I1, lateral (select avg(salary) as avg_salary
                             from instructor I2
                             where I2.dept_name = I1.dept_name);
```

## `with` --- 临时关系

拥有最大预算的系：

```sql
with max_budget (value)  -- 创建临时关系 max_budget，其唯一的属性名为 value
	as (select max(budget) from department)
select dept_name
from department, max_budget
where department.budget = max budget.value;
```

通常比嵌套的子查询更清晰，且临时关系可在多处复用。

可以创建多个临时关系：

```sql
with
	/* 临时关系 1 */dept_total (dept_name, value)
		as (select dept_name, sum(salary) from instructor group by dept_name),
	/* 临时关系 2 */dept_total_avg(value)
		as (select avg(value) from dept_total)
select dept_name
from dept_total, dept_total_avg
where dept_total.value > dept_total_avg.value;  -- 总工资 > 平均总工资
```

## 标量子查询

返回单值（之集）的子查询，可用在 `select`-, `where`-, `having`-clauses 中接收单值的地方。

查询各系及其讲师人数：

```sql
select dept_name,
  (select count(*)
   from instructor
   where department.dept_name = instructor.dept_name
  ) as num_instructors/* 该系讲师人数 */
from department;
```

# Modification of Database

若含有 `where`-clause，则先完成该 clause，再修改 relation。

## `delete`

与 `select` 类似：

```sql
delete from relation where predicate;
```

每次只能从一个 relation 中删除 tuples。

`where`-clause 可以含子查询：

```sql
delete from instructor
where salary < (select avg (salary) from instructor);
```

## `insert`

按 attributes 在 schema 中的顺序插入 values：

```sql
insert into course -- attributes 依次为 course_id, title, dept_name, credits
values ('CS-437', 'Database Systems', 'Comp. Sci.', 4);
```

或显式给定顺序（可以与 schema 中的不一致）：

```sql
insert into course (title, course id, credits, dept_name)
values ('Database Systems', 'CS-437', 4, 'Comp. Sci.');
```

更一般的，可以插入查询结果：

```sql
-- 从 student 中找到音乐系总学分超过 144 的学生，将他们插入 instructor
insert into instructor
  select ID, name, dept_name, 18000
  from student
  where dept_name = 'Music' and tot_cred > 144;
```

## `update`

所有讲师涨薪 5%：

```sql
update instructor
set salary = salary * 1.05;
```

收入小于平均收入的讲师涨薪 5%：

```sql
update instructor
set salary = salary * 1.05
where salary < (select avg (salary) from instructor);
```

条件分支：

```sql
update instructor
set salary =
	case
    when salary <= 50000 then salary * 1.05  -- [0, 50000]
    when salary <= 100000 then salary * 1.03 -- (50000, 100000]
    else salary * 1.01  -- (100000, infty)
	end
```

[标量子查询](#标量子查询)可用于 `set`-clause：

```sql
-- 将每个 student 的 tot_cred 更新为已通过（grade 非空不等于 F）课程的学分之和
update student
set tot cred = (
  select sum(credits)  -- 若未通过任何课程，则返回 null
  from takes, course
  where student.ID = takes.ID and takes.course_id = course.course_id
  	and takes.grade <> 'F' and takes.grade is not null);
```

