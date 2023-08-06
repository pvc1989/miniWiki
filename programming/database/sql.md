---
title: SQL (Structured Query Language)
---

# Softwares

## MySQL

- [Tips on using MySQL](https://www.db-book.com/university-lab-dir/mysql-tips.html)

## PostgreSQL

- [Tips on using PostgreSQL](https://www.db-book.com/university-lab-dir/postgresql-tips.html)
- [PostgreSQL (Current) Documentation](https://www.postgresql.org/docs/current/index.html)

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

其中 `not null` 规定该 attribute 不能取空值，`integrity_constraint_i` 可以是任意 [integrity constraints](#integrity)，例如：

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

## `coalesce`

以任意多个相同类型为输入，返回第一个非空值：

```sql
select ID, coalesce(salary, 0/* 将 null 替换为 0 */) as salary
from instructor;
```

## `decode` in Oracle

不要求类型相同，按第一个匹配替换：

```sql
decode (value,
        match_1, replacement_1,
        ...,
        match_n, replacement_n,
        default_replacement);
```

⚠️ 与一般情形不同，`null` 与 `null` 视为相等。

将 `null` 替换为 `N/A`：

```sql
select ID, decode (salary, null, 'N/A', salary) as salary
from instructor
```

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

⚠️ <a href id="null=null"></a>若 $t_1$ 与 $t_2$ 至少有一个同名 attribute 的值均为 `null`，其余同名 attributes 的值均非空且相等，则 $t_1=t_2$ 返回 `unknown`；而 `unique` 当且仅当存在 $t_1=t_2$ 为 `true` 时才返回 `false`；故在此情形下，`unique` 依然返回 `true`。

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

## `with` --- Temporary Relations<a href id="with"></a>

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

# Join Expressions

## `cross join`

表示 Cartesian product，可以用 `,` 代替：

```sql
select count(*) from student cross join takes;
-- 等价于
select count(*) from student, takes;
```

## `natural join`

只保留 Cartesian product 中同名 attributes 取相同值的 tuples，且同名 attributes 只保留一个。

```sql
select name, course_id from student, takes where student.ID = takes.ID;
-- 等价于
select name, course_id from student natural join takes;
```

可以用 `join r using (a)` 指定与 `r` 连接时需相等的 attribute(s)：

```sql
-- (student natural join takes) 与 course 有两个同名 attributes (course_id, dept_name)
select name, title from (student natural join takes)
  join course using (course_id);  -- 保留 course_id 相等的 tuples
select name, title from (student natural join takes)
  natural join course;  -- 保留 dept_name, course_id 均相等的 tuples
```

## `on` --- Conditional Join

```sql
select * from student, takes where student.ID = takes.ID;
-- 等价于
select * from student join takes on student.ID = takes.ID;  -- 同名 attributes 均保留
-- 几乎等价于
select * from student natural join takes;  -- 同名 attributes 只保留一个
```

## `inner join`

以上 `join`s 都是 `inner join`，其中 `inner` 可以省略。

## `outer join`

`outer join` 为没有参与 `inner join` 的单侧 `tuple` 提供 `null` 值配对，即：允许来自一侧 tuple 在另一侧中缺少与之匹配的 tuple。在连接后的 tuple 中，缺失的值置为 `null`。

在连接结果中保留没有选课的学生，其选课信息置为 `null`：

```sql
-- left outer join 允许 left tuple 缺少与之匹配的 right tuple
select * from student natural left outer join takes;
-- right outer join 允许 right tuple 缺少与之匹配的 left tuple
select * from takes natural right outer join student;
```

```sql
A full outer join B
-- 等价于
(A left outer join B) union (A right outer join B)
```

`outer join` 也可以配合 `on` 使用：

```sql
select * from student left outer join takes on student.ID = takes.ID;  -- 除 ID 保留两次外，几乎等价于 natural left outer join
select * from student left outer join takes on (1 = 1);  -- 等价于 cross join（所有 tuples 均参与 inner join，不提供 null 值配对）
select * from student left outer join takes on (1 = 1) where student.ID = takes.ID;  -- 等价于 natural join
```

# Views --- Virtual Relations

[`with`](#with)-clause 可在单个 query 内创建临时关系。

## `create view`

```sql
create view view_name as <query_expression>;
create view view_name(attribute_1, ..., attribute_n) as <query_expression>;
```

各系系名及该系讲师的总工资：

```sql
create view department_total_salary(dept_name, total_salary) as
  select dept_name, sum (salary) from instructor group by dept_name;
```

## Materialized Views

为避免数据过期，view 通常在被使用时才会去执行 query。

为节省时间，某些系统允许预存 view，并负责（在 query 中的 relation(s) 被更新时）更新 view 中的数据。存在多种更新策略：

- immediately：
- lazily：
- periodically：

## Updatable Views

满足以下条件的 view 可以被更新：

- `from`-clause 只含 1 个实际 relation
- `select`-clause 只含 attribute names，不含表达式、聚合函数、`distinct` 修饰
- 未列出的 attributes 接受 `null` 值
- query 中不含 `group by` 或 `having`

💡 推荐用 trigger 机制更新 view。

# Transactions

每个 transaction 由一组不可分的 statements 构成，整体效果为 all-or-nothing，只能以以下两种方式之一结束：

- `commit work`
- `rollback work`

MySQL、PostgreSQL 默认将每一条 statement 视为一个 transaction，且执行完后自动提交。

为创建含多条 statements 的 transaction，必须关闭自动提交机制。

- SQL-1999、SQL Server 支持将多条 statements 置于 `begin atomic ... end` 中，以创建 transaction。
- MySQL、PostgreSQL 支持 `begin` 但不支持 `end`，必须以 `commit` 或 `rollback` 结尾。

## PostgreSQL

从 Alice's 账户向 Bob's 账户转账 100 元，所涉及的两步 `update` 操作是不可分的：

```postgresql
BEGIN;
UPDATE accounts SET balance = balance - 100.00 WHERE name = 'Alice';
UPDATE accounts SET balance = balance + 100.00 WHERE name = 'Bob';
COMMIT;  -- 如果 Alice's 账户余额为负或其他故障，可以用 ROLLBACK 回滚到交易前的状态
```

PostgreSQL 支持更精细的提交/回滚控制：

```postgresql
BEGIN;
UPDATE accounts SET balance = balance - 100.00 WHERE name = 'Alice';
SAVEPOINT my_savepoint;
UPDATE accounts SET balance = balance + 100.00 WHERE name = 'Bob';
-- oops ... forget that and use Wally's account
ROLLBACK TO my_savepoint;  -- 在 my_savepoint 之后的 savepoints 将被自动释放
UPDATE accounts SET balance = balance + 100.00 WHERE name = 'Wally';
COMMIT;
```

# Integrity Constraints<a href id="integrity"></a>

可以在 `create table` 时给定，也可以向已有的 relation 中添加：

```sql
alter table relation add <integrity_constraint>;
```

## `not null` --- 非空值

默认 `null` 属于所有 domains；若要从某个 domain 中排除 `null`，可在 domain 后加 `not null`：

```sql
name varchar(20) not null
budget numeric(12,2) not null
```

`primary key` 默认为 `not null`。

## `unique` --- Superkey

```sql
unique (A_1, ..., A_n)  -- 这组 attributes 构成一个 superkey，即不同 tuples 的取值不能重复
```

⚠️ `null` 不等于任何值，参见 [`null = null`](#null=null)。

## `check` --- 条件检查<a href id="check"></a>

```sql
CREATE TABLE department
  (..., 
   budget numeric(12,2) check (budget > 0)/* 预算值必须为正 */,
   ...);
create table section
  (...,
   semester varchar (6),
   check (semester in ('Fall', 'Winter', 'Spring', 'Summer')),
   ...); 
```

⚠️ 除 `check(true)` 外，`check(unknown)` 亦返回 `true`。

⚠️ SQL 标准支持 `check` 中含 subquery，但多数系统尚未支持。

## `references` --- 外键约束<a href id="foreign"></a>

```sql
foreign key (dept_name) references department  -- primary key by default
foreign key (dept_name) references department(dept_name/* primary key or superkey */)
```

亦可在 attribute 定义中使用：

```sql
CREATE TABLE course
  (...,
   dept_name varchar(20) references department,
   ...);
```

违反约束的操作默认被拒绝（transaction 回滚），但 `foreign key` 允许设置 `cascade` 等操作：

```sql
foreign key (dept_name) references department
  on delete cascade/* 若 department 中的某个 tuple 被删除，则 course 中相应的 tuples 亦被删除 */
  on update cascade/* 若 department 中的某个 tuple 被更新，则 course 中相应的 tuples 亦被更新 */
```

除 `cascade` 外，还支持 `set null` 或 `set default` 操作。

⚠️ 含有 `null` 的 tuple 默认满足约束。

## `constraint` --- 约束命名

```sql
create table instructor
  (...,
   salary numeric(8,2), /* 命名的约束 */constraint minsalary check (salary > 29000),
   ...);
alter table instructor drop constraint minsalary;  -- 删除该约束
```

## 延迟检查

某些场景必须临时违反约束，例如：

```sql
-- 夫妻二人均以对方姓名为外键，先 insert 任何一人都会违反外键约束
create table person
  (name varchar(20),
   spouse varchar(20),
   primary key (name),
   foreign key (spouse) references person(name)
  );
```

SQL 标准支持

- 用 `initially deferred` 修饰约束，表示该约束延迟到 transaction 末尾才检查。
- 用 `deferrable` 修饰约束，表示该约束默认立即检查，但可以在 transaction 中用 `set constraints <constraint_1, ..., constraint_n> defered` 延迟到末尾。

## `assertion`

```sql
create assertion <assertion_name> check <predicate>;
```

$\forall$ 学生，其 `tot_cred` = 其已通过课程的学分之和：

```sql
create assertion credits_earned_constraint check
  (not exists (select ID from student
               where tot_cred <>
                 (select coalesce(sum(credits), 0)
                  from takes natural join course
                  where student.ID = takes.ID
                    and grade is not null
                    and grade<> 'F'
                 )
              )
  );
```

💡 SQL 不支持 $\forall x, P(x)$，但可以等价的表示为 $\nexists x, \lnot P(x)$。

⚠️ 因开销巨大，多数系统尚未支持 `assertion`。

# Data Types and Schemas

## 时间相关类型

```sql
date '2018-04-25'
time '09:30:00'  -- time(3) 表示秒精确到 3 位小数，默认 0 位小数
timestamp '2018-04-25 10:29:01.45'  -- 默认 6 位小数
```

抽取信息：

```sql
extract(f/* year, month, day, hour, minute, second */ from d/* date or time */)
```

获取当前时间：

```sql
current_date
current_time  -- 含时区信息
localtime  -- 不含时区信息
current_timestamp
localtimestamp
```

## 类型转换

`cast(e as t)` 将表达式 `e` 转化为类型`t`：

```sql
select cast(ID/* 原为 varchar(5) */ as numeric(5)) as inst_id
from instructor
order by inst_id  -- 按数值比较
```

## 格式转换

MySQL：

```mysql
format
```

[PostgreSQL](https://www.postgresql.org/docs/current/functions-formatting.html#FUNCTIONS-FORMATTING-TABLE)：

```postgresql
to_char(timestamp '2002-04-20 17:31:12.66', 'HH12:MI:SS') → 05:31:12
to_char(interval '15h 2m 12s', 'HH24:MI:SS') → 15:02:12
to_char(125, '999') → 125
to_char(125.8::real, '999D9') → 125.8
to_char(-125.8, '999D99S') → 125.80-
to_date('05 Dec 2000', 'DD Mon YYYY') → 2000-12-05
to_number('12,454.8-', '99G999D9S') → -12454.8
to_timestamp('05 Dec 2000', 'DD Mon YYYY') → 2000-12-05 00:00:00-05
```

## `default` --- 默认值

```sql
create table student
  (ID varchar (5),
   name varchar (20) not null,
   dept_name varchar (20), 
   tot_cred numeric(3,0) default 0,
   primary key (ID)
  );
insert into student(ID, name, dept_name)
  values ('12789', 'Newman', 'Comp. Sci.'/* 缺省 tot_cred 值，以 0 补之 */);
```

## `*lob` --- Large OBject

- `clob` --- Character LOB
- `blob` --- Binary LOB

可以定义 LOB attributes：

```sql
book_review clob(10KB)
image blob(10MB)
movie blob(2GB)
```

⚠️ LOB 的读写效率很低，一般以其 locator 作为 attribute，而非对象本身。

## 用户定义类型

### `create type`

美元与英镑不应当能直接比较、算术运算，可通过定义类型加以区分：

```sql
create type Dollars as numeric(12,2) final;
create type  Pounds as numeric(12,2) final;
create table department
  (dept_name varchar (20),
   building varchar (15),
   budget Dollars);
```

### `create domain`

SQL-92 支持自定义 domain，以施加[完整性约束](#integrity)、默认值：

```sql
create domain DDollars as numeric(12,2) not null;
create domain YearlySalary numeric(8,2)
  constraint salary_value_test check(value >= 29000.00);
```

⚠️ 不同自定义 domain 的值直接可以直接比较、算术运算。

## 生成唯一键值

### Oracle

```sql
create table instructor (
  ID number(5) generated always as identity/* 总是由系统自动生成 ID 值 */,
  ...,
  primary key (ID);
);
insert into instructor(name, dept_name, salary) 
  values ('Newprof', 'Comp. Sci.', 100000);  -- 缺省 ID 值
```

若 `always` 替换为 `by default`，则允许用户给定 ID 值。

### MySQL

```mysql
create table instructor (
  ID number(5) auto_increment,
  ...,
  primary key (ID);
);
```

### PostgreSQL

```postgresql
create table instructor (
  ID serial,
  ...,
  primary key (ID);
);
```

相当于

```sql
CREATE SEQUENCE inst_id_seq AS integer;
CREATE TABLE instructor (
  ID integer DEFAULT nextval('inst_id_seq')
  ...,
  primary key (ID);
);
ALTER SEQUENCE inst_id_seq OWNED BY instructor.ID;
```

## 复用 Schema

```sql
create table temp_instructor like instructor;  -- ⚠️ 尚未实现
```

由查询结果推断 schema：

```sql
create table t1 as (select * from instructor where dept_name = 'Music')
with data/* 多数实现默认带数据，哪怕 with data 被省略 */;
```

## `create schema`

|           数据库系统           |         操作系统          |
| :----------------------------: | :-----------------------: |
|            catalog             | home directory of a user  |
|             schema             |    a directory in `~`     |
|        relations, views        |           files           |
|        connect to a DBS        |       log into a OS       |
|    default catalog & schema    |            `~`            |
| `catalog5.univ_schema.course ` | `/home/username/filename` |

```postgresql
CREATE SCHEMA hollywood
    CREATE TABLE films (title text, release date, awards text[])
    CREATE VIEW winners AS
        SELECT title, release FROM films WHERE awards IS NOT NULL;
DROP SCHEMA hollywood;
```

等价于

```postgresql
CREATE SCHEMA hollywood;
CREATE TABLE hollywood.films (title text, release date, awards text[]);
CREATE VIEW hollywood.winners AS
    SELECT title, release FROM hollywood.films WHERE awards IS NOT NULL;
DROP SCHEMA hollywood;
```

# Indexing

Index 将一组 attributes 组合为一个 search key，用来避免遍历所有 tuples 从而加速查找。

Index 与物理层相关，而 SQL 标准限于逻辑层，故没有提供 index 定义命令；但很多数据库系统提供了以下命令：

```sql
create index <index_name> on <relation_name> (<attribute_list>);
drop index <index_name>;
```

# Authorization

最高权限属于***数据库管理员 (DataBase Administrator, DBA)***，其权限包括授权、重构数据库等。

## Privileges

```sql
grant <privilege_list>
on <relation_name/view_name>
to <user_list/role_list>;

revoke <privilege_list>
on <relation_name/view_name>
from <user_list/role_list>;
```

其中

- `privilege_list` 可以包括

  - `select`，相当于文件系统中的 read 权限。
  - `insert`，可以在其后附加 `(attribute_list)`，表示 `insert` 时只允许提供这些 attributes 的值。
  - `update`，可以在其后附加 `(attribute_list)`，表示 `update` 时只允许修改这些 attributes 的值。
  - `references`，可以在其后附加 `(attribute_list)`，表示这些 attributes 可以被用作 [foreign key](#foreign) 或出现在 [`check`](#check) 约束中。
  - `delete`
  - 相当于以上之和的 `all privileges`（创建 `relation` 的 `user` 自动获得 `all privileges`）。
- `user_list` 可以包括
  - 具体的用户名
  - `public`，表示当前及将来所有用户

## Roles

同类用户应当拥有相同权限。

```sql
create role instructor;
grant select on takes to instructor;
```

Role 可以被赋予某个具体的 user 或其他 role：

```sql
create role dean;
grant instructor to dean;  -- 继承 instructor 的权限
grant dean to Robert;
```

默认当前 session 的 role 为 `null`，但可显式指定：

```sql
set role role_name;
```

此后赋权时可附加 ` granted by current_role`，以避免 cascading revocation。

## 传递权限

默认不允许转移权限，但可以用 `with grant option` 赋予某个 user/role 传递权限的权限：

```sql
grant select on department to Alice with grant option;
revoke option grant for select on department from Alice;
```

某个权限的权限传递关系构成一个 directed graph：以 users/roles 为 nodes（其中 DBA 为 root）、以权限传递关系为 edges，每个 user/role 有一条或多条来自 root 的路径。

撤回某个 user/role 的权限可能导致其下游 users/roles 的权限亦被撤销：

```sql
revoke select on department from Alice;  -- 允许 cascading revocation
revoke select on department from Alice restrict;  -- 如有 cascading revocation 则报错
```

# SQL in Programming Languages

# Functions and Procedures

# Triggers

# Recursive Queries

# Advanced Aggregation Features