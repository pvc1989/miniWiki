---
title: SQL (Structured Query Language)
---

# Softwares

## MySQL

- [Tips ON using MySQL](https://www.db-book.com/university-lab-dir/mysql-tips.html)

## PostgreSQL

- [Tips ON using PostgreSQL](https://www.db-book.com/university-lab-dir/postgresql-tips.html)
- [PostgreSQL (Current) Documentation](https://www.postgresql.org/docs/current/index.html)

## SQLite

### `sql.js`

- [Online SQL interpreter ON db-book.com](https://www.db-book.com/university-lab-dir/sqljs.html)
- [Try SQL at w3schools.com](https://www.w3schools.com/sql/trysql.asp?filename=trysql_select_all)

# Data Definition

## Basic Types

|                内置类型                |                    含义                     |
| :------------------------------------: | :-----------------------------------------: |
|      `CHAR(n)` or `CHARACTER(n)`       |                定长度字符串                 |
| `VARCHAR(n)` or `CHARACTER VARYING(n)` |       变长度字符串（最大长度为 `n`）        |
|           `INT` or `INTEGER`           |                                             |
|                  `S`                   |                                             |
|            `NUMERIC(p, d)`             | 定点数（`p` 位十进制小数，小数点后 `d` 位） |
|       `REAL`, `DOUBLE PRECISION`       |                   浮点数                    |
|               `float(n)`               |       浮点数（至少 `n` 位十进制小数）       |

## Basic Schema Definition

创建 relation：

```sql
CREATE TABLE r (
  attribute_1 domain_1 <NOT NULL>, ..., attribute_n domain_n <NOT NULL>,
  <integrity_constraint_1>, ..., <integrity_constraint_1>
);
```

其中 `NOT NULL` 规定该 attribute 不能取空值，`integrity_constraint_i` 可以是任意 [integrity constraints](#integrity)，例如：

```sql
PRIMARY KEY (A_{j_1}, ..., A_{j_m}) -- 规定 r 的 m 个 attributes 为 r 的主键，其值唯一且不能为空
FOREIGN KEY (A_{k_1}, ..., A_{k_n}) REFERENCES s -- 规定 r 的 n 个 attributes 值必须为 s 的主键值
```

删除 relation：

```sql
DROP TABLE r; -- 删除 r 及其 schema
DELETE FROM r; -- 只删除 r 中的 tuples
```

增删 attributes：

```sql
ALTER TABLE r ADD Attribute Domain; -- 增加一列，各 tuples 的该属性值为 NULL
ALTER TABLE r DROP Attribute; -- 删除一列
```

# Basic Structure of SQL Queries

## Queries ON a Single Relation

查询单一 attribute：

```sql
SELECT dept_name FROM instructor; -- 结果可能含重复 tuples
SELECT DISTINCT dept_name FROM instructor; -- 从上述结果中去除重复
```

查询多个 attributes 并做算术运算：

```sql
SELECT ID, name, salary * 1.1 FROM instructor;
```

带条件（可用 `AND`, `or`, `NOT` 相连）查询：

```sql
SELECT name FROM instructor WHERE dept_name = 'Comp. Sci.' AND salary > 70000;
```

## Queries ON Multiple Relations

不同 relations 的同名 attributes 以 `relation.attribute` 的方式区分：

```sql
SELECT name, instructor.dept_name, building
FROM instructor, department
WHERE instructor.dept_name = department.dept_name;
```

一般形式的查询由三个 clauses 构成：

```sql
SELECT attribute_1, ..., attribute_n
FROM relation_1, ..., relation_m
WHERE predicate;
```

逻辑上分三步：

- 构造 `FROM`-clause 中的 `relation`s 的 Cartesian product
- 利用 `WHERE`-clause 中的 `predicate` 筛选上述 Cartesian product 的 tuples
- 输出 `SELECT`-clause 中的 `attribute`s of 上述 tuples（可以用 `*` 表示所有 attributes）

# Basic Operations

## `AS` --- 重命名

重命名 attribute：

```sql
SELECT name AS instructor_name, course_id
FROM instructor, teaches
WHERE instructor.ID = teaches.ID;
```

重命名 relations：

```sql
SELECT T.name, S.course_id
FROM instructor AS T, teaches AS S
WHERE T.ID = S.ID;
```

## `LIKE` --- 字符串匹配

字符串用单引号界定，字符串内的单引号用双引号代替。

模式匹配：

- `%` 匹配任意子字符串
- `_` 匹配任意字符
- `\` 表示转义字符

例如

```sql
SELECT dept_name FROM department
WHERE building LIKE '%Watson%'; -- 含 Watson 的 building

SELECT dept_name FROM department
WHERE building NOT LIKE '%Watson%'; -- 不含 Watson 的 building
```

## `ORDER BY` --- 输出排序

按某个 attribute 升序排列：

```sql
SELECT name FROM instructor WHERE dept_name = 'Physics' ORDER BY name;
```

按多个 attributes 依次排列：

```sql
-- 先按 salary 降序排列，再对相同 salary 的 tuples 按 name 升序排列
SELECT * FROM instructor ORDER BY salary DESC, name ASC;
```

## `BETWEEN` --- 数值范围

```sql
SELECT name FROM instructor WHERE salary BETWEEN 90000 AND 100000;
-- 等价于
SELECT name FROM instructor WHERE salary <= 100000 AND salary >= 90000;
```

## Row Constructor

```sql
SELECT name, course id FROM instructor, teaches
WHERE instructor.ID = teaches.ID AND dept_name = 'Biology';
-- 等价于
SELECT name, course id FROM instructor, teaches
WHERE (instructor.ID, dept_name) = (teaches.ID, 'Biology');
```

# Set Operations

## `UNION`

```sql
(SELECT ...) UNION (SELECT ...);  -- 集合并运算，结果不含重复的 tuples
(SELECT ...) UNION ALL (SELECT ...);  -- 结果保留重复的 tuples，重复次数 = SUM(各 queries 中的重复次数)
```

## `INTERSECT`

```sql
(SELECT ...) INTERSECT (SELECT ...);  -- 集合交运算，结果不含重复的 tuples
(SELECT ...) INTERSECT ALL (SELECT ...);  -- 结果保留重复的 tuples，重复次数 = min(各 queries 中的重复次数)
```

⚠️ MySQL 不支持 `INTERSECT`。

## `EXCEPT`

```sql
(SELECT ...) EXCEPT (SELECT ...);  -- 集合差运算，结果不含重复的 tuples
(SELECT ...) EXCEPT ALL (SELECT ...);  -- 结果保留重复的 tuples，重复次数 = max(0, (query1 中的重复次数) - (query2 中的重复次数))
```

⚠️ MySQL 不支持 `EXCEPT`；Oracle 用 `MINUS` 代替 `EXCEPT`；Oracle-12c 用 `MULTISET EXCEPT` 代替 `EXCEPT ALL`。

# Null Values

`WHERE`-clause 中

- 含 `NULL` 的算术运算，结果为 `NULL`；
- 含 `NULL` 的比较运算，结果为 `UNKNOWN`。

```sql
TRUE  AND UNKNOWN  -- 结果为 UNKNOWN
FALSE AND UNKNOWN  -- 结果为 FALSE
TRUE   OR UNKNOWN  -- 结果为 TRUE
FALSE  OR UNKNOWN  -- 结果为 UNKNOWN
      NOT UNKNOWN  -- 结果为 UNKNOWN
```

`SELECT`-clause 中的 `DISTINCT` 将两个 `NULL` 视为相同的值。

## `COALESCE`

以任意多个相同类型为输入，返回第一个非空值：

```sql
SELECT ID, COALESCE(salary, 0/* 将 NULL 替换为 0 */) AS salary
FROM instructor;
```

## `DECODE` in Oracle

不要求类型相同，按第一个匹配替换：

```sql
DECODE(value,
       match_1, replacement_1,
       ...,
       match_n, replacement_n,
       default_replacement);
```

⚠️ 与一般情形不同，`NULL` 与 `NULL` 视为相等。

将 `NULL` 替换为 `N/A`：

```sql
SELECT ID, DECODE(salary, NULL, 'N/A', salary) AS salary
FROM instructor;
```

# Aggregate Functions

SQL 提供 5 个基本聚合函数，它们以集合为输入，以单值（的集合）为输出。

- `AVG`, `SUM` 的输入必须是数值的集合
- `MIN`, `MAX`, `COUNT` 的输入可以是其他类型数据的集合

除 `COUNT(*)` 外，均忽略 `NULL`；作用于空集时，`COUNT` 返回 `0`，其余返回 `NULL`。

## Basis Aggregation

```sql
SELECT AVG(salary) AS avg_salary
FROM instructor WHERE dept_name = 'Comp. Sci.';

SELECT COUNT(DISTINCT ID)
FROM teaches WHERE semester = 'Spring' AND year = 2018;

SELECT COUNT(*) FROM course;
```

## `GROUP BY` --- 分组

按 `dept_name` 分组，计算各组的 `AVG(salary)`：

```sql
SELECT dept_name, AVG(salary) AS avg_salary
FROM instructor GROUP BY dept_name;
```

⚠️ 未出现在 `GROUP BY`-clause 里的 attributes，在 `SELECT`-clause 中只能作为聚合函数的输入，不能作为输出的 attributes。

## `HAVING` --- 组条件<a href id="HAVING"></a>

平均工资大于 42000 的系：

```sql
SELECT dept_name, AVG(salary) AS avg_salary
FROM instructor
GROUP BY dept_name
HAVING AVG(salary) > 42000;
```

逻辑顺序：

- 先由 `FROM`-clause 构造 Cartesian product
- 利用 `WHERE`-clause 筛选 tuples
- 利用 `GROUP BY`-clause 分组（默认为一组）
- 利用 `HAVING`-clause 对各 groups 进行筛选
- 输出 `SELECT`-clause 指定的 attributes (of groups)

# Nested Subqueries

## `IN` --- $\in$

这里的“集合”可以是形如 `(SELECT ...)` 的子查询结果，或形如 `(v_1, ..., v_n)` 的枚举集。

与 `INTERSECT` 等价：

```sql
SELECT DISTINCT course_id FROM section
WHERE semester = 'Fall' AND year = 2017 AND
  course_id IN (SELECT course_id FROM section
                WHERE semester = 'Spring' AND year = 2018);
```

与 `EXCEPT` 等价：

```sql
SELECT DISTINCT course_id FROM section
WHERE semester = 'Fall' AND year = 2017 AND
  course_id NOT IN (SELECT course_id FROM section
                    WHERE semester = 'Spring' AND year = 2018);
```

## `SOME` --- $\exists$

```sql
-- salary 大于子查询结果中的某个 salary
SELECT name FROM instructor
WHERE salary > SOME (SELECT salary FROM instructor WHERE dept_name = 'Biology');
```

⚠️ 与 `ANY` 为同义词，早期版本的 SQL 只支持 `ANY`。

## `ALL` --- $\forall$

```sql
-- salary 大于子查询结果中的所有 salary
SELECT name FROM instructor
WHERE salary > ALL (SELECT salary FROM instructor WHERE dept_name = 'Biology');
```

## `EXISTS` --- 集合非空

```sql
SELECT course_id FROM section AS S
WHERE semester = 'Fall' AND year = 2017 AND
  EXISTS (SELECT * FROM section AS T
          WHERE semester = 'Spring' AND year = 2018 AND S.course_id = T.course_id);
```

其中 `S` 在外层查询定义，可以在内层子查询中使用。作用域规则与高级编程语言类似。

$A\supset B$ 可以表示为

```sql
NOT EXISTS (B EXCEPT A)
```

上过生物系所有课程的学生：

```sql
SELECT S.ID, S.name FROM student AS S
WHERE NOT EXISTS (
  (SELECT course_id FROM course WHERE dept_name = 'Biology')  -- Biology 的所有课程
  EXCEPT
  (SELECT T.course_id FROM takes AS T WHERE S.ID = T.ID) -- 学号为 S.ID 的学生上过的课程
);
```

## `UNIQUE` --- 无重复

2017 年至多开过一次的课程：

```sql
SELECT T.course_id FROM course AS T
WHERE UNIQUE (SELECT R.course_id FROM section AS R
              WHERE T.course_id = R.course_id AND R.year = 2017);
```

等价于

```sql
SELECT T.course_id FROM course AS T
WHERE 1 >= (SELECT COUNT(R.course_id) FROM section AS R
            WHERE T.course_id = R.course_id AND R.year = 2017);
```

⚠️ <a href id="NULL=NULL"></a>若 $t_1$ 与 $t_2$ 至少有一个同名 attribute 的值均为 `NULL`，其余同名 attributes 的值均非空且相等，则 $t_1=t_2$ 返回 `UNKNOWN`；而 `UNIQUE` 当且仅当存在 $t_1=t_2$ 为 `TRUE` 时才返回 `FALSE`；故在此情形下，`UNIQUE` 依然返回 `TRUE`。

## `FROM`-clause 中的子查询

与 [`HAVING`](#HAVING) 等价的写法：

```sql
SELECT dept_name, avg_salary
FROM (SELECT dept_name, AVG(salary) AS avg_salary
      FROM instructor GROUP BY dept_name)
WHERE avg_salary > 42000;
```

子查询结果是一个 relation，可将其命名为 `dept_avg`，它含有 `dept_name`, `avg_salary` 这两个 attributes：

```sql
SELECT dept_name, avg_salary
FROM (SELECT dept_name, AVG(salary) FROM instructor GROUP BY dept_name)
  AS dept_avg (dept_name, avg_salary)
WHERE avg_salary > 42000;
```

⚠️ MySQL 及 PostgreSQL 规定 `FROM`-clause 中的子查询结果必须被命名。

自 SQL-2003 起，支持用 `LATERAL` 访问 `FROM`-clause 中已出现过的 relation：

```sql
SELECT name, salary, avg_salary
FROM instructor I1, LATERAL (SELECT AVG(salary) AS avg_salary
                             FROM instructor I2
                             WHERE I2.dept_name = I1.dept_name);
```

## `WITH` --- Temporary Relations<a href id="with"></a>

拥有最大预算的系：

```sql
with max_budget (value)  -- 创建临时关系 max_budget，其唯一的属性名为 value
  AS (SELECT MAX(budget) FROM department)
SELECT dept_name
FROM department, max_budget
WHERE department.budget = max_budget.value;
```

通常比嵌套的子查询更清晰，且临时关系可在多处复用。

可以创建多个临时关系：

```sql
WITH
  /* 临时关系 1 */dept_total (dept_name, value)
    AS (SELECT dept_name, SUM(salary) FROM instructor GROUP BY dept_name),
  /* 临时关系 2 */dept_total_avg(value)
    AS (SELECT AVG(value) FROM dept_total)
SELECT dept_name
FROM dept_total, dept_total_avg
WHERE dept_total.value > dept_total_avg.value;  -- 总工资 > 平均总工资
```

## 标量子查询

返回单值（之集）的子查询，可用在 `SELECT`-, `WHERE`-, `HAVING`-clauses 中接收单值的地方。

查询各系及其讲师人数：

```sql
SELECT dept_name,
  (SELECT COUNT(*) FROM instructor
   WHERE department.dept_name = instructor.dept_name
  ) AS num_instructors/* 该系讲师人数 */
FROM department;
```

# Modification of Database

若含有 `WHERE`-clause，则先完成该 clause，再修改 relation。

## `DELETE FROM`

与 `SELECT` 类似：

```sql
DELETE FROM relation WHERE predicate;
```

每次只能从一个 relation 中删除 tuples。

`WHERE`-clause 可以含子查询：

```sql
DELETE FROM instructor
WHERE salary < (SELECT AVG(salary) FROM instructor);
```

## `INSERT INTO`

按 attributes 在 schema 中的顺序插入 values：

```sql
INSERT INTO course -- attributes 依次为 course_id, title, dept_name, credits
VALUES ('CS-437', 'Database Systems', 'Comp. Sci.', 4);
```

或显式给定顺序（可以与 schema 中的不一致）：

```sql
INSERT INTO course (title, course_id, credits, dept_name)
VALUES ('Database Systems', 'CS-437', 4, 'Comp. Sci.');
```

更一般的，可以插入查询结果：

```sql
-- 从 student 中找到音乐系总学分超过 144 的学生，将他们插入 instructor
INSERT INTO instructor
  SELECT ID, name, dept_name, 18000
  FROM student
  WHERE dept_name = 'Music' AND tot_cred > 144;
```

## `UPDATE ... SET`

所有讲师涨薪 5%：

```sql
UPDATE instructor
SET salary = salary * 1.05;
```

收入小于平均收入的讲师涨薪 5%：

```sql
UPDATE instructor
SET salary = salary * 1.05
WHERE salary < (SELECT AVG(salary) FROM instructor);
```

条件分支：

```sql
UPDATE instructor
SET salary =
  CASE
    WHEN salary <= 50000 THEN salary * 1.05  -- [0, 50000]
    WHEN salary <= 100000 THEN salary * 1.03 -- (50000, 100000]
    ELSE salary * 1.01  -- (100000, infty)
  END
```

[标量子查询](#标量子查询)可用于 `SET`-clause：

```sql
-- 将每个 student 的 tot_cred 更新为已通过（grade 非空不等于 F）课程的学分之和
UPDATE student
SET tot_cred = (
  SELECT SUM(credits)  -- 若未通过任何课程，则返回 NULL
  FROM takes, course
  WHERE student.ID = takes.ID AND takes.course_id = course.course_id
    AND takes.grade <> 'F' AND takes.grade IS NOT NULL);
```

# Join Expressions

## `CROSS JOIN`

表示 Cartesian product，可以用 `,` 代替：

```sql
SELECT COUNT(*) FROM student CROSS JOIN takes;
-- 等价于
SELECT COUNT(*) FROM student, takes;
```

## `NATURAL JOIN`

只保留 Cartesian product 中同名 attributes 取相同值的 tuples，且同名 attributes 只保留一个。

```sql
SELECT name, course_id FROM student, takes WHERE student.ID = takes.ID;
-- 等价于
SELECT name, course_id FROM student NATURAL JOIN takes;
```

可以用 `JOIN r USING (a)` 指定与 `r` 连接时需相等的 attribute(s)：

```sql
-- (student NATURAL JOIN takes) 与 course 有两个同名 attributes (course_id, dept_name)
SELECT name, title FROM (student NATURAL JOIN takes)
  JOIN course using (course_id);  -- 保留 course_id 相等的 tuples
SELECT name, title FROM (student NATURAL JOIN takes)
  NATURAL JOIN course;  -- 保留 dept_name, course_id 均相等的 tuples
```

## `ON` --- Conditional Join

```sql
SELECT * FROM student, takes WHERE student.ID = takes.ID;
-- 等价于
SELECT * FROM student JOIN takes ON student.ID = takes.ID;  -- 同名 attributes 均保留
-- 几乎等价于
SELECT * FROM student NATURAL JOIN takes;  -- 同名 attributes 只保留一个
```

## `INNER JOIN`

以上 `JOIN`s 都是 `INNER JOIN`，其中 `INNER` 可以省略。

## `OUTER JOIN`

`OUTER JOIN` 为没有参与 `INNER JOIN` 的单侧 tuple 提供 `NULL` 值配对，即：允许来自一侧 tuple 在另一侧中缺少与之匹配的 tuple。在连接后的 tuple 中，缺失的值置为 `NULL`。

在连接结果中保留没有选课的学生，其选课信息置为 `NULL`：

```sql
-- LEFT OUTER JOIN 允许 left tuple 缺少与之匹配的 right tuple
SELECT * FROM student NATURAL LEFT OUTER JOIN takes;
-- RIGHT OUTER JOIN 允许 right tuple 缺少与之匹配的 left tuple
SELECT * FROM takes NATURAL RIGHT OUTER JOIN student;
```

```sql
x FULL OUTER JOIN y
-- 等价于
(x LEFT OUTER JOIN y) UNION (x RIGHT OUTER JOIN y)
```

`OUTER JOIN` 也可以配合 `ON` 使用：

```sql
SELECT * FROM student LEFT OUTER JOIN takes ON student.ID = takes.ID;  -- 除 ID 保留两次外，几乎等价于 NATURAL LEFT OUTER JOIN
SELECT * FROM student LEFT OUTER JOIN takes ON (1 = 1);  -- 等价于 cross join（所有 tuples 均参与 inner join，不提供 NULL 值配对）
SELECT * FROM student LEFT OUTER JOIN takes ON (1 = 1) WHERE student.ID = takes.ID;  -- 等价于 NATURAL JOIN
```

# Views --- Virtual Relations<a href id="view"></a>

[`with`](#with)-clause 可在单个 query 内创建临时关系。

## `CREATE VIEW`

```sql
CREATE VIEW view_name AS <query_expression>;
CREATE VIEW view_name(attribute_1, ..., attribute_n) AS <query_expression>;
```

各系系名及该系讲师的总工资：

```sql
CREATE VIEW department_total_salary(dept_name, total_salary) AS
  SELECT dept_name, SUM(salary) FROM instructor GROUP BY dept_name;
```

## Materialized Views

为避免数据过期，view 通常在被使用时才会去执行 query。

为节省时间，某些数据库系统支持 materialized view，以负责预存并（在 query 中的 relation(s) 被更新时）更新 view 中的数据。存在多种更新策略：

- immediately：
- lazily：
- periodically：

## Updatable Views

满足以下条件的 view 可以被更新：

- `FROM`-clause 只含 1 个实际 relation
- `SELECT`-clause 只含 attribute names，不含表达式、聚合函数、`DISTINCT` 修饰
- 未列出的 attributes 接受 `NULL` 值
- query 中不含 `GROUP BY` 或 `HAVING`

💡 推荐用 trigger 机制更新 view。

# Transactions

每个 transaction 由一组不可分的 statements 构成，整体效果为 all-or-nothing，只能以以下两种方式之一结束：

- commit work
- rollback work

MySQL、PostgreSQL 默认将每一条 statement 视为一个 transaction，且执行完后自动提交。

为创建含多条 statements 的 transaction，必须关闭自动提交机制。

- SQL-1999、SQL Server 支持将多条 statements 置于 `BEGIN ATOMIC ... END` 中，以创建 transaction。
- MySQL、PostgreSQL 支持 `BEGIN` 但不支持 `END`，必须以 `COMMIT` 或 `ROLLBACK` 结尾。

## PostgreSQL

从 Alice's 账户向 Bob's 账户转账 100 元，所涉及的两步 `UPDATE` 操作是不可分的：

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
-- oops ... forget that AND use Wally's account
ROLLBACK TO my_savepoint;  -- 在 my_savepoint 之后的 savepoints 将被自动释放
UPDATE accounts SET balance = balance + 100.00 WHERE name = 'Wally';
COMMIT;
```

# Integrity Constraints<a href id="integrity"></a>

可以在 `CREATE TABLE` 时给定，也可以向已有的 relation 中添加：

```sql
ALTER TABLE relation ADD <integrity_constraint>;
```

## `NOT NULL` --- 非空值

默认 `NULL` 属于所有 domains；若要从某个 domain 中排除 `NULL`，可在 domain 后加 `NOT NULL`：

```sql
name VARCHAR(20) NOT NULL
budget NUMERIC(12,2) NOT NULL
```

`PRIMARY KEY` 默认为 `NOT NULL`。

## `UNIQUE` --- Superkey

```sql
UNIQUE (A_1, ..., A_n)  -- 这组 attributes 构成一个 superkey，即不同 tuples 的取值不能重复
```

⚠️ `NULL` 不等于任何值，参见 [`NULL = NULL`](#NULL=NULL)。

## `CHECK` --- 条件检查<a href id="CHECK"></a>

```sql
CREATE TABLE department
  (..., 
   budget NUMERIC(12,2) CHECK (budget > 0)/* 预算值必须为正 */,
   ...);
CREATE TABLE section
  (...,
   semester VARCHAR(6),
   CHECK (semester IN ('Fall', 'Winter', 'Spring', 'Summer')),
   ...); 
```

⚠️ 除 `CHECK(TRUE)` 外，`CHECK(UNKNOWN)` 亦返回 `TRUE`。

⚠️ SQL 标准支持 `CHECK` 中含 subquery，但多数系统尚未支持。

## `REFERENCES` --- 外键约束<a href id="foreign"></a>

```sql
FOREIGN KEY (dept_name) REFERENCES department  -- PRIMARY KEY by default
FOREIGN KEY (dept_name) REFERENCES department(dept_name/* PRIMARY KEY or superkey */)
```

亦可在 attribute 定义中使用：

```sql
CREATE TABLE course (
  ...,
  dept_name VARCHAR(20) REFERENCES department,
  ...
);
```

违反约束的操作默认被拒绝（transaction 回滚），但 `FOREIGN KEY` 允许设置 `CASCADE` 等操作：

```sql
FOREIGN KEY (dept_name) REFERENCES department
  ON DELETE CASCADE/* 若 department 中的某个 tuple 被删除，则 course 中相应的 tuples 亦被删除 */
  ON UPDATE CASCADE/* 若 department 中的某个 tuple 被更新，则 course 中相应的 tuples 亦被更新 */
```

除 `CASCADE` 外，还支持 `SET NULL` 或 `SET DEFAULT` 操作。

⚠️ 含有 `NULL` 的 tuple 默认满足约束。

💡 借助 [triggers](#Triggers) 可实现更一般的 [referential integrity](#referential) constraints。

## `CONSTRAINT` --- 约束命名

```sql
CREATE TABLE instructor
  (...,
   salary NUMERIC(8,2), /* 命名的约束 */CONSTRAINT minsalary CHECK (salary > 29000),
   ...);
ALTER TABLE instructor DROP CONSTRAINT minsalary;  -- 删除该约束
```

## 延迟检查

某些场景必须临时违反约束，例如：

```sql
-- 夫妻二人均以对方姓名为外键，先 insert 任何一人都会违反外键约束
CREATE TABLE person (
  name VARCHAR(20),
  spouse VARCHAR(20),
  PRIMARY KEY (name),
  FOREIGN KEY (spouse) REFERENCES person(name)
);
```

SQL 标准支持

- 用 `INITIALLY DEFERRED` 修饰约束，表示该约束延迟到 transaction 末尾才检查。
- 用 `DEFERRABLE` 修饰约束，表示该约束默认立即检查，但可以在 transaction 中用
  ```sql
  SET CONSTRAINTS <constraint_1, ..., constraint_n> DEFERRED
  ```
  延迟到末尾。

## `ASSERTION`

```sql
CREATE ASSERTION <assertion_name> CHECK <predicate>;
```

$\forall$ 学生，其 `tot_cred` = 其已通过课程的学分之和：

```sql
CREATE ASSERTION credits_earned_constraint CHECK (
  NOT EXISTS (
    SELECT ID FROM student WHERE tot_cred <> (
      SELECT COALESCE(SUM(credits), 0)
      FROM takes NATURAL JOIN course
      WHERE student.ID = takes.ID
        AND grade IS NOT NULL AND grade<> 'F'
    )
  )
);
```

💡 SQL 不支持 $\forall x, P(x)$，但可以等价的表示为 $\nexists x, \lnot P(x)$。

⚠️ 因开销巨大，多数系统尚未支持 `ASSERTION`。

# Data Types AND Schemas

## 时间相关类型

```sql
DATE '2018-04-25'
TIME '09:30:00'  -- time(3) 表示秒精确到 3 位小数，默认 0 位小数
TIMESTAMP '2018-04-25 10:29:01.45'  -- 默认 6 位小数
```

抽取信息：

```sql
EXTRACT(f/* year, month, day, hour, minute, second */ FROM d/* date or time */)
```

获取当前时间：

```sql
CURRENT_DATE
CURRENT_TIME  -- 含时区信息
LOCALTIME  -- 不含时区信息
CURRENT_TIMESTAMP
LOCALTIMESTAMP
```

## 类型转换

`CAST(e AS t)` 将表达式 `e` 转化为类型`t`：

```sql
SELECT CAST(ID/* 原为 VARCHAR(5) */ AS NUMERIC(5)) AS inst_id
FROM instructor
ORDER BY inst_id  -- 按数值比较
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

## `DEFAULT` --- 默认值

```sql
CREATE TABLE student (
  ID VARCHAR (5),
  name VARCHAR (20) NOT NULL,
  dept_name VARCHAR (20), 
  tot_cred NUMERIC(3,0) DEFAULT 0,
  PRIMARY KEY (ID)
);
INSERT INTO student(ID, name, dept_name)
  VALUES ('12789', 'Newman', 'Comp. Sci.'/* 缺省 tot_cred 值，以 0 补之 */);
```

## `*LOB` --- Large OBject

- `CLOB` --- Character LOB
- `BLOB` --- Binary LOB

可以定义 LOB attributes：

```sql
book_review CLOB(10KB)
image BLOB(10MB)
movie BLOB(2GB)
```

⚠️ LOB 的读写效率很低，一般以其 locator 作为 attribute，而非对象本身。

## 用户定义类型

### `CREATE TYPE`

美元与英镑不应当能直接比较、算术运算，可通过定义类型加以区分：

```sql
CREATE TYPE Dollars AS NUMERIC(12,2) final;
CREATE TYPE  Pounds AS NUMERIC(12,2) final;
CREATE TABLE department (
  dept_name VARCHAR (20),
  building VARCHAR (15),
  budget Dollars
);
```

### `CREATE domain`

SQL-92 支持自定义 domain，以施加[完整性约束](#integrity)、默认值：

```sql
CREATE domain DDollars AS NUMERIC(12,2) NOT NULL;
CREATE domain YearlySalary NUMERIC(8,2)
  CONSTRAINT salary_value_test CHECK(value >= 29000.00);
```

⚠️ 不同自定义 domain 的值直接可以直接比较、算术运算。

## 生成唯一键值

### Oracle

```sql
CREATE TABLE instructor (
  ID number(5) GENERATED ALWAYS AS IDENTITY/* 总是由系统自动生成 ID 值 */,
  ...,
  PRIMARY KEY (ID)
);
INSERT INTO instructor(name, dept_name, salary) 
  VALUES ('Newprof', 'Comp. Sci.', 100000);  -- 缺省 ID 值
```

若 `always` 替换为 `BY DEFAULT`，则允许用户给定 ID 值。

### MySQL

```mysql
CREATE TABLE instructor (
  ID number(5) auto_increment,
  ...,
  PRIMARY KEY (ID)
);
```

### PostgreSQL

```postgresql
CREATE TABLE instructor (
  ID SERIAL,
  ...,
  PRIMARY KEY (ID)
);
```

相当于

```sql
CREATE SEQUENCE inst_id_seq AS INTEGER;
CREATE TABLE instructor (
  ID INTEGER DEFAULT nextval('inst_id_seq')
  ...,
  PRIMARY KEY (ID)
);
ALTER SEQUENCE inst_id_seq OWNED BY instructor.ID;
```

## 复用 Schema

```sql
CREATE TABLE temp_instructor LIKE instructor;  -- ⚠️ 尚未实现
```

由查询结果推断 schema：

```sql
CREATE TABLE t1 AS (SELECT * FROM instructor WHERE dept_name = 'Music')
WITH DATA/* 多数实现默认带数据，哪怕 WITH DATA 被省略 */;
```

## `CREATE SCHEMA`

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
CREATE INDEX <index_name> ON <relation_name> (<attribute_list>);
DROP INDEX <index_name>;
```

# Authorization

最高权限属于***数据库管理员 (DataBase Administrator, DBA)***，其权限包括授权、重构数据库等。

## Privileges

```sql
GRANT <privilege_list>
ON <relation_name/view_name>
TO <user_list/role_list>;

REVOKE <privilege_list>
ON <relation_name/view_name>
FROM <user_list/role_list>;
```

其中

- `privilege_list` 可以包括

  - `SELECT`，相当于文件系统中的 read 权限。
  - `INSERT`，可以在其后附加 `(attribute_list)`，表示 `INSERT` 时只允许提供这些 attributes 的值。
  - `UPDATE`，可以在其后附加 `(attribute_list)`，表示 `UPDATE` 时只允许修改这些 attributes 的值。
  - `REFERENCES`，可以在其后附加 `(attribute_list)`，表示这些 attributes 可以被用作 [`FOREIGN KEY`](#foreign) 或出现在 [`CHECK`](#CHECK) 约束中。
  - `DELETE`
  - 相当于以上之和的 `ALL PRIVILEGES`（创建 `relation` 的 `user` 自动获得 `ALL PRIVILEGES`）。
- `user_list` 可以包括
  - 具体的用户名
  - `PUBLIC`，表示当前及将来所有用户

## Roles

同类用户应当拥有相同权限。

```sql
CREATE ROLE instructor;
GRANT SELECT ON takes TO instructor;
```

Role 可以被赋予某个具体的 user 或其他 role：

```sql
CREATE ROLE dean;
GRANT instructor TO dean;  -- 继承 instructor 的权限
GRANT dean TO Robert;
```

默认当前 session 的 role 为 `NULL`，但可显式指定：

```sql
SET ROLE role_name;
```

此后赋权时可附加 `GRANTED BY CURRENT_ROLE`，以避免 cascading revocation。

## 传递权限

默认不允许转移权限，但可以用 `WITH GRANT OPTION` 赋予某个 user/role 传递权限的权限：

```sql
GRANT SELECT ON department TO Alice WITH GRANT OPTION;
REVOKE OPTION GRANT FOR SELECT ON department FROM Alice;
```

某个权限的权限传递关系构成一个 directed graph：以 users/roles 为 nodes（其中 DBA 为 root）、以权限传递关系为 edges，每个 user/role 有一条或多条来自 root 的路径。

撤回某个 user/role 的权限可能导致其下游 users/roles 的权限亦被撤销：

```sql
REVOKE SELECT ON department FROM Alice;  -- 允许 cascading revocation
REVOKE SELECT ON department FROM Alice restrict;  -- 如有 cascading revocation 则报错
```

# In Programming Languages

- Dynamic SQL：在*运行期*以字符串形式构造并提交 SQL 语句。
- Embedded SQL：由预处理器在*编译期*将查询需求编译为函数调用。

## Java

Java DataBase Connectivity (JDBC)

- [Java JDBC API - Oracle](https://docs.oracle.com/javase/8/docs/technotes/guides/jdbc/)
- [Microsoft JDBC Driver for SQL Server](https://learn.microsoft.com/en-us/sql/connect/jdbc/microsoft-jdbc-driver-for-sql-server)

```java
import java.sql.*;

public static void JDBCexample(String userid, String passwd) {
  try (
    /* try-with-resources since Java 7 */
    Connection conn = DriverManager.getConnection(
        "<protocol>@<url>:<port>:<database>", userid, passwd);
    Statement stmt = conn.createStatement();
    /* 否则需要手动 conn.close(); stmt.close(); */
  ) {
    try {
      stmt.executeUpdate("<INSERT|UPDATE|DELETE_statement>");
    } catch (SQLException sqle) {
      System.out.println("Could NOT insert tuple. " + sqle);
    }
    ResultSet rset = stmt.executeQuery("<SELECT_statement>");
    while (rset.next()/* for each tuple */) {
      System.out.println(rset.getString("<attribute_name>") + " " 
                         + rset.getFloat(2/* (1-based) 2nd attribute */));
    }
  } catch (Exception sqle) {
    System.out.println("Exception : " + sqle);
  }
} 
```

若要在 Java 程序中推断某个 relation 的 schema，可以从 `ResultSet` 对象中提取元数据：

```java
ResultSetMetaData rsmd = rset.getMetaData();
for(int i = 1; i <= rsmd.getColumnCount(); i++) {
  System.out.println(rsmd.getColumnName(i));
  System.out.println(rsmd.getColumnTypeName(i));
} 
```

💡 推荐用 `prepareStatement` 方法（由 SQL 系统完成代入并处理转义），以替代更危险的 `String` 串联操作：

```java
PreparedStatement pStmt = conn.prepareStatement(
    "INSERT INTO instructor VALUES(?, ?, ?, ?)");
pStmt.setString(1, "88877");
pStmt.setString(2, "Perry");
pStmt.setString(3, "Finance");
pStmt.setInt(4, 125000);
pStmt.executeUpdate();  // INSERT INTO instructor VALUES(88877, Perry, Finance, 125000);
pStmt.setString(1, "88878");
pStmt.executeUpdate();  // INSERT INTO instructor VALUES(88878, Perry, Finance, 125000);
```

类似地，可参数化 SQL 函数、过程调用：

```java
// 需用 registerOutParameter() 注册返回类型
CallableStatement cStmt1 = conn.prepareCall("{? = call some function(?)}"); CallableStatement cStmt2 = conn.prepareCall("{call some procedure(?, ?)}");
```

## Python

- [`psycopg2`](https://www.psycopg.org/docs/) is the most popular PostgreSQL database adapter for the Python programming language.
  - [Passing parameters to SQL queries](https://www.psycopg.org/docs/usage.html#query-parameters)
- [`pyodbc`](https://github.com/mkleehammer/pyodbc/wiki) is an open source Python module that makes accessing ODBC databases simple.

```python3
import psycopg2

def PythonDatabaseExample(userid, passwd):
    try:
        conn = psycopg2.connect(host, port, dbname, user, password)
        cur = conn.cursor()
        try:
            cur.execute("INSERT INTO instructor VALUES(%s, %s, %s, %s)",
                        ("77987", "Kim", "Physics", 98000))
            conn.commit()
        EXCEPT Exception AS sqle:
            print("Could NOT insert tuple. ", sqle)
            conn.rollback()
        cur.execute("""SELECT dept_name, AVG(salary)
                       FROM instructor GROUP BY dept_name""")
        for dept in cur:
            print dept[0], dept[1]
    except exception as sqle:
        print("Exception : ", sqle) 
```

## C

Open Database Connectivity (ODBC)

- [Microsoft ODBC Specification](https://github.com/Microsoft/ODBC-Specification)
- [Microsoft ODBC Driver for SQL Server](https://learn.microsoft.com/en-us/sql/connect/odbc/microsoft-odbc-driver-for-sql-server)

```c
void ODBCexample() {
  RETCODE error;
  HENV env; SQLAllocEnv(&env); /* environment */
  HDBC conn; SQLAllocConnect(env, &conn); /* database connection */
  SQLConnect(conn,
             "db.yale.edu", SQL_NTS/* 表示前一个实参是以 '\0' 结尾的字符串 */,
             "avi", SQL_NTS, "avipasswd", SQL_NTS);
  {
    HSTMT stmt; SQLAllocStmt(conn, &stmt); /* statement */
    char * sqlquery = "SELECT dept_name, SUM(salary) FROM instructor GROUP BY dept_name";
    error = SQLExecDirect(stmt, sqlquery, SQL_NTS);
    if (error == SQL_SUCCESS) {
      char deptname[80]; int lenOut1;
      SQLBindCol(stmt, 1/* 第 1 个 attribute */, SQL_C_CHAR, deptname,
                 80/* 最大长度 */, &lenOut1/* 实际长度（负值表示 null）的地址 */);
      float salary; int lenOut2;
      SQLBindCol(stmt, 2/* 第 2 个 attribute */, SQL_C_FLOAT, &salary, 0, &lenOut2);
      while (SQLFetch(stmt) == SQL_SUCCESS) {
        printf(" %s %g∖n", deptname, salary);
      }
    }
    SQLFreeStmt(stmt, SQL_DROP);  /* 所有 allocated 资源都要被 freed */
  }
  SQLDisconnect(conn);
  SQLFreeConnect(conn);
  SQLFreeEnv(env);
}
```

Transactions 相关：

```c
SQLSetConnectOption(conn, SQL_AUTOCOMMIT, 0);
SQLTransact(conn, SQL_COMMIT);
SQLTransact(conn, SQL_ROLLBACK);
```

# Functions AND Procedures

⚠️ 实际数据库系统给出的具体实现不同于 SQL 标准（本节）。

## 基本语法

```sql
DECLARE <variable_name> <type>;  -- 声明变量
SET <variable_name> = <value>  -- 变量赋值
BEGIN <SQL_statements> END  -- 复合语句
BEGIN ATOMIC <SQL_transaction> END  -- 不可分的复合语句
```

循环（`LEAVE` 相当于 `break`，`ITERATE` 相当于 `continue`）：

```sql
WHILE boolean_expression DO
  sequence_of_statements;
END WHILE

REPEAT
  sequence_of_statements;
UNTIL boolean_expression
END REPEAT

DECLARE n INTEGER DEFAULT 0;
FOR r AS  -- for each row in the table
  SELECT budget FROM department;
DO
  SET n = n - r.budget;
END FOR
```

条件分支：

```sql
IF boolean_expression THEN
  statement_or_compound_statement
ELSEIF boolean_expression THEN
  statement_or_compound_statement
ELSE
  statement_or_compound_statement
END IF
```

## 异常机制

```sql
DECLARE out_of_classroom_seats CONDINTION  -- 内置 SQLEXCEPTION, SQLWARNING, NOT FOUND. 
DECLARE EXIT/* 或 CONTINUE */ HANDLER FOR out_of_classroom_seats

BEGIN
  ...
  SIGNAL out_of_classroom_seats  -- 抛出异常
  ...
END 
```

## 可调用对象

输出某系讲师人数：

```sql
CREATE FUNCTION dept_count(dept_name VARCHAR(20))
  RETURNS INTEGER
BEGIN
  DECLARE d_count INTEGER;
    SELECT COUNT(*) INTO d_count
    FROM instructor
    WHERE instructor.dept_name = dept_name
  RETURN d_count;
END
-- 或等价的 PROCEDURE
CREATE PROCEDURE dept_count_proc(IN dept_name VARCHAR(20),
                                 OUT d_count INTEGER)
BEGIN
  SELECT COUNT(*) INTO d_count
  FROM instructor
  WHERE instructor.dept_name = dept_count_proc.dept_name
END
-- 调用 PROCEDURE 前，需先声明返回值：
DECLARE d_count INTEGER;
CALL dept_count_proc('Physics', d_count);
```

输出某系讲师信息：

```sql
CREATE FUNCTION instructor_of(dept_name VARCHAR(20))
  RETURNS TABLE (ID VARCHAR (5), name VARCHAR (20),
                 dept_name VARCHAR (20), salary NUMERIC (8,2))
  RETURN TABLE (
    SELECT ID, name, dept_name, salary
    FROM instructor
    WHERE instructor.dept_name = instructor_of.dept_name
  );
```

⚠️ 可以同名：

- 同名 `PROCEDURE`s 的 arguments 个数必须不同。
- 同名 `FUNCTION`s 的 arguments 个数可以相同，但至少有一个 argument 的类型不同。

## External Language Routines

```sql
CREATE FUNCTION dept_count(dept_name VARCHAR(20))
  RETURNS INTEGER
  LANGUAGE C
  EXTERNAL NAME 'path_to_dept_count'

CREATE PROCEDURE dept_count_proc(IN dept_name VARCHAR(20),
                                 OUT d_count INTEGER)
  LANGUAGE C
  EXTERNAL NAME 'path_to_dept_count_proc'
```

# Triggers

用例：规定某商品库存的最小值，当售出该商品导致库存量小于最小值时，自动下单订购该商品。

定义 trigger 需指定：

- Event: 触发 trigger 的事件（售出商品）
- Condition: 执行 actions 的条件（库存量小于最小值）
- Actions: 需要执行的操作（自动下单）

## Referential Integrity<a href id="referential"></a>

```sql
CREATE TRIGGER timeslot_check1
/* Event: */AFTER INSERT ON section
REFERENCING NEW ROW AS nrow FOR EACH ROW  -- 遍历 each inserted row
/* Condition: */WHEN (
  /* inserted time_slot_id 不属于 time_slot */
  nrow.time_slot_id NOT IN (SELECT time_slot_id FROM time_slot)
)
/* Action: */BEGIN ROLLBACK END;

CREATE TRIGGER timeslot_check2
/* Event: */AFTER DELETE ON timeslot
REFERENCING OLD ROW AS orow FOR EACH ROW  -- 遍历 each deleted row
/* Condition: */WHEN (
  /* deleted time_slot_id 不属于 time_slot */
  orow.time_slot_id NOT IN (SELECT time_slot_id FROM time_slot)
  AND
  /* 且依然被 section 中的 tuple(s) 引用 */
  orow.time_slot_id IN (SELECT time_slot_id FROM section)
)
/* Action: */BEGIN ROLLBACK END;
```

## 更新关联数据

`UPDATE` 触发的 trigger 可以指定 attributes：

```sql
CREATE TRIGGER credits_earned 
AFTER UPDATE OF takes ON grade
REFERENCING NEW ROW AS nrow
REFERENCING OLD ROW AS orow
FOR EACH ROW
WHEN (/* 新成绩及格且非空 */nrow.grade <> 'F' AND nrow.grade IS NOT NULL)
  AND (/* 旧成绩不及格或为空 */orow.grade = 'F' OR orow.grade IS NULL)
BEGIN ATOMIC
  UPDATE student SET tot_cred = tot_cred +
    (SELECT credits FROM course WHERE course.course_id = nrow.course_id)
  WHERE student.id = nrow.id;
END;
```

## Transition Tables

涉及的所有称为 transition tables：

```sql
REFERENCING NEW TABLE AS ntbl
REFERENCING OLD TABLE AS otbl
FOR EACH STATEMENT
```

⚠️ 只能用于 `AFTER` triggres。

## `DISABLE` AND `ENABLE`

Triggers 在创建时默认为启用的。可手动停用或启用：

```sql
ALTER TRIGGER <trigger_name> DISABLE;
ALTER TRIGGER <trigger_name> ENABLE;
```

# Recursive Queries

用例：找到某一课程的所有（直接或间接）先修课程。

创建递归的[临时表](#with)：

```sql
WITH RECURSIVE rec_prereq(course_id, prereq_id) AS (
  /* base query */SELECT course_id, prereq_id FROM prereq
  UNION
  /* recursive query */SELECT rec_prereq.course_id, prereq.prereq_id
    FROM rec_prereq, prereq
    WHERE rec_prereq.prereq_id = prereq.course_id
)
SELECT * FROM rec_prereq WHERE rec_prereq.course_id = 'CS-347';
```

若以 `CREATE RECURSIVE VIEW` 代替 `WITH RECURSIVE`，则创建递归的 [view](#view)。

某些数据库系统允许省略 `RECURSIVE`。

Recursive query 必须是单调的，即 $V_1\subset V_2 \implies f(V_1)\subset f(V_2)$，因此不能含有

- 以 recursive view 为输入的聚合函数
- `NOT EXISTS` 作用在用到 recursive view 的 subquery 上
- `EXCEPT` 右端项含有 recursive view

# Advanced Aggregation Features

## Ranking

假设 `studentgr_grades` 有每个学生的 `ID` 及其 `GPA`，按 `GPA` 降序排序并输出排名：

```sql
SELECT ID, RANK() OVER (ORDER BY (GPA) DESC) AS s_rank
FROM student_grades ORDER BY s_rank;
```

默认将 `NULL` 视为最大值，可手动设为最小值：

```sql
SELECT ID, RANK() OVER (ORDER BY (GPA) DESC NULLS LAST) AS s_rank
FROM student_grades ORDER BY s_rank;
```

假设有 `dept_grades(ID, dept_name, GPA)`，则可先按 `dept_name` 分组，再对各组按 `GPA` 排名：

```sql
SELECT ID, dept_name,
  RANK() OVER (PARTITION BY dept_name ORDER BY GPA DESC) AS dept_rank
FROM dept_grades
ORDER BY dept_name, dept_rank;
```

其他排名函数：

- `PERCENT_RANK` 定义为分数 $(r-1)/(n-1)$，其中 $n$ 为 tuples 个数，$r$ 为 `RANK` 结果。
- `CUME_DIST` 定义为 $p/n$，其中 $n$ 为 tuples 个数，$p$ 为排名 $\le$ 当前值的个数。
- `ROW_NUMBER` 相当于先对各 rows 排序，在输出各 row 的序号。
- `NTILE(n)` 将 tuples 按顺序均匀（各桶 tuples 数量至多相差 `1`）分入 `n` 个桶，返回每个 tuple 的桶号。

## Windowing

假设 `tot_credits(year, num_credits)` 含有每年的总学分。

对 `(year-3, year]` 的值取平均：

```sql
SELECT year,
  AVG(num_credits) OVER (ORDER BY year ROWS 3 PRECEDING)
  AS avg_total_credits
FROM tot_credits;
```

对 `(year-3, year+2)` 的值取平均：

```sql
SELECT year,
  AVG(num_credits) OVER (ORDER BY year ROWS BETWEEN 3 PRECEDING AND 2 FOLLOWING)
  AS avg_total_credits
FROM tot_credits;
```

对每年及之前所有年份的值取平均：

```sql
SELECT year,
  AVG(num_credits) OVER (ORDER BY year ROWS UNBOUNDED PRECEDING) AS avg_total_credits
FROM tot_credits;
```

Windowing 也支持按 `PARTITION` 执行：

```sql
SELECT dept_name, year,
  AVG(num_credits)
  OVER (PARTITION BY dept_name ORDER BY year ROWS BETWEEN 3 PRECEDING AND CURRENT ROW)
  AS avg_total_credits
from tot_credits_dept;
```

## Pivoting

- Cross-tabulation/pivot-table：由某个 relation `R` 导出的 table `T`，其中 `R` 的某个 attribute `A` 的值被 `T` 用作 attribute names，相应的值通常取某些聚合函数的返回值。
- Pivot attribute：上述 attribute `A`。

假设有 `sales(name, size, color, quantity)`，以下语句得到以 `(name, size, dark, pastel, white)` 为 attributes 的 pivot-table：

```sql
SELECT * FROM sales
PIVOT(
  SUM(quantity)  -- operations for getting new attribute values
  FOR color  -- the pivot attribute
  IN ('dark', 'pastel', 'white')  -- new attribute names
);
```

