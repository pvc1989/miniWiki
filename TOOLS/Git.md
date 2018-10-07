# Git

## 基本概念

**版本控制系统 (version control system, VCS)** 是一种用来追踪文件修改历史的软件, 是软件开发过程中管理源代码的必备工具. 目前最流行的 VCS 是由 Linux 之父  [Linus Torvalds](https://en.wikipedia.org/wiki/Linus_Torvalds) 发起的 [Git](https://git-scm.com/).

一个项目中所有被 Git 追踪的文件所组成的集合称为一个**仓库 (repository)**. 程序员把修改的内容(连同个人信息及备注)**提交 (commit)** 给 Git, 由 Git 将当前仓库状态保存为一个**快照 (snapshot)**. 凡是被保存为快照的仓库状态, 几乎总能通过 Git 来恢复. 由于一次提交总是对应于一个快照, 因此程序员在讨论时, 往往用"某次提交"表示"某个快照".

Git 的一大特色是支持多**分支 (branch)** 并行开发. 一个仓库至少有一个分支, 代表开发主线, 因此习惯上命名为`master`, 其最新状态一般代表项目当前的稳定版. 以`master`作为根结点, 可以分出一系列相互独立的子分支. 这些子分支又可以作为新的根节点, 分出新的子分支. 全部分支及相互间的主从关系在逻辑上构成一棵树. 在子分支上做过一些开发和提交后, 如果需要在主分支上采纳这些修改, 则需**签出 (checkout)** 到主分支, 然后将子分支上提交的内容**合并 (merge)** 到主分支中.

Git 是一种分布式的 VCS, 每个项目参与者(甚至用户和旁观者)都可以在本地主机上获得代码仓库的一份副本. 结合分支机制及代码仓库托管网站(如[GitHub](https://github.com/), [Bitbucket](https://bitbucket.org)), 可以很容易地实现多人远程合作.

## 常用命令

### 配置

```shell
# 设置提交时所附加的用户名
git config --global user.name "[name]"
```

```shell
# 设置提交时所附加的邮箱地址
git config --global user.email "[email address]"
```

```shell
# 开启命令行自动高亮
git config --global color.ui auto
```

### 新建仓库

```shell
# 在本地新建名为 project-name 的仓库
git init [project-name]
```

```shell
# 从指定远程位置获取仓库, 创建本地副本
git clone [url]
```

### 提交

```shell
# 查看自上次提交之后所做的修改
git status
```

```shell
# 暂存修改, 等待提交
git add [file]
```

```shell
# 查看未暂存的修改
git diff
```

```shell
# 查看已暂存的修改
git diff --staged
```

```shell
# 从暂存区中撤回某文件, 但保留对其所做的修改
git reset [file]
```

```shell
# 将暂存的内容提交
git commit -m "[descriptive message]"
```

### 撤销

```shell
# 撤销自某次提交以后的所有提交, 但保留对文件所做的修改
git reset [commit]
```

```shell
# 撤销自某次提交以后的所有提交, 并丢弃对文件所做的修改
# !!! 极其危险 !!!
git reset --hard [commit]
```

### 删除或移动文件

```shell
# 删除文件, 并暂存该操作
git rm [file]
```

```shell
# 将文件从版本控制系统中删除, 但仍保留在磁盘中
git rm --cached [file]
```

```shell
# 移动或重命名文件
git mv [file-original] [file-renamed]
```

### 分支

```shell
# 查看当前仓库的所有分支
git branch
```

```shell
# 新建名为 branch-name 的分支
git branch [branch-name]
```

```shell
# 切换到指定分支
git checkout [branch-name]
```

```shell
# 将指定分支上的历史合并到当前分支
git merge [branch]
```

```shell
# 删除指定分支
git branch -d [branch-name]
```

### 查看历史

```shell
# 查看当前分支的版本历史
git log
```

```shell
# 查看指定文件的版本历史
git log --follow [file]
```

```shell
# 查看两个分支之间的差异
git diff [first-branch]...[second-branch]
```

```shell
# 查看某次提交的内容
git show [commit]
```

### 同步

```shell
# 从远程仓库获取更新
git fetch [bookmark]
```

```shell
# 从远程仓库获取更新, 并合并到当前分支
git pull
# 等价于
git fetch
git merge FETCH_HEAD
```

```shell
# 将远程仓库的某个分支合并到本地当前分支
git merge [bookmark]/[branch]
```

```shell
# 将本地分支推送到 alias 所指向的代码托管网站
git push [alias] [branch]
```



