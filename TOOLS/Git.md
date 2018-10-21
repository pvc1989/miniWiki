# Git

## 基本概念

*版本控制系统 (Version Control System, VCS)* 是一种用来追踪文件修改历史的软件, 是软件开发过程中管理源代码的必备工具. 目前最流行的 VCS 是诞生于 2005 年的开源软件 [Git](https://git-scm.com/). 它是由 [Linus Torvalds](https://en.wikipedia.org/wiki/Linus_Torvalds) 为了管理 Linux 内核而创建, 并与其他代码贡献者一同开发的.

一个项目中所有被 Git 追踪的文件及修改历史所组成的集合称为一个*仓库 (repository)*. 程序员把修改的内容 (连同个人信息及备注) *提交 (commit)* 给 Git, 由 Git 将当前仓库状态保存为一个*快照 (snapshot)*. 凡是被保存为快照的状态, 几乎总能通过 Git 来恢复. 一次提交总是对应于一个快照, 因此程序员在讨论时, 往往不加区分地混用 commit 与 snapshot.

Git 的一大特色是支持多*分支 (branch)* 并行开发. 一个仓库至少有一个分支, 代表开发主线, 因此习惯上命名为 `master`, 其最新状态一般代表项目当前的稳定版. 以 `master` 作为根结点, 可以分出一系列相互独立的子分支. 这些子分支又可以作为新的根节点, 分出新的子分支. 全部分支及相互间的主从关系在逻辑上构成一棵树. 在子分支上做过一些开发和提交后, 如果想要在主分支上采纳这些修改, 那么需要先*签出 (checkout)* 到主分支, 再将子分支上提交的内容*合并 (merge)* 到主分支中.

Git 是一种分布式的 VCS, 每个代码贡献者 (甚至用户和第三方观众) 都可以在本地主机上获得代码仓库的一份副本. 结合分支机制及代码仓库托管网站 (如 [GitHub](https://github.com/), [Bitbucket](https://bitbucket.org)), 可以很容易地实现多人远程合作.

[学堂在线](http://www.xuetangx.com/)上的[软件工程](http://www.xuetangx.com/courses/course-v1:TsinghuaX+34100325_X+sp/)课程中有一节专门介绍了 Git, 注册后可以[在线观看](http://www.xuetangx.com/courses/course-v1:TsinghuaX+34100325_X+sp/courseware/e007fce98a8c437b9ce97909117ceba2/35663f6870b24a639231fe00a119b18c/).

## 常用命令

尽管我们可以通过 GUI (如 [GitHub Desktop](https://desktop.github.com), [SourceTree](https://www.sourcetreeapp.com)) 来完成许多常用 Git 操作, 但学习 Git 命令有助于提高对 GUI 中相应操作的理解; 除此之外, 有一些 Git 操作只能通过 CLI 来完成. 因此, 建议 Git 新手以 CLI 为学习重点, 在对常用命令比较熟悉之后, 再根据需要选用 GUI.

### 配置

```shell
# 查看配置信息
git config --list
# 设置提交时所附加的用户名
git config --global user.name "[name]"
# 设置提交时所附加的邮箱地址
git config --global user.email "[email address]"
# 开启命令行自动高亮
git config --global color.ui auto
```

### 获取帮助

```shell
# 通过以下三种方式, 都可以获得关于 git command 的帮助信息
git help <command>
git <command> --help
man git-<command>
```

### 新建仓库

```shell
# 在本地新建名为 project-name 的仓库
git init [project-name]
```

```shell
# 从指定远程服务器获取仓库, 创建本地副本
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
# 将所有修改暂存并提交
git commit -a -m "[descriptive message]"
```

```shell
# 补交到最近一次提交
git add [forgotten file]
git commit --amend
```

### 撤销

```shell
# 撤销自某次提交以后的所有提交, 但保留对文件所做的修改
git reset [commit]
# 撤销自某次提交以后的所有提交, 并丢弃对文件所做的修改
# !!! 极其危险 !!!
git reset --hard [commit]
```

### 删除或移动文件

```shell
# 删除文件, 并暂存该操作
git rm [file]
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
# 新建名为 branch-name 的分支
git branch [branch-name]
# 删除指定分支
git branch -d [branch-name]
```

```shell
# 切换到指定分支
git checkout [branch-name]
```

```shell
# 将指定分支上的历史合并到当前分支
git merge [branch]
```

### 查看历史

```shell
# 查看当前分支的版本历史
git log
# 或
git log -p -2
git log --oneline --decorate --graph --all
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

## 忽略规则

默认情况下, Git 会尝试跟踪一个仓库的各级目录下的所有文件. 在软件开发过程中, 经常会生成一些临时文件. 如果想要让 Git 忽略这些文件, 那么需要在仓库根目录下的 `.gitignore` 文件里列举出这些文件名 (可以使用通配符, 以使忽略规则作用到同一类文件). [GitHub](https://github.com/github/gitignore) 给出了一些常用编程语言的 `.gitignore` 范例.

## GitHub Guides

[GitHub](https://github.com/) 是目前最流行的 Git 项目托管网站. [GitHub Guides](https://guides.github.com) 介绍了依托该网站进行项目开发的流程和技巧. 这里小结 (不是完整翻译) 其中几篇的要点.

### GitHub Flow

[GitHub Flow](https://guides.github.com/introduction/flow/) 是一种基于 Git 的分支机制和 GitHub 网站进行软件开发的流程. 按时间顺序, 主要包括以下几个步骤.

#### 新建分支

在 Git 项目中, 任何人在任何时候都可以从任何一个分支上分出一个子分支. 在子分支中所做的修改, 不会立刻影响到主分支, 而是要经过主分支维护者所主导的代码审查, 才会被主分支合并. 在所有分支中, `master` 分支上的代码应当总是处于*可部署的 (deployable)* 状态.

#### 提交修改

源代码的修改历史也是源代码的一部分, 因此任何修改都应当被如实提交给 Git. 类似于[单一责任原则](https://en.wikipedia.org/wiki/Single_responsibility_principle), 一次提交应当只做一件事, 代表一组相关且内聚的操作, 并且有简洁而清晰的注释, 这样有助于追踪修改历史.

#### 请求拉取

*拉取请求 (pull request)* 是指由**子分支开发者**与**主分支维护者**进行对话所发送的消息, 一般用于申请代码审查, 或者交流其他信息. 利用 GitHub 的 `@mention` 机制, 可以在 pull request 消息中直接与指定的人员或团队进行交流.

#### 审查代码

在 pull request 中, 开发者和维护者可以就代码内容进行交流. 在讨论过程中, 开发者可以随时在子分支上继续提交和推送, GitHub 会动态地显示这些变化.

#### 集成测试

利用 GitHub, 可以在合并前对子分支中的代码进行验证. 在经过代码审查并且通过分支内的单元测试后, 可以将这些修改部署到产品中, 进行系统集成测试.

#### 合并修改

经过验证后, 主分支维护者就可以将子分支中的修改合并到主分支上. 在 GitHub 上, 可以在 pull request 里嵌入一些关键词, 用以关联一些*问题 (issue)*. 当 pull request 被合并后, 相关的 issue 也随之而被关闭. 关键词使用方法参见 [Closing issues using keywords](https://help.github.com/articles/closing-issues-using-keywords/).

