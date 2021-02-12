---
title: 版本控制
---

# Git

## 参考资料
- 《[Pro Git](https://git-scm.com/book/en/v2)》系统深入地介绍了 Git 的原理及操作。
- 《[软件工程](http://www.xuetangx.com/courses/course-v1:TsinghuaX+34100325_X+sp/)》的 6.5 节介绍了 Git 的概念及操作，注册后可以在线观看。

## 基本概念
『版本控制系统 (version control system, VCS)』是一种用来追踪文件修改历史的软件，是软件开发过程中管理源代码的必备工具。目前最流行的 VCS 是诞生于 2005 年的开源软件 [Git](https://git-scm.com/)。它是由 [Linus Torvalds](https://en.wikipedia.org/wiki/Linus_Torvalds) 为了管理 Linux 内核而创建，并与其他代码贡献者一同开发的。

一个项目中所有被 Git 追踪的文件（及修改历史）所组成的集合称为一个『仓库 (repository)』。程序员把修改的内容（连同个人信息及备注）『提交 (commit)』给 Git，由 Git 将当前仓库状态保存为一个『快照 (snapshot)』。凡是被保存为快照的状态，几乎总能通过 Git 来恢复。一次 *提交* 总是对应于一个 *快照*，因此程序员在讨论时，往往不加区分地混用这两个词。

Git 的一大特色是支持多『分支 (branch)』平行开发。一个仓库至少有一个分支，代表开发主线，因此习惯上命名为 `master`，其最新状态一般代表项目当前的『稳定版』。以 `master` 作为根结点，可以分出一系列相互独立的子分支。这些子分支又可以作为新的根节点，分出新的子分支。全部分支及相互间的主从关系在逻辑上构成一棵『分支树 (tree of branches)』。如果要将 `branch1` 上的提交合并到 `branch2` 中，需要先『签出 (checkout)』到 `branch2`，再将 `branch1` 上的提交『合并 (merge)』到 `branch2` 中。

Git 是一种分布式的 VCS，每个代码贡献者及用户都可以在本地获得代码仓库的一份副本。结合分支机制及 [GitHub](#GitHub) 等代码仓库托管网站，可以很容易地实现多人远程合作。

## 常用命令
尽管我们可以通过 [GitHub Desktop](https://desktop.github.com) 等 GUI 来完成绝大多数常用 Git 操作，但有一些高级功能只能通过 CLI 来完成。Git 新手应当以 CLI 为学习重点，在对常用命令比较熟悉之后，再根据需要选用 GUI。

### 系统配置

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
# 通过以下三种方式，都可以获得关于 git command 的帮助信息
git help <command>
git <command> --help
man git-<command>
```

### 创建仓库

```shell
# 在本地新建名为 directory 的仓库
git init [directory]
```

```shell
# 从指定远程服务器获取仓库，创建本地副本
git clone <repository> [<directory>]
```

```shell
# 克隆远程仓库作为当前仓库的子模块
git submodule add <repository> [<path>]
# 克隆含有子模块的项目
git clone --recurse-submodules[=<pathspec>] <repository> [<directory>]
# 相当于
git clone <repository> [<directory>]
git submodule update --init --recursive
# 更新所有 submodule
git submodule update --remote --recursive
```

### 提交修改

```shell
# 查看自上次提交之后所做的修改
git status
```

```shell
# 暂存修改，等待提交
git add [file]
```

```shell
# 查看未暂存的修改
git diff
# 查看已暂存的修改
git diff --staged
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

### 撤销修改

```shell
# 撤销对某文件的修改（如果文件已暂存，撤销无效）
# （注：需要 Git v2.23+）
git restore [file]
# 从暂存区中撤回某文件，但保留对其所做的修改
git restore --staged [file]
```

```shell
# 撤销自某次提交以后的所有提交，但保留对文件所做的修改
git reset [commit]
# 撤销自某次提交以后的所有提交，并丢弃对文件所做的修改
# !!! 极其危险 !!!
git reset --hard [commit]
```

### 删除文件

```shell
# 删除文件，并暂存该操作
git rm [file]
# 将文件从版本控制系统中删除，但仍保留在磁盘中
git rm --cached [file]
```

### 移动文件
```shell
# 移动或重命名文件
git mv [file-original] [file-renamed]
```

### 分支管理

```shell
# 查看当前仓库的所有分支
git branch
# 新建名为 branch-name 的分支
git branch [branch-name]
# 删除指定分支
git branch -d [branch-name]
# 签出到指定分支
git checkout [branch-name]
```

```shell 
# 将指定分支上的历史合并到当前分支
git merge [branch]
# 以『变基 (rebase)』代替『合并 (merge)』
# 将 topic-branch 上的提交追加到 base-branch 上的最新提交之后
git checkout <topic-branch>
git rebase <base-branch>
git checkout <base-branch>
git merge <topic-branch>
```

### 版本标签

```shell
# 查看现有标签
git tag --list
git tag --list v4.1*
# 为最近一次提交添加轻量标签（不含其他信息）
git tag v4.1.2
# 为最近一次提交添加附注标签
git tag -a v4.1.3 -m "my version 4.1.3"
# 为某次历史提交添加标签
git tag -a <tagname> <commit>
# 将标签推送到远程仓库
git push origin <tagname>
git push origin --tags
# 删除标签
git tag -d <tagname>
git push origin --delete <tagname>
# 签出到指定标签
git checkout v4.1.1
```

### 历史记录

```shell
# 查看当前分支的版本历史
git log
git log -p -2
git log --oneline --decorate --graph --all
# 查看指定文件的版本历史
git log --follow [file]
# 查看某次提交的内容
git show [commit]
# 查看两次提交之间的差异
git diff [<options>] <commit> <commit> [--] [<path>]
# 查看两个分支之间当前的差异
git diff develop master
git diff develop..master
# 查看 master 在分出 develop 以后的提交
git diff develop...master
```

### 远程同步

```shell
# 创建并签出到远程分支 <remote>/<branch> 的本地追踪分支 <branch>
# <remote>/<branch> 称作 <branch> 的『上游 (upstream)』分支
git checkout -b <branch> <remote>/<branch>
# 该命令可以简写为
git checkout --track <remote>/<branch>
# 如果本地不存在 <branch> 分支，并且只有一个远程仓库含有 <branch> 分支，
# 则上述命令还可以进一步简写为
git checkout <branch>
```

```shell
# 从远程仓库『获取 (fetch)』更新
git fetch [remote]
# 将远程仓库的某个分支『合并 (merge)』到本地当前分支
git merge [remote]/[branch]
# 以上两步可以合并为『拉取 (pull)』操作
git pull
```

```shell
# 将本地分支推送到 bookmark 所指向的远程仓库
git push [remote] [branch]
```

## 忽略规则

默认情况下，Git 会尝试跟踪一个仓库的各级目录下的所有文件。在软件开发过程中，经常会生成一些临时文件。如果想要让 Git 忽略这些文件，那么需要在仓库根目录下的 `.gitignore` 文件里列举出这些文件名（可以使用通配符，以使忽略规则作用到同一类文件）。[GitHub](https://github.com/github/gitignore) 给出了一些常用编程语言的 `.gitignore` 范例。

# GitHub

《[GitHub Guides](https://guides.github.com)》简明扼要地介绍了依托 [GitHub](https://github.com/) 进行项目开发的流程和技巧，其中 《[Understanding the GitHub flow](https://guides.github.com/introduction/flow/)》《[Hello World](https://guides.github.com/activities/hello-world/)》《[Mastering Markdown](https://guides.github.com/features/mastering-markdown/)》《[Forking Projects](https://guides.github.com/activities/forking/)》是新手必读的简易入门教程。

## GitHub Flow
[*GitHub Flow*](https://guides.github.com/introduction/flow/) 是一种基于 Git 分支机制及 GitHub 代码仓库托管网站进行软件开发的流程，主要包括以下几个步骤。

### 新建分支
在 Git 项目中，任何人都可以从（属于自己的仓库的）任何一个分支上分出一个 *子分支*。在 *子分支* 中所做的修改，不会立刻影响到 *主分支*，而是要经过 *主分支维护者* 所主导的代码审查，才会被主分支 *合并*。在所有分支中，`master` 分支上的代码应当总是处于『可部署的 (deployable)』状态。

### 提交修改
源代码的修改历史也是源代码的一部分，因此任何修改都应当被如实提交给 Git。一次提交应当只做一件事，代表一组相关且内聚的操作，并且有简洁而清晰的注释，这样有助于追踪修改历史。

### 请求拉取
在 GitHub 上，每个注册用户都可以将其他用户的仓库作为『主仓库』，利用 [『分叉 (fork)』](https://guides.github.com/activities/forking/) 创建属于自己的『子仓库』。在 *子仓库* 中所做的修改，不会立刻影响到 *主仓库*，而是要经过 *主仓库维护者* 所主导的代码审查，才会被主仓库 *合并*。

『拉取请求 (pull request, PR)』是指由 *子仓库或子分支开发者* 与 *主仓库或主分支维护者* 进行对话所发送的消息，一般用于申请代码审查，或者交流其他信息。利用 GitHub 的 `@mention` 机制，可以在 PR 消息中直接与指定的人员或团队进行交流。

### 审查代码

在 PR 中，开发者和维护者可以就代码内容进行交流。在讨论过程中，开发者可以随时在子分支上继续提交和推送，GitHub 会动态地显示这些变化。

### 集成测试
利用 GitHub，可以在合并前对子分支中的代码进行验证。在经过代码审查并且通过分支内的单元测试后，可以将这些修改部署到产品中，进行系统集成测试。

### 合并修改
经过验证后，主分支维护者就可以将子分支中的修改合并到主分支上。在 GitHub 上，可以在 PR 里嵌入一些关键词，用以关联一些『问题 (issue)』。当 PR 被合并后，相关的 issue 也随之而被关闭。关键词使用方法参见《[Closing issues using keywords](https://help.github.com/articles/closing-issues-using-keywords/)》。

## GitHub Actions

### 基本概念

- [About continuous integration](https://docs.github.com/en/articles/about-continuous-integration)
- [Introduction to GitHub Actions](https://docs.github.com/en/actions/learn-github-actions/introduction-to-github-actions)
- [Core concepts for GitHub Actions](https://docs.github.com/en/github/automating-your-workflow-with-github-actions/core-concepts-for-github-actions)

|       概念        |         含义         |                             示例                             |
| :---------------: | :------------------: | :----------------------------------------------------------: |
|   事件 (event)    | 触发工作流的某次提交 |             推送 (push)、拉取请求 (pull request)             |
|   行动 (action)   |       单个命令       |           编译 (compile)、链接 (link)、运行 (run)            |
|    步骤 (step)    |       一组行动       |         配置 (configure)、构建 (build)、测试 (test)          |
|    任务 (job)     |       一组步骤       |                 在 Linux 系统下构建整个仓库                  |
| 工作流 (workflow) |       一组任务       |                 在多个系统下分别构建整个仓库                 |
|  运行器 (runner)  |  运行工作流的服务器  | [GitHub 提供的虚拟环境](https://docs.github.com/en/actions/reference/virtual-environments-for-github-hosted-runners) |

### 简单示例

对于用主流语言编写的项目，GitHub Actions 提供了预制的 `.yml` 文件。更一般的：

1. 在仓库根目录中创建 `.github/workflows` 子目录（只需做一次）。
2. 在该子目录中创建 `WORKFLOW_NAME.yml` 文件，例如

   ```yaml
   name: learn-github-actions  # 当前 workflow 的名称
   on: [push]  # 触发当前 workflow 的事件
   jobs:
     check-bats-version:  # 当前 job 的名称
       runs-on: ubuntu-latest  # 指定 runner 的操作系统
       steps:
         - uses: actions/checkout@v2    # 在 runner 中下载当前仓库
         - uses: actions/setup-node@v1  # 在 runner 中安装 node
         - run: npm install -g bats     # 在 runner 中用 npm 安装 bats
         - run: bats -v                 # 在 runner 中运行 bats
   ```

3. 在 GitHub Actions 页面[查看任务完成情况](https://docs.github.com/en/actions/learn-github-actions/introduction-to-github-actions#viewing-the-jobs-activity)。

### 常用语法

- [Learn YAML in five minutes](https://www.codeproject.com/Articles/1214409/Learn-YAML-in-five-minutes)
  - 缩进全部用空格实现，不要用制表符！
- [Workflow syntax for GitHub Actions](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions)

```yaml
jobs:
  build:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macOS-latest, windows-latest]
        node: [6, 8, 10]
    steps:
      - uses: actions/setup-node@v1
        with:
          node-version: ${{ matrix.node }}
```

### 缓存结果

- [Caching and storing workflow data](https://docs.github.com/en/free-pro-team@latest/actions/guides#caching-and-storing-workflow-data)
- [`actions/cache`](https://github.com/actions/cache) on GitHub

## SSH 免密连接

### SSH 原理

参见《[Secure SHell (SSH)](./linux/ssh.md)》。

### [Error: Permission denied (publickey)](https://docs.github.com/en/free-pro-team@latest/github/authenticating-to-github/error-permission-denied-publickey)

可能原因及修复办法：

1. 使用 `sudo` 连接。
2. 服务端无可用公钥。可按《[Add the SSH key to your GitHub account](https://docs.github.com/en/free-pro-team@latest/articles/adding-a-new-ssh-key-to-your-github-account)》修复。
3. 本地 `ssh-agent` 进程未启动或无相应私钥（常见于重启系统后）。可按《[Adding your SSH key to the ssh-agent](https://docs.github.com/en/free-pro-team@latest/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent#adding-your-ssh-key-to-the-ssh-agent)》修复。

命令概要：

```shell
# 检查连接：
$ ssh -T git@github.com
# 检查已加载的私钥：
$ ssh-add -l
# 加载指定私钥：
$ ssh-add -K ~/.ssh/<filename>
```
