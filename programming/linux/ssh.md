---
title: Secure SHell (SSH)
---

本文参考了[鳥哥](http://linux.vbird.org/vbird/)的《[文字介面連線伺服器：SSH 伺服器](http://linux.vbird.org/linux_server/0310telnetssh.php#ssh_server)》。
原文以 CentOS 6 为例讲解，本文根据 Ubuntu 16.04 LTS 的特点做了一些修改。
对于不同的 Linux 发行版，大多数命令是一致的，只在『软件包管理命令』等细节上略有区别。

# SSH 加密通信原理

『SSH (**S**ecure **SH**ell)』是一种用于远程访问 Linux 主机的 CLI 软件。
它通过对通信数据加密与解密，使得本地主机可以『安全地 (securely)』访问远程主机上的『终端 (shell)』，从而使用其资源。

SSH 对数据的加密与解密主要是依靠成对的『公钥 (public key)』和『私钥 (private key)』来实现的：

- A 将自己的公钥提供给 B。
- B 利用 A 的公钥对数据进行加密，再将加密过的数据发送给 A。
- A 利用自己的私钥，对 B 发送过来的加密数据进行解密。

建立 SSH 连接主要包括以下几步：

1. 服务端开启 SSH 服务，检查 `/etc/ssh/ssh_host_key` 等文件是否存在。如果不存在，则创建这些文件。
2. 客户端发送连接请求。
3. 服务端接到请求后，将自己的公钥（明文）发送给客户端。
4. 客户端将收到的服务端公钥与 `~/.ssh/known_hosts` 中的记录进行对比：
   - 如果是首次连接，则新建一条记录。
   - 否则检查是否与已有的记录一致：
     - 如果一致，则继续连接。
     - 否则发出警告（认为服务端被别人伪装了）并退出。
5. 客户端随机生成自己的公钥和私钥，并将公钥（明文）传送给服务端。
6. 当服务端和客户端分别拥有自己的公私钥和对方的公钥后，便建立了 SSH 连接，可以开始互相传送数据。

# 在服务端开启 SSH 服务

```shell
# 检查 SSH 服务的状态:
systemctl status ssh
# 如果需要, 安装 SSH 服务端软件:
sudo apt install openssh-server
# 开启 SSH 服务:
sudo systemctl start ssh
sudo systemctl enable ssh
# 再次检查 SSH 服务的状态:
systemctl status ssh
```

# 从客户端登入远程主机

## 从 Linux 主机访问远程 Linux 主机

```shell
# 如果需要，安装 SSH 客户端软件：
sudo apt install openssh-server
```
登入远程主机上的指定用户：
```shell
ssh user@address
# 按提示输入正确的密码后，即可登入远程主机。
```
或者，登入远程主机后，在远程主机上运行某程序：
```shell
# 等程序执行完成后，才切换回本地终端：
ssh user@address do_something
# 不等程序执行完，立即切换回本地终端：
ssh -f user@address do_something
```
输入 `exit` 则结束本次 SSH 连接，但一般不会关闭远程主机。

## 从 Windows 主机访问远程 Linux 主机

首先需要在 Windows 主机上安装 SSH 客户端软件，例如 [PuTTY](https://www.chiark.greenend.org.uk/~sgtatham/putty/)。
启动后，在地址栏输入远程 Linux 主机的『IP 地址』和『端口号（默认为 `22`）』，然后会弹出一个虚拟终端，在以下提示信息后面输入用户名，并按提示输入密码，即可建立 SSH 连接：

``` shell
login as:
```

## 修改远程 Linux 主机上当前用户的密码

建立远程连接后，在终端中输入以下命令：

```shell
passwd
```

然后根据提示输入原密码和新密码，这样在下次连接时就需要使用新密码了。

# 传输文件

## Secure File Transfer Program (SFTP)

与 `ssh` 命令类似，登入远程主机上的指定用户：

```shell
sftp user@address
# 按提示输入正确的密码后，即可登入远程主机。
# 终端提示符变为：
sftp> 
```

在 `sftp>` 后，可以输入一般的 shell 命令：

```shell
sftp> pwd   # 在远程主机上执行
sftp> lpwd  # 在本地主机上执行
```

传输文件需用以下命令：

```bash
# 上传到指定远程目录
sftp> put local_file remote_dir
# 上传到当前远程目录
sftp> put local_file
# 下载到指定本地目录
sftp> get remote_file local_dir
# 上传到当前本地目录
sftp> get local_file
```

## Secure Copy Program (SCP)

```shell
# 上传
scp local_file user@address:dir
scp -r local_dir user@address:dir
# 下载
scp user@address:file local_dir
scp -r user@address:dir local_dir
```

# 免密访问

默认情况下，每次建立 SSH 连接都需要输入远程主机上指定用户的密码。当需要频繁建立连接时，我们希望免去这一步骤。这一需求可以通过 *将客户端公钥写入服务端的 `~/.ssh/authorized_keys` 文件中* 来实现。

首先，在客户端制作密钥：

```shell
# 切换到 ~ 目录
cd ~
# 生成 SSH 密钥
ssh-keygen
# 连按三次 [Enter] 以接受默认设置
# 切换到 ~/.ssh 目录
cd .ssh
# 将公钥文件 id_rsa.pub 传送给服务端指定用户
scp id_rsa.pub user@address:~  # 需要输入密码
```

然后，在服务端添加授权信息：

```shell
# 切换到 ~ 目录
cd ~
# 检查 ~/.ssh 是否存在
ls .ssh
# 如果不存在, 建立该目录
mkdir .ssh
chmod 700 .ssh
# 将客户端公钥添加进 authorized_keys
cat id_rsa.pub >> .ssh/authorized_keys
rm id_rsa.pub
# 更改 authorized_keys 的访问权限
chmod 644 .ssh/authorized_keys 
```

至此，应当可以从客户端通过 `ssh` 命令免密登入远程主机上的该用户，或者通过 `scp` 命令免密上传或下载数据。

作为特例，如果希望能够通过 SSH 免密访问本地主机上的当前用户，只需在本地主机上开启 SSH 服务，并将当前用户的公钥写入自己的 `~/.ssh/authorized_keys` 文件。

# 在远程主机运行任务

```shell
$ nohup command [options]    # 在远程主机的前台运行任务
$ nohup command [options] &  # 在远程主机的后台运行任务
```
