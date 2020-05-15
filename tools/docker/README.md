# Docker

## 体系结构

![](https://docs.docker.com/engine/images/architecture.svg)

### Daemon

名为 `dockerd` 的可执行程序，启动后为寄生在宿主机上的后台进程，提供对 *镜像 (image)*、*容器 (container)* 等 Docker 对象的管理服务。

### Client

名为 `docker` 的可执行程序，提供了用户与 Docker 系统互动的命令行接口。

### Registry

存储 Docker 镜像的站点，默认为 https://hub.docker.com，也可以配置成其他站点。

### Images

用于创建容器的只读模板（类似于软件安装包）。新镜像通过在旧镜像之上逐层添加配置指令而获得。通常将较为复杂的配置写在 Dockerfile 中，以便在其他主机上复现。

### Containers

镜像的可执行实例（类似于安装后的软件实例）。单个<镜像>|<安装包>可以在同一台主机上<生成多个容器>|<安装多个实例>，不同<名称的容器>|<位置的实例>运行时互不干扰。

|                          容器                          |                     虚拟机                      |
| :----------------------------------------------------: | :---------------------------------------------: |
| ![](https://docs.docker.com/images/Container%402x.png) | ![](https://docs.docker.com/images/VM%402x.png) |



## 常用命令

```shell
docker --version
docker --help
docker COMMAND --help
```

### Images

```shell
# List images:
docker images [OPTIONS] [REPOSITORY[:TAG]]
# Manage images:
docker image COMMAND
docker image ls  # List images
docker image rm [OPTIONS] IMAGE [IMAGE...]  # Remove one or more images
# Search the Docker Hub for images:
docker search [OPTIONS] TERM
docker search --filter is-official=true ubuntu
# Pull an image or a repository from a registry:
docker pull [OPTIONS] NAME[:TAG|@DIGEST]
docker pull ubuntu:16.04  # TAG 默认为 latest
```

### Containers

每个 CONTAINER 都可以通过其 ID（的头几位）或 NAME 来指明。

```shell
# List containers:
docker ps [OPTIONS]
docker ps  # 列出活动容器
docker ps -a  # 列出所有容器# Start one or more stopped containers:
# Start/Stop one or more stopped containers:
docker start [OPTIONS] CONTAINER [CONTAINER...]
docker stop [OPTIONS] CONTAINER [CONTAINER...]
# ⚠️ Kill one or more running containers:
docker kill [OPTIONS] CONTAINER [CONTAINER...]
# Attach local stdin, stdout, and stderr to a running container:
docker attach [OPTIONS] CONTAINER
# Remove one or more containers:
docker rm [OPTIONS] CONTAINER [CONTAINER...]
docker rm -f $(docker ps -qa)  # ⚠️ 强制删除所有容器
```

### Run

```shell
# Run a command in a new container:
docker run [OPTIONS] IMAGE [COMMAND] [ARG...]
docker run -it ubuntu  # 以互动模式启动虚拟终端
docker run -w DIR ubuntu  # 启动容器后以 DIR 为工作目录
docker run -v HOST_DIR:DIR ubuntu # 容器读写 DIR == 宿主读写 HOST_DIR
docker run -v `pwd`:`pwd` -w `pwd` -it ubuntu  # 挂载并进入同名目录
```

## 参考资料

- 官方文档
  - [Overview](https://docs.docker.com/get-started/overview/)
  - [Get started](https://docs.docker.com/get-started/)
  - [Reference](https://docs.docker.com/reference/)