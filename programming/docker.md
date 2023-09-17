---
title: Docker
---

# 体系结构

![](https://docs.docker.com/engine/images/architecture.svg)

## Daemon

名为 `dockerd` 的可执行程序，启动后为寄生在宿主机上的后台进程，提供对『镜像 (image)』、『容器 (container)』等 Docker 对象的管理服务。

## Client

名为 `docker` 的可执行程序，提供了用户与 Docker 系统互动的命令行接口。

## Registry

存储 Docker 镜像的站点，默认为 https://hub.docker.com，也可以配置成其他站点。

## Images

用于创建容器的只读模板（类似于软件安装包）。新镜像通过在旧镜像之上逐层添加配置指令而获得。通常将较为复杂的配置写在 Dockerfile 中，以便在其他主机上复现。

## Containers

镜像的可执行实例（类似于安装后的软件实例）。单个【镜像】/【安装包】可以在同一台主机上【生成多个容器】/【安装多个实例】，不同【名称的容器】/【位置的实例】运行时互不干扰。

# 常用命令

```shell
docker --version
docker --help
docker COMMAND --help
```

某些常用（但不是所有）选项支持简化写法：

```shell
docker -v  # docker --version
docker -h  # docker --hep
```

初学时应掌握完整写法，熟练后再用简化写法。

## Images

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

## Containers

```shell
# List containers:
docker ps [OPTIONS]
docker ps  # 列出活动容器
docker ps --all  # 列出所有容器
# Start/Stop one or more stopped containers:
docker start [OPTIONS] CONTAINER [CONTAINER...]
docker stop  [OPTIONS] CONTAINER [CONTAINER...]
# ⚠️ Kill one or more running containers:
docker kill [OPTIONS] CONTAINER [CONTAINER...]
# Attach local stdin, stdout, and stderr to a running container:
docker attach [OPTIONS] CONTAINER
docker start --attach CONTAINER
# Remove one or more containers:
docker rm [OPTIONS] CONTAINER [CONTAINER...]
docker rm --force $(docker ps -qa)  # ⚠️ 强制删除所有容器
```

每个 `CONTAINER` 都可以通过其『`ID` 的前几位』或『`NAME`』来指明。

## Run

```shell
# Run a command in a new container:
docker run [OPTIONS] IMAGE [COMMAND] [ARG...]
docker run --interactive --tty IMAGE  # 以互动模式启动虚拟终端
docker run --workdir WORK_DIR IMAGE  # 启动容器后以 DIR 为工作目录
docker run --volume HOST_DIR:WORK_DIR IMAGE # 将 HOST_DIR 挂载为 WORK_DIR
# 简化写法：
docker run -it IMAGE
docker run -w WORK_DIR IMAGE
docker run -v HOST_DIR:WORK_DIR IMAGE
```
所谓『挂载 (mount)』是指『容器读写 `WORK_DIR`』等效于『宿主读写 `HOST_DIR`』。

【常用场景】将宿主的当前目录挂载到容器的同名目录：

```shell
docker run -v `pwd`:`pwd` -w `pwd` -it IMAGE  # 路径不能有空格
docker run --mount type=bind,src="$(pwd)",dst="$(pwd)" -w "$(pwd)" -it IMAGE  # 【推荐】可读性更强、路径更灵活
```

# 参考资料

- 官方文档
  - [Overview](https://docs.docker.com/get-started/overview/)
  - [Get started](https://docs.docker.com/get-started/)
  - [Reference](https://docs.docker.com/reference/)
