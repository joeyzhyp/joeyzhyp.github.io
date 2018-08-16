---
layout:     post
title:      "Docker Basic Usage-Docker Command"
subtitle:   "Framework，docker"
date:       2018-04-10
author:     "ZHYP"
tags:
    - framework
    - docker
---



## Docker的基本使用

> [参考链接:官网CLI教程](https://docs.docker.com/engine/reference/commandline/cli/)
>
> [参考博客](http://www.ruanyifeng.com/blog/2018/02/docker-tutorial.html)

### Image镜像

**Docker 把应用程序及其依赖，打包在 image 文件里面。**只有通过这个文件，才能生成 Docker 容器。image 文件可以看作是容器的模板。Docker 根据 image 文件生成容器的实例。同一个 image 文件，可以生成多个同时运行的容器实例。

image 是二进制文件。实际开发中，一个 image 文件往往通过继承另一个 image 文件，加上一些个性化设置而生成。举例来说，你可以在 Ubuntu 的 image 基础上，往里面加入 Apache 服务器，形成你的 image。

```bash
# 列出本机的所有 image 文件。
$ docker image ls
# 删除 image 文件
$ docker image rm [IMAGE]
```

image 文件是通用的，一台机器的 image 文件拷贝到另一台机器，照样可以使用。一般来说，为了节省时间，我们应该尽量使用别人制作好的 image 文件，而不是自己制作。即使要定制，也应该基于别人的 image 文件进行加工，而不是从零开始制作。

**创建Docker镜像**

1. 通过dockerfile创建

   具体内容见下一章内容: **通过通过dockerfile创建镜像**

2. 通过docker commit创建

   当我们在制作自己的镜像的时候，会在container中安装一些工具、修改配置，如果不做commit保存起来，那么container停止以后再启动，这些更改就消失了。于是我们可以通过`docker commit`命令,将修改保存为一个新的image镜像.

   ```bash
   $ docker commit [container] [repo:tag]
   ```

   repo:tag 表示你要保存的image的 `REPOSITORY` 和它的 `TAG`

3. 通过docker pull从Docker Hub获取

   ```bash
   $ docker pull NAME[:TAG|@DIGEST]
   ```

**给镜像打TAG**

tag的作用主要有两点：一是为镜像起一个容易理解的名字，二是可以通过`docker tag`来重新指定镜像的仓库，这样在`push`时自动提交到仓库。

- `docker image tag`

  ```bash
  $ docker image tag SOURCE_IMAGE[:TAG] TARGET_IMAGE[:TAG]
  ```

  ​

- `docker tag`

  ```bash
  $ docker tag SOURCE_IMAGE[:TAG] TARGET_IMAGE[:TAG]
  ```

  **Examples**

  - Tag an image referenced by ID

    To tag a local image with ID “0e5574283393” into the “fedora” repository with “version1.0”:

    ```bash
    $ docker tag 0e5574283393 fedora/httpd:version1.0
    ```

    ​

  - Tag an image referenced by Name

    To tag a local image with name “httpd” into the “fedora” repository with “version1.0”:

    ```bash
    $ docker tag httpd fedora/httpd:version1.0
    ```

    Note that since the tag name is not specified, the alias is created for an existing local version `httpd:latest`

  - Tag an image referenced by Name and Tag

    To tag a local image with name “httpd” and tag “test” into the “fedora” repository with “version1.0.test”:

    ```bash
    $ docker tag httpd:test fedora/httpd:version1.0.test
    ```

  - Tag an image for a private repository

    To push an image to a private registry and not the central Docker registry you must tag it with the registry hostname and port (if needed).

    ```bash
    $ docker tag 0e5574283393 myregistryhost:5000/fedora/httpd:version1.0
    ```

    ​

### Container容器

**image 文件生成的容器实例，本身也是一个文件，称为容器文件。**也就是说，一旦容器生成，就会同时存在两个文件： image 文件和容器文件。而且关闭容器并不会删除容器文件，只是容器停止运行而已。

```bash
# 列出本机正在运行的容器
$ docker container ls
$ docker ps

# 列出本机所有容器，包括终止运行的容器
$ docker container ls --all
$ docker ps --all
```

**Run**

```bash
# 从 image 文件，生成一个正在运行的容器实例
$ docker run [IMAGE]
```

注意，`docker run` 命令具有自动抓取 image 文件的功能。如果发现本地没有指定的 image 文件，就会从仓库自动抓取。

```bash
$ docker run -p 8888:8888 --name XXX -it tensorflow/tensorflow:latest-gpu /bin/bash
```

docker run 参数含义详解:

> - -p: Publish a container’s port(s) to the host
>
> - -name: 指定启动后的容器名字，如果不指定则docker会帮我们取一个名字
>
> - -it: 容器的 Shell 映射到当前的 Shell，然后你在本机窗口输入的命令，就会传入容器
>
> - tensorflow/tensorflow: 镜像, 也可以用 Image ID代替
>
> - /bin/bash: 容器启动以后，内部第一个执行的命令。这里是启动 Bash，保证用户可以使用 Shell
>
> - -v:  docker volume 数据卷
>
>   一些实用操作:
>
>   - -v "/tmp/.X11-unix:/tmp/.X11-unix"   通过挂载本机x11目录到容器实现x11 socket共享
>   - -v "/etc/localtime:/etc/localtime:ro"   确保容器使用的是宿主机的时间
>   - -v "/dev/input:/dev/input"   挂载宿主机键盘输入到容器
>
> - --privileged: 使用privileged参数,容器内的root才真正拥有root权限, 否则容器内的root仅仅是外部的一个普通用户权限
>
> - -e: 设置环境变量

如果一切正常，运行上面的命令以后，就会返回一个命令行提示符:

```bash
root@de0561e28230:/notebooks# 
```

这表示你已经在容器里面了，返回的提示符就是容器内部的 Shell 提示符.

**Start**

前面的`docker run`命令是新建容器，每运行一次，就会新建一个容器。同样的命令运行两次，就会生成两个一模一样的容器文件。如果希望重复使用容器，就要使用`docker start`命令，它用来启动已经生成、已经停止运行的容器文件。

```bash
$ docker start [containerID or containerName]
```

这样的命令启动后, 容器在后台运行, 不会在shell中显示. 如何想要在shell中显示,看后续的`exec`  命令.

**Kill**

在容器的命令行，按下 Ctrl + c 停止进程，然后按下 Ctrl + d （或者输入 exit）退出容器。此外，也可以用`docker kill`终止容器运行。

```bash
# 在本机的另一个终端窗口，查出容器的 ID
$ docker container ls

# 停止指定的容器运行
$ docker kill [containerID or containerName]
```

**Stop**

前面的`docker kill`命令终止容器运行，相当于向容器里面的主进程发出 SIGKILL 信号。而`docker stop`命令也是用来终止容器运行，相当于向容器里面的主进程发出 SIGTERM 信号，然后过一段时间再发出 SIGKILL 信号。

```bash
$ dock stop [containerID or containerName]
```

这两个信号的差别是，应用程序收到 SIGTERM 信号以后，可以自行进行收尾清理工作，但也可以不理会这个信号。如果收到 SIGKILL 信号，就会强行立即终止，那些正在进行中的操作会全部丢失.

**Rm**

终止运行的容器文件，依然会占据硬盘空间，可以使用`docker rm`命令删除。

```bash
# 查出容器的 ID
$ docker container ls --all

# 删除指定的容器文件
$ docker rm [containerID or containerName]
```

运行上面的命令之后，再使用`docker container ls --all`命令，就会发现被删除的容器文件已经消失了。

也可以使用`docker run`命令的`--rm`参数，在容器终止运行后自动删除容器文件。

```bash
$ docker run --rm -p 8000:3000 -it tensorflow/tensorflow:latest-gpu /bin/bash
```

**注意**

> 删除容器之后，所有对容器进行的更改，包括增删的文件，安装的库，会全部消失！不会被保存至镜像中 ！！！！！！！！！！！！！！！！！

**Logs**

`docker container logs`命令用来查看 docker 容器的输出，即容器里面 Shell 的标准输出。如果`docker run`命令运行容器的时候，没有使用`-it`参数，就要用这个命令查看输出。

```bash
$ docker container logs [containerID]
```

**Exec**

`docker exec`命令用于进入一个正在运行的 docker 容器。如果`docker run`命令运行容器的时候，没有使用`-it`参数，就要用这个命令进入容器。一旦进入了容器，就可以在容器的 Shell 执行命令了。

```bash
$ docker exec -it [containerID] /bin/bash
```

**Cp**  有待修正!!!!!!!!!!!

`docker container cp`命令用于从正在运行的 Docker 容器里面，将文件拷贝到本机。下面是拷贝到当前目录的写法。

```bash
# 从容器复制到主机
$ docker container cp [containID]:SRC_PATH DEST_PATH
# 从主机复制到容器
$ docker cp SRC_PATH|- [containID]:DEST_PATH
```



### docker镜像的迁移

**从镜像迁移镜像**

```bash
$ docker image save [IMAGE] > xxx.tar
```

这样我们就得到了一个镜像的备份, 将他  `scp` 到另一个主机上, 再从备份恢复:

- 方式一:

  ```bash
  $ docker image load < xxx.tar
  ```

  这样恢复的镜像, 名字仍然是原来的名字, TAG也是原来的标签.

- 方式二:

  ```bash
  cat xxx.tar | docker import - XXX:TAG2
  ```

  此时,输入命令 ` docker images` 即可看到你刚恢复的镜像, 名字为 XXX, tag为 TAG2.

**从容器迁移镜像**

导出容器到本地文件:

```bash
$ docker container export [containID] > yyy.tar
```

这样我们就得到了一个容器的备份, 将他  `scp` 到另一个主机上, 再从备份恢复为镜像:

```bash
cat yyy.tar | docker import - YYY:TAG3
```

此时,输入命令 ` docker images` 即可看到你刚恢复的镜像, 名字为 YYY, tag为 TAG3.

### nvidia-docker基本命令

使用nvidia-docker, 在GPU上运行只需要把上面命令中的`docker` 替换成 `nvidia-docker`即可。

### Docker volume

为了解决docker中的以下几个问题而提出volume:

- 容器中的文件在宿主机上存在形式复杂，不能在宿主机上很方便的对容器中的文件进行访问
- 多个容器之间的数据无法共享
- 当删除容器时，容器产生的数据将丢失

正是为了解决这些问题，Docker引入了数据卷（volume）机制。volume是存在一个或多个容器中的特定文件或文件夹，这个目录能够独立于联合文件系统的形式在宿主机中存在，并为数据的共享与持久提供一下便利:

- volume在容器创建时就初始化，在容器运行时就可以使用其中的文件
- volume能在不同的容器之间共享和重用
- 对volume中的数据的操作会马上生效
- 对volume中数据操作不会影响到镜像(image)本身
- volume的生存周期独立于容器的生存周期，即使删除容器，volume仍然会存在，没有任何容器使用的volume也不会被Docker删除

如何使用volume:

1. 从容器中挂在volume(-v /path)

2. 从宿主机挂载volume(-v /host-path:/container-path)

   将主机的文件或文件夹作为volume挂载时，可以用多个 -v标签为容器添加多个volume，还可以使用:ro指定该volume为只读。

   注意: 利用docker commit生成新镜像，然后docker run -it 运行新镜像，发现容器挂载目录下没有任何文件了。说明生成新镜像时，是不保存挂载文件的。

3. 使用Dockerfile添加volume

   - 单个: VOLUME /data
   - 多个: VOLUME ["/data1","/data2"]

   **注意，dockerfile中使用volume是不能挂载宿主机中指定的文件夹。这是为了保证Dockerfile的可移植性，因为不能保证所有的宿主机都有对应的文件夹。**

   而且, 在Dockerfile中使用VOLUME指令后，如果尝试对这个volume进行修改，这些 **修改指令都不会生效** .

4. 共享volume/数据卷容器(--volumes-from)

   **未完待续**