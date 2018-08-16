---
layout:     post
title:      "Docker Basic Usage-Docker Command"
subtitle:   "Framework，docker"
date:       2018-04-12
author:     "ZHYP"
tags:
    - framework
    - docker
---



## Use Dockerfile to Build Image

### Dockerfile

1. 编写Dockerfile

   首先，在项目的根目录下，新建一个文本文件`.dockerignore`，写入下面的内容:

   ```txt
   .git
   xxx.log
   ```

   上面代码表示，这几个路径要排除，不要打包进入 image 文件。如果你没有路径要排除，这个文件可以不新建。

   然后，在项目的根目录下，新建一个文本文件 Dockerfile，写入下面的内容:

   ```dockerfile
   FROM tensorflow/tensorflow:1.4.1-gpu-py3
   COPY . /app
   WORKDIR /app
   RUN npm install --registry=https://registry.npm.taobao.org
   COPY run.sh /
   EXPOSE 3000
   ```

   含义如下:

   > - FROM tensorflow/tensorflow:1.4.1-gpu-py3：该 image 文件继承自tensorflow的docker镜像，冒号表示标签，这里标签是`1.4.1-gpu-py3`
   > - COPY .  /app：将当前目录下的所有文件（除了`.dockerignore`排除的路径），都拷贝进入 image 文件的`/app`目录。
   > - WORKDIR /app：指定接下来的工作路径为`/app`。
   > - RUN npm install：在`/app`目录下，运行`npm install`命令安装依赖。注意，安装后所有的依赖，都将打包进入 image 文件。
   > - EXPOSE 3000：将容器 3000 端口暴露出来， 允许外部连接这个端口。

   ​

2. 创建 image 文件

   有了 Dockerfile 文件以后，就可以使用`docker image build`命令创建 image 文件了。

   ```bash
   $ docker image build -t my-demo .
   # 或者
   $ docker image build -t my-demo:0.0.1 .
   ```

   上面代码中，`-t`参数用来指定 image 文件的名字，后面还可以用冒号指定标签。如果不指定，默认的标签就是`latest`。最后的那个点表示 Dockerfile 文件所在的路径，上例是当前路径，所以是一个点。

   如果运行成功，就可以看到新生成的 image 文件`my-demo`了。

   ```bash
   $ docker image ls
   ```

   ​

3. 生成容器

   `docker container run`命令会从 image 文件生成容器。

4. CMD 命令

   容器启动以后，需要手动输入命令; 我们可以把这个命令写在 Dockerfile 里面，这样容器启动以后，这个命令就已经执行了，不用再手动输入了。

   ```dockerfile
   FROM tensorflow/tensorflow:1.4.1-gpu-py3
   COPY . /app
   WORKDIR /app
   RUN npm install --registry=https://registry.npm.taobao.org
   COPY run.sh /
   EXPOSE 3000
   CMD ["/run.sh", "--allow-root"]
   ```

   上面的 Dockerfile 里面，多了最后一行`CMD ["/run.sh", "--allow-root"]`，它表示容器启动后自动执行`run.sh`。

   Tips:  `RUN`命令在 image 文件的构建阶段执行，执行结果都会打包进入 image 文件；`CMD`命令则是在容器启动后执行。另外，一个 Dockerfile 可以包含多个`RUN`命令，但是只能有一个`CMD`命令。

   注意，指定了`CMD`命令以后，`docker container run`命令就不能附加命令了（比如前面的`/bin/bash`），否则它会覆盖`CMD`命令。

   ​

### Dockerfile 常用指令Tips

#### 准则

1. 尽量将Dockerfile放在空目录中，如果目录中必须有其他文件，则使用.dockerignore文件。
2. 避免安装不必须的包。
3. 每个容器应该只关注一个功能点。
4. 最小化镜像的层数。
5. 多行参数时应该分类。这样更清晰直白，便于阅读和review，另外，在每个换行符\前都增加一个空格。
6. 对构建缓存要有清楚的认识。

#### 指令注意事项

##### FROM

任何时候，尽量使用官方镜像源作为你镜像的基础镜像。

```
FROM必须是除了注释以外的第一行；
可以有多个FROM语句，来创建多个image；
```

##### RUN

RUN语句由两种方式:

1. apt-get

   尽量避免使用RUN apt-get upgrade或者dist-upgrade，因为基础镜像的很多核心包不会再未授权的容器中升级。

   apt get时, 要结合RUN apt-get update和apt-get install在同一个RUN语句下一起使用。如果将update和install分开使用，执行多个Dockerfile时，会引起缓存问题，导致后面执行的install语句会失败。

   另外，执行完apt-get语句后，最后最好加上删除安装包的语句，以减小镜像的体积。如：

   ```dockerfile
   RUN apt-get update && apt-get install -y \
       aufs-tools \
       automake \
       build-essential \
    && rm -rf /var/lib/apt/lists/*
   ```

   **注意**：官方的Debian和Ubuntu镜像会自动执行“RUN apt-get clean”，所以不需要明确地删除指令。

2. 管道使用

   很多RUN命令都需要使用到管道，如：

   ```dockerfile
   RUN wget -O - https://some.site | wc -l > /number
   ```

##### CMD

CMD语句与RUN不同，RUN是在build镜像的时候运行，而CMD语句是在build结束后运行。一个Dockerfile钟可以有多个RUN语句，虽然也可以有多个CMD语句，但是却只有最后一条CMD语句会执行。CMD语句格式为：

```dockerfile
CMD [“executable”, “param1”, “param2”…]
```

##### ENV

用于设置环境变量，设置后，后面的RUM指令就可以使用之前的环境变量了。同时，还可以通过docker run --env key=value，在容器启动时设置环境变量。如：

```dockerfile
ENV PG_MAJOR 9.3
ENV PG_VERSION 9.3.4
RUN curl -SL http://example.com/postgres-$PG_VERSION.tar.xz | tar -xJC /usr/src/postgress && …
ENV PATH /usr/local/postgres-$PG_MAJOR/bin:$PATH
```

##### EXPOSE

指明容器会监听链接的端口

##### ADD COPY

虽然ADD和COPY功能相似，但一般来讲，更建议使用COPY。因为COPY比ADD更透明，COPY只支持从本地文件到容器的拷贝，但是ADD还有一些其他不明显的特性（比如本地tar包解压缩和远程URL支持）。因此，ADD的最优用处是本地tar包自动解压缩到镜像中。如：ADD rootfs.tar.xz /。

如果有多个Dockerfile步骤用于处理不同的文件，建议分开COPY它们，而不是一次性拷贝。这可以保证每个步骤的build缓存只在对应的文件改变时才无效。比如：

```dockerfile
COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt
COPY . /tmp/
```

镜像的大小很重要，因此不鼓励使用ADD从远端URL获取包；可以使用curl或者wget来代替。这种方式你可以删除不再需要的文件，如解压缩后的tar包，从而不需要再添加额外的layer到镜像中。

##### ENTRYPOINT

语法如下：

1. ENTRYPOINT ["executable", "param1", "param2"]
2. ENTRYPOINT command param1 param2

与`CMD` 命令对比:

1. 相同点:
   - 只能写一条 `CMD` or `ENTRYPOINT` 命令. 如果写了多条，那么只有最后一条生效
   - 容器启动时才运行，运行时机相同
2. 不同点:
   - ENTRYPOINT不会被运行的command覆盖，而CMD则会被覆盖
   - 如果我们在Dockerfile种同时写了ENTRYPOINT和CMD，并且CMD指令不是一个完整的可执行命令，那么CMD指定的内容将会作为ENTRYPOINT的参数
   - 如果我们在Dockerfile种同时写了ENTRYPOINT和CMD，并且CMD是一个完整的指令，那么它们两个会互相覆盖，谁在最后谁生效

##### WORKDIR

设置工作目录，对RUN,CMD,ENTRYPOINT,COPY,ADD生效。如果不存在则会创建，也可以设置多次。

如：

```dockerfile
WORKDIR /a
WORKDIR b
WORKDIR c
RUN pwd
```

pwd执行的结果是/a/b/c