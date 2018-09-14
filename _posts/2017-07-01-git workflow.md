---
layout:     post
title:      "Git WorkFlow"
subtitle:   ""
date:       2017-07-01
author:     "Joey"
catalog:      true
tags:
    - Git
---

## Git 工作流

> [Git教程参考](http://www.yiibai.com/git/)

### git的基本使用

1. 当前工作区的文件修改状态：`git status`
2. 将文件添加到待提交区：`git add` 
3. 提交所有文件修改并且为此次提交备注：`git commit -m "commit note"`
4. 拉取remote上源的改动：`git fetch`， `git pull`， `git merge`
   - 你的local repo不一定和remote repo完全一致的，因为其他人也在push或者merge，用这个命令是拉取remote repo上的更新和改变的**信息**
   - 但是，只会拉取信息，不会直接将源代码或者其他文件拉取到本地，这是`fetch`和`pull`命令的本质区别
   - 比如，你自己在gitlab的网页（亦即remote repo）上新建了分支，但是你的本地是没有该分支的信息的，这种情况就用`fetch`，然后就可以`checkout`到你新建的分支然后作出修改了。
   - `git pull=git fetch + git merge`，多用`fetch + merge`，少用`pull`。特别是在协作的时候。
5. 复原还在工作区的文件修改（还没有提交的修改）：`git reset HEAD filename`
6. 将新的提交推送到remote repo：`git push origin branch_name`

**注意**：

设置自己的git信息

```
git config --global user.name "abc" # 设置你自己的名字
git config --global user.email "abc@piec.com.cn" # 设置你自己的邮箱
```



### Git fetch和git pull的区别

Git中从远程的分支获取最新的版本到本地有这样2个命令：

1. git fetch：相当于是从远程获取最新版本到本地，不会自动merge

```
git fetch origin master
git log -p master..origin/master
git merge origin/master
```

 以上命令的含义：

- 首先从远程的origin的master主分支下载最新的版本到origin/master分支上

- 然后比较本地的master分支和origin/master分支的差别

- 最后进行合并

上述过程其实可以用以下更清晰的方式来进行：

```
git fetch origin master:tmp
git diff tmp 
git merge tmp
```

 从远程获取最新的版本到本地的test分支上，之后再进行比较合并

2. git pull：相当于是从远程获取最新版本并merge到本地

```
git pull origin master
```

上述命令其实相当于git fetch 和 git merge

在实际使用中，git fetch更安全一些， 因为在merge前，我们可以查看更新情况，然后再决定是否合并。



### 配合gitlab展开的工作流/协作方式

1. master分支为production/release/发布分支，master上永远是可以直接上线运行，经过了大部分测试，并且有对应文档的版本。（永久分支）
2. develop分支是开发分支，最新的bug修改，feature的更新，都会整合到这个分支。（永久分支）
3. bug分支/feature分支，以修改bug和增加feature为目的而存在的临时分支，当bug修复完毕或feature开发完毕，就会删除。（临时分支）
4. 以issue为导向的bug/feature分支建立：在任务管理工具（Tower）上获取开发相关的任务->新建对应issue->gitlab上从issue建立对应分支，记得要从develop分支开辟分支，不是从master上。gitlab现在是不能选择从哪个分支建立分支，默认的是master，所以要在gitlab上将默认分支设置成develop。
5. 权限设置：在gitlab上，repo管理员可以设置protected branches，每个branch可以设置allow to push和allow to merge两种权限，master和develop分支的两个权限都要限定为repo管理员，其他开发成员不能直接push和merge到这两个分支。防止错误的操作。
6. 用merge request合并bug/feature：当开发人员bug/feature**开发完毕后**，在gitlab上提交merge request，source branch选择新建立的分支，target branch选择develop分支。这样的好处是，消除了多人协作分支合并时，每次要git pull拉取最新develop的操作，因为在remote repo上，develop永远是最新的，而merge的动作直接操作在remote repo而不是local repo。
7. 以下展示了一次bug/feature从建立分支到merge request的流程：
   - 建立issue
   - 从issue建立branch
   - 在本地`git fetch`，将新建的分支信息拉取下来
   - 转到新的分支进行开发，`git checkout yourbranch`，运行这句命令后，可以看到分支已经显示了当前换到了新分支。
   - 然后就是开发，等开发结束，将所有改动提交到该分支。`git push origin yourbranch`
   - 转到gitlab该repo的Merge Requests页面，点击Create merge request,编写信息并submit
   - 之后就是管理员查看merge请求，解决冲突。

