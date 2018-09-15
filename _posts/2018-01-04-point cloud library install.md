---
layout:     post
title:      "Point Cloud Library Installation"
subtitle:   ""
date:       2018-01-04
author:     "Joey"
catalog:      true
tags:
    - Point Cloud
---



## PCL安装

[参考文献](http://robotica.unileon.es/index.php/PCL/OpenNI_tutorial_1:_Installing_and_testing)

### Install dependencies

```
sudo add-apt-repository ppa:v-launchpad-jochen-sprickerhof-de/pcl
sudo apt-get update
```

此时terminal中如果出现：

```
gpg: symbol lookup error: /usr/local/lib/libreadline.so.6: undefined symbol: UP
```

解决方法：

```
cd /usr/local/lib
sudo mkdir temp
sudo mv /usr/local/lib/libreadline* temp
ldconfig
```

之后再添加源并更新：

```
sudo add-apt-repository ppa:v-launchpad-jochen-sprickerhof-de/pcl
sudo apt-get update
sudo rm -rf /usr/local/lib/temp
```

之后安装：

```
sudo apt-get install build-essential cmake libpcl-1.7-all libpcl-1.7-all-dev libopenni-dev libopenni-sensor-primesense-dev
```

```
sudo apt-get install build-essential cmake cmake-curses-gui libboost-all-dev libeigen3-dev libflann-dev libvtk5-dev libvtk5-qt4-dev libglew-dev libxmu-dev libsuitesparse-dev libqhull-dev libpcap-dev libxmu-dev libxi-dev libgtest-dev libqt4-opengl-dev
```



### JDK

解压 `jdk-8u144-linux-x64.tar.gz`（网上下载或者从同事处copy）

```
sudo mkdir -p /usr/lib/jvm/
sudo cp -r jdk1.8.0_144 /usr/lib/jvm/
```

配置默认选项，版本号根据自己下载的自行更改：

```
sudo update-alternatives --install "/usr/bin/java" "java" "/usr/lib/jvm/jdk1.8.0_144/bin/java" 1
sudo update-alternatives --install "/usr/bin/javac" "javac" "/usr/lib/jvm/jdk1.8.0_144/bin/javac" 1
sudo update-alternatives --install "/usr/bin/jar" "jar" "/usr/lib/jvm/jdk1.8.0_144/bin/jar" 1
```

使用以下命令手动选择合适的选项

```
sudo update-alternatives --config java
sudo update-alternatives --config javac
sudo update-alternatives --config jar
```

查看version，确认是你install的：

```
java -version
javac -version
```

![jdk_version](/image/in-post/pcl/jdk_version.png)



### OpenNI

解压`OpenNI-master.zip` ， `Sensor-master.zip`（网上下载或者从同事处copy）

install dependencies：

```
sudo apt-get install python libusb-1.0-0-dev freeglut3-dev doxygen graphviz
```

进入解压后的OpenNI-master文件夹，打开terminal：

```
cd Platform/Linux/CreateRedist/
./RedistMaker
```

结束后，进入 `Platform/Linux/Redist/OpenNI-Bin-Dev-Linux-x64-v1.5.7.10/ ` 文件夹下：

```
sudo ./install.sh
```

同样进入解压后的`Sensor-master`文件夹，

```
cd Platform/Linux/CreateRedist/
./RedistMaker
cd Platform/Linux/Redist/Sensor-Bin-Linux-x64-v5.1.6.6/
sudo ./install.sh
```

### CUDA

默认你已经装过CUDA了

### PCL

**获取PCL source code**：

```
sudo apt-get install git
git clone https://github.com/PointCloudLibrary/pcl PCL-trunk-Source
```

可从同事处copy

**Compiling：**

进入你的pcl directory（pcl-pcl-1.7.2）进行编译：

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
ccmake .
```

在跳出的界面上，可更改设置，但一般我们选择默认选项；

按'C'键进行configure，直到下方出现：

> Press [g] to generate and exit

press 'G';

然后在terminal中继续输入：

```
make
```

等待编译通过

**Installing：**

```
sudo make install
```



**Testing:**

去同事出copy `pcl_simul`，删除build里除了`model_uniform.ply` 的所有文件，在pcl_simul/删除CMakeLists.txt文件，然后新建一个和之前内容完全一样的文件：

因为直接copy过来，会由于路径错误而make不过的情况。

之后打开cmake gui：

```
zhyp@zhyp:~/pcl_simul$ cmake-gui
```

在gui中输入CMakeLists.txt所在目录以及build目录，按config，下方提示config done之后，按generate，提示generate done后即可退出。

然后进入build文件夹：

```
zhyp@zhyp:~/pcl_simul/build$ make
Scanning dependencies of target pcl_simul
[100%] Building CXX object CMakeFiles/pcl_simul.dir/src/pcl_simul.cpp.o
Linking CXX executable pcl_simul
[100%] Built target pcl_simul
```

然后即可运行：

```
zhyp@zhyp:~/pcl_simul/build$ ./pcl_simul model_uniform.ply 2 10
```

如果报错：

```
./pcl_simul: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.20' not found (required by ./pcl_simul)
```

这是由于 libstdc++.so.6的版本过低造成的；再重新找一个高版本的就可以了

首先观察/usr/lib/x86_64-linux-gnu/文件夹下的库里面的libstdc++.so.6是什么？

```
zhyp@zhyp:/usr/lib/x86_64-linux-gnu$ ls -l | grep libstdc++.so.6
lrwxrwxrwx 1 root root       19  9月  7 13:19 libstdc++.so.6 -> libstdc++.so.6.0.19
-rw-r--r-- 1 root root   979056  5月  7  2016 libstdc++.so.6.0.19
```

看出：libstdc++.so.6是一个软链接，它链接到了实际的动态库文件：libstdc++.so.6.0.19;

然后看一下libstdc++.so.6.0.19里有什么样的版本的GLIBCXX？

```
zhyp@zhyp:/usr/lib/x86_64-linux-gnu$ strings libstdc++.so.6 | grep GLIBCXX
GLIBCXX_3.4
GLIBCXX_3.4.1
GLIBCXX_3.4.2
GLIBCXX_3.4.3
GLIBCXX_3.4.4
GLIBCXX_3.4.5
GLIBCXX_3.4.6
GLIBCXX_3.4.7
GLIBCXX_3.4.8
GLIBCXX_3.4.9
GLIBCXX_3.4.10
GLIBCXX_3.4.11
GLIBCXX_3.4.12
GLIBCXX_3.4.13
GLIBCXX_3.4.14
GLIBCXX_3.4.15
GLIBCXX_3.4.16
GLIBCXX_3.4.17
GLIBCXX_3.4.18
GLIBCXX_3.4.19
GLIBCXX_DEBUG_MESSAGE_LENGTH
```

里面确实没有 version ’GLIBCXX_3.4.20'，需要找一个新的

找一个更高版本的libstdc++.so.6:  使用locate命令来查查本地有没有：

```
zhyp@zhyp:/usr/lib/x86_64-linux-gnu$ locate libstdc++.so.6
/home/zhyp/.local/share/Trash/files/rr_robot/rr_robot_control/RR_RC/dist/Main/libstdc++.so.6
/home/zhyp/Desktop/rr_robot/rr_robot_control/RR_RC/dist/Main/libstdc++.so.6
/home/zhyp/catkin_ws/src/rr_robot/rr_robot_control/RR_RC/dist/Main/libstdc++.so.6
/usr/lib/x86_64-linux-gnu/libstdc++.so.6
/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.19
/usr/share/gdb/auto-load/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.19-gdb.py
```

没有

去网上下载 libstdc++.so.6.0.20，然后打开terminal，手动更新软连接：

```
zhyp@zhyp:~/Downloads$ sudo cp libstdc++.so.6.0.20 /usr/lib/x86_64-linux-gnu/

zhyp@zhyp:/usr/lib/x86_64-linux-gnu$ ls -l | grep libstdc++.so.6
lrwxrwxrwx 1 root root       19  9月  7 13:19 libstdc++.so.6 -> libstdc++.so.6.0.19
-rw-r--r-- 1 root root   979056  5月  7  2016 libstdc++.so.6.0.19
-rw-r--r-- 1 root root  1011824 12月 28 09:09 libstdc++.so.6.0.20

zhyp@zhyp:/usr/lib/x86_64-linux-gnu$ sudo rm -rf /usr/lib/x86_64-linux-gnu/libstdc++.so.6
zhyp@zhyp:/usr/lib/x86_64-linux-gnu$ sudo ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.20 /usr/lib/x86_64-linux-gnu/libstdc++.so.6

```

再看一下GLIBCXX的版本：

```
zhyp@zhyp:/usr/lib/x86_64-linux-gnu$ sudo strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX
GLIBCXX_3.4
GLIBCXX_3.4.1
GLIBCXX_3.4.2
GLIBCXX_3.4.3
GLIBCXX_3.4.4
GLIBCXX_3.4.5
GLIBCXX_3.4.6
GLIBCXX_3.4.7
GLIBCXX_3.4.8
GLIBCXX_3.4.9
GLIBCXX_3.4.10
GLIBCXX_3.4.11
GLIBCXX_3.4.12
GLIBCXX_3.4.13
GLIBCXX_3.4.14
GLIBCXX_3.4.15
GLIBCXX_3.4.16
GLIBCXX_3.4.17
GLIBCXX_3.4.18
GLIBCXX_3.4.19
GLIBCXX_3.4.20
GLIBCXX_DEBUG_MESSAGE_LENGTH
```

已经有 version GLIBCXX_3.4.20了

再运行

```
zhyp@zhyp:~/pcl_simul/build$ ./pcl_simul model_uniform.ply 2 10
```

即可。