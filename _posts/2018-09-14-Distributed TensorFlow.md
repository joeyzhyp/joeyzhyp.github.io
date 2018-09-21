---
layout:     post
title:      "Distributed TensorFlow"
subtitle:   ""
date:       2018-09-14
author:     "Joey"
catalog:      true
tags:
    - TensorFlow
    - Framework
---



# Distributed Tensorflow

## 详解

> 参考： [官网docs](https://www.tensorflow.org/deploy/distributed)

TensorFlow的两种不同的分布式方法：

1. 在许多GPU（和服务器）上运行并行实验来搜索好的超参数
2. 通过多个GPU（和服务器）分布式训练单个网络，减少训练时间

### PS-Worker架构

tensorflow将模型维护和训练计算解耦合，将模型训练分为两个job：

- 模型相关：模型参数存储、分发、汇总、更新，由PS执行
- 训练相关：包含推理计算、梯度计算（正向/反向传播），由worker执行

该架构下，所有的woker共享PS上的参数，并按照相同的数据流图传播不同batch的数据，计算出不同的梯度，交由PS汇总、更新新的模型参数，大体逻辑如下：

1. pull：各个woker根据数据流图拓扑结构从PS获取最新的模型参数
2. feed：各个worker根据定义的规则填充各自batch的数据
3. compute：各个worker使用第一步的模型参数计算各自的batch数据，求出各自batch的梯度
4. push：各个worker将各自的梯度推送到PS
5. update：PS汇总来自n个worker的n份梯度，来更新模型参数

分布式经典架构PS-worker会重复上面步骤，直到损失到达阈值或者轮数到达阈值。

体系架构如图所示：

![ps-worker-architecture](/img/in-post/tensorflow/ps-worker-architecture.png)

### 模型并行与数据并行

**何谓模型并行？**

> 切分模型，模型不同层执行在不同设备上，一个批次样本可以在不同设备同时执行。TensorFlow尽量让相邻计算在同一台设备上完成节省网络开销。

![tf-模型并行](/img/in-post/tensorflow/tf-模型并行.png)

**何谓数据并行？**

> 切分数据，每个设备上都有相同的模型，但是每个设备都使用不同的训练样本进行模型训练。

![tf-数据并行](/img/in-post/tensorflow/tf-数据并行.png)

我们接下来重点关注 **数据并行** 。

### 图内复制和图间复制

**何谓图内复制？**

> In this approach, the client builds a single [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph) that contains one set of parameters (in [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable) nodes pinned to `/job:ps`); and multiple copies of the compute-intensive part of the model, each pinned to a different task in `/job:worker`.

简而言之，只构建一个Graph，Graph中含有一套模型参数，放置在ps上；同时Graph中关于模型计算部分的多个副本，每个副本放置在一个Worker上，这样，多个Worker就可以同时训练同一个模型。这与tensorflow的单机多卡训练类似，单机多卡中每个GPU上的计算子图是相同的，但他们属于同一个Graph。

使用图内复制时，所有op都在同一个图中，用一个client来生成图，把所有操作分配到集群所有ps和Worker上。图内复制和单机多卡类似，如果扩展到多机多卡，那么数据分发也要是在客户端的一个节点上。

所以此方式配置起来非常简单，只需要一个client生成，其他Worker只需要join，暴露一个网络接口，等在那里接受任务就好。另外，由于图内复制各个Worker的计算子图都属于同一个Graph，所以实现同步训练会比较简单。但是，缺点就是训练数据的分发在一个Worker上，要把训练数据分到不同的机器上，严重影响了并发的训练速度。而且一旦负责生成Graph的这个client挂掉，那么整个系统就全部崩溃了，容错能力很差。

总结：

`优势`，计算节点只需要调用join()函数等待任务，客户端随时提交数据就可以训练；容易实现同步训练。

`劣势`，训练数据分发在一个Worker上，要分发给不同Worker，严重影响并发训练速度且容错能力差。

代码示例：

```python
with tf.device("/job:ps/task:0"):
  x = tf.placeholder(tf.float32, [None, n_input])
  y = tf.placeholder(tf.float32, [None, n_classes])
  keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


  weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
  }

  biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
  }


with tf.device("/job:worker/task:0"):
  pred = conv_net(x, weights, biases, keep_prob)

  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

  # Evaluate model
  correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.device("/job:worker/task:1"):
  pred2 = conv_net(x, weights, biases, keep_prob)

  cost2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred2, labels=y))
  # in-graph需要自己归并cost之类的。。。
  cost3 = tf.add(cost,cost2)

  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost3)

  # Evaluate model
  correct_pred2 = tf.equal(tf.argmax(pred2, 1), tf.argmax(y, 1))
  accuracy2 = tf.reduce_mean(tf.cast(correct_pred2, tf.float32))
  accuracy3 = tf.add(accuracy2,accuracy)

  # Initializing the variables
  init = tf.global_variables_initializer()
```



**何谓图间复制？**

> In this approach, there is a separate client for each `/job:worker` task, typically in the same process as the worker task. Each client builds a similar graph containing the parameters (pinned to `/job:ps`as before using [`tf.train.replica_device_setter`](https://www.tensorflow.org/api_docs/python/tf/train/replica_device_setter) to map them deterministically to the same tasks); and a single copy of the compute-intensive part of the model, pinned to the local task in `/job:worker`.

使用图间复制时，每一个Worker各创建一个图，训练参数保存在ps，各个工作节点独立计算，计算完成把要更新的参数发给ps，ps更新参数即可。

`优势` ：不需要数据分发，各个工作节点都创建图和读取数据训练，某个Worker挂掉不影响其他Worker训练。

`劣势` ：Worker既是图创建者又是计算任务执行者。

### 同步更新和异步更新

同步更新和异步更新是数据并行模式按照参数更新机制分类而成；

**何谓同步更新？**

> 所有worker完成本轮计算后，汇总梯度，更新模型，计算能力强的worker需要阻塞等待其他worker

synchronous training with stochastic gradient descent (SGD):

![syn-sgd](/img/in-post/tensorflow/syn-sgd.png)

**同步更新，只有在所有设备都成功计算并向PS发送了梯度后，模型才会更新。**

因此，这意味着，如果存在一个拖后腿的设备（计算速度慢或者网络连接速度慢），训练过程会严重停滞。

`优势`  ：每个训练批次考虑所有Worker训练情况，损失下降稳定， 收敛快。

`劣势`  ：性能瓶颈在最慢的Worker。Worker之间性能不同，劣势明显。

**何谓异步更新？**

> 各个worker独立训练，计算出梯度后即刻更新参数，不需要等待其他worker完成计算

Asynchronous training with stochastic gradient descent (SGD):

![asyn-sgd](/img/in-post/tensorflow/asyn-sgd.png)

**异步更新，任何设备都不会等待来自任何其他设备的模型更新；计算出梯度后立即更新。**

`优势` ：性能不存在瓶颈。

`劣势` ：每个Worker计算梯度值发回PS有参数更新冲突，影响算法收敛速度，损失下降过程抖动较大，不稳定。

**总结：**

> 同步更新、异步更新实现区别于更新参数服务器参数策略。
>
> 数据量小，各节点计算能力较均衡，用同步模型。
>
> 数据量大，各机器计算性能参差不齐，用异步模式。

### 角色分配

Tensorflow分布式是有一个`Cluster`组成， 它包含一个或者多个PS，以及多个Worker；

- ps：作为分布式训练的参数服务器，等待各个Worker来连接，聚合梯度，把更新后的weights（model）发送出去；

- worker：得到任务并进行计算；

- chief supervisors：在所有Worker中选择一个作为`主Worker`， 负责协调模型训练，模型初始化，已完成训练步骤的统计，会话监控，TensorBoard的日志保存，为从故障中恢复进行模型断点的保存和恢复。主 worker也会管理故障，在一个worker或者参数服务器失效的情况下确保容错能力。如果主worker自己宕机，那么需要从最近的模型断点开始恢复训练过程。

  **注意：** Tensorflow分布式必须显式的管理各个设备的启停。这意味着，要跟踪程序中所有TensorFlow服务器的IP地址和端口，并且手动启停这些服务器。

### 训练步骤

以图间复制，异步模式为例。

#### 1. 创建一个Cluster 

`Cluster` 就是一组job，tensorflow一般将job分为两类：

- ps：用于存储参数；
- Worker：用于执行具体的计算。

第一步，我们就要创建一个Cluster，用 `tf.train.ClusterSpec`来描述Cluster中所有的job，并且用`tf.train.Server`指定本机的任务和job类型。代码如下：

```python
# 参数服务器parameter server节点
flags.DEFINE_string('ps_hosts', '192.168.16.115:22221', 'Comma-separated list of hostname:port pairs')
# 两个worker节点
flags.DEFINE_string('worker_hosts', '192.168.16.115:22222,192.168.16.115:22223',
                    'Comma-separated list of hostname:port pairs')
# 设置job name参数
flags.DEFINE_string('job_name', None, 'job name: worker or ps')
# 设置任务的索引
flags.DEFINE_integer('task_index', None, 'Index of task within the job')
FLAGS = flags.FLAGS
def main(unused_argv):
    ps_spec = FLAGS.ps_hosts.split(',')
    worker_spec = FLAGS.worker_hosts.split(',')

    # 创建集群
    num_worker = len(worker_spec)
    cluster = tf.train.ClusterSpec({'ps': ps_spec, 'worker': worker_spec})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    
if __name__ == '__main__':
    tf.app.run()
```

 `tf.train.Server` 为每一个设备构建一个server，而且它使得每一个在当前集群（Cluster）中的server都知道其他server在做什么。

> A tf.train.Server instance encapsulates a set of devices and a tf.Session target that can participate in distributed training. A server belongs to a cluster (specified by a tf.train.ClusterSpec), and corresponds to a particular task in a named job. The server can communicate with any other server in the same cluster.

在后续过程中，我们需要用到`server.target`使得`tf.Session`能够连接到此server：

```python
server = tf.train.Server(...)
with tf.Session(server.target):
  # ...
```

#### 2. 定义PS操作

tensorflow中的参数服务器只需要管理tensorflow中的变量，不需要执行训练的过程，所以如果当前机器的Job是PS，那么就只需要调用`server.join()`即可。

```python
if FLAGS.job_name == 'ps':
  	server.join()
```

这里用到了`server.join()`  : Blocks until the server has shut down. This method currently blocks forever.

#### 3. 定义Chief Worker

在所有的Worker中，有且只有一个是Chief Worker，它除了负责计算外，还负责输出日志，保存模型等；这里设置task index为0的机器为chief worker：

```python
is_chief = (FLAGS.task_index == 0)
```

#### 4. 图间复制分配参数和计算

tensorflow中的`tf.train.replica_device_setter` 函数会自动将**此句之后所定义的**所有的参数Variables以及Variables ops分配到ps上，而将non-Variable ops分配到当前的Worker上。如果有多个ps，就轮流循环分配：

```python
with tf.device(tf.train.replica_device_setter(
            cluster=cluster
    )):
  	v1 = tf.Variable(...)  # assigned to /job:ps/task:0
  	v2 = tf.Variable(...)  # assigned to /job:ps/task:1
  	v3 = tf.Variable(...)  # assigned to /job:ps/task:0
    a = v1 + v2  # assigned to /job:worker
```

即：variables以及对variables初始化，更新等op将会被自动分配在ps上；而其他计算将被分配在Worker上；

可以自行确认：

```python
with tf.device(tf.train.replica_device_setter(cluster=cluster)):
      v = tf.Variable([1, 2])
      w = tf.Variable([2, 1])
      a = v + w
      self.assertDeviceEqual("/job:ps/task:0", v.device)
      self.assertDeviceEqual("/job:ps/task:0", v.initializer.device)
      self.assertDeviceEqual("/job:ps/task:1", w.device)
      self.assertDeviceEqual("/job:ps/task:1", w.initializer.device)
      self.assertDeviceEqual("/job:worker", a.device)
```

or：

```python
with tf.device(tf.train.replica_device_setter(
                cluster=cluster)):
            w = tf.Variable(0.0, name="weight")
            b = tf.Variable(0.0, name="bias")
            loss = tf.square(Y - tf.multiply(X, w) - b)
            global_step = tf.Variable(0)
            train_op = tf.train.AdagradOptimizer(0.01).minimize(
                loss, global_step=global_step)
            saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()
            
            # test device replica:
            print('define w in ',  w.device)
            print('define b in ',  b.device)
            print('define loss in ',  loss.device)
            print('define global_step in ',  global_step.device)
            print('define train_op in ',  train_op.device)
            print('define init_op in ',  init_op.device)
            print('init w in ',  w.initializer.device)
            print('init b in ',  b.initializer.device)
            print('init global_step in ',  global_step.initializer.device)
            
            # -------------output ----------------------
            #define w in  /job:ps/task:0
            #define b in  /job:ps/task:0
            #define loss in  /job:worker
            #define global_step in  /job:ps/task:0
            #define train_op in  /job:ps/task:0
            #define init_op in  /job:ps/task:0
            #init w in  /job:ps/task:0
            #init b in  /job:ps/task:0
            #init global_step in  /job:ps/task:0
             # -------------output ----------------------
```

此外，你也可以在`with tf.device(tf.train.replica_device_setter(cluster=cluster)):` 之外自行分配参数和计算。如：

```python
with tf.device("/job:ps/task:0"):
	X = tf.placeholder(tf.float32, [100,128,128,3], name="X")
with tf.device("/job:worker/task:0"):
	... #training ops definition
    train_step = (
            tf.train.AdamOptimizer(learning_rate)
            .minimize(loss, global_step=global_step)
```



#### 5. 定义计算图

与单机一样：

```python
loss = ...
train_op = ...
init_op = ...
saver = ...
summary_op = ...
...
```

#### 6. 创建Supervisor，管理session 

`tf.train.Supervisor`能管理训练深度学习模型的通用功能：队列操作，模型保存，日志输出，会话生成等。它其实是对Saver（模型参数存储恢复）、Coordinator（多线程服务生命周期管理）、SessionManager（单机以及分布式会话管理）三个类的封装：

`Coordinator`会监测程序的线程是否运行正常，任何异常的出现都会向Supervisor报告，此时Coordinator讲程序的停止条件设置为True，Supervisor停止训练并清理工作（关闭会话、回收内存等），其他服务检测到True后会各自关闭服务，终止线程。

`SessionManager`帮助用户创建管理单机或是分布式会话，以便简化数据流图的生命周期和维护逻辑，同时负责将checkpoint文件中加载出的数据恢复到数据流图中。

流程逻辑如下：

1. 创建Supervisor实例，构造方法需要传入checkpoint文件和summary文件存储目录（Supervisor的logdir参数）
2. 调用`tf.train.Supervisor.managed_session`，从Supervisor实例获取会话实例
3. 使用该会话执行训练，训练中需要检查停止条件，保证训练正确性。

获取managed_session时，Supervisor会通过QueueRunner同时启动一下三个服务：

- 检查点服务：将数据流图中的参数定期保存，默认10min保存一次，且会识别global_step（Supervisor的global_step参数）
- 汇总服务：默认2min一次
- 步数计数器服务：向汇总添加global_step/sec，2min一次

使用managed_session创建会话时，会**自动恢复上一次的结果并继续训练**。

代码如下：

```python
sv = tf.train.Supervisor(is_chief=is_chief, 
                         logdir=train_dir, 
                         init_op=init_op, 
                         summary_op=summary_op,
                         saver=saver,
                         save_model_secs=60,
                         save_summaries_secs=60,
                         recovery_wait_secs=1, 
                         global_step=global_step，)

config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index]
    )

# 异步：
with sv.managed_session(server.target, config=config) as sess:
```

参数介绍：

- tf.train.Supervisor:
  - is_chief：是否为chief supervisor角色
  - logdir：checkpoint文件和summary文件保存的路径
  - init_op：初始化op
  - summary_op：生成日志op，将会自动保存Summary，如果不想自动保存，设为None即可；
  - saver：将保存checkpoint的saver对象传入，supervisor就会自动保存checkpoint，如果不想自动保存，就将其设为None；
  - global_step：当前迭代的轮数，用于生成保存模型文件的文件名
  - save_model_secs：自动保存模型的时间间隔，
  - save_summaries_secs：自动日志输出的时间间隔

- tf.ConfigProto:

  - allow_soft_placement：为了避免手动指定的设备不存在这种情况, 将allow_soft_placement设置为 True, 这样 tensorFlow 会自动选择一个存在并且支持的设备来运行 operation.
  - log_device_placement：记录operations 和 Tensor 被指派到哪个设备上运行
  - device_filters：硬件过滤器，如果被设置的话，会话会忽略掉所有不匹配过滤器的硬件。

- managed_session：

  异步接口，用于生成会话。参数初始化之后即可运行

- prepare_or_wait_for_session：

  同步模式，用于生成会话。参数初始化完成并且chief supervisor准备好了才可以运行

如果你去深入了解`managed_session` 的Python实现，你会发现，它只是对`prepare_or_wait_for_session` 的一个调用！两者最大的区别其实只是：

- `prepare_or_wait_for_session` ： return sess
- `managed_session` ：yield sess

所以tf此处的异步，其实是通过yield同步函数来实现的。

#### 7. 迭代训练

```python
with sv.managed_session(server.target, config=config) as sess:
	while not sv.should_stop():
      	sess.run([train_op, global_step])
```

然后用这同一份代码在不同的机器上运行：

```bash
python worker.py --job_name=ps --task_index=0
python worker.py --job_name=worker --task_index=0
python worker.py --job_name=worker --task_index=1
```

#### 附：同步模式区别

##### 1. 定义计算图时

和异步模式定义计算图时的区别为：同步模式需要使用 `tf.train.SyncReplicasOptimizer` 函数处理同步更新；其次 `opt.minimize()`或`opt.apply_gradients()`的时候一定要传入`global_step`(用来同步的)。代码示例如下：

```python
opt = tf.train.SyncReplicasOptimizer(
		tf.train.GradientDescentOptimizer(lr),
		replicas_to_aggregate=num_worker,
  		total_num_replicas=num_worker,
  		replica_id=FLAGS.task_index
)
train_op=opt.minimizie(loss, global_step=global_step)
```

**参数说明：**

- tf.train.GradientDescentOptimizer(lr)：定义优化方法
- replicas_to_aggregate：每一轮更新参数需要多少个Worker计算出的梯度，即每一步更新中并行的Worker数。
- total_num_replicas：指定的Worker的总数量
- replica_id：当前Worker的index

根据 `replicas_to_aggregate` 和 `total_num_replicas` 的释义，可以得出：

- replicas_to_aggregate=total_num_replicas：全民参与，一个worker领取一个batch数据
- replicas_to_aggregate>total_num_replicas：能者多劳，先完成自己batch的worker会继续领取未训练数据，PS会等到梯度份数到达并行数后进行模型参数计算
- replicas_to_aggregate<total_num_replicas：替补等位，存在空闲的worker，取代可能出现的异常worker，确保训练过程不停滞。

##### 2. 定义Chief Worker时

在同步模式下，chief worker需要协调不同worker计算得到的参数梯度，并最终更新参数，这就需要它做一些额外的初始化工作：

```python
if is_chief:
  	chief_queue_runner = opt.get_chief_queue_runner()
    init_tokens_op = opt.get_init_tokers_op(0)
```

chief woker做额外工作的原因见此目录附。

##### 3. 创建Supervisor时

与异步模式使用`managed_session` 不同，同步模式使用`prepare_or_wait_for_session` 创建会话。

##### 4. 迭代训练时，chief worker有额外动作

在开始训练前，chief worker需要启动协调同步更新的队列并执行初始化操作。

```python
if is_chief:
  sv.start_queue_runners(sess, [chief_queue_runner])
  sess.run(init_tokens_op)
```

其他与异步模式相同

chief woker做额外工作的原因见此目录附。

##### 附：chief worker在同步模式的额外动作

同步模式有两个概念：

**梯度聚合器**：

每一个模型参数有一个自己队列，收集来自不同worker的梯度值，梯度聚合器包含M个队列对应M个模型参数，每个队列收集来自N个worker计算出来的N个梯度值。

**同步标记队列：**

存储同步标记，实际上就是N个global_step值，每个worker领取一个，用于控制同步

以所有Worker参与同步，即`replicas_to_aggregate=total_num_replicas` ，为例：

**worker工作模式**：

1. 从同步标记队列领取一个global_step，表示全局训练步数的同步标记
2. 将同步标记值赋予worker的本地训练步数local_step
3. 从PS获取最新模型参数
4. 计算出M个梯度值
5. 将M个梯度值推送到PS上的M个梯度队列中

**PS工作模式**：

1. 从梯度聚合器上收集worker推送过来的梯度值，每个队列收集N份（对应N个global_step下训练值）后，计算均值，收集齐M个均值后，得到M对{模型参数，梯度值}的聚合元组
2. 更新模型参数
3. 向同步标记队列推送N个global_step+1标记

聚合器收集梯度值并校验local_step是否符合global_step，是则接收梯度值，计算能力强的worker提交梯度后由于没有同步标记可以领取所以被阻塞，PS集齐N份后更新参数，发布下次N个同步标记，开始下一步训练。

由于初始PS不会更新参数发布同步标记，所以需要初始化同步标记队列——sync_init_op，直接向队列注入N个0标记。

> 同步标记队列即是由chief Worker来实现

## 注意

按`ctrl z` 退出程序，并且需要执行 `ps -a ` 以及 `kill -9 xxx` 来杀掉进程

## 问题

### 1. CreateSession still waiting for response from worker

所有分布式进程都启动后，chief worker进程不断在打印如下信息，没有开始训练:

```bash
I tensorflow/core/distributed_runtime/master.cc:221] CreateSession still waiting for response from worker: /job:ps/replica:0/task:0
I tensorflow/core/distributed_runtime/master.cc:221] CreateSession still waiting for response from worker: /job:worker/replica:0/task:1
I tensorflow/core/distributed_runtime/master.cc:221] CreateSession still waiting for response from worker: /job:ps/replica:0/task:0
I tensorflow/core/distributed_runtime/master.cc:221] CreateSession still waiting for response from worker: /job:worker/replica:0/task:1
```

**解决方案：**

首先保证job_name,task_index,ps_hosts,worker_hosts这四个参数都是正确的,考虑以下这种情况是不正确的：

在一个IP为192.168.1.100的机器上启动ps或worker进程：

--job_name=worker

--task_index=1

--ps_hosts=192.168.1.100:2222,192.168.1.101:2222

--worker_hosts=192.168.1.100:2223,192.168.1.101:2223

因为该进程启动位置是192.168.1.100，但是运行参数中指定的task_index为1，对应的IP地址是ps_hosts或worker_hosts的第二项（第一项的task_index为0)，也就是192.168.1.101，和进程本身所在机器的IP不一致。

如果以上正确，那么考虑使用 ` device_filters` :

启动分布式TF集群时，每个节点都会启动一个Server. 默认情况下，每个节点都会跟其他节点进行通信，然后再开始创建Session. 在集群节点多时，会带来两个问题：

1. 由于每个节点两两通信，造成训练的启动会比较慢。
2. 当某些worker挂掉重启（例如因为内存消耗过多），如果其他某些worker已经跑完结束了，重启后的worker就会卡住，一直等待其他的worker。此时会显示log: `CreateSession still waiting for response from worker: /job:worker/replica:0/task:x`.

解决这个问题的方法是使用device filter. 例如:

```python
config = tf.ConfigProto(
        log_device_placement=True,
        device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index]
    )

with sv.managed_session(server.target, config=config) as sess:
```

另外一种情况也会导致该问题的发生，从TensorFlow-1.4开始，分布式会自动使用环境变量中的代理去连接，如果运行的节点之间不需要代理互连，那么将代理的环境变量移除即可，在脚本的开始位置添加代码：

```python
# 注意这段代码必须写在import tensorflow as tf或者import moxing.tensorflow as mox之前

import os

os.enrivon.pop('http_proxy')
os.enrivon.pop('https_proxy')
```



> 参考链接：
>
> [1. 使用device_filters ](http://blog.codescv.com/2018/05/20/dist-tf-for-sparse-models.html)
>
> [2. http_proxy ](https://bbs.huaweicloud.com/blogs/463145f7a1d111e89fc57ca23e93a89f)
>
> [3. GitHub Issue](https://github.com/tensorflow/tensorflow/issues/12745)