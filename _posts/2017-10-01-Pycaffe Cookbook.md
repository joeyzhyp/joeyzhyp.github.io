---
layout:     post
title:      "PyCaffe Cookbook"
subtitle:   "The Installation and Usage of PyCaffe and the General Steps of Training and Testing"
date:       2017-10-01
author:     "Joey, WFZ"
catalog:      true
tags:
    - Caffe
    - Framework
---



# PyCaffe Cookbook (Version 1.0)

Last updated: Oct 1st, 2017 2:56 PM
## Installation
You may have a look at the installation guide in the official website: http://caffe.berkeleyvision.org/installation.html

If you were using ubuntu system, you can follow the instructions below. You may find some related software in ./Installation folder.
1. General dependencies
```bash
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev
sudo apt-get install libhdf5-serial-dev protobuf-compiler
sudo apt-get install --no-install-recommends libboost-all-dev
sudo apt-get install liblmdb-dev libgoogle-glog-dev libgflags-dev
```
2. OpenCV (recommend via source code)
  Download OpenCV from the official website: http://opencv.org/downloads.html
  Follow instruction guide here: http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html
3. Blas
```bash
sudo apt-get install libatlas-base-dev
```
4. Python
```bash
sudo apt-get install python-dev
```
5. CUDA (recommend .run file)
  Download CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads. Select runfile(local) as the installer type. The downloaded file should be like cuda_8.0.61_375.26_linux.run.
  Some libraries (including OpenGL, GLU, GLUT, X11) have to be installed to run CUDA samples:
```bash
sudo apt-get install freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev
```
Quit the X session by pressing ctrl+alt+(F1~F6). In command line mode:
```bash
sudo service lightdm stop
```
Then go to the folder containing the .run file and type in:
```bash
chmod +x cuda_8.0.61_375.26_linux.run
sudo ./cuda_8.0.61_375.26_linux.run
```
Here you are supposed to have installed the graphic driver already. Hit the Space until the end of the instructions. Then type in accept to contiunue. Note that **DON'T** install NVidia graphic driver again!
The environment variables need to be changed so that your system can find CUDA's libraries and binaries:
```bash
sudo nano /etc/ld.so.conf
```
Append the following line in the .conf file:
```bash
/usr/local/cuda/lib64
```
Save with ctrl+O and Enter, then exit with ctrl+X. Reload the cache of the dynamic linker with:
```bash
sudo ldconfig
```
Now, append CUDA's bin directory to your PATH:
```bash
sudo nano ~/.bashrc
```
Append the following line at the end of the file:
```bash
export PATH=$PATH:/usr/local/cuda/bin
```
CUDA is now installed and you can build CUDA samples and run the tests.
6. CuDNN:
  Download the CuDNN from: https://developer.nvidia.com/cudnn
  Extract the zip file. Go to the unzipped folder and copy the header and lib files:
```bash
cd cuda/include/
sudo cp cudnn.h /usr/local/cuda/include/
cd ../lib64
sudo cp lib* /usr/local/cuda/lib64/
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
```
Then link the files. Note that the version should be adjusted accordingly.
```bash
cd /usr/local/cuda/lib64/
sudo rm -rf libcudnn.so libcudnn.so.5
sudo ln -s libcudnn.so.5.1.5 libcudnn.so.5
sudo ln -s libcudnn.so.5 libcudnn.so
sudo ldconfig
```
7. Python dependencies
```bash
sudo apt-get install python-pip
```
Go to the folder caffe/python and open a terminal:
```bash
for req in $(cat requirements.txt); do sudo pip install $req; done
```

## Complilation
Go to /caffe and make a copy of MaMakefile.config.examples:
```bash
cp Makefile.config.example Makefile.config
```
If you were using Ubuntu 16.04, you need to find the folder containing libhdf5_serial.so.10.1.0 (according to your hdf5 version) and do the file linking:
```bash
sudo ln libhdf5_serial.so.10.1.0 libhdf5.so
sudo ln libhdf5_serial_hl.so.10.0.2 libhdf5_hl.so
sudo ldconfig
```
And change the INCLUDE_DIRS in Makefile.config as:
```bash
INCLUDE_DIRS:= $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial
```
Notes on Makefile.config:
- For CPU&GPU Caffe, no further changes are needed.
- For adding CuDNN acceleration, uncomment the switch:
```
USE_CUDNN:=1
```
- For CPU-only Caffe, uncomment the switch (**NOT** recommended):
```
CPU_ONLY:=1.
```
After all these having done, you can make the Caffe now:
```bash
make all -j4
make test -j4
make runtest -j4
make pycaffe -j4
make distribute
```

You can add PyCaffe to your path:
```bash
sudo nano ~/.bashrc
```
And append the following line in the end:
```bash
export PYTHONPATH=~/path/to/caffe/python:$PYTHONPATH
```
## Prerequisites
This cookbook intends to provide basic usage guidances to PyCaffe, a Python interface of Caffe. It assumes that the readers have gone through the official Caffe tutorial and have basic knowledge of Google's Protocol Buffers. If the readers are not familiar with them, please refer to:
** Official Caffe tutorial **: http://caffe.berkeleyvision.org/tutorial/
** Google's Protocol Buffers **: https://developers.google.com/protocol-buffers/
You may also refer to cs231n Lecture 12, available at:
http://cs231n.stanford.edu/slides/winter1516_lecture12.pdf

## General Steps of Training and Testing
1. Preparing training and testing data
2. Creating LMDB or other data source formats
3. Data preprocessing (section **Miscellaneous**)
4. Defining network structures
  - for visualizing network structures, refer to section **Miscellaneous**
5. Defining solver
6. Training and testing
  - for fine-tuning, refer to section 3.2
  - for online training, refer to section 3.4
7. Deploying (section 4)

[TOC]
## 1. Preparing Data
Here we provide training and testing data in ./Data. The data was downloaded form Kaggle. Images of cats and dogs are contained in the dataset with keywords 'cat' and 'dog' respectively in their filenames.
## 2. Creating Database
Caffe supports many methods for data input. However, to get more out of its amazing GPU accelerated training, you certainly don't want to let the file I/O slow you down, which is why a database system is normally employed. The most common option is to use LMDB. You can query any file from it using a scripting language, and read big chunks of data from it much faster than reading a file.
Technically LMDB provides key-vale storage, where each key-value pair will be a sample in our dataset. The key will simply be a string version of an ID value, and the value will be a serialized version of the Datum class (which are built using protobuf) in Caffe. For detailed LMDB introductions, you may refer to the official website: https://lmdb.readthedocs.io/en/release/#
However, we would like to show an example here to illustrate its usage in Python.To understand the code better, you may need to read the Google Protocol Buffer tutorial: https://developers.google.com/protocol-buffers/docs/pythontutorial.
### 2.1 Create LMDB
The source code is available at ./Code/Section2/create_lmdb.py. You may read the functions from the comments. If you run the program like this:
```bash
python create_lmdb.py
```
Be default train_lmdb and valid_lmdb will be created in folder ./Result/Section2/. You can change the source and target names. Call helper function to display more info:
```bash
python create_lmdb.py -h
```
Some source code from caffe source code may be helpful in understanding the code:
- Datum: /caffe/src/caffe/proto/caffe.proto
- array_to_datum: /caffe/python/caffe/io.py

### 2.2 Retrieve Data from LMDB
You can also retrieve data from previously created lmdb database. The source code is available at ./Code/Section2/retrieve_lmdb.py. You can run the code like this:
```bash
python retrieve_lmdb.py
```
Some source code from caffe source code may be helpful in understanding the code:
- datum_to_array: /caffe/python/caffe/io.py

## 3. Data Preprocessing
Data is often pre-processed before it is passed to Caffe. There are many methods for preprocess data. Herein, we only list two basic ones: Input Data Preprocessing and Input Mean Calculation.
###3.1 Input Data Preprocessing
Input data preprocessing can be done either by calling a helper function named transformer defined in /caffe/python/caffe/io.py or by defining transform_param in the net prototxt, depending on the usage. For the first method, we append a code below, readers can adjust the parameters to make it runnable and see the effect:
```python
import caffe

# instantiating a transformer object by specifying the data blob size
#  you can define multiple transformers identified by keys in a dictionary
#  in this case, 'data' is an identifier of the transformer
#  note that Caffe stores images in the format of channel * height * width (C*H*W)
transformer = caffe.io.Transformer({'data': [batch_size, channel, height, width]})

# setting up the transformer
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension, i.e. from H*W*C to C*H*W
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255] if the original image is represented by numbers in [0, 1] (scale before subtracting the mean)
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR (if necessary)
transformer.set_input_scale('data', 0.5)  # scale after subtracting the mean, not sure what this is for

# loading image
image = caffe.io.load_image('image.jpg')

# transforming the image
transformed_image = transformer.preprocess('data', image)

# if you want to visualize the images in a data blob
# you may use deprocess to reverse the process
original_image = transformer.deprocess('data', transformed_image)
```
For the second method, we will discuss it further in Section 3.
###3.2 Input Mean Calculation
Herein, we are particular interested in calculating image mean of the traing set, which will be used later in Section 4.
####3.2.1 Generate Image Mean
Usually, the input image for training neural networks has zero mean. Caffe provides a useful tool for computing mean value of images contained in a database. Type in the following command to see its usage:
```bash
/path/to/caffe/build/tools/compute_image_mean
```
You should create a mean.binaryproto file yourself and save it in /Result/Section3/.

####3.2.2 Read Image Mean
The generated file from 3.2.1 is binary. But you can tranform it to numpy array using convert_bp_to_array.py in ./Code/Section3.

## 4. Defining Networks
A neural network architecture is defined in a protocol buffer definition file (prototxt) in Caffe. This file can be written manually or it can be created by calling some PyCaffe APIs. We will create a network called CaffeNet in this section using PyCaffe. Instead of mixing two phases (train and validation) together, we created two networks with the same structure but for different phases. The code is available at ./Code/Section4/.
- CaffeNet_train.py is used to produce training net.
- CaffeNet_valid.py is used to produce validation net.

### 4.1 Import
If you did not append the caffe PATH, you need to add it to system:
```python
# pycaffe root needs to be added to system path before importing caffe
import sys
sys.path.append("/home/username/caffe/python")
# import caffe
import caffe
# layers: caffe built-in layers
# params: caffe pre-defined parameters
from caffe import layers as L, params as P
```
- The functions in caffe python module can be seen in /caffe/python/caffe/\_\_init\_\_.py
- layers and params are defined in /caffe/python/caffe/net_spec.py

### 4.2 Defining Layers
Different types of layers are defined in the same format but with different layer type names and parameters. Convolutional layer is used as an example here to illustrate how to define a layer.
```python
conv = L.Convolution(bottom, **kw)
# conv: this variable points to the object of the layer.
# Convolution: this is the layer type name pre-defined in Caffe.
# bottom: this is the name of the layer at the bottom of the current layer. Note that input layers do not have bottoms. Some layers accept multiple bottoms.
# **kw: key word parameters that depend on the type of layer.
```
The file caffe_root/src/caffe/proto/caffe.proto contains everything you need to know about layers, including their type names and parameters. The readers are strongly recommended to familiarize themselves with the structure of this document.
** Important Notes:**
- If a parameter is defined as **repeated** in caffe.proto, it means multiple values can be given to it, and you should use a python list object to wrap these values.
- If a parameter itself is a **message** defined in caffe.proto, you need to find the definition of this **message** and wrap its parameters in a python dictionary.
- All the layers are inheritated from the class Layer, therefore, parameters in **LayerParameter** are applicable to all types of layers.

Some commonly used layers are introduced here:
- **Input**: a simple input layer usually used in deployment when input data (e.g. images) is fed into the network in realtime (i.e. not stored in file systems or databases such as LMDB).
  *Parameters*:
  --** shape** [BlobShape]: shape itself is an instantiation of Blobshape which contains one **repeated** parameter **dim**.
  *Example*:
```python
input = L.Input(shape=dict(dim=[50, 3, 240, 240]))
```
- **Data**: an input layer used for data stored in databases (caffe supports LevelDB and LMDB). You can specify both input data and label data, or just one of them,  in this layer.
  *Parameters*:
  --**source** [string]: a path to the database folder.
  --**batch_size** [uint32]: batch size
  --**backend** [DB]: choose a database backend (LEVELDB or LMDB)
  --**transform_param** [TransformationParameter]: this is a parameter from the parent class **Layer**, it is used for image pre-processing.
  --**ntop**[uint32]: number of top layers. It is defined in net_spec.py. It should be emphasized here that if ntop > 1, the returned object from a layer definition is a tuple of sets of 'Top' objects of size ntop.
  *Example*:
```python
input, label = L.Data(source=path_to_db, batch_size=50, backend=P.Data.LMDB, transform_param=dict(mean_value=[104, 117, 123]), ntop=2)
# transform the image by subtracting the mean value
```
- **DummyData**: dummy data layer, used for debugging. The content in this layer is generated by a filler.
  *Parameters*:
  --**shape** [BlobShape]: the same as in the **Input** layer.
  --**data_filler** [FillerParameter]: used to fill in the content of this layer.
  *Example*:
```python
dummy_data = L.DummyData(shape=dict(dim=[1, 3, 227, 227]), data_filler=dict(type='constant', value=1.0))
# fill in this layer with all ones, you can find more information about fillers in caffe.proto
```
- **Convolution**: the convolutional layer
  *Parameters*:
  --**num_output** [uint32]: The number of outputs for the layer
  --**kernel_size** [uint32]: kernel size
  --**stride** [uint32]: stride, defaults to 1
  --**pad** [uint32]: padding size, defaults to 0
  --**param** [ParamSpec]: this is a parameter from the parent class Layer, it is used to control the updating of individual layers based on the global updating rule.
  --**weight_filler** [FillerParameter]: the filler used for initializing the weights.
  --**bias_filler** [FillerParameter]: the filler used for initializing the bias.
  *Example*:
```python
conv = L.Convolution(bottom, kernel_size=7, stride=2, num_output=64, pad=1, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0.0))
# param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]: the first dict is for weights and the second one is for bias. lr_mult is a multiplier applied to the global learning rate. If you want to freeze some layers, you can set lr_mult=0 to stop them from updating. decay_mult is a multiplier applied to the global regularization strength.
# weight_filler=dict(type='gaussian', std=0.01): weights are initialized randomly with a Gaussian distribution with std=0.01.
# bias_filler=dict(type='constant', value=0.0): biases are initialized to constant 0.0.
```
- **ReLU**: ReLu layer
  *Parameters*:
  --**in_place** [bool]: this parameter cannot be found in caffe.proto. If it is True, its output will overwrite the output of its bottom layer for the purpose of saving memory.
  *Example*:
```python
relu = L.ReLU(bottom, in_place=True)
```
- **Pooling**
```python
pool = L.Pooling(bottom, pooling_param=dict(pool=P.Pooling.MAX, kernel_size=3, stride=2))
```
- **InnerProduct**： inner product layer
  *Parameters*:
  --**num_output** [uint32]: The number of outputs for the layer
  --**param** [ParamSpec]: this is a parameter from the parent class Layer, it is used to control the updating of individual layers based on the global updating rule.
  --**weight_filler** [FillerParameter]: the filler used for initializing the weights.
  --**bias_filler** [FillerParameter]: the filler used for initializing the bias.
  *Example*:
```python
fc = L.InnerProduct(bottom, num_output=128, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0.0))
```
- **Softmax**: 1D softmax layer
  *Parameters*:
  --**axis** [int32]: the axis along which to perform the operation, defaults to 1.
  *Example*:
```python
softmax = L.Softmax(bottom, axis=1)
```
- **EuclideanLoss**: a type of loss layer used at the end of a network for training and testing. It calculates the Euclidean distance between network output and label. Euclidean loss is commonly used in regression analysis. This layer is not required in deployment.
  *Parameters*:
  This layer does not have any key word parameter and it cannot be found in caffe.proto.
  *Example*:
```python
loss = L.EuclideanLoss(bottom=['output', 'label'])
# this layer accepts two input bottoms, the bottom names should be supplied as a list of strings
```
- **SoftmaxWithLoss**
```python
loss  = L.SoftmaxWithLoss(bottom1, bottom2)
```

##### Python Layer
Python layer is a special type of layer designed for developing user-defined layers in Python.
```python
# TODO
```

### 4.3 Defining Networks
PyCaffe provides a wrapper class that simplifies the definition of a network as well as the generation of a network prototxt file.
```python
n = caffe.NetSpec()
# NetSpec is a shell that wraps all the layers, it does nothing to those layers
# NetSpec provides a method called to_proto, which converts all the layers inside it to a single proto object
```
After instantiating a NetSpec object, you can fill in the layers you need into this object. Here is an example:
```python
# the first layer must be an input data layer
n.data, n.label = L.Data(batch_size=50, backend=P.Data.LMDB, source=path_to_lmdb, transform_param=dict(scale=1./255), ntop=2)
# pay attention to how these layers are connected
n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
n.fc1 =   L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
n.relu1 = L.ReLU(n.fc1, in_place=True)
n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
n.loss =  L.SoftmaxWithLoss(n.score, n.label)
```
** Use Python reserved symbols in layer names**
Sometimes you may need to use some ASCII symbols which cannot be used in a Python variable name to name a layer. This usually happens when you want to do fine-tuning. For example, the first convolutional layer of GoogLeNet is named 'conv1/7x7_s2'. In order to do fine-tuning, you need to name the first convolutional layer in your network the same name 'conv1/7x7_s2'. This problem can be solved as follows:
```python
# do not fill conv into the network
conv = L.Convolution(bottom, kernel_size=7, stride=2, num_output=64)
# fill in the layer by giving it another name
n.__setattr__('conv1/7x7_s2', conv)
```
If you want to have a deeper understanding of how the layers and networks are constructed. You may have a closer look at the file in /caffe/python/caffe/net\_spec.py. Before doing that, you should be familiar with the python build-in functions \_\_getattr\_\_, \_\_setattr\_\_, \_\_getitem\_\_ and \_\_setitem\_\_.

### 4.4 Generating Network Prototxt
Finally, we need to generate a prototxt file for the network.
```python
# convert the network into a proto object
proto = n.to_proto()
# write to file
# you need to convert the proto object to a string
with open(filename, 'w') as f:
    f.write(str(proto))
```
###4.5 Network Visualization
If you want to verify the topology of the network, you may use Caffe's draw function to get an intuitive view of the network.
When you have the proto object of a network, you can use a simple python function provided by caffe to draw the network layout.
```python
import caffe.draw

caffe.draw.draw_net_to_file(proto, filename, rank)
# arguments:
# 	proto: network proto object
# 	filename: network layout file name, including path
# 	rank: orientation of the network
# 		'TB': from top to bottom
# 		'BT': from bottom to top
# 		'LR': from left to right
# 		'RL': from right to left
```
If you have a prototxt file instead of a proto object, you need to read in the file and create a proto object in order to call the above funtion. This process can be a little complex. Luckily, the python file located at caffe/python/draw_net.py does this for you. This simplest way of using this .py file is as follows:
```bash
python draw_net.py --rankdir 'TB' source_prototxt_file target_image_file
```
For more information about the usage of this file, read it!

## 5. Defining Solver
A solver is also defined in a prototxt file. Specifically, we tell Caffe what our network looks like and what the learning strategy is in this file. Again, the prototxt can be generated by python. It is a function provided by protobuf because SolverParameter is just a message in caffe.proto. The source code is in ./Code/Section5/.

### 5.1 Import
```python
# this module contains everything we need to define a solver
from caffe.proto.caffe_pb2 import SolverParameter
```

### 5.2 Solver Parameters
To define a solver, we instantiate a SolverParameter object and fill in the parameters necessary for training. All the available parameters can be found in caffe.proto (message SolverParameter).
```python
solver_param = SolverParameter()

# tell Caffe where to find the network architecture
# note that a solver can have ONLY ONE training network but multiple testing networks
# use the following one of two if training and testing networks are defined in a single proto
solver_param.net = path_to_prototxt_file
solver_param.net_param = proto_object
# use the following two of four if training and testing networks are defined in separate protos
# note that testing networks are contained in a list, hence the append operation
solver_param.train_net = path_to_prototxt_file
solver_param.test_net.append(path_to_prototxt_file)
solver_param.train_net_param = proto_object
solver_param.test_net_param.append(proto_object)

# define how many iterations to run for testing
# one value for each testing network
solver_param.test_iter.append(...)

# define the number of iterations between two testing phases
# one value for all testing networks
solver_param.test_interval = ...

# define the number of iterations between displaying info. If display = 0, no info will be displayed.
solver_param.display = ...

# define the maximum number of iterations
# usually useful when you use command line interface to run training
# if you use Python, you can use a loop to control the number of iterations
solver_param.max_iter = ...

# define global learning rate
solver_param.base_lr = ...

# regularization weight
solver_param.weight_decay = ...

# define the learning rate decay policy
# please refer to caffe.proto and Caffe official tutorial for details of policies and their corresponding parameters
solver_param.lr_policy = ...

# number of iterations between snapshots
solver_param.snapshot = ...

# prefix of snapshot file name
solver_param.snapshot_prefix = ...
```

### 5.3 Generating prototxt
```python
# write the solver_param object to file
with open(filename, 'w') as f:
    f.write(str(solver_param))
```

## 6. Training and Testing
The source code is available in ./Code/Section6/
### 6.1 Preparation
Before training the network, we need to choose a device (CPU or GPU) and initialize the solver.
```python
import caffe

# choose the following one of two modes
caffe.set_mode_cpu() # use CPU

caffe.set_mode_gpu() # use GPU
caffe.set_device(0) # ID of the GPU you want to use, 0 for the first one
# it seems that while caffe c++ supports multiple GPUs, pycaffe only supports a single GPU
# as explained here: https://github.com/BVLC/caffe/issues/4253

# initialize a solver from a prototxt file
solver = caffe.get_solver(path_to_solver_prototxt)
```
### 6.2 Training
#### 6.2.1 Automatic Training
- The source code is Automatic\_Training.py
- It would be very useful to save the info displayed in the terminal to a logfile. You may use the following command:```python Automatic_Training.py 2>&1 | tee -a your_log.log```
- You may check the tee command: ```tee --help``` for more info.

The simplest way of training the network requires only one command:
```python
solver.solve()
```
This commands Caffe to train the network until the maximum number of iterations defined in the solver prototxt file. During training, Caffe automatically fetches data from the source, and does the forward and backward processes iteratively.
####6.2.2 Step Training
- The source code is Step\_Training.py
- You may take snapshot at any time by calling solver.snapshot()

One drawback of automatic training is that it does not allow us to interrupt the training process. Usually, during training we want to fetch some information about the network states, such as training and validation loss. To do so, we need to call the **step** function to train the network. This function allows one to specify the number of iterations to run.
```python
# train the network for one iteration
# you can put other positive integers in the bracket
solver.step(6)
solver.snapshot()
```
Now, you can implement a loop to train the network. Inside the loop you can fetch information about the network or do something else you want.
```python
num_of_iter = 3000 # number of iterations
for i in range(num_of_iter):
	# ... do something ...
	solver.step(1)
	# ... do something ...
```
####6.2.3 Resume Training
- The source code is Resume\_Training.py
- To run the file, you need to have a snapshot of a terminated training process stored in ./Result/Section6/

During training, Caffe regularly saves snapshots of solver and network status into hard drives. The frequency of saving snapshots is defined in the solver prototxt file. Each snapshot consists of one solver state file (suffix .solverstate) and one model file (suffix .caffemodel). The former one stores information about the solver, such as iteration, learning rate, etc. The latter one constains the trained parameters of all the layers in the network. We need these two files to resume training if the training was interrupted intentionally or accidentally.
```python
# starting from initializing the solver
solver = caffe.get_solver(path_to_solver_prototxt)

# restore solver state
solverstate_file = './snapshot_iter_1000.solverstate'
solver.restore(solverstate_file)

# restore model parameters
model_file = './snapshot_iter_1000.caffemodel'
solver.net.copy_from(model_file)
```
### 6.3 Analyze Training Process
This section shows you how to fetch network parameters, output and contents of blobs to help you analyse the training process and the performance of a trained network.

#### 6.3.1 Accessing Blobs and Network Parameters
- Sample code is available Access\_Net.py.
- To run the sample code, it is assumed that you have a snapshot of previously trained network.

There are several rules to accessing blobs and network parameters:
- to access the training network, use `solver.net`.
- to access the testing networks, use `solver.test_nets[0]`. Use the index to specify which test network you want to access if you have multiple test networks.
- to access the blobs of a training network, use `solver.net.blobs['blob_name'].data`. Note the name of a blob is the same as the name of its input layer. The final `.data` allows you to access the data of a blob, if this is missing, you get an object of that blob.
- to access the layer parameters of a training network, use `solver.net.params['layer_name'][0].data`. If the index is 0, we get the weights; if the index is 1, we get the bias.
- the contents of blobs, as well as parameters, are all in the format of numpy arrays.
- the difference of availability of blob data and parameters is that when you load a pretrained network, you have valid parameters but you won't have blob data until you pass data in the network and start training.
- you may find python API in python/caffe/_caffe.cpp, a little bit tedious though.

Here is an example of how to capture training and validation losses:
```python
import numpy as np
num_of_iter = 3000 # number of iterations
train_loss = np.zeros(num_of_iter)
val_loss = np.zeros(num_of_iter)
for i in range(num_of_iter):
	solver.step(1)
	# must use copy()
	train_loss[i] = solver.net.blobs['loss'].data.copy()
	val_loss[i] = solver.test_nets[0].blobs['loss'].data.copy()
```
####6.3.2 Parsing Logs
- For demostration purpose, we created a sample log in ./Result/Section6/Sample.log.

Caffe provides a useful tool for parsing a log: /caffe/tools/extra/parse_log.py
```bash
cd path/to/caffe/tools/extra
python parse_log.py logfile_path output_dir
```
It would create .log.test and .log.train files. You may create tables (e.g. using pandas.read_csv())and plot curves (e.g. using matplot) as you wish.
### 6.4 Testing and Online Training
The training and validation processes fetch data from data sources (such as LMDB) pre-defined in the network prototxt file. The contents in data sources are pre-generated and usually  you do not want to mess it up by adding some new data.
What if you want to use some new data to train or test the network?
One of the benefit of pycaffe is that you can directly change the content of an input data blob and run the forward and backward processes to train or test the network with new data online.
Here is an example of how to realize this:
```python
import numpy as np

batch_size = 10
new_data = np.random.randn(batch_size, 3, 32, 32)
new_label = np.random.randn(batch_size)

# after some data pre-processing
transformed_data = preprocess(new_data)
# write directly into input and label blobs
solver.net.blobs['input'].data[:] = transformed_data
solver.net.blobs['label'].data[:] = new_label

# train the network
solver.step(1)
# or test the network
solver.net.forward()

# get the predicted output and loss
output = solver.net.blobs['output'].data
loss = solver.net.blobs['loss'].data
```
## 7 Fine-tuning
The idea of fine-tuning is to migrate part of a trained network (both network structure and parameters) to a new network. In order to realize this, the layers to be migrated should be defined in the new network in the same way (i.e. the same names and structures) as it was defined in the old (trained) network. For example, if you want to use the first convolution layer (named 'conv1/7x7_s2') of GoogLeNet in your own network, you need to define a convolution layer named 'conv1/7x7_s2' with kernel size 7 and stride 2 in your own network. Sometimes you may want to freeze the migrated layer from updating, as mentioned before this can be done by setting `lr_mult=0` in the **param** parameter of that layer.
In order to copy the parameters of the trained network to your new network, you need to have the trained network model file (with suffix .caffemodel).
Once we have the network and model file ready, we can initialize the solver and network.
```python
# create solver and initialize weights
solver = caffe.get_solver(path_to_solver_prototxt)

# trained network model file
trained_net_model = './googlenet.caffemodel'
# copy weights to the new network
solver.net.copy_from(trained_net_model)
```
Now you can train the network using the same methods described above.

## 8. Deploying
```python
# TODO
```