# Triplet loss 训练

本页描述如何用Triplet Loss训练Inception-Resnet-v1模型。必须指出训练triplet loss比训练softmax复杂。但是训练集非常大（超过100000），softmax 本身就会变得很大，而triplet loss还能很好地运行。需要注意的是，本指南决不包含如何使用triplet loss来训练模型的最终配方，而应被视为正在进行中的项目。

训练更好性能的模型，请参考 [Classifier training of Inception-ResNet-v1](https://github.com/davidsandberg/facenet/wiki/Classifier-training-of-inception-resnet-v1.md)


# 1、安装Tensorflow

当前FaceNet项目运行需要Tensorflow r1.0 版本，它能通过 [pip](https://www.tensorflow.org/get_started/os_setup#pip_installation) 或 [源码](https://www.tensorflow.org/get_started/os_setup#installing_from_sources) 安装。

由于深神经网络的训练是非常密集的计算，建议使用CUDA启用GPU。Tensorflow的安装页面也详细描述了如何安装CUDA。

# 2、克隆FaceNet到本地

使用命令：

> git clone https://github.com/davidsandberg/facenet.git

# 3、设置 python 路径

设置 PYTHONPATH 环境变量，指向 facenet项目的src目录，例如：

> export PYTHONPATH=[...]/facenet/src

[...]代表facenet在本地的目录。

# 4、准备训练的数据集

## 数据集结构

假定训练数据集被安排如下，即每一个类是包含该类的训练示例的子目录（编者注：训练数据目录下有以各个人名命名的子目录，人名目录下放该人的头像图片）。

<pre>
Aaron_Eckhart
    Aaron_Eckhart_0001.jpg

Aaron_Guiel
    Aaron_Guiel_0001.jpg

Aaron_Patterson
    Aaron_Patterson_0001.jpg

Aaron_Peirsol
    Aaron_Peirsol_0001.jpg
    Aaron_Peirsol_0002.jpg
    Aaron_Peirsol_0003.jpg
    Aaron_Peirsol_0004.jpg
    ...
</pre>

## 人脸对齐

人脸对齐推荐使用经证实有非常好性能的[MTCNN](https://github.com/kpzhang93/MTCNN_face_detection_alignment) 训练集。作者已以提供一个基于MATLAB和Caffe的MTCNN实现。此外，使用此实现对数据集进行对齐的MATLAB脚本可以在[这里](https://github.com/davidsandberg/facenet/blob/master/tmp/align_dataset.m)找到。

简化使用这个项目[提供](https://github.com/davidsandberg/facenet/tree/master/src/align) python/tensorflow的MTCNN实现。这个实现没有任何依赖除了Tensorflow，在LFW上运行类似于Matlab实现。 

> python src/align/align_dataset_mtcnn.py ~/datasets/casia/CASIA-maxpy-clean/ ~/datasets/casia/casia_maxpy_mtcnnpy_182 --image_size 182 --margin 44

通过上述命令生成182x182像素的人脸缩略图。Inception-ResNet-v1模型的输入是使用随机数的160x160像素范围。作为实验，在Inception-ResNet-v1模型上各追加32像素。这样做的原因是扩大人脸对齐所提供的范围盒，并给CNN添加一些上下文信息。然而，这个参数的设置还没有被研究过，而且很可能是其他的差数导致了更好的性能。

为提高对齐进程，上述命令能运行在多进程。下面的命令能跑4个进程。限制每个Tensorflow session内存使用的gpu_memory_fraction参数设置为0.25，表示每个session最大使用25%的GPU内存。如果下面的命令导致GPU内存耗尽，尽量减少并行进程的数量，增加每个session的GPU内存比例。

> for N in {1..4}; do python src/align/align_dataset_mtcnn.py ~/datasets/casia/CASIA-maxpy-clean/ ~/datasets/casia/casia_maxpy_mtcnnpy_182 --image_size 182 --margin 44 --random_order --gpu_memory_fraction 0.25 & done

# 5、开始分类器训练

运行train_tripletloss.py开始训练，命令如下：

> python src/train_tripletloss.py --logs_base_dir ~/logs/facenet/ --models_base_dir ~/models/facenet/ --data_dir ~/datasets/casia/casia_maxpy_mtcnnalign_182_160 --image_size 160 --model_def models.inception_resnet_v1 --lfw_dir ~/datasets/lfw/lfw_mtcnnalign_160 --optimizer RMSPROP --learning_rate 0.01 --weight_decay 1e-4 --max_nrof_epochs 500

当训练开始时，为训练session在log_base_dir 和 models_base_dir 目录下创建格式为yyyymmdd-hhmm的子目录。参数 data_dir 用于指定训练数据集的位置。需要注意的是，可以使用多个数据集的联合使用冒号分隔路径。 最后，对推理网络的描述是由 model_def 参数确定。在上面的例子中，models.inception_resnet_v1中的模型指向在models包中的inception_resnet_v1模块。该模块必须定义一个 inference(images, ...) 函数，参数images是输入图像的占位符(在Inception-ResNet-v1例子中尺寸为<?,160,160,3>)，并返回一个 embeddings 变量的引用。

如果参数 lfw_dir 设置为指向的LFW数据集的基本目录，该模型已经在LFW上被评估1000批次。如何评估存在的LFW模型，请参考[Validate-on-LFW](https://github.com/davidsandberg/facenet/wiki/Validate-on-LFW) 页面。如果期望不评估LFW在训练期间，可以去掉 lfw_dir 参数。但是，请注意，这里使用LFW数据应该被排列在相同的训练数据集。

# 6、运行TensorBoard

用[TensorBoard](https://www.tensorflow.org/how_tos/summaries_and_tensorboard/#launching-tensorboard) 能监控训练FaceNet的学习过程。运行命令开始TensorBoard：

> tensorboard --logdir=~/logs/facenet --port 6006

接着就可以用浏览器访问了：
http://localhost:6006/
