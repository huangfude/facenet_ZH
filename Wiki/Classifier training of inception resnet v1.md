# Classifier training of inception resnet v1

本页描述如何分类器训练[Inception-Resnet-v1](https://arxiv.org/abs/1602.07261) 模型，即不使用在[Facenet](http://arxiv.org/abs/1503.03832) 论文中描述的三重损失。[如这](http://www.robots.ox.ac.uk/%7Evgg/publications/2015/Parkhi15/parkhi15.pdf)所述，作为分类器的训练使训练变得更容易、更快。Facenet论文也使用Inception体系结构的non-ResNet版。在用CASIA/Facescrub数据集训练时，这些网络似乎很困难去训练并且不太集中。没有正规使用训练集时，会有相当大的误差，意味着该模型不能过度使用。使用Inception-Resnet-v1的例子解决了聚合问题并且无论是看准确性和验证率，结果明显提升在LFW的性能（VAL@FAR=10^-3）。 

# 1、安装Tensorflow

当前FaceNet项目运行需要Tensorflow r1.0 版本，它能通过 [pip](https://www.tensorflow.org/get_started/os_setup#pip_installation) 或 [源码](https://www.tensorflow.org/get_started/os_setup#installing_from_sources) 安装。

由于深神经网络的训练是非常密集的计算，建议使用CUDA启用GPU。Tensorflow的安装页面也详细描述了如何安装CUDA。

# 2、Clone the FaceNet repo

使用命令：
<pre>
git clone https://github.com/davidsandberg/facenet.git
</pre>

# 3、设置 python 路径

设置 PYTHONPATH 环境变量，指向 facenet项目的src目录，例如：
<pre>
export PYTHONPATH=[...]/facenet/src
</pre>
[...]代表facenet在本地的目录。

# 4、准备训练的数据集

## 数据集结构

假定训练数据集被安排如下，即每一个类是包含该类的训练示例的子目录。

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

<pre>
python src/align/align_dataset_mtcnn.py ~/datasets/casia/CASIA-maxpy-clean/ ~/datasets/casia/casia_maxpy_mtcnnpy_182 --image_size 182 --margin 44
</pre>

通过上述命令生成182x182像素的人脸缩略图。Inception-ResNet-v1模型的输入是使用随机数的160x160像素范围。作为实验，在Inception-ResNet-v1模型上各追加32像素。这样做的原因是扩大人脸对齐所提供的范围盒，并给CNN添加一些上下文信息。然而，这个参数的设置还没有被研究过，而且很可能是其他的差数导致了更好的性能。

为提高对齐进程，上述命令能运行在多进程。下面的命令能跑4个进程。限制每个Tensorflow session内存使用的gpu_memory_fraction参数设置为0.25，表示每个session最大使用25%的GPU内存。如果下面的命令导致GPU内存耗尽，尽量减少并行进程的数量，增加每个session的GPU内存比例。

<pre>
for N in {1..4}; do python src/align/align_dataset_mtcnn.py ~/datasets/casia/CASIA-maxpy-clean/ ~/datasets/casia/casia_maxpy_mtcnnpy_182 --image_size 182 --margin 44 --random_order --gpu_memory_fraction 0.25 & done
</pre>

# 5、开始分类器训练

运行train_softmax.py开始训练，命令如下：

<pre>
python src/train_softmax.py --logs_base_dir ~/logs/facenet/ --models_base_dir ~/models/facenet/ --data_dir ~/datasets/casia/casia_maxpy_mtcnnalign_182 --image_size 160 --model_def models.inception_resnet_v1 --lfw_dir ~/datasets/lfw/lfw_mtcnnalign_160 --optimizer RMSPROP --learning_rate -1 --max_nrof_epochs 80 --keep_probability 0.8 --random_crop --random_flip --learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt --weight_decay 5e-5 --center_loss_factor 1e-2 --center_loss_alfa 0.9
</pre>

当训练开始时，为训练session在log_base_dir 和 models_base_dir 目录下创建格式为yyyymmdd-hhmm的子目录。参数 data_dir 用于指定训练数据集的位置。需要注意的是，可以使用多个数据集的联合使用冒号分隔路径。 最后，对推理网络的描述是由 model_def 参数确定。在上面的例子中，models.inception_resnet_v1中的模型指向在models包中的inception_resnet_v1模块。该模块必须定义一个 inference(images, ...) 函数，参数images是输入图像的占位符(在Inception-ResNet-v1例子中尺寸为<?,160,160,3>)，并返回一个 embeddings 变量的引用。


如果参数 lfw_dir 设置为指向的LFW数据集的基本目录，该模型已经在LFW上被评估1000批次。如何评估存在的LFW模型，请参考[Validate-on-LFW](https://github.com/davidsandberg/facenet/wiki/Validate-on-LFW) 页面。如果期望不评估LFW在训练期间，可以去掉 lfw_dir 参数。但是，请注意，这里使用LFW数据应该被排列在相同的训练数据集。

达到max_nrof_epochs值时训练停止，本例设置为80次训练次数。Nvidia Pascal Titan X GPU，tensorflow R1.0，CUDA 8，cudnn 5.1.5和inception-resnet-v1模型，大约需要12小时。

为了提高最终模型的性能，当训练开始聚合时，学习率降低了10倍。通过参数 learning_rate_schedule_file 在一个文本文件中定义学习率任务安排同时设置参数learning_rate的负值。例如在项目[data/learning_rate_schedule_classifier_casia.txt](https://github.com/davidsandberg/facenet/blob/master/data/learning_rate_schedule_classifier_casia.txt) 中的简单使用，像这样：

<pre>
# Learning rate schedule
# Maps an epoch number to a learning rate
0:  0.1
65: 0.01
77: 0.001
1000: 0.0001
</pre>

在这里，第一列是编号，第二列是学习率，这意味着当编号在65…76的范围内时，学习率设置为0.01。