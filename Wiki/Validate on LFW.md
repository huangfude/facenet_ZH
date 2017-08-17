#  LFW 验证

# 1、安装依赖

假设下面的已经做了：
a) Tensorflow 已经[安装](https://github.com/davidsandberg/facenet/wiki/Running-training#1-install-tensorflow)
b) Facenet[资源](https://github.com/davidsandberg/facenet.git) 已经克隆到本地
c) 必须的[python模块](https://github.com/davidsandberg/facenet/blob/master/requirements.txt) 已经安装

# 2、下载LFW数据集

1、[这里下载](http://vis-www.cs.umass.edu/lfw/lfw.tgz) 没有对齐的图片
    本例下载到 ~/Downloads 目录

2、解压下载好的文件
    假设存在 ~/datasets 目录
<pre>
cd ~/datasets
mkdir -p lfw/raw
tar xvf ~/Downloads/lfw.tgz -C /lfw/raw --strip-components=1
</pre>

# 3、设置python路径

设置 PYTHONPATH 环境变量，指向facenet项目的src目录，例如：

> export PYTHONPATH=[...]/facenet/src

[...]代表facenet在本地的目录。

# 4、对齐LFW数据集

在 align 模块中使用 align_dataset_mtcnn 对齐LFW数据集，执行命令如：

> for N in {1..4}; do python src/align/align_dataset_mtcnn.py ~/datasets/lfw/raw ~/datasets/lfw/lfw_mtcnnpy_160 --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.25 & done

参数 margin 控制相对于bounding盒子裁剪的宽度。32像素是一个160像素图片在边缘的对应于182像素图像的大小，这图像的大小已被用于训练的模型下。

# 5、下载预训练的模型（可选）

如果你没有自己的训练模型，可以在[这里](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk) 下载一个去测试。本例下载到 ~/models/facenet/ 目录，解压后产生 20170512-110547 目录文件，里面内容如下：

<pre>
20170512-110547.pb
model-20170512-110547.ckpt-250000.data-00000-of-00001
model-20170512-110547.ckpt-250000.index
model-20170512-110547.meta
</pre>

# 6、测试

用 validate_on_lfw 测试：

> python src/validate_on_lfw.py ~/datasets/lfw/lfw_mtcnnpy_160 ~/models/facenet/20170512-110547

它将
a) 加载模型
b) 加载并解析图像文件
c) 计算在所有测试图片中的将维
d) 计算精度、验证率(@FAR=-10e-3)、曲线下面积（AUC）和相等错误率（EER）性能。

通常输出如下：

<pre>
Model directory: /home/david/models/20170512-110547
Metagraph file: model-20170512-110547.meta
Checkpoint file: model-20170512-110547.ckpt-250000
Runnning forward pass on LFW images
Accuracy: 0.992+-0.003
Validation rate: 0.97733+-0.01340 @ FAR=0.00100
Area Under Curve (AUC): 1.000
Equal Error Rate (EER): 0.007
</pre>

