# 基于Tensorflow的人脸识别

这是一个介绍tensorflow实现人脸识别的论文[FaceNet: A Unified Embedding for Face Recognition and Clustering](arxiv.org/abs/1503.03832)。该项目还使用了[A Discriminative Feature Learning Approach for Deep Face Recognition](http://ydwen.github.io/papers/WenECCV16.pdf)和牛津[视觉几何组织](http://www.robots.ox.ac.uk/%7Evgg/)的[Deep Face Recognition](http://www.robots.ox.ac.uk/%7Evgg/publications/2015/Parkhi15/parkhi15.pdf)论文。

# Tensorflow 版本

当前repo兼容Tensorflow r1.0

# News

暂略

# 已预先培训好的模型

<table>
    <tr>
        <th>模型名称</th>
        <th>LFW 精度</th>
        <th>训练数据集</th>
        <th>结构</th>
    </tr>
    <tr>
        <td><a href='https://drive.google.com/file/d/0B5MzpY9kBtDVOTVnU3NIaUdySFE'>20170511-185253</a></td>
        <td>0.987</td>
        <td>CASIA-WebFace</td>
        <td><a href='https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py'>Inception ResNet v1</a></td>
    </tr>
    <tr>
        <td><a href='https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk'>20170512-110547</a></td>
        <td>0.992</td>
        <td>MS-Celeb-1M</td>
        <td><a href='https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py'>Inception ResNet v1</a></td>
    </tr>
</table>

# 灵感

大量借鉴[OpenFace](https://github.com/cmusatyalab/openface)的思想

# 训练数据

[CASIA-WebFace](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html)数据集已被用于训练。该训练集包括453453个图像及10575个以上人脸识别后的身份。如果在训练前对数据集进行过滤，就能提高一些性能。关于这项工作如何完成的更多信息稍后说明。表现最好的模型已经被训练在[MS-Celeb-1M](https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/)的一个子集上。但此数据集大得多，而且也包含了更多的标签噪声，因此应用数据集过滤非常关键。 

# 预处理

## 用MTCNN进行人脸对齐

上述人脸检测的数据的缺少一些特殊的例子（遮挡，有轮廓，等）。这使得训练设置为“容易”，导致模型在其他基准上表现差劲。为解决这个问题，人脸里程碑式的识别已经被测试过。这个用[Multi-task CNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html)设置的人脸里程碑式的识别已经被证实效果非常好。[这里](https://github.com/kpzhang93/MTCNN_face_detection_alignment)是Matlab/Caffe实现的非常好的结果。一个用Python/Tensorflow实现MTCNN的在[这里](https://github.com/davidsandberg/facenet/tree/master/src/align)。这个实现没有得到和Matlab/Caffe实现一样的结果，但性能非常类似。

## 进行训练

目前，最好的训练方法是以[中心损失](http://ydwen.github.io/papers/WenECCV16.pdf)作为分类器进行训练。详情见[Classifier training of Inception-ResNet-v1](https://github.com/davidsandberg/facenet/wiki/Classifier-training-of-inception-resnet-v1)

# 预训练模型

## Inception-ResNet-v1 model

目前，表现最好的模型是在CASIA-Webface用MTCNN对齐的Inception-Resnet-v1训练模型。

# 性能（表现）

这个[20170512-110547](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk)模型的LFW精度在0.992+-0.003之间。如何运行这个测试详情见[Validate on LFW](https://github.com/davidsandberg/facenet/wiki/Validate-on-lfw)。
