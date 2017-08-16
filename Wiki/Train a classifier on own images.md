# 用自己图片训练分类器

本页描述怎么用自己的数据集去训练你自己的分类器。假设你已经参照[Validate on LFW](https://github.com/davidsandberg/facenet/wiki/Validate-on-lfw) 的页面去做了这些步骤，克隆FaceNet项目，下载LFW数据集，设置python路径和校准LFW数据集（LFW试验的最后一步）。在下面例子中使用 20170216-091149 的冻结模型图，使用冻结的图能明显提高模型的加载速度。

# 在LFW训练分类器

这个实验，我们训练一个分类器使用LFW图像的子集。LFW数据集分成训练集和测试集，然后预训练的模型加载，并将该模型用于生成选定的图像特征。该预训练模型通常训练一个更大的性能良好的数据集（当前例子用MS-Celeb-1M数据的一个子集）。

 * 数据集分割成训练集和测试集
 * 加载特征提取的预训练模型
 * 计算数据中嵌入的图像
 * mode=TRAIN:
    - 训练用嵌入数据集的训练部分的分类器
    - 作为python pickle模块保存训练的分类模型
 *  mode=CLASSIFY: 
    - 加载分类模型
    - 测试用嵌入数据集的测试部分的分类器

在训练数据集上训练分类器，命令为：

> python src/classifier.py TRAIN ~/datasets/lfw/lfw_mtcnnalign_160 ~/models/model-20170216-091149.pb ~/models/lfw_classifier.pkl --batch_size 1000 --min_nrof_images_per_class 40 --nrof_train_images_per_class 35 --use_split_dataset

输出如下：

<pre>
Number of classes: 19
Number of images: 665
Loading feature extraction model
Model filename: /home/david/models/model-20170216-091149.pb
Calculating features for images
Training classifier
Saved classifier model to file "/home/david/models/lfw_classifier.pkl"
</pre>

训练好的分类器可以被用于使用测试集的分类。 ：

> python src/classifier.py CLASSIFY ~/datasets/lfw/lfw_mtcnnalign_160 ~/models/model-20170216-091149.pb ~/models/lfw_classifier.pkl --batch_size 1000 --min_nrof_images_per_class 40 --nrof_train_images_per_class 35 --use_split_dataset

这里使用数据集的测试集部分进行分类，并将分类结果与分类概率一起显示出来。这个子集的分类精度~ 0.98。

<pre>
Number of classes: 19
Number of images: 1202
Loading feature extraction model
Model filename: /home/david/models/export/model-20170216-091149.pb
Calculating features for images
Testing classifier
Loaded classifier model from file "/home/david/lfw_classifier.pkl"
   0  Ariel Sharon: 0.583
   1  Ariel Sharon: 0.611
   2  Ariel Sharon: 0.670
...
...
...
1198  Vladimir Putin: 0.588
1199  Vladimir Putin: 0.623
1200  Vladimir Putin: 0.566
1201  Vladimir Putin: 0.651
Accuracy: 0.978
</pre>

# 在你自己的数据集训练分类器

也许你想自动分类你的私人相册，或者你有一个安全相机想识别出你的家庭成员。那么很有可能你想用你自己的数据集训练一个分类器，本例classifier.py程序能实现这些。通过复制LFW数据集，我已经创建了自己的训练和测试集。在这个例子中，前5个图像用于训练，后5个图像用于测试。
他们是：

    Ariel_Sharon
    Arnold_Schwarzenegger
    Colin_Powell
    Donald_Rumsfeld
    George_W_Bush
    Gerhard_Schroeder
    Hugo_Chavez
    Jacques_Chirac
    Tony_Blair
    Vladimir_Putin

分类器的训练方法类似：

> python src/classifier.py TRAIN ~/datasets/my_dataset/train/ ~/models/model-20170216-091149.pb ~/models/my_classifier.pkl --batch_size 1000

这个分类器的训练要运行几秒（加载预训练模型后），输出如下。因为数据集简单，准确性是非常好的。

<pre>
Number of classes: 10
Number of images: 50
Loading feature extraction model
Model filename: /home/david/models/model-20170216-091149.pb
Calculating features for images
Training classifier
Saved classifier model to file "/home/david/models/my_classifier.pkl"
</pre>

测试数据集的目录结构如下：

<pre>
/home/david/datasets/my_dataset/test
├── Ariel_Sharon
│   ├── Ariel_Sharon_0006.png
│   ├── Ariel_Sharon_0007.png
│   ├── Ariel_Sharon_0008.png
│   ├── Ariel_Sharon_0009.png
│   └── Ariel_Sharon_0010.png
├── Arnold_Schwarzenegger
│   ├── Arnold_Schwarzenegger_0006.png
│   ├── Arnold_Schwarzenegger_0007.png
│   ├── Arnold_Schwarzenegger_0008.png
│   ├── Arnold_Schwarzenegger_0009.png
│   └── Arnold_Schwarzenegger_0010.png
├── Colin_Powell
│   ├── Colin_Powell_0006.png
│   ├── Colin_Powell_0007.png
...
...
</pre>

在测试集上运行分类器：

> python src/classifier.py CLASSIFY ~/datasets/my_dataset/test/ ~/models/model-20170216-091149.pb ~/models/my_classifier.pkl --batch_size 1000

<pre>
Number of classes: 10
Number of images: 50
Loading feature extraction model
Model filename: /home/david/models/model-20170216-091149.pb
Calculating features for images
Testing classifier
Loaded classifier model from file "/home/david/models/my_classifier.pkl"
   0  Ariel Sharon: 0.452
   1  Ariel Sharon: 0.376
   2  Ariel Sharon: 0.426
...
...
...
  47  Vladimir Putin: 0.418
  48  Vladimir Putin: 0.453
  49  Vladimir Putin: 0.378
Accuracy: 1.000
</pre>

这段代码的目的是提供一些关于如何使用面部识别器的灵感和想法，但它本身并不是一个有用的应用程序。在现实生活中的应用可能还需要一些额外的东西：

 * 包含人脸检测和分类管道
 * 使用分类概率的阈值来查找未知的人，而不是只使用概率最大的类。 