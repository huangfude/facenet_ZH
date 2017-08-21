# 变分自编码器（VAE）

该页面描述如何去：

1、基于facenet感知损失训练变分自编码器，论文见“[Deep Feature Consistent Variational Autoencoder](https://arxiv.org/abs/1610.00291) ”。

2、计算基于CelebA数据集属性的属性向量。

3、通过将属性向量添加到图像的潜在变量中，向脸添加笑脸。来自CelebA数据集的潜在变量用来产生未修改的人脸图像。

在开始之前假设你已经：

 * 克隆项目资源
 * 设置python路径
 * 对齐数据集
 * 从[20170512-110547](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk) 下载并打出感知模型。

# 1、训练变分自编码器

本节介绍了如何使用知觉损失训练变自编码（VAE）,而不是依赖于训练图像和重建图像之间的像素差异的损失，知觉损失试图生成一个具有类似的空间特征的图像作为源图像。

为了实现这一点，在预训练的人脸识别模型中训练一些中间层激活的L2差异。确切地说，哪些层用于这一点并没有被认真调查，因此在这方面可能还有改进的余地。但是，在感知模型中不同层次训练的两种模型的性能比较是不容易的，因此必须手动检查结果。可以通过在感知模型中使用更高层的方法避免产生奇怪人脸的错误案例。

此外，通过调查或优化有利于重建损失和Kullback Leibler发散损失的因素。

以下的命令训练产生64x64像素的VAE。图片的尺寸是由VAE实现决定的，VAE实现在generative.models的包中，包括实现编码器、解码器和获取VAE图片的方法。

一般地，有效的模型：
<table>
<tr>
    <th>模型</th>
    <th>生成的图片大小</th>
</tr>
<tr>
    <td>dfc_vae</td>
    <td>64x64 pixels</td>
</tr>
<tr>
    <td>dfc_vae_resnet</td>
    <td>64x64 pixels</td>
</tr>
<tr>
    <td>dfc_vae_large</td>
    <td>128x128 pixels</td>
</tr>
</table>

这个命令训练使用50000步感知损失的VAE和20170512-110547模型。这个模型在CASIA-WebFace上被训练，但是它貌似能跑的更好在更大的数据集（例如 MS-Celeb-1M）。

> python src/generative/train_vae.py \
src.generative.models.dfc_vae \
~/datasets/casia/casia_maxpy_mtcnnpy_182 \
src.models.inception_resnet_v1 \
~/models/export/20170512-110547/model-20170512-110547.ckpt-250000 \
--models_base_dir ~/vae/ \
--reconstruction_loss_type PERCEPTUAL \
--loss_features 'Conv2d_1a_3x3,Conv2d_2a_3x3,Conv2d_2b_3x3' \
--max_nrof_steps 50000 \
--batch_size 128 \
--latent_var_size 100 \
--initial_learning_rate 0.0002 \
--alfa 1.0 \
--beta 0.5

用这些参数做5000步训练在Pascal Titan X上跑了4小时。每500步训练脚本保存一组改造的人脸png文件，最后一组图片像这样：

![Reconstructed faces](https://github.com/davidsandberg/facenet/wiki/20170708-150701-reconstructed_050000.png)

包括下面第2步计算属性向量训练好的模型可以从[这里](https://drive.google.com/open?id=0B5MzpY9kBtDVS2R5Wm5IRFpST2M) 下载。

# 2、计算属性向量

这步使用[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) 数据集计算一些属性在潜变量空间向量。该数据集包含200k有标注40种不同的属性的图片，像 Blond Hair 和 Mustache （下面是一个完整的属性列表）。但是，本例只使用‘Smiling’属性，脚本是给CelebA 数据集中的所有图片计算潜在变量。计算每一个存在或不存在的平均潜在变量，例如，查找不存在“微笑”属性且所有相同属性的图像。属性向量为两平均潜变量之间不同的计算。


VAE模型检查点应指向第1步的训练关卡。在运行CelebA数据集对齐之前，list_attr_celeba.txt文件包含每一图片的40个属性，可下载数据本身。

> python src/generative/calculate_attribute_vectors.py \
src.generative.models.dfc_vae \
~/vae/20170708-150701/model.ckpt-50000 \
~/datasets/celeba/img_celeba \
~/datasets/celeba/list_attr_celeba.txt \
~/vae/20170708-150701/attribute_vectors.h5 \
--batch_size 128 \
--image_size 160 \
--latent_var_size 100

迭代到最后，会有个像这样的警告信息：

<pre>
Out of range: FIFOQueue '_0_input_producer/input_producer/fraction_of_32_full/fraction_of_32_full' is closed and has insufficient elements (requested 1, current size 0)
</pre>

这是因为 有小部分的例子中，不能填满一组图片。

# 3、给人脸添加微笑

你可以跳过第1、2步，直接从[这里](https://drive.google.com/open?id=0B5MzpY9kBtDVS2R5Wm5IRFpST2M) 下载处理好的CelebA数据集。

证明使用VAE我们能修改图片的属性，用我们简单在第2步已经计算好潜在变量的图片。我们选择一些不存在"Smiling"属性的人脸，接着添加不同“Smiling”属性矢量（也可以通过第2步计算），并把结果写入一个文件。

> python src/generative/modify_attribute.py \
src.generative.models.dfc_vae \
~/vae/20170708-150701/model.ckpt-50000 \
~/vae/20170708-150701/attribute_vectors.h5 \
~/vae/20170708-150701/add_smile.png

上面脚本创建一些不同人脸的图片，每个人脸被加入不同的微笑矢量，结果如图所示：

![Faces with different amount of added smile](https://github.com/davidsandberg/facenet/wiki/20170708-150701-add_smile.png)

需要注意的是这参数没有优化很多，能通过调节参数改善感知损失（loss_features 参数）。

例子中，我们用的是已经计算好潜在变量的CelebA数据集，但是，它是直接修改脚本去输入图片。运行编码器(VAE)计算潜在变量，加减属性向量修改潜在变量，接着解码器产生一个新图片。

# CelebA属性

CelebA完整的属性列表如下：

<pre>
5_o_Clock_Shadow, Arched_Eyebrows, Attractive, Bags_Under_Eyes, Bald, Bangs, Big_Lips, Big_Nose, Black_Hair, Blond_Hair, Blurry, Brown_Hair, Bushy_Eyebrows, Chubby, Double_Chin, Eyeglasses, Goatee, Gray_Hair, Heavy_Makeup, High_Cheekbones, Male, Mouth_Slightly_Open, Mustache, Narrow_Eyes, No_Beard, Oval_Face, Pale_Skin, Pointy_Nose, Receding_Hairline, Rosy_Cheeks, Sideburns, Smiling, Straight_Hair, Wavy_Hair, Wearing_Earrings, Wearing_Hat, Wearing_Lipstick, Wearing_Necklace, Wearing_Necktie, Young
</pre> 
