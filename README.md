# 人脸识别

## 算法简介

我们的算法可以分成两个部分，识别人脸位置和确定人脸分类。这两个部分可以看成：
1.检测出人脸之间相似性。
2.检测出人脸之间不同性。
由于这两项工作截然相反，所以我们使用了两个网络来分别完成这两项工作。

## 人脸检测

### 简述

我们的人脸检测网络采用了和Faster RCNN类似的策略，但我们在ROI Polling上进行了创新，兼顾了小目标检测和大目标检测，为此，我们还使用了改进后的RESNET101_V2的网络，使我们的网络对于小目标更加敏感。在增加了少量的运算单元后，我们的网络可以识别24*24像素下的人脸(甚至于更低!)。我们调整了网络结构，并没有采用传统的卷积网络(提取特征)+全连接层(分类)的结构，而是采用了全卷积结构，这让我们的识别网络的速度远远高于传统的神经网络识别方法，识别精度也高于传统的算子和特征值人脸识别算法。

### 数据集介绍

pass

### 算法介绍

```
img = tf.constant(img,shape = (1,h,w,mod),dtype = tf.float32) # 图像原始数据

# 使用无pool1&pool5的RESNET 101
net, endpoints = my_resnet(img,global_pool = False,num_classes=None,is_training=True,reuse = tf.compat.v1.AUTO_REUSE) # net's w&h = original_img's w&h / 8
```

我们进行模型搭建和使用的平台为windows10-python3.6.2-tensorflow-gpu。
首先，我们的图像(img_batch = [batch_size,h,w,mod],batch_size为图像的数量,h为图像高度,w为图像宽度,mod为图像通道数，这里我们处理的均为RGB三色彩图，所以我们的通道数均为3)通过我们改进版的RESNET101_V2网络，传统的RESNET101_V2的网络结构如下:

![resnet101](pic_result\\resnet101.jpg)

而我们的网络去掉了pool1和pool5层，使网络放缩系数从32下降到了8。这使我们的网络对于小目标更加的敏感。通过了该网络后，我们得到了卷积后的信息图:img_batch_conv = [batch_size,h/8,w/8,2048]

```
weights = {
            'down':tf.compat.v1.get_variable(name = 'w_down',shape = [1,1,2048,1024]),# 降采样
            'feature':tf.compat.v1.get_variable(name = 'w_feature',shape = [1,1,1024,K*K*2])
            }
biases = {
            'down':tf.compat.v1.get_variable(name = 'b_down',shape = [1024,]), # 降采样
            'feature':tf.compat.v1.get_variable(name = 'b_feature',shape = [K*K*2,])
        }
```

img_batch_conv首先通过一个shape = [1,1,2048,1024]的卷积算子，该算子的作用是进一步的特征提取和降采样。我们采用多步特征提取的策略是由于在RCNN[1]一文中，作者提出在VOC2012测试集下，三层的特征识别网络比一层特征识别网络的正确率要高。

通过该算子我们得到了一个[batch_size,h/8,w/8,1024]结构的数据，将该数据通过一个[1,1,1024,K\*K\*C]的算子得到特征图feature_map。feature_map的概念在Faster RCNN[2]的文章内提出，提取特征图的算子的K\*K代表着每一块feature_map有K\*K个bin，而C代表着识别物体的类别+1(背景),这里我们的K取3，C取2(只有两个分类:人脸和背景)。

```
    with tf.compat.v1.variable_scope('RPN', reuse=tf.compat.v1.AUTO_REUSE):
        weights = {
            'rpn_1':tf.compat.v1.get_variable(name = 'w_rpn_1_1',shape = [3,3,1024,1]), # 高:宽 1:1的卷积
            'rpn_2':tf.compat.v1.get_variable(name = 'w_rpn_1_2',shape = [3,6,1024,1]), # 高:宽 1:2的卷积
            'rpn_3':tf.compat.v1.get_variable(name = 'w_rpn_2_1',shape = [6,3,1024,1]), # 高:宽 2:1的卷积
            'rpn_4':tf.compat.v1.get_variable(name = 'w_rpn_2_2',shape = [6,6,1024,1]),
            'rpn_5':tf.compat.v1.get_variable(name = 'w_rpn_2_4',shape = [6,12,1024,1]),
            'rpn_6':tf.compat.v1.get_variable(name = 'w_rpn_4_2',shape = [12,6,1024,1]),
            'rpn_7':tf.compat.v1.get_variable(name = 'w_rpn_4_4',shape = [12,12,1024,1]),
            'rpn_8':tf.compat.v1.get_variable(name = 'w_rpn_4_8',shape = [12,24,1024,1]),
            'rpn_9':tf.compat.v1.get_variable(name = 'w_rpn_8_4',shape = [24,12,1024,1])
        }
        biases = {
            'rpn_1':tf.compat.v1.get_variable(name = 'b_rpn_1_1',shape = [1,]),
            'rpn_2':tf.compat.v1.get_variable(name = 'b_rpn_1_2',shape = [1,]),
            'rpn_3':tf.compat.v1.get_variable(name = 'b_rpn_2_1',shape = [1,]),
            'rpn_4':tf.compat.v1.get_variable(name = 'b_rpn_2_2',shape = [1,]),
            'rpn_5':tf.compat.v1.get_variable(name = 'b_rpn_2_4',shape = [1,]),
            'rpn_6':tf.compat.v1.get_variable(name = 'b_rpn_4_2',shape = [1,]),
            'rpn_7':tf.compat.v1.get_variable(name = 'b_rpn_4_4',shape = [1,]),
            'rpn_8':tf.compat.v1.get_variable(name = 'b_rpn_4_8',shape = [1,]),
            'rpn_9':tf.compat.v1.get_variable(name = 'b_rpn_8_4',shape = [1,])
        }
```

我们将得到的feature_map = [batch_size,h/8,w/8,K\*K\*C]使用三种不同形状，三种不同大小，一共九种不同形状或大小的卷积核对我们的网络进行卷积，得到了9种不同形状或大小的archors中是否存在人脸的概率。这里我们虽然沿用了Faster RCNN[2]中anchor的概念，但我们并没有使用ROI Pooling而是只使用了ROI。因为Tensorflow采用的是流图计算，增加ROI Pooling反而会让每个anchor独立计算，大大增加了我们的计算量，而且不同大小的anchor进行Pooling后均会生成K*K形状的数据，不方便我们的网络对于不同大小的anchor进行不同状态的识别。而且由于我们只用分成背景和人脸两个类别，所以即使不进行ROI Pooling，我们网络所需的运算单元也不会增加太多。所以我们采用了9种不同形状的卷积核对应九种anchor的策略。通过RPN评价层后，我们可以得到对于每个区域是否存在人脸的评价。

这里我们训练采用了YOLO[3]而不是Faster RCNN[2]的训练策略,即我们对于不同比例的正例和负例采用比例因子去平衡它。这是因为我们是一张一张图去训练的，训练的图中正例的数量远小于负例。

```
loss_rpn[i] = -(up*tf.math.log(pred_rpn[i])*rpn_view[i] + (1 - rpn_view[i])*tf.math.log(1 - pred_rpn[i]))
```

这里loss_rpn[i]为第i个anchor的loss，up为正例激励因子，pred_rpn[i]为网络预测的第i个anchor的结果，rp_view[i]为第i个anchor的真实结果。我们的损失函数使用的是交叉熵。

之后，我们将选出来的区域的feature_map，通过ROI Pooling。在Bounding Box回归之前通过ROI Pooling一方面是由于Bounding Box回归只针对正例（选出来的区域），区域的个数较少，即使创建独立运算的anchor数量也不多，训练压力不大；二是对于Bounding Box回归，anchor的大小和形状不具有太大的意义。Bounding Box回归的计算规则如下:
```
dx = x - x'
dy = y - y'
kw = w / w'
kh = h / h'

# 计算损失
loss_bbox = (pre_bbox - select_bbox) * data_type 
loss_bbox = tf.reduce_mean(loss_bbox*loss_bbox)
```
x,y为中心点坐标，w,h为宽高。
通过网络我们可以得到[dx,dy,dw,dh]
loss_bbox即Bounding Box回归的loss，损失函数我们采用的是平方函数。

我们首先通过RPN评价层得到评分最高的N个anchor，每个anchor都带有[ax,ay,aw,ah]的属性，ax,ay为网络放缩下的左上角坐标，aw,ah为网络放缩下的宽高。所以首先要对放缩后的图像区域进行恢复:
```
x' = ax * NET_SCALE
y' = ay * NET_SCALE
w' = aw * NET_SCALE
h' = ah * NET_SCALE
```
这里我们的网络放缩系数NET_SACLE为8。

然后进行Bounding Box回归:
```
x = x' + dx
y = y' + dy
w = w' * kw
h = h' * kh
```
得到[x,y,w,h]，如果该区域与其他比它得分高的区域的IOU>0.5的情况下，该区域会被抑制(NMS非极大值抑制)。

### Reference

[1]Ross Girshick, Jeff Donahue, Trevor Darrell. Rich feature hierarchies for accurate object detection and semantic segmentation[J]. 2013.

[2]Ren, Shaoqing, He, Kaiming, Girshick, Ross,等. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks[J]. IEEE Transactions on Pattern Analysis & Machine Intelligence, 2015, 39(6):1137-1149.

[3]Redmon, Joseph, Divvala, Santosh, Girshick, Ross,等. You Only Look Once: Unified, Real-Time Object Detection[J].