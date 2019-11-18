from tensorflow.contrib.slim import nets
from data_dealer import *
from my_resnet import my_resnet
from path import *
import tensorflow as tf
import numpy
# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

slim = tf.contrib.slim

def trainRPN(img,rpn_view):
    """
    img:图像经过网络数据张量
    rpn_view:rpn分类结果张量
    return:
    分类准确度
    各个anchor的分类情况
    """ 
    # print(net)
    # RPN操作
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

    r = [None] * 9
    pred_rpn = [None]*9
    loss_rpn = [None]*9
    for i in range(9):
        r[i] = tf.nn.conv2d(input = img,filter = weights['rpn_' + str(i+1)],strides = [1, 1, 1, 1],padding = 'VALID')
        r[i] = tf.reshape(r[i],r[i].get_shape().as_list()[1:-1])
        # rpn_view[i] = tf.reshape(rpn_view[i],(-1,))
        pred_rpn[i] = tf.add(r[i],biases['rpn_' + str(i+1)])
        pred_rpn[i] = tf.sigmoid(pred_rpn[i])
        # 防止梯度爆炸
        pred_rpn[i] = tf.clip_by_value(pred_rpn[i],1e-7,1.0-1e-7)
        loss_rpn[i] = -(tf.math.log(pred_rpn[i])*rpn_view[i] + (1 - rpn_view[i])*tf.math.log(1 - pred_rpn[i]))
        loss_rpn[i] = tf.reduce_mean(loss_rpn[i],name = 'loss_rpn_' + str(i+1))

    # 计算准确率
    ac_rpn = tf.abs(pred_rpn[0] - rpn_view[0])
    accuracy_rpn = 1 - tf.reduce_mean(ac_rpn,name = 'accuracy_rpn')
    for i in range(1,9):
        ac_rpn = tf.abs(pred_rpn[i] - rpn_view[i])
        accuracy_rpn = accuracy_rpn + 1 - tf.reduce_mean(ac_rpn,name = 'accuracy_rpn')
    accuracy_rpn = accuracy_rpn / 9

    # 计算损失
    for i in range(9):
        tf.compat.v1.losses.add_loss(tf.reduce_mean(loss_rpn[i])/9)

    return accuracy_rpn,pred_rpn