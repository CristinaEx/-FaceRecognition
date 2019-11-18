from define import *
import tensorflow as tf

def trainBbox(select,select_img,select_bbox,select_label):
    """
    select_img:选择的图像feature map数据
    select_label:选择的标签数据
    select_bbox:选择的bbox数据
    select:选取数据
    """
    with tf.compat.v1.variable_scope('BBOX', reuse=tf.compat.v1.AUTO_REUSE):
        weights = {
            'bbox':tf.compat.v1.get_variable(name = 'w_bbox',shape = [K,K,K*K*2,4]) # 分类
        }
        biases = {
            'bbox':tf.compat.v1.get_variable(name = 'b_bbox',shape = [4,]) # 分类
        }
    
    select_img = tf.reshape(select_img,(-1,K,K,K*K*2))
    pre_bbox = tf.nn.conv2d(select_img,weights['bbox'],[1,1,1,1],padding = 'VALID')
    pre_bbox = tf.add(pre_bbox,biases['bbox'])
    pre_bbox = tf.reshape(pre_bbox,shape = (len(select),1,4))

    select_label = tf.constant(select_label,dtype = tf.float32)
    data_type = tf.reshape(select_label,(1,len(select),1,1))
    data_type = tf.nn.conv2d(data_type,tf.ones((1,1,1,4)),[1,1,1,1],padding = 'VALID')
    data_type = tf.reshape(data_type,(len(select),1,4))

    # 计算损失
    loss_bbox = (pre_bbox - select_bbox) * data_type 
    loss_bbox = tf.reduce_mean(loss_bbox*loss_bbox)
    tf.compat.v1.losses.add_loss(loss_bbox) 

    # 计算准确率
    accuracy_bbox = 1 - loss_bbox

    return accuracy_bbox