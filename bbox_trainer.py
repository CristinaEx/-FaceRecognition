from define import *
import tensorflow as tf

def trainBbox(img,bbox,select):
    """
    img:图像feature map数据
    bbox:bbox数据
    select:选取数据
    """
    weights = {
        'bbox':tf.compat.v1.get_variable(name = 'w_bbox',shape = [K,K,K*K,4]) # 分类
    }
    biases = {
        'bbox':tf.compat.v1.get_variable(name = 'b_bbox',shape = [4,]) # 分类
    }

    data_bbox = []
    for s in select:
        data_bbox.append(bbox[s[0]][s[1]][s[2]])
    data_bbox = tf.constant(data_bbox,dtype = tf.float32) 


    pre_bbox = tf.nn.conv2d(img_data,weights['bbox'],[1,1,1,1],padding = 'VALID')
    pre_bbox = tf.add(pre_bbox,biases['bbox'])
    pre_bbox = tf.reshape(pre_bbox,shape = (len(select),2,4))

    data_type = tf.reshape(data_type,(1,len(select),2,1))
    data_type = tf.nn.conv2d(data_type,tf.ones((1,1,1,4)),[1,1,1,1],padding = 'VALID')
    data_type = tf.reshape(data_type,(len(select),2,4))

    if DEBUG:
        print('BBOX DEBUG:')
        print(img_data)
        print(pre_bbox)
        print(data_bbox)
        print(data_type)

    # 计算损失
    loss_bbox = tf.abs(pre_bbox - data_bbox) * data_type 
    loss_bbox = tf.reduce_mean(loss_bbox,name = 'loss_bbox')
    tf.compat.v1.losses.add_loss(loss_bbox) 

    # 计算准确率
    accuracy_bbox = 1 - loss_bbox

    return accuracy_bbox