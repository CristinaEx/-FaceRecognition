from my_resnet import my_resnet
from data_dealer import DataDealer
from path import *
from define import *
from RPN_trainer import trainRPN
from bbox_trainer import trainBbox
import tensorflow as tf
import numpy
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

slim = tf.contrib.slim

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
plt.ion()
plt.show()

class RPNTrainer:

    def __init__(self,model_path = RPN_BATCH_PATH,init_model_path = RES_NET_101):
        self.dealer = DataDealer()
        self.model_path = model_path
        self.init_model_path = init_model_path

    def train(self,round_,batch_size,init_learning_rate,learning_rate_loss):
        """
        round_:次数
        batch_size:图片数
        init_learning_rate:初始化学习率
        learning_rate_loss：学习率衰减
        """
        accuracy = []
        rounds = []
        for i in range(2):
            accuracy.append([])
        rate = init_learning_rate
        for i in range(round_):
            tf.compat.v1.reset_default_graph() # 重置计算图
            try:
                ac = self.trainOne(batch_size,rate)
            except:
                print('ERROR')
                continue
            else:
                pass
            for j in range(2):
                accuracy[j].append(ac[j])
            rounds.append(i)
            plt.figure(1)
            plt.clf()          
            plt.subplot(211)
            plt.plot(rounds,accuracy[0],'-r')
            plt.subplot(212)
            plt.plot(rounds,accuracy[1],'-b')
            plt.pause(60)        
            rate = rate * learning_rate_loss

    def trainOne(self,batch_size,learning_rate,rounds = 10):
        """
        batch_size:图片数
        learning_rate:学习率
        rounds:运算次数
        """
        img_batch,label_batch,bbox_batch = self.dealer.getRandomTrainBatch(batch_size)

        if not os.path.exists(os.path.dirname(self.model_path)):
            # os.makedirs(os.path.dirname(self.model_path))
            RESTORE = False
        else:
            RESTORE = True

        weights = {
            'down':tf.compat.v1.get_variable(name = 'w_down',shape = [1,1,2048,1024]),# 降采样
            'feature':tf.compat.v1.get_variable(name = 'w_feature',shape = [1,1,1024,K*K*2])
            }
        biases = {
            'down':tf.compat.v1.get_variable(name = 'b_down',shape = [1024,]), # 降采样
            'feature':tf.compat.v1.get_variable(name = 'b_feature',shape = [K*K*2,])
        }

        for index in range(len(img_batch)):
            img = img_batch[index]
            label = label_batch[index]
            bbox = bbox_batch[index]   

            rpn_view = [None]*9
            for i in range(9):
                rpn_view[i] = tf.constant(label[i],dtype = tf.float32) # 高宽比1:1 1:2 2:1

            h,w,mod = numpy.shape(img)
            img = tf.constant(img,shape = (1,h,w,mod),dtype = tf.float32) # 图像原始数据

            # 使用无pool1&pool5的RESNET 101
            net, endpoints = my_resnet(img,global_pool = False,num_classes=None,is_training=True,reuse = tf.compat.v1.AUTO_REUSE) # net's w&h = original_img's w&h / 16

            net = tf.nn.conv2d(input = net,filter = weights['down'],strides = [1, 1, 1, 1],padding = 'VALID')
            net = tf.add(net,biases['down'])

            # 生成feature_map
            feature_map = tf.nn.conv2d(input = net,filter = weights['feature'],strides = [1, 1, 1, 1],padding = 'VALID')
            feature_map = tf.add(feature_map,biases['feature'])

            # 训练RPN网络
            rpn_accuracy,rpn_result = trainRPN(feature_map,rpn_view)

            # 获取选取的anchors index
            select = DataDealer.chooseClassficationData(label)

            select_img = []
            select_label= []
            select_bbox = []
            anchor_type = [[int(x[0]/NET_SCALE/K),int(x[1]/NET_SCALE/K)] for x in ANTHORS_TYPIES]
            for s in select:
                img_ = feature_map[0,s[1]:(s[1]+anchor_type[s[0]][1]*K),s[2]:(s[2]+anchor_type[s[0]][0]*K)]
                img_ = tf.expand_dims(img_,0)
                select_img.append(tf.nn.avg_pool2d(img_,[1,anchor_type[s[0]][1],anchor_type[s[0]][0],1],[1,anchor_type[s[0]][1],anchor_type[s[0]][0],1],padding = 'VALID'))
                select_label.append(label[s[0]][s[1]][s[2]])
                select_bbox.append(bbox[s[0]][s[1]][s[2]])
            
            # 训练bounding_box
            bbox_accuracy = trainBbox(select,select_img,select_bbox,select_label)

        # 建立优化器 随机梯度下降
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = learning_rate)

        # 减少误差，提升准确度
        train = optimizer.minimize(tf.compat.v1.losses.get_total_loss())

        saver = tf.compat.v1.train.Saver(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES))

        with tf.compat.v1.Session() as sess:
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)

            if RESTORE:
                saver.restore(sess,self.model_path)
            else:
                slim.assign_from_checkpoint_fn(self.init_model_path, slim.get_variables_to_restore(), # 第一次path = RES_NET_101 后来:model_path
                               ignore_missing_vars=False,
                               reshape_variables=False)            
            for i in range(rounds):
                if i == 0:
                    ac = sess.run([rpn_accuracy,bbox_accuracy])
                sess.run(train)         
                # print(sess.run([rpn_accuracy,bbox_accuracy,tf.compat.v1.losses.get_total_loss()]))
            if not RESTORE:
                os.makedirs(os.path.dirname(self.model_path))
            saver.save(sess, self.model_path)
            
        return ac

if __name__ == '__main__':
    trainer = RPNTrainer()
    trainer.train(round_ = 1000,batch_size = 1,init_learning_rate = 0.000001,learning_rate_loss = 0.999)