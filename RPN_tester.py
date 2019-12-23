from define import *
from path import *
from my_resnet import my_resnet
from PIL import Image
from define import *
from math import floor

import tensorflow as tf
import numpy
import cv2
import time

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

slim = tf.contrib.slim


class RPNTester:

    def __init__(self,w = 192,h = 384):
        """
        w:测试图像宽度
        h:测试图像高度
        """
        self.w = w
        self.h = h
        self.RPN_RESULT_NUM = 4

    def loadModel(self,model_path = RPN_BATCH_PATH):
        """
        从model_path中加载模型
        """
        with tf.compat.v1.variable_scope('RPN', reuse=tf.compat.v1.AUTO_REUSE):
            weights = {
                'rpn_1':tf.compat.v1.get_variable(name = 'w_rpn_1_1',shape = [3,3,K*K*2,1]), # 高:宽 1:1的卷积
                'rpn_2':tf.compat.v1.get_variable(name = 'w_rpn_1_2',shape = [3,6,K*K*2,1]), # 高:宽 1:2的卷积
                'rpn_3':tf.compat.v1.get_variable(name = 'w_rpn_2_1',shape = [6,3,K*K*2,1]), # 高:宽 2:1的卷积
                'rpn_4':tf.compat.v1.get_variable(name = 'w_rpn_2_2',shape = [6,6,K*K*2,1]),
                'rpn_5':tf.compat.v1.get_variable(name = 'w_rpn_2_4',shape = [6,12,K*K*2,1]),
                'rpn_6':tf.compat.v1.get_variable(name = 'w_rpn_4_2',shape = [12,6,K*K*2,1]),
                'rpn_7':tf.compat.v1.get_variable(name = 'w_rpn_4_4',shape = [12,12,K*K*2,1]),
                'rpn_8':tf.compat.v1.get_variable(name = 'w_rpn_4_8',shape = [12,24,K*K*2,1]),
                'rpn_9':tf.compat.v1.get_variable(name = 'w_rpn_8_4',shape = [24,12,K*K*2,1])
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
        with tf.compat.v1.variable_scope('BBOX', reuse=tf.compat.v1.AUTO_REUSE):
            weights['bbox'] = tf.compat.v1.get_variable(name = 'w_bbox',shape = [K,K,K*K*2,4]) # 分类
            biases['bbox'] = tf.compat.v1.get_variable(name = 'b_bbox',shape = [4,]) # 分类

        weights['down'] = tf.compat.v1.get_variable(name = 'w_down',shape = [1,1,2048,1024])# 降采样
        weights['feature'] = tf.compat.v1.get_variable(name = 'w_feature',shape = [1,1,1024,K*K*2])
        biases['down'] = tf.compat.v1.get_variable(name = 'b_down',shape = [1024,]) # 降采样
        biases['feature'] = tf.compat.v1.get_variable(name = 'b_feature',shape = [K*K*2,])

        self.img = tf.compat.v1.placeholder(dtype = tf.float32,shape = (1,self.h,self.w,3))

        # 使用无pool1&pool5的RESNET 101
        net, endpoints = my_resnet(self.img,global_pool = False,num_classes=None,is_training=True,reuse = tf.compat.v1.AUTO_REUSE) # net's w&h = original_img's w&h / 16

        net = tf.nn.conv2d(input = net,filter = weights['down'],strides = [1, 1, 1, 1],padding = 'VALID')
        net = tf.add(net,biases['down'])

        # 生成feature_map
        self.feature_map = tf.nn.conv2d(input = net,filter = weights['feature'],strides = [1, 1, 1, 1],padding = 'VALID')
        self.feature_map = tf.add(self.feature_map,biases['feature'])

        self.pred_rpn = [None]*9
        for i in range(9):
            r = tf.nn.conv2d(input = self.feature_map,filter = weights['rpn_' + str(i+1)],strides = [1, 1, 1, 1],padding = 'VALID')
            r = tf.reshape(r,r.get_shape().as_list()[1:-1])
            self.pred_rpn[i] = tf.add(r,biases['rpn_' + str(i+1)])
            self.pred_rpn[i] = tf.sigmoid(self.pred_rpn[i])

        self.select = tf.compat.v1.placeholder(dtype = tf.float32,shape = (self.RPN_RESULT_NUM,K,K,K*K*2))
        self.pre_bbox = tf.nn.conv2d(self.select,weights['bbox'],[1,1,1,1],padding = 'VALID')
        self.pre_bbox = tf.add(self.pre_bbox,biases['bbox'])
        self.pre_bbox = tf.reshape(self.pre_bbox,shape = (self.RPN_RESULT_NUM,4))

        saver = tf.compat.v1.train.Saver(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES))

        self.sess =  tf.compat.v1.Session()
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

        saver.restore(self.sess,RPN_BATCH_PATH)

        
    def __getRPNTopList(self,rpn_result,top_num):
        """
        return top_list: value,[i,y,x]
        """
        top_list = []
        for t in range(top_num):
            top_list.append([0,[0,0,0]])
        for i in range(len(rpn_result)):
            for y in range(len(rpn_result[i])):
                for x in range(len(rpn_result[i][y])):
                    for t in range(top_num):
                        if rpn_result[i][y][x] > top_list[t][0]:
                            top_list = top_list[:t] + [[rpn_result[i][y][x],[i,y,x]]] + top_list[t:-1]
                            break
        return top_list

    def __markRPNResult(self,img,rpn_result,top_num):
        """
        img:图像矩阵
        rpn_result:结果
        top_num:个数
        将rpn的结果标注在图像上，选择top_num个结果
        输出图像
        return 
        """
        top_list = self.__getRPNTopList(rpn_result,top_num)

        img_cv = numpy.reshape(img,(self.h,self.w,3))
        img_cv = cv2.cvtColor(img_cv,cv2.COLOR_RGB2BGR)  

        color_rect = (0,0,255)
        color_text = (0,255,0)
        for top in top_list: 
            i,y,x = top[1]
            x = x*NET_SCALE
            y = y*NET_SCALE
            cv2.rectangle(img_cv, (x,y), (x+ANTHORS_TYPIES[i][0],y+ANTHORS_TYPIES[i][1]),color_rect,3)
            cv2.putText(img_cv, 'RPN', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, color_text,2)
        
        image = Image.fromarray(cv2.cvtColor(img_cv,cv2.COLOR_BGR2RGB))  
        image.show()  
        
    def __roiPool2d(self,data,ksize):
        """
        ROI AVERAGE POOLING
        STRIDE = KSIZE
        data : (h,w,mod)
        ksize : (hk,wk)
        roi -> 
        return 
        (hk,wk,mod)
        """
        h0,w0,mod = numpy.shape(data)
        h,w = ksize
        hk = int(h0/h)
        wk = int(w0/w)
        result = numpy.zeros((h,w,mod))
        for y in range(h):
            for x in range(w):
                for i in range(hk):
                    for j in range(wk):
                        result[y][x] = result[y][x] + data[y*hk+i][x*wk+j]
        result = result / hk / wk
                
        return result

    def __markResult(self,img,rpn_top,bbox_top):
        img_cv = numpy.reshape(img,(self.h,self.w,3))
        img_cv = cv2.cvtColor(img_cv,cv2.COLOR_RGB2BGR)  

        color_rect = (0,0,255)
        color_text = (0,255,0)
        # NMS非极大值抑制
        rpn_rect = list(map(lambda x:self.__getRect(x[1]),rpn_top))
        book = []

        for i in range(self.RPN_RESULT_NUM): 
            # 判断是否需要抑制
            # 重叠度>0.5的情况下抑制
            if len(list(filter(lambda x:self.__calculateIOU(rpn_rect[i],x) > 0.5,book))) > 0:
                continue

            x,y,w,h = rpn_rect[i]
            
            mx = x + w/2
            my = y + h/2
            mx = bbox_top[i][0] + mx
            my = bbox_top[i][1] + my
            w = bbox_top[i][2] * w
            h = bbox_top[i][3] * h
            x = mx - w/2
            y = my - h/2

            x = int(min(max(0,x),self.w))
            y = int(min(max(0,y),self.h))
            w = floor(max(0,min(w,self.w - x)))
            h = floor(max(0,min(h,self.h - y)))

            if x == self.w or y == self.h or w == 0 or h == 0:
                continue

            cv2.rectangle(img_cv, (x,y), (x+w,y+h),color_rect,3)
            cv2.putText(img_cv, str(rpn_top[i][0]), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, color_text,2)
            book.append([x,y,w,h])
        
        image = Image.fromarray(cv2.cvtColor(img_cv,cv2.COLOR_BGR2RGB))  
        image.show()  

    def __getRect(self,rpn_box):
        i,y,x = rpn_box
        x = x*NET_SCALE
        y = y*NET_SCALE
        w = ANTHORS_TYPIES[i][0]
        h = ANTHORS_TYPIES[i][1]
        return [x,y,w,h]

    def __calculateIOU(self,box1,box2):
        """
        box1:[x,y,w,h](x,y为左上角坐标)
        box2:[x,y,w,h](x,y为左上角坐标)
        return IOU
        """
        # IOU = 0 (未重叠)
        # IOU = (w1 * h1 + w2 * h2 - S(重叠部分面积)) / (w1 * h1 + w2 * h2)
        # S(重叠部分面积) = （min(x1+w1,x2+w2) - max(x1,x2))*(min(y1+h1,y2+h2) - max(y1,y2))
        w_S = min(box1[0]+box1[2],box2[0]+box2[2]) - max(box1[0],box2[0])
        h_S = min(box1[1]+box1[3],box2[1]+box2[3]) - max(box1[1],box2[1])
        # 若存在重叠
        if w_S > 0 and h_S > 0:
            S = w_S*h_S
            union = box1[2]*box1[3] + box2[2]*box2[3] - S 
            return w_S*h_S/union
        return 0

    def testPicVisual(self,img):
        """
        img:图像
        可视化测试图片
        """
        img = img.resize((self.w,self.h))
        img = numpy.reshape(img,(1,self.h,self.w,3))
        
        # RPN
        pred_rpn = self.sess.run([self.feature_map] + self.pred_rpn,feed_dict = {self.img:img})
        # 1,h,w,K*K*2
        feature_map = pred_rpn[0]
        pred_rpn = pred_rpn[1:]
        rpn_top_list = self.__getRPNTopList(pred_rpn,self.RPN_RESULT_NUM) # 获得候选区域
        print(rpn_top_list)
        self.__markRPNResult(img,pred_rpn,self.RPN_RESULT_NUM)

        # BBOX
        select = []
        anchor_type = [[int(x[0]/NET_SCALE),int(x[1]/NET_SCALE)] for x in ANTHORS_TYPIES]
        for top in rpn_top_list:
            data = feature_map[0,top[1][1]:(top[1][1]+anchor_type[top[1][0]][1]),top[1][2]:(top[1][2]+anchor_type[top[1][0]][0])]
            data = self.__roiPool2d(data,(K,K))
            select.append(data)
        select = numpy.array(select)
        pre_bbox = self.sess.run([self.pre_bbox],feed_dict = {self.select:select})
        pre_bbox = numpy.array(pre_bbox)
        pre_bbox = numpy.reshape(pre_bbox,(self.RPN_RESULT_NUM,4))

        # BBOX
        bbox_top_list = []
        for i in range(self.RPN_RESULT_NUM):
            bbox = pre_bbox[i]
            bbox_top_list.append(bbox)
        self.__markResult(img,rpn_top_list,bbox_top_list)

    def getImgRPN(self,img,top_num):
        """
        img:图片
        top_nmm:最多返回区域个数
        return [[x,y,w,h]...]
        """
        img = img.resize((self.w,self.h))
        img = numpy.reshape(img,(1,self.h,self.w,3))
        
        # RPN
        pred_rpn = self.sess.run([self.feature_map] + self.pred_rpn,feed_dict = {self.img:img})
        # 1,h,w,K*K*2
        feature_map = pred_rpn[0]
        pred_rpn = pred_rpn[1:]
        rpn_top_list = self.__getRPNTopList(pred_rpn,top_num) # 获得候选区域

        # BBOX
        select = []
        anchor_type = [[int(x[0]/NET_SCALE),int(x[1]/NET_SCALE)] for x in ANTHORS_TYPIES]
        for top in rpn_top_list:
            data = feature_map[0,top[1][1]:(top[1][1]+anchor_type[top[1][0]][1]),top[1][2]:(top[1][2]+anchor_type[top[1][0]][0])]
            data = self.__roiPool2d(data,(K,K))
            select.append(data)
        select = numpy.array(select)
        pre_bbox = self.sess.run([self.pre_bbox],feed_dict = {self.select:select})
        pre_bbox = numpy.array(pre_bbox)
        pre_bbox = numpy.reshape(pre_bbox,(top_num,4))

        # BBOX
        bbox_top_list = []
        for i in range(top_num):
            bbox = pre_bbox[i]
            bbox_top_list.append(bbox)
        
        # NMS非极大值抑制
        rpn_rect = list(map(lambda x:self.__getRect(x[1]),rpn_top_list))

        book = []
        for i in range(self.RPN_RESULT_NUM): 
            # 判断是否需要抑制
            # 重叠度>0.5的情况下抑制
            if len(list(filter(lambda x:self.__calculateIOU(rpn_rect[i],x) > 0.5,book))) > 0:
                continue

            x,y,w,h = rpn_rect[i]
            
            mx = x + w/2
            my = y + h/2
            mx = bbox_top_list[i][0] + mx
            my = bbox_top_list[i][1] + my
            w = bbox_top_list[i][2] * w
            h = bbox_top_list[i][3] * h
            x = mx - w/2
            y = my - h/2

            x = int(min(max(0,x),self.w))
            y = int(min(max(0,y),self.h))
            w = floor(max(0,min(w,self.w - x)))
            h = floor(max(0,min(h,self.h - y)))

            if x == self.w or y == self.h or w == 0 or h == 0:
                continue
            book.append([x,y,w,h])
        return book

    def __del__(self):
        print('SESS CLOSE')
        self.sess.close()

if __name__ == '__main__':
    tester = RPNTester()
    tester.loadModel()
    pic_list = ['D:\\Face\\data\\originalPics\\2003\\02\\02\\big\\img_806.jpg',
    'D:\\Face\\data\\originalPics\\2003\\02\\02\\big\\img_138.jpg',
    'D:\\\Face\\data\\originalPics\\2003\\02\\02\\big\\img_1005.jpg',
    'D:\\\Face\\data\\originalPics\\2003\\02\\03\\big\\img_922.jpg']
    for pic in pic_list:
        img = Image.open(pic)
        t_ = time.time()
        tester.testPicVisual(img)
        print('use ' + str(time.time() - t_) + ' seconds')
