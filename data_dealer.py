from data_reader import DataReader
from define import *
import random
import math
import numpy

class DataDealer(DataReader):

    def __init__(self):
        DataReader.__init__(self)

    def getRandomImgIds(self,num):
        """
        num:正整数
        return num个随机的img_index
        """
        img_index = []
        while len(img_index) != num:
            index = random.randint(0,len(self.imgs)-1)
            if index in img_index:
                continue
            img_index.append(index)
        return img_index

    def getRandomTrainBatch(self,num):
        """
        num:正整数
        return img_batch label_batch
        """
        return self.getTrainBatch(self.getRandomImgIds(num))

    def __ellipse2Rect(self,ellipse):
        """
        ellipse:[ra, rb, Θ, cx, cy, s]
        ra，rb：半长轴、半短轴
        cx, cy：椭圆中心点坐标
        Θ：长轴与水平轴夹角（头往左偏Θ为正，头往右偏Θ为负）
        s：置信度得分
        return rect = [x,y,w,h](x,y为左上角坐标)
        """
        ra, rb, theta, cx, cy, s = ellipse
        w = 2*max([abs(ra*math.sin(theta)),abs(rb*math.cos(theta))])
        h = 2*max([abs(ra*math.cos(theta)),abs(rb*math.sin(theta))])
        rect = [cx-w/2,cy-h/2,w,h]
        return rect

    def __calculateBbox(self,box1,box2):
        """
        box1:[x,y,w,h](x,y为左上角坐标)
        box2:[x,y,w,h](x,y为左上角坐标)
        return Bounding Box Regression
        """
        box1 = [box1[0]+box1[2]/2,box1[1]+box1[3]/2,box1[2],box1[3]]
        box2 = [box2[0]+box2[2]/2,box2[1]+box2[3]/2,box2[2],box2[3]]
        return [box2[0]-box1[0],box2[1]-box1[1],box2[2]/box1[2],box2[3]/box1[3]]

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

    def __calculateAllIOU(self,box,anno):
        """
        box:[x,y,w,h](x,y为左上角坐标)
        anno = [[x,y,w,h]...]
        return IOU
        """
        result = []
        index = 0
        for a in anno:
            # 全部在放缩的网络上进行IOU计算a
            box1 = [int(x/NET_SCALE) for x in box]
            box2 = [int(x/NET_SCALE) for x in a]
            IOU = self.__calculateIOU(box1,box2)
            result.append(IOU)
            index += 1
        return result

    def __calculateBbox(self,box1,box2):
        """
        box1:[x,y,w,h](x,y为左上角坐标)
        box2:[x,y,w,h](x,y为左上角坐标)
        return Bounding Box Regression
        """
        box1 = [box1[0]+box1[2]/2,box1[1]+box1[3]/2,box1[2],box1[3]]
        box2 = [box2[0]+box2[2]/2,box2[1]+box2[3]/2,box2[2],box2[3]]
        return [box2[0]-box1[0],box2[1]-box1[1],box2[2]/box1[2],box2[3]/box1[3]]

    def getTrainBatch(self,img_index):
        """
        img_index:img_index列表
        return img_batch label_batch bbox_batch
        """
        img_batch = []
        label_batch = []
        bbox_batch = []
        for index in img_index:
            img,anno = self.getOneImgAndAnno(index)
            # 选择规范的数据
            try:
                height,width,mod = numpy.shape(img)
            except:
                # 二维图像跳过
                continue
            # 不规范的图像跳过
            if height < MIN_TRAIN_PIC_HEIGHT or width < MIN_TRAIN_PIC_WIDTH or not mod == 3:
                continue
            img_batch.append(img)
            data = [] # 二级数据,记录每个anchor的所属类别及IOU
            label = []
            bbox = []
            book = []
            # 计算各个区间的IOU
            for i in range(ANTHORS_TYPE_NUM*3):
                data.append([])
                label.append([])
                bbox.append([])
                for y in range(0,math.ceil(height/NET_SCALE) - int(ANTHORS_TYPIES[i][1]/NET_SCALE) + 1):
                    data[i].append([])
                    label[i].append([])
                    bbox[i].append([])
                    for x in range(0,math.ceil(width/NET_SCALE) - int(ANTHORS_TYPIES[i][0]/NET_SCALE) + 1):
                        box1 = [x*NET_SCALE,y*NET_SCALE,ANTHORS_TYPIES[i][0],ANTHORS_TYPIES[i][1]]
                        result = self.__calculateAllIOU(box1,[self.__ellipse2Rect(a) for a in anno])
                        max_index = 0
                        for index in range(len(result)):
                            if result[index] > result[max_index]:
                                max_index = index
                        data[i][y].append(result)
                        if not max_index in book:
                            book.append(max_index)
                        if result[max_index] > IOU_POSITIVE:
                            # print(data[i][y][x][max_index])
                            label[i][y].append(1)
                            bbox[i][y].append(self.__calculateBbox(box1,self.__ellipse2Rect(anno[max_index])))
                        else:
                            label[i][y].append(0)
                            bbox[i][y].append([0,0,0,0])
            # 若某张脸不存在正例，则选择和它重合度最大的
            for index in range(len(anno)):
                if index not in book:
                    max_i = 0
                    max_y = 0
                    max_x = 0
                    for i in range(ANTHORS_TYPE_NUM*3):
                        for y in range(0,math.ceil(height/NET_SCALE) - int(ANTHORS_TYPIES[i][1]/NET_SCALE) + 1):
                            for x in range(0,math.ceil(width/NET_SCALE) - int(ANTHORS_TYPIES[i][0]/NET_SCALE) + 1):
                                if data[i][y][x][index] > data[max_i][max_y][max_x][index]:
                                    max_i = i
                                    max_y = y
                                    max_x = x
                    label[max_i][max_y][max_x] = 1
                    box1 = [max_x*NET_SCALE,max_y*NET_SCALE,ANTHORS_TYPIES[max_i][0],ANTHORS_TYPIES[max_i][1]]
                    bbox[max_i][max_y][max_x] = self.__calculateBbox(box1,self.__ellipse2Rect(anno[index]))
                    # print(data[max_i][max_y][max_x][index])
            label_batch.append(label)
            bbox_batch.append(bbox)
        img_batch = numpy.array(img_batch)
        label_batch = numpy.array(label_batch)
        bbox_batch = numpy.array(bbox_batch)
        return img_batch,label_batch,bbox_batch

    @staticmethod
    def chooseClassficationData(view,num = 128):
        """
        view:前景分类
        num:返回数据矩阵个数
        return data
        """
        data = []
        for i in range(len(view)):
            for y in range(len(view[i])):
                for x in range(len(view[i][y])): 
                    if len(data) >= 128:
                        return data
                    # 若为前景
                    if view[i][y][x] == 1:
                        data.append([i,y,x])
        while len(data) < 128:
            i = random.randint(0,len(view)-1)
            data.append([i,random.randint(0,len(view[i])-1),random.randint(0,len(view[i][0])-1)])
        return data


if __name__ == '__main__':
    dealer = DataDealer()
    img_batch,label_batch,bbox_batch = dealer.getRandomTrainBatch(10)
    print(numpy.shape(label_batch[0][0]))
    print(numpy.shape(bbox_batch[0][0]))