from path import *
from PIL import Image
import numpy
import os

class DataReader:
    # 读取数据集内容

    def __init__(self):
        self.imgs = []
        self.book = {}
        for root, dirs, files in os.walk(ANNOTATIONS_PATH):  
            pass
        files = list(filter(lambda x:len(x) >= 20,files))
        for file_ in files:
            with open(ANNOTATIONS_PATH + '\\' + file_,'r') as f:
                while True:
                    img = f.readline()[:-1]
                    img = img.replace('/','\\',5)
                    if not img:
                        break
                    self.imgs.append(img)
                    self.book[img] = []
                    num = int(f.readline())
                    for i in range(num):
                        self.book[img].append(list(map(lambda x:float(x),f.readline().split())))

    def getOneImgAndAnno(self,index):
        """
        获取一个图片和Anno的匹配
        返回 Img矩阵 Anno
        """
        img = Image.open(IMG_PATH + '\\' + self.imgs[index] + '.jpg')
        # print(IMG_PATH + '\\' + self.imgs[index] + '.jpg')
        anno = self.book[self.imgs[index]]
        return numpy.array(img),anno
                

if __name__ == '__main__':
    reader = DataReader()
    img,anno = reader.getOneImgAndAnno(3)
    print(numpy.shape(img))
