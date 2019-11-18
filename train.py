from my_resnet import my_resnet
from data_dealer import DataDealer
import tensorflow as tf

class NetTrainer:

    def __init__(self):
        pass

if __name__ == '__main__':
    for i in range(8):
        tf.compat.v1.reset_default_graph() # 重置计算图
        input_ = tf.get_variable(name = 'hi',shape = (10,480+i,480+i,3),dtype = tf.float32)
        output_ = my_resnet(input_)
        print(output_[0])