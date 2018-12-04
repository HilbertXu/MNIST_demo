# -*- coding: utf-8 -*-
#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
#core layers
from keras.layers import Dense, Dropout, Activation, Flatten

#Conv layers & Maxpooling layers
from keras.layers import Convolution2D, MaxPool2D

#BN layers
from keras.layers.normalization import BatchNormalization

#used to generate one-hot code
from keras.utils import np_utils

from keras.models import load_model
from keras.optimizers import SGD
from keras import backend as K

batch_size =128
nb_classes = 10
nb_epoch = 12
IMAGE_SIZE = 28

#卷积层使用卷积核的数目
nb_filter = 6
#池化层操作范围
pool_size = (2,2)
#卷积核大小
kernel_size = (5,5)

MODEL_PATH = '../model/MNIST_demo.h5'
HISTORY_PATH = '../History/Train_History.txt'
FIGURE_PATH  = '../History'
DATASET_PATH = '../DataSet/mnist.npz'

class dataset:
    def __init__(self):
        #训练集的样本和标签(x 为图像样本, y 为每一张图片对应的标签)
        self.x_train     = None
        self.y_train     = None

        #测试集的样本和标签
        self.x_test      = None
        self.y_test      = None

        #输入网络层的图像的格式
        self.input_shape = None

    def load(self, image_rows = IMAGE_SIZE, image_cols = IMAGE_SIZE,
                    image_channels = 1, predifined_classes = nb_classes):
        '''
        由此方法加载数据集需要从外网下载
        Downloading data from https://s3.amazonaws.com/img-datasets/mnist.npz
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        但是这个url被墙了，所以只能手动下载后自己加载数据集
        '''
        mnist_data = np.load(DATASET_PATH)
        x_train, y_train = mnist_data['x_train'], mnist_data['y_train']
        x_test, y_test = mnist_data['x_test'], mnist_data['y_test']
        mnist_data.close()

        #一样的要根据后端的不同来调整图像三通道的顺序
        #tensorflow后端下要调整图像三通道顺序为(100,28,28,3)
        #即(图像数量，image_rows, image_cols, color)
        #加上一个判断语句以防万一
        if K.image_dim_ordering() == 'th':
            print 'Theano detected'
            x_train = x_train.reshape(x_train.shape[0],image_channels, image_rows, image_cols)
            x_test  = x_test.reshape(x_test.shape[0], image_channels, image_rows, image_cols)
            self.input_shape = (image_channels, image_rows, image_cols)

            #这一段是使用全连接网络时使用的数据输入格式
            #x_train = x_train.reshape(x_train.shape[0], 784)
            #x_test  = x_test.reshape(x_test.shape[0], 784)

        if K.image_dim_ordering() == 'tf':
            print 'Tensorflow detected'
            x_train = x_train.reshape(x_train.shape[0], image_rows, image_cols, image_channels)
            x_test  = x_test.reshape(x_test.shape[0], image_rows, image_cols, image_channels)
            self.input_shape = (image_rows, image_cols, image_channels)

            #这一段是使用全连接网络时使用的数据输入格式
            #x_train = x_train.reshape(x_train.shape[0], 784)
            #x_test  = x_test.reshape(x_test.shape[0], 784)

        #9314~
        #np.random.seed(9314)
        #random.shuffle(index)



        #训练集中样本和标签为float32型
        x_train = x_train.astype('float32')
        x_test  = x_test.astype('float32')

        #归一化
        x_train /= 255
        x_test /= 255

        print ('x_train shape', x_train.shape)
        print ('train samples', x_train.shape[0])
        print ('test sample', x_test.shape[0])

        #利用np_utils生成训练师需要的one-hot code
        #一般多分类问题样本的标签需要使用one-hot code
        #即有多少种分类状态(nb_classes)就生成一个多少维(nb_classes)的标签，每一个样本对应的标签
        #形式为: [0.......1........0]
        #其中有(nb_classes-1)个0, 在该样本所属类别对应的那一个位置(k)
        #标签对应的(nb_classes)维行向量的第k个元素为1, 指示了该样本的所属类别
        #也就是该样本对应的那个类别它火起来了呀~所以才叫独热码(`・ω・´)
        #使用如下的to_categorical来生成所需要的one-hot code

        #-----------!!!!!---------
        #注意在生成独热码之前的初始标签应该是一个np.array型的数组
        #该数组应该从0开始顺序编号，从0一直到(nb_classes-1)
        y_train = np_utils.to_categorical(y_train, 10)
        y_test  = np_utils.to_categorical(y_test, 10)

        self.x_train = x_train
        self.x_test = x_test

        self.y_train = y_train
        self.y_test  = y_test

class Model:
    def __init__(self):
        self.model = None

    def build_model(self, dataset, nb_classes = 10):
        print 'the dataset input shape is :'
        print dataset.input_shape
        print 'now we start to build our CNN  <(~︶~)>'
        #建立一个空的序贯式模型
        #因为要拿CNN讲所以把原来的全连接模型改成了LeNet-5网络
        self.model = Sequential()
        
        self.model.add(Convolution2D(nb_filter, kernel_size[0], kernel_size[1], border_mode = 'valid', input_shape = dataset.input_shape))
        self.model.add(Activation('tanh'))
        #选用relu激活函数
        #f(x) = max(0, x)
        #即只要x大于0该网络层就会被激活

        self.model.add(Convolution2D(16, 3, 3, border_mode = 'valid'))
        self.model.add(Activation('tanh'))

        #self.model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9,
        #                                    weights=None, beta_init='zero', gamma_init='one'))
        #池化层 & Dropout层
        self.model.add(MaxPool2D(pool_size=(2,2)))
        self.model.add(Dropout(0.25))
        #在池化操作之后，以0.25为比例，随机地断开现有的神经元
        #这样其他的神经元会被迫地学习被断开部分神经元的内容
        #这样可以使神经元学习到更广的内容，能有效地避免过拟合现象

        #Flatten层，将卷积层的多维输出一维化，常用在卷积层到全连接层的过度
        self.model.add(Flatten())

        #全连接层
        self.model.add(Dense(120, activation='tanh'))
        self.model.add(Dense(84, activation='tanh'))
        self.model.add(Dropout(0.5))

        #输出层，激活函数为‘softmax’
        #softmax是一个线性分类器
        #可以将输出转化为对应每一个种类的置信度
        self.model.add(Dense(nb_classes))
        self.model.add(Activation('softmax'))

        #输出模型的参数信息
        self.model.summary()

        #全连接网络样例
        '''
        self.model = Sequential()

        self.model.add(Dense(500, input_shape=(784,)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(500))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(10))
        self.model.add(Activation('softmax'))

        self.model.summary()
        '''


    def train(self, dataset, batch_size = batch_size, nb_epoch = nb_epoch,
              data_augmentation = False):
        sgd = SGD(lr = 0.05, decay = 1e-6,
                momentum = 0.9, nesterov = True) #采用SGD+momentum的优化器进行训练，首先生成一个优化器对象
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='rmsprop',#优化器可以选择rmsprop也可以选择自己生成的SGD优化器
                           #optimizer='sgd',    #如果使用SGD都能跑地效果很好那说明网络已经搭建地很好了~
                           metrics=['accuracy'])
        #不使用数据提升，所谓的提升就是从我们提供的训练数据中利用旋转、翻转、加噪声等方法
        #创造新的训练数据，有意识的提升训练数据规模，增加模型训练量
        if not data_augmentation:
            #recording loss, loss_val, accuracy, accuracy_val
            #and wirte it to Train_History.txt

            #在fit之前手动打乱数据集，寻找一个提高val_acc的解决方法
            #index = [i for i in range(len(dataset))]

            hist = self.model.fit(dataset.x_train,
                           dataset.y_train,
                           batch_size = batch_size,
                           nb_epoch = nb_epoch,
                           validation_split = 0.33,
                           shuffle = True)
            with open(HISTORY_PATH,'w') as f:
                f.write(str(hist.history))

            #visualization
            #store the output history
            model_val_loss = hist.history['val_loss']
            model_val_acc  = hist.history['val_acc']
            model_loss     = hist.history['loss']
            model_acc      = hist.history['acc']

            #Using matplotlib to visualize
            epochs = np.arange(nb_epoch)+1
            plt.figure()
            plt.plot(epochs, model_val_loss, label = 'model_val_loss')
            plt.plot(epochs, model_loss, label = 'model_loss')
            plt.title('visualize the training process')
            plt.xlabel('Epoch #')
            plt.ylabel('Validation Loss & Train Loss')
            plt.legend()
            plt.savefig(FIGURE_PATH+'/loss_figure.png')
            plt.show()

            plt.figure()
            plt.plot(epochs, model_val_acc, label = 'model_val_acc')
            plt.plot(epochs, model_acc, label = 'model_acc')
            plt.title('visualize the training process')
            plt.xlabel('Epoch #')
            plt.ylabel('Validation accuracy & Train accuracy')
            plt.legend()
            plt.savefig(FIGURE_PATH+'/acc_figure.png')
            plt.show()

        #使用实时数据提升
        else:
            #定义数据生成器用于数据提升，其返回一个生成器对象datagen，datagen每被调用一
            #次其生成一组数据（顺序生成），节省内存，其实就是python的数据生成器
            datagen = ImageDataGenerator(
                featurewise_center = False,             #是否使输入数据去中心化（均值为0），
                samplewise_center  = False,             #是否使输入数据的每个样本均值为0
                featurewise_std_normalization = False,  #是否数据标准化（输入数据除以数据集的标准差）
                samplewise_std_normalization  = False,  #是否将每个样本数据除以自身的标准差
                zca_whitening = False,                  #是否对输入数据施以ZCA白化
                rotation_range = 20,                    #数据提升时图片随机转动的角度(范围为0～180)
                width_shift_range  = 0.2,               #数据提升时图片水平偏移的幅度（单位为图片宽度的占比，0~1之间的浮点数）
                height_shift_range = 0.2,               #同上，只不过这里是垂直
                horizontal_flip = True,                 #是否进行随机水平翻转
                vertical_flip = False)                  #是否进行随机垂直翻转

            #计算整个训练样本集的数量以用于特征值归一化、ZCA白化等处理
            datagen.fit(dataset.x_train)

            #利用生成器开始训练模型
            hist = self.model.fit_generator(datagen.flow(dataset.x_train, dataset.y_train,
                                                   batch_size = batch_size),
                                     samples_per_epoch = dataset.x_train.shape[0],
                                     nb_epoch = nb_epoch,
                                     validation_split = 0.33,
                                     shuffle = True)
            with open(HISTORY_PATH,'w') as f:
                f.write(str(hist.history))

            #visualization
            #store the output history
            model_val_loss = hist.history['val_loss']
            model_val_acc  = hist.history['val_acc']
            model_loss     = hist.history['loss']
            model_acc      = hist.history['acc']

            #Using matplotlib to visualize
            epochs = np.arange(nb_epoch)+1
            plt.figure()
            plt.plot(epochs, model_val_loss, label = 'model_val_loss')
            plt.plot(epochs, model_loss, label = 'model_loss')
            plt.title('visualize the training process')
            plt.xlabel('Epoch #')
            plt.ylabel('Validation Loss & Train Loss')
            plt.legend()
            plt.savefig(FIGURE_PATH+'/loss_figure.png')
            plt.show()

            plt.figure()
            plt.plot(epochs, model_val_acc, label = 'model_val_acc')
            plt.plot(epochs, model_acc, label = 'model_acc')
            plt.title('visualize the training process')
            plt.xlabel('Epoch #')
            plt.ylabel('Validation accuracy & Train accuracy')
            plt.legend()
            plt.savefig(FIGURE_PATH+'/acc_figure.png')
            plt.show()

    def save_model(self, file_path = MODEL_PATH):
        self.model.save(file_path)

    def evaluate(self, dataset):
        score = self.model.evaluate(dataset.x_test, dataset.y_test, verbose=1)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))

    def load_model(self, file_path = MODEL_PATH):
        self.model = load_model(file_path)

    #识别人脸
    def predict(self, image):
        #依然是根据后端系统确定维度顺序
        if K.image_dim_ordering() == 'th' and image.shape != (1, 1, IMAGE_SIZE, IMAGE_SIZE):
            #image = resize_image(image)                             #尺寸必须与训练集一致都应该是IMAGE_SIZE x IMAGE_SIZE
            image = image.reshape((1, 1, IMAGE_SIZE, IMAGE_SIZE))   #与模型训练不同，这次只是针对1张图片进行预测
        elif K.image_dim_ordering() == 'tf' and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 1):
            #image = resize_image(image)
            image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 1))

        #浮点并归一化
        image = image.astype('float32')
        image /= 255

        #给出输入属于各个类别的概率
        result = self.model.predict_proba(image)
        print('result:', result)

        #给出类别预测：每一个类别的置信度
        result = self.model.predict_classes(image)

        #返回类别预测结果
        return result[0]


if __name__ == '__main__':
    dataset = dataset()
    dataset.load()

    model = Model()
    model.build_model(dataset)
    model.train(dataset)
    model.save_model(MODEL_PATH)

    model = Model()
    model.load_model(MODEL_PATH)
    model.evaluate(dataset)
