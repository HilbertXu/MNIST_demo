准备了一个MNIST的demo，有模型构建，训练，测试以及可视化的功能，注释都写了

#已经解决，在自己训练的时候一定要注意打乱数据集
但是也有过拟合的问题，不过我先在怀疑是软件的问题，而不是网络和数据集的问题
晚些时候用全连接网络进行测试

/src/MNIST_demo.py 以完成从图像输入和读取视频流的功能
以下是命令行输入语句
！！！！当前路径应该在/home/kamerider/Machine Intelligent/MNIST_demo/src下
using image:
python MNIST_demo.py --image_path ../test_image/0.png

using camera:
python MNIST_demo.py --camera_id 0
(camera_id 0 意味着使用电脑自带的摄像头，请确保摄像头能够正常工作)

train model:
python train_MNIST.py

！！！！在使用训练好的模型进行识别的时候，一定要注意输入模型的测试图片要压缩到28x28的大小，，和网络输入层要求的格式一致（28,28,1）
