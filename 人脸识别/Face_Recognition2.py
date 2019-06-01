import paddle
import os
import paddle.fluid as fluid

import numpy

import sys

from multiprocessing import cpu_count

# 定义训练的mapper

# train_mapper函数的作用是用来对训练集的图像进行处理修剪和数组变换，返回img数组和标签

# sample是一个python元组，里面保存着图片的地址和标签。 ('../images/face/zhangziyi/20181206145348.png', 2)

def train_mapper(sample):

    img, label = sample

    # 进行图片的读取，由于数据集的像素维度各不相同，需要进一步处理对图像进行变换

    img = paddle.dataset.image.load_image(img)

    #进行了简单的图像变换，这里对图像进行crop修剪操作，输出img的维度为(3, 100, 100)

    img = paddle.dataset.image.simple_transform(im=img,          #输入图片是HWC

                                                resize_size=100, # 剪裁图片

                                                crop_size=100,

                                                is_color=True,  #彩色图像

                                                is_train=True)

    #将img数组进行进行归一化处理，得到0到1之间的数值

    img= img.flatten().astype('float32')/255.0

    return img, label

# 对自定义数据集创建训练集train的reader

def train_r(train_list, buffered_size=1024):

    def reader():

        with open(train_list, 'r') as f:

            # 将train.list里面的标签和图片的地址方法一个list列表里面，中间用\t隔开'

            #../images/face/jiangwen/0b1937e2-f929-11e8-8a8a-005056c00008.jpg\t0'

            lines = [line.strip() for line in f]

            for line in lines:

                # 图像的路径和标签是以\t来分割的,所以我们在生成这个列表的时候,使用\t就可以了

                img_path, lab = line.strip().split('\t')

                yield img_path, int(lab)

    # 创建自定义数据训练集的train_reader

    return paddle.reader.xmap_readers(train_mapper, reader,cpu_count(), buffered_size)


# sample是一个python元组，里面保存着图片的地址和标签。 ('../images/face/zhangziyi/20181206145348.png', 2)

def test_mapper(sample):

    img, label = sample

    img = paddle.dataset.image.load_image(img)

    img = paddle.dataset.image.simple_transform(im=img, resize_size=100, crop_size=100, is_color=True, is_train=False)

    img= img.flatten().astype('float32')/255.0

    return img, label


# 对自定义数据集创建验证集test的reader

def test_r(test_list, buffered_size=1024):

    def reader():

        with open(test_list, 'r') as f:

            lines = [line.strip() for line in f]

            for line in lines:

                #图像的路径和标签是以\t来分割的,所以我们在生成这个列表的时候,使用\t就可以了

                img_path, lab = line.strip().split('\t')

                yield img_path, int(lab)


    return paddle.reader.xmap_readers(test_mapper, reader,cpu_count(), buffered_size)

BATCH_SIZE = 32

# 把图片数据生成reader

trainer_reader = train_r(train_list="/home/aistudio/data/data2394/face/trainer.list")

train_reader = paddle.batch(

    paddle.reader.shuffle(

        reader=trainer_reader,buf_size=300),

    batch_size=BATCH_SIZE)


tester_reader = test_r(test_list="/home/aistudio/data/data2394/face/test.list")

test_reader = paddle.batch(

     tester_reader, batch_size=BATCH_SIZE)
temp_reader = paddle.batch(trainer_reader,

                            batch_size=3)

temp_data=next(temp_reader())

print(temp_data)
image = fluid.layers.data(name='image', shape=[3, 100, 100], dtype='float32')#[3, 100, 100]，表示为三通道，100*100的RGB图

label = fluid.layers.data(name='label', shape=[1], dtype='int64')

print('image_shape:',image.shape)


def convolutional_neural_network(image, type_size):
    # 第一个卷积--池化层

    conv_pool_1 = fluid.nets.simple_img_conv_pool(input=image,  # 输入图像

                                                  filter_size=3,  # 滤波器的大小

                                                  num_filters=32,  # filter 的数量。它与输出的通道相同

                                                  pool_size=2,  # 池化层大小2*2

                                                  pool_stride=2,  # 池化层步长

                                                  act='relu')  # 激活类型

    # Dropout主要作用是减少过拟合，随机让某些权重不更新

    # Dropout是一种正则化技术，通过在训练过程中阻止神经元节点间的联合适应性来减少过拟合。

    # 根据给定的丢弃概率dropout随机将一些神经元输出设置为0，其他的仍保持不变。

    drop = fluid.layers.dropout(x=conv_pool_1, dropout_prob=0.5)

    # 第二个卷积--池化层

    conv_pool_2 = fluid.nets.simple_img_conv_pool(input=drop,

                                                  filter_size=3,

                                                  num_filters=64,

                                                  pool_size=2,

                                                  pool_stride=2,

                                                  act='relu')

    # 减少过拟合，随机让某些权重不更新

    drop = fluid.layers.dropout(x=conv_pool_2, dropout_prob=0.5)

    # 第三个卷积--池化层

    conv_pool_3 = fluid.nets.simple_img_conv_pool(input=drop,

                                                  filter_size=3,

                                                  num_filters=64,

                                                  pool_size=2,

                                                  pool_stride=2,

                                                  act='relu')

    # 减少过拟合，随机让某些权重不更新

    drop = fluid.layers.dropout(x=conv_pool_3, dropout_prob=0.5)

    # 全连接层

    fc = fluid.layers.fc(input=drop, size=512, act='relu')

    # 减少过拟合，随机让某些权重不更新

    drop = fluid.layers.dropout(x=fc, dropout_prob=0.5)

    # 输出层 以softmax为激活函数的全连接输出层，输出层的大小为图像类别type_size个数

    predict = fluid.layers.fc(input=drop, size=type_size, act='softmax')

    return predict


def vgg_bn_drop(image, type_size):
    def conv_block(ipt, num_filter, groups, dropouts):
        return fluid.nets.img_conv_group(

            input=ipt,  # 具有[N，C，H，W]格式的输入图像

            pool_size=2,

            pool_stride=2,

            conv_num_filter=[num_filter] * groups,  # 过滤器个数

            conv_filter_size=3,  # 过滤器大小

            conv_act='relu',

            conv_with_batchnorm=True,  # 表示在 Conv2d Layer 之后是否使用 BatchNorm

            conv_batchnorm_drop_rate=dropouts,  # 表示 BatchNorm 之后的 Dropout Layer 的丢弃概率

            pool_type='max')  # 最大池化

    conv1 = conv_block(image, 64, 2, [0.0, 0])

    conv2 = conv_block(conv1, 128, 2, [0.0, 0])

    conv3 = conv_block(conv2, 256, 3, [0.0, 0.0, 0])

    conv4 = conv_block(conv3, 512, 3, [0.0, 0.0, 0])

    conv5 = conv_block(conv4, 512, 3, [0.0, 0.0, 0])

    drop = fluid.layers.dropout(x=conv5, dropout_prob=0.5)

    fc1 = fluid.layers.fc(input=drop, size=512, act=None)

    bn = fluid.layers.batch_norm(input=fc1, act='relu')

    drop2 = fluid.layers.dropout(x=bn, dropout_prob=0.0)

    fc2 = fluid.layers.fc(input=drop2, size=512, act=None)

    predict = fluid.layers.fc(input=fc2, size=type_size, act='softmax')

    return predict

# ##### 获取分类器，用cnn或者vgg网络进行分类type_size要和训练的类别一致 ########

predict = convolutional_neural_network(image=image, type_size=4)

# predict = vgg_bn_drop(image=image, type_size=4)

# 获取损失函数和准确率

cost = fluid.layers.cross_entropy(input=predict, label=label)

# 计算cost中所有元素的平均值

avg_cost = fluid.layers.mean(cost)

#计算准确率

accuracy = fluid.layers.accuracy(input=predict, label=label)

# 定义优化方法

optimizer = fluid.optimizer.Adam(learning_rate=0.001)    # Adam是一阶基于梯度下降的算法，基于自适应低阶矩估计该函数实现了自适应矩估计优化器

optimizer.minimize(avg_cost)                             # 取局部最优化的平均损失

print(type(accuracy))
# 使用CPU进行训练

place = fluid.CPUPlace()

# 创建一个executor

exe = fluid.Executor(place)

# 对program进行参数初始化1.网络模型2.损失函数3.优化函数

exe.run(fluid.default_startup_program())

# 定义输入数据的维度,DataFeeder 负责将reader(读取器)返回的数据转成一种特殊的数据结构，使它们可以输入到 Executor

feeder = fluid.DataFeeder(feed_list=[image, label], place=place)#定义输入数据的维度，第一个是图片数据，第二个是图片对应的标签。
# 训练的轮数

EPOCH_NUM = 5

print('开始训练...')

for pass_id in range(EPOCH_NUM):

    train_cost = 0

    for batch_id, data in enumerate(train_reader()):  # 遍历train_reader的迭代器，并为数据加上索引batch_id

        train_cost, train_acc = exe.run(

            program=fluid.default_main_program(),  # 运行主程序

            feed=feeder.feed(data),  # 喂入一个batch的数据

            fetch_list=[avg_cost, accuracy])  # fetch均方误差和准确率

        if batch_id % 10 == 0:  # 每10次batch打印一次训练、进行一次测试

            print("\nPass %d, Step %d, Cost %f, Acc %f" %

                  (pass_id, batch_id, train_cost[0], train_acc[0]))

    # 开始测试

    test_accs = []  # 测试的损失值

    test_costs = []  # 测试的准确率

    # 每训练一轮 进行一次测试

    for batch_id, data in enumerate(test_reader()):  # 遍历test_reader

        test_cost, test_acc = exe.run(program=fluid.default_main_program(),  # #运行测试主程序

                                      feed=feeder.feed(data),  # 喂入一个batch的数据

                                      fetch_list=[avg_cost, accuracy])  # fetch均方误差、准确率

        test_accs.append(test_acc[0])  # 记录每个batch的误差

        test_costs.append(test_cost[0])  # 记录每个batch的准确率

    test_cost = (sum(test_costs) / len(test_costs))  # 每轮的平均误差

    test_acc = (sum(test_accs) / len(test_accs))  # 每轮的平均准确率

    print('Test:%d, Cost:%0.5f, ACC:%0.5f' % (pass_id, test_cost, test_acc))

    # 两种方法，用两个不同的路径分别保存训练的模型

    # model_save_dir = "/home/aistudio/data/data2394/model_vgg"

    model_save_dir = "/home/aistudio/data/data2394/model_cnn"

    # 如果保存路径不存在就创建

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # 保存训练的模型，executor 把所有相关参数保存到 dirname 中

    fluid.io.save_inference_model(dirname=model_save_dir,

                                  feeded_var_names=["image"],

                                  target_vars=[predict],

                                  executor=exe)

print('训练模型保存完成！')