import keras
import math
import numpy as np
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, AveragePooling2D, GlobalAveragePooling2D, Lambda, concatenate
from keras.initializers import he_normal
from keras.layers.merge import Concatenate
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Model
from keras import optimizers, regularizers

growth_rate        = 12 
depth              = 20
compression        = 0.5

img_rows, img_cols = 32, 32
img_channels       = 3
num_classes        = 10
batch_size         = 64         # 64 or 32 or other
epochs             = 300
iterations         = 782       
weight_decay       = 1e-4

mean = [125.307, 122.95, 113.865]
std  = [62.9932, 62.0887, 66.7048]

from keras import backend as K
if('tensorflow' == K.backend()):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

def scheduler(epoch):
    if epoch < 150:
        return 0.1
    if epoch < 225:
        return 0.01
    return 0.001

def densenet(img_input,classes_num):   #输入图片和类别
    def conv(x, out_filters, k_size):  #定义卷积函数（Conv2D（默认stride=1，dialation=1））
        #conv返回一个和输入一样的大小，改变其中filter也就是channe的数量
        return Conv2D(filters=out_filters,
                      kernel_size=k_size,
                      strides=(1,1),
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(weight_decay),
                      use_bias=False)(x)

    def dense_layer(x):
        #对输入的x执行softmax全链接，输出10个
        return Dense(units=classes_num,
                     activation='softmax',
                     kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l2(weight_decay))(x)

    def bn_relu(x):
        #批归一化+激活
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = Activation('relu')(x)
        return x

    def bottleneck(x):
        #定义一个瓶颈网络
        channels = growth_rate * 4
        x = bn_relu(x)
        x = conv(x, channels, (1,1)) #x通过1x1网络channel变多
        x = bn_relu(x)
        x = conv(x, growth_rate, (3,3))#3*3网络，变回原channel
        return x

    def single(x):
        #x经过3*3卷积，输出原尺寸大小
        x = bn_relu(x)
        x = conv(x, growth_rate, (3,3))
        return x

    def transition(x, inchannels):
        outchannels = int(inchannels * compression)   #compression=0.5
        x = bn_relu(x)
        x = conv(x, outchannels, (1,1))   #x输出变成inchannel的一半
        x = AveragePooling2D((2,2), strides=(2, 2))(x)  #池化（output-2+1）/2
        return x, outchannels
    #这个densenet有问题，，没有之前所有层都连接到这层，只是前一层连接到本层
    def dense_block(x,blocks,nchannels):
        concat = x
        for i in range(blocks):
            x = bottleneck(concat)
            concat = concatenate([x,concat], axis=-1)   #在channel那个维度相加
            nchannels += growth_rate #nchannel是bottleneck叠加的次数*growth_rate
        return concat, nchannels


    nblocks = (depth - 4) // 6  #depth=100 可以计算出堆叠个数
    nchannels = growth_rate * 2  #growth_rate=12


    x = conv(img_input, nchannels, (3,3))  #
    x, nchannels = dense_block(x,nblocks,nchannels)
    x, nchannels = transition(x,nchannels)
    x, nchannels = dense_block(x,nblocks,nchannels)
    x, nchannels = transition(x,nchannels)
    x, nchannels = dense_block(x,nblocks,nchannels)
    x = bn_relu(x)
    x = GlobalAveragePooling2D()(x)
    x = dense_layer(x)
    return x


if __name__ == '__main__':

    # load data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test  = keras.utils.to_categorical(y_test, num_classes)
    x_train = x_train.astype('float32')
    x_test  = x_test.astype('float32')
    
    # - mean / std
    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
        x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]

    # build network
    img_input = Input(shape=(img_rows,img_cols,img_channels))# input给定输入的shape默认给shape添加batchs，
    #即返回（batch，）
    output    = densenet(img_input,num_classes)
    model     = Model(img_input, output)
    
    # model.load_weights('ckpt.h5')

    print(model.summary())

    # from keras.utils import plot_model
    # plot_model(model, show_shapes=True, to_file='model.png')

    # set optimizer
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # set callback
    tb_cb     = TensorBoard(log_dir='./densenet/', histogram_freq=0)
    change_lr = LearningRateScheduler(scheduler)
    ckpt      = ModelCheckpoint('./ckpt.h5', save_best_only=False, mode='auto', period=10)
    cbks      = [change_lr,tb_cb,ckpt]

    # set data augmentation
    print('Using real-time data augmentation.')
    datagen   = ImageDataGenerator(horizontal_flip=True,width_shift_range=0.125,height_shift_range=0.125,fill_mode='constant',cval=0.)

    datagen.fit(x_train)

    # start training
    model.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size), steps_per_epoch=iterations, epochs=epochs, callbacks=cbks,validation_data=(x_test, y_test))
    model.save('densenet.h5')
