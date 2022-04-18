from keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from keras.layers import *
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.resnet_v2 import ResNet152V2


def pretrained_model(pretrain=1):
    # VGG16 Pretrained model
    inputs = Input((512, 512, 3))
    model_vgg = VGG16(include_top=False, input_tensor=inputs, pooling=None)
    model_vgg.trainable = False
    vgg = Model(model_vgg.input, model_vgg.get_layer('block1_conv2').output)
    model = vgg
    # ResNet50V2
    model_ResNet = ResNet152V2(include_top=False, input_tensor=inputs, pooling=None)
    model_ResNet.trainable = False
    model_ResNet = Model(model_ResNet.input, model_vgg.get_layer('block1_conv2').output)
    # Associate Model
    if (pretrain == 2):
        model = model_ResNet
    # Print Model Summary
    print(model.summary())
    # return model
    # from keras.applications.vgg16 import VGG16
    # from keras.applications.resnet_v2 import ResNet152V2
    # inputs = Input((512, 512, 3))
    # model_vgg = VGG16(include_top=False, input_tensor=inputs, pooling=None)
    # model_vgg.trainable = False
    # vgg = Model(model_vgg.input, model_vgg.get_layer('block1_conv2').output)
    # # print(vgg.summary())
    # model_ResNet = ResNet152V2(include_top=False, input_tensor=inputs, pooling=None)
    # model_ResNet.trainable = False
    # vgg = Model(model_ResNet.input, model_vgg.get_layer('block1_conv2').output)
    # print(vgg.summary())

def convLayer2pool(inputs, num_filter, classes, pool_kernal: bool = False, drop = 0):

    conv = Conv2D(num_filter, classes, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(num_filter, classes, activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    # if drop have drop generalization
    if (drop > 0):
        conv1 = Dropout(drop)(conv1)
    # if has pooling
    # if (pool_kernal):
        # conv1 = MaxPooling2D(pool_size=pool_kernal)(conv1)
    return conv1

def droping(convL):
    dropedL = Dropout(0.5)(convL)
    return dropedL

def poolingL(conv, pool_kernal):
    pooling = MaxPooling2D(pool_size=pool_kernal)(conv1)
    return pooling

def BCDU_net_D3_1(input_size = (512, 512, 3), N = 255):

    inputs = Input(input_size)

    conv1 = convLayer2pool(inputs, 64, 3)
    pool1 = poolingL(conv1, (2,2))
    conv2 = convLayer2pool(pool1, 128, 3)
    pool2 = poolingL(conv2, (2,2))
    conv3 = convLayer2pool(pool2, 128, 3, drop = 0.5)
    pool3 = poolingL(conv3, (2,2))

    # Downsampling1
    d1 = convLayer2pool(pool3, 512, 3, drop = 0.5)
    # Downsampling2
    d2 = convLayer2pool(d1, 512, 3, drop = 0.5)
    # Downsampling3
    merge_dense = concatenate([d1, d2], axis=3)
    d3 = convLayer2pool(merge_dense, 512, 3, drop = 0.5)

    up6 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(d3)
    up6 = BatchNormalization(axis=3)(up6)
    up6 = Activation('relu')(up6)

    x1 = Reshape(target_shape=(1, np.int32(N / 4), np.int32(N / 4), 256))(conv3)
    x2 = Reshape(target_shape=(1, np.int32(N / 4), np.int32(N / 4), 256))(up6)
    merge6 = concatenate([x1, x2], axis=1)
    merge6 = ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', return_sequences=False, go_backwards=True,
                        kernel_initializer='he_normal')(merge6)

    conv6 = convLayer2pool(merge6, 256, 3)  
    up7 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(conv6)
    up7 = BatchNormalization(axis=3)(up7)
    up7 = Activation('relu')(up7)

    x1 = Reshape(target_shape=(1, np.int32(N / 2), np.int32(N / 2), 128))(conv2)
    x2 = Reshape(target_shape=(1, np.int32(N / 2), np.int32(N / 2), 128))(up7)
    merge7 = concatenate([x1, x2], axis=1)
    merge7 = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=False, go_backwards=True,
                        kernel_initializer='he_normal')(merge7)

    conv7 = convLayer2pool(merge7, 128, 3)

    up8 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(conv7)
    up8 = BatchNormalization(axis=3)(up8)
    up8 = Activation('relu')(up8)

    x1 = Reshape(target_shape=(1, N, N, 64))(conv1)
    x2 = Reshape(target_shape=(1, N, N, 64))(up8)
    merge8 = concatenate([x1, x2], axis=1)
    merge8 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=False, go_backwards=True,
                        kernel_initializer='he_normal')(merge8)

    conv8 = convLayer2pool(merge8, 64, 3)
    
    conv8 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv9 = Conv2D(3, 1, activation='sigmoid')(conv8)

    model = Model(inputs, conv9)

    return model

def LSTM(input_size = (512, 512, 3), N = 255):
    inputs = Input(input_size)

    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # D1
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4_1 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4_1 = Dropout(0.5)(conv4_1)

    up6 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(conv4_1)
    up6 = BatchNormalization(axis=3)(up6)
    up6 = Activation('relu')(up6)

    x1 = Reshape(target_shape=(1, np.int32(N / 4), np.int32(N / 4), 256))(drop3)
    x2 = Reshape(target_shape=(1, np.int32(N / 4), np.int32(N / 4), 256))(up6)
    merge6 = concatenate([x1, x2], axis=1)
    merge6 = ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', return_sequences=False, go_backwards=True,
                        kernel_initializer='he_normal')(merge6)

    conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(conv6)
    up7 = BatchNormalization(axis=3)(up7)
    up7 = Activation('relu')(up7)

    x1 = Reshape(target_shape=(1, np.int32(N / 2), np.int32(N / 2), 128))(conv2)
    x2 = Reshape(target_shape=(1, np.int32(N / 2), np.int32(N / 2), 128))(up7)
    merge7 = concatenate([x1, x2], axis=1)
    merge7 = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=False, go_backwards=True,
                        kernel_initializer='he_normal')(merge7)

    conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(conv7)
    up8 = BatchNormalization(axis=3)(up8)
    up8 = Activation('relu')(up8)

    x1 = Reshape(target_shape=(1, N, N, 64))(conv1)
    x2 = Reshape(target_shape=(1, N, N, 64))(up8)
    merge8 = concatenate([x1, x2], axis=1)
    merge8 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=False, go_backwards=True,
                        kernel_initializer='he_normal')(merge8)

    conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv9 = Conv2D(3, 1, activation='sigmoid')(conv8)

    model = Model(inputs, conv9)
    return model


if __name__ == "__main__":
    pretrained_model(1)
