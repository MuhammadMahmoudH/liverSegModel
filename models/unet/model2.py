from keras.models import Model
from keras.layers import *
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.applications.resnet_v2 import ResNet152V2

def vgg16(inputs):
    model_vgg = VGG16(include_top=False, input_shape=(512, 512, 3), pooling=None)
    model_vgg.trainable = False
    vgg = Model(model_vgg.input, model_vgg.get_layer('block1_conv2').output)
    vgg=vgg(inputs)
    return vgg

def resnet50(inputs):
    model_vgg = VGG16(include_top=False, input_shape=(512, 512, 3), pooling=None)
    model_vgg.trainable = False
    vgg = Model(model_vgg.input, model_vgg.get_layer('block1_conv2').output)
    vgg=vgg(inputs)
    return vgg


# print(vgg.summary())

def double_conv_block(x, n_filters):
    # Conv2D then ReLU activation
    x = Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    # Conv2D then ReLU activation
    x = Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)

    return x


def downsample_block(x, n_filters):
    f = double_conv_block(x, n_filters)
    p = MaxPool2D(2)(f)
    p = Dropout(0.3)(p)

    return f, p


def upsample_block(x, conv_features, n_filters):
    # upsample
    x = Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    # concatenate
    x = concatenate([x, conv_features])
    # dropout
    x = Dropout(0.3)(x)
    # Conv2D twice with ReLU activation
    x = double_conv_block(x, n_filters)

    return x


def preTrainedUNet(train_dataset, valid_dataset):
    inputs = Input(shape=(512, 512, 3))
    pretrain = vgg16(inputs)
    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(pretrain, 64)
    # 2 - downsample
    f2, p2 = downsample_block(p1, 128)
    # 3 - downsample
    f3, p3 = downsample_block(p2, 256)
    # 4 - downsample
    f4, p4 = downsample_block(p3, 512)

    # 5 - bottleneck
    bottleneck = double_conv_block(p4, 1024)

    # decoder: expanding path - upsample
    # 6 - upsample
    u6 = upsample_block(bottleneck, f4, 512)
    # 7 - upsample
    u7 = upsample_block(u6, f3, 256)
    # 8 - upsample
    u8 = upsample_block(u7, f2, 128)
    # 9 - upsample
    u9 = upsample_block(u8, f1, 64)

    # outputs
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(u9)

    unet_model = Model(inputs, outputs, name="VGG19_U-Net")

    return unet_model
