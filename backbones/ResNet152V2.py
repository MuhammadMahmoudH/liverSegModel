from keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from keras.layers import *
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.resnet_v2 import ResNet152V2


def pretrained_model(pretrain=1):
    model_vgg = VGG16(include_top=False, input_shape=(512, 512, 3), pooling=None)
    model_vgg.trainable = False
    vgg = Model(model_vgg.input, model_vgg.get_layer('block1_conv2').output)
    model = vgg
    model_ResNet = ResNet152V2(include_top=False, input_shape=(512, 512, 3), pooling=None)
    model_ResNet.trainable = False
    model_ResNet = Model(model_ResNet.input, model_vgg.get_layer('block1_conv2').output)
    if (pretrain == 2):
        model = model_ResNet

    print(model.summary())
    return model