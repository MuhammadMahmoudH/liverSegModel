from tensorflow.keras.layers import concatenate, UpSampling2D, MaxPooling2D, Dense, Conv2D, MaxPool2D, Dropout, Flatten, BatchNormalization, Input
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D, UpSampling2D, Reshape, Dense
from tensorflow.keras.models import Model

from keras.applications.densenet import DenseNet121

input_size = (512,512,3)
inputs = Input(input_size)
def  UnetDenseNet121():
  denseNet121 = DenseNet121(include_top=False,input_shape=(512,512,3),pooling=None)
  denseNet121.trainable = False
  zero_padding2d = denseNet121.get_layer('zero_padding2d')(inputs)
  conv1conv = denseNet121.get_layer('conv1/conv')(zero_padding2d)#concate with 256
  conv1bn = denseNet121.get_layer('conv1/bn')(conv1conv)
  conv1relu = denseNet121.get_layer('conv1/relu')(conv1bn)
  zero_padding2d_1 = denseNet121.get_layer('zero_padding2d_1')(conv1relu)
  pool1 = denseNet121.get_layer('pool1')(zero_padding2d_1)
  #---------------------------------------------------------------
  conv2_block1_0_bn = denseNet121.get_layer('conv2_block1_0_bn')(pool1)#concate with 128
  conv2_block1_0_relu = denseNet121.get_layer('conv2_block1_0_relu')(conv2_block1_0_bn)
  conv2_block1_1_conv = denseNet121.get_layer('conv2_block1_1_conv')(conv2_block1_0_relu)
  conv2_block1_1_bn = denseNet121.get_layer('conv2_block1_1_bn')(conv2_block1_1_conv)
  conv2_block1_1_relu = denseNet121.get_layer('conv2_block1_1_relu')(conv2_block1_1_bn)
  conv2_block1_2_conv = denseNet121.get_layer('conv2_block1_2_conv')(conv2_block1_1_relu)
  conv2_block1_concat = denseNet121.get_layer('conv2_block1_concat')([pool1,conv2_block1_2_conv])

  conv2_block2_0_bn = denseNet121.get_layer('conv2_block2_0_bn')(conv2_block1_concat)
  conv2_block2_0_relu = denseNet121.get_layer('conv2_block2_0_relu')(conv2_block2_0_bn)
  conv2_block2_1_conv = denseNet121.get_layer('conv2_block2_1_conv')(conv2_block2_0_relu)
  conv2_block2_1_bn = denseNet121.get_layer('conv2_block2_1_bn')(conv2_block2_1_conv)
  conv2_block2_1_relu = denseNet121.get_layer('conv2_block2_1_relu')(conv2_block2_1_bn)
  conv2_block2_2_conv = denseNet121.get_layer('conv2_block2_2_conv')(conv2_block2_1_relu)
  conv2_block2_concat = denseNet121.get_layer('conv2_block2_concat')([conv2_block1_concat,conv2_block2_2_conv])

  conv2_block3_0_bn = denseNet121.get_layer('conv2_block3_0_bn')(conv2_block2_concat)
  conv2_block3_0_relu = denseNet121.get_layer('conv2_block3_0_relu')(conv2_block3_0_bn)
  conv2_block3_1_conv = denseNet121.get_layer('conv2_block3_1_conv')(conv2_block3_0_relu)
  conv2_block3_1_bn = denseNet121.get_layer('conv2_block3_1_bn')(conv2_block3_1_conv)
  conv2_block3_1_relu = denseNet121.get_layer('conv2_block3_1_relu')(conv2_block3_1_bn)
  conv2_block3_2_conv = denseNet121.get_layer('conv2_block3_2_conv')(conv2_block3_1_relu)
  conv2_block3_concat = denseNet121.get_layer('conv2_block3_concat')([conv2_block2_concat,conv2_block3_2_conv])

  conv2_block4_0_bn = denseNet121.get_layer('conv2_block4_0_bn')(conv2_block3_concat)
  conv2_block4_0_relu = denseNet121.get_layer('conv2_block4_0_relu')(conv2_block4_0_bn)
  conv2_block4_1_conv = denseNet121.get_layer('conv2_block4_1_conv')(conv2_block4_0_relu)
  conv2_block4_1_bn = denseNet121.get_layer('conv2_block4_1_bn')(conv2_block4_1_conv)
  conv2_block4_1_relu = denseNet121.get_layer('conv2_block4_1_relu')(conv2_block4_1_bn)
  conv2_block4_2_conv = denseNet121.get_layer('conv2_block4_2_conv')(conv2_block4_1_relu)
  conv2_block4_concat = denseNet121.get_layer('conv2_block4_concat')([conv2_block3_concat,conv2_block4_2_conv])

  conv2_block5_0_bn = denseNet121.get_layer('conv2_block5_0_bn')(conv2_block4_concat)
  conv2_block5_0_relu = denseNet121.get_layer('conv2_block5_0_relu')(conv2_block5_0_bn)
  conv2_block5_1_conv = denseNet121.get_layer('conv2_block5_1_conv')(conv2_block5_0_relu)
  conv2_block5_1_bn = denseNet121.get_layer('conv2_block5_1_bn')(conv2_block5_1_conv)
  conv2_block5_1_relu = denseNet121.get_layer('conv2_block5_1_relu')(conv2_block5_1_bn)
  conv2_block5_2_conv = denseNet121.get_layer('conv2_block5_2_conv')(conv2_block5_1_relu)
  conv2_block5_concat = denseNet121.get_layer('conv2_block5_concat')([conv2_block4_concat,conv2_block5_2_conv])

  conv2_block6_0_bn = denseNet121.get_layer('conv2_block6_0_bn')(conv2_block5_concat)
  conv2_block6_0_relu = denseNet121.get_layer('conv2_block6_0_relu')(conv2_block6_0_bn)
  conv2_block6_1_conv = denseNet121.get_layer('conv2_block6_1_conv')(conv2_block6_0_relu)
  conv2_block6_1_bn = denseNet121.get_layer('conv2_block6_1_bn')(conv2_block6_1_conv)
  conv2_block6_1_relu = denseNet121.get_layer('conv2_block6_1_relu')(conv2_block6_1_bn)
  conv2_block6_2_conv = denseNet121.get_layer('conv2_block6_2_conv')(conv2_block6_1_relu)
  conv2_block6_concat = denseNet121.get_layer('conv2_block6_concat')([conv2_block5_concat,conv2_block6_2_conv])

  pool2_bn = denseNet121.get_layer('pool2_bn')(conv2_block6_concat)
  pool2_relu = denseNet121.get_layer('pool2_relu')(pool2_bn)
  pool2_conv = denseNet121.get_layer('pool2_conv')(pool2_relu)
  pool2_pool = denseNet121.get_layer('pool2_pool')(pool2_conv)
  #-------------------------------------------------------------------------
  conv3_block1_0_bn = denseNet121.get_layer('conv3_block1_0_bn')(pool2_pool)#concate with 128
  conv3_block1_0_relu = denseNet121.get_layer('conv3_block1_0_relu')(conv3_block1_0_bn)
  conv3_block1_1_conv = denseNet121.get_layer('conv3_block1_1_conv')(conv3_block1_0_relu)
  conv3_block1_1_bn = denseNet121.get_layer('conv3_block1_1_bn')(conv3_block1_1_conv)
  conv3_block1_1_relu = denseNet121.get_layer('conv3_block1_1_relu')(conv3_block1_1_bn)
  conv3_block1_2_conv = denseNet121.get_layer('conv3_block1_2_conv')(conv3_block1_1_relu)
  conv3_block1_concat = denseNet121.get_layer('conv3_block1_concat')([pool2_pool,conv3_block1_2_conv])

  conv3_block2_0_bn = denseNet121.get_layer('conv3_block2_0_bn')(conv3_block1_concat)
  conv3_block2_0_relu = denseNet121.get_layer('conv3_block2_0_relu')(conv3_block2_0_bn)
  conv3_block2_1_conv = denseNet121.get_layer('conv3_block2_1_conv')(conv3_block2_0_relu)
  conv3_block2_1_bn = denseNet121.get_layer('conv3_block2_1_bn')(conv3_block2_1_conv)
  conv3_block2_1_relu = denseNet121.get_layer('conv3_block2_1_relu')(conv3_block2_1_bn)
  conv3_block2_2_conv = denseNet121.get_layer('conv3_block2_2_conv')(conv3_block2_1_relu)
  conv3_block2_concat = denseNet121.get_layer('conv3_block2_concat')([conv3_block1_concat,conv3_block2_2_conv])

  conv3_block3_0_bn = denseNet121.get_layer('conv3_block3_0_bn')(conv3_block2_concat)
  conv3_block3_0_relu = denseNet121.get_layer('conv3_block3_0_relu')(conv3_block3_0_bn)
  conv3_block3_1_conv = denseNet121.get_layer('conv3_block3_1_conv')(conv3_block3_0_relu)
  conv3_block3_1_bn = denseNet121.get_layer('conv3_block3_1_bn')(conv3_block3_1_conv)
  conv3_block3_1_relu = denseNet121.get_layer('conv3_block3_1_relu')(conv3_block3_1_bn)
  conv3_block3_2_conv = denseNet121.get_layer('conv3_block3_2_conv')(conv3_block3_1_relu)
  conv3_block3_concat = denseNet121.get_layer('conv3_block3_concat')([conv3_block2_concat,conv3_block3_2_conv])

  conv3_block4_0_bn = denseNet121.get_layer('conv3_block4_0_bn')(conv3_block3_concat)
  conv3_block4_0_relu = denseNet121.get_layer('conv3_block4_0_relu')(conv3_block4_0_bn)
  conv3_block4_1_conv = denseNet121.get_layer('conv3_block4_1_conv')(conv3_block4_0_relu)
  conv3_block4_1_bn = denseNet121.get_layer('conv3_block4_1_bn')(conv3_block4_1_conv)
  conv3_block4_1_relu = denseNet121.get_layer('conv3_block4_1_relu')(conv3_block4_1_bn)
  conv3_block4_2_conv = denseNet121.get_layer('conv3_block4_2_conv')(conv3_block4_1_relu)
  conv3_block4_concat = denseNet121.get_layer('conv3_block4_concat')([conv3_block3_concat,conv3_block4_2_conv])

  conv3_block5_0_bn = denseNet121.get_layer('conv3_block5_0_bn')(conv3_block4_concat)
  conv3_block5_0_relu = denseNet121.get_layer('conv3_block5_0_relu')(conv3_block5_0_bn)
  conv3_block5_1_conv = denseNet121.get_layer('conv3_block5_1_conv')(conv3_block5_0_relu)
  conv3_block5_1_bn = denseNet121.get_layer('conv3_block5_1_bn')(conv3_block5_1_conv)
  conv3_block5_1_relu = denseNet121.get_layer('conv3_block5_1_relu')(conv3_block5_1_bn)
  conv3_block5_2_conv = denseNet121.get_layer('conv3_block5_2_conv')(conv3_block5_1_relu)
  conv3_block5_concat = denseNet121.get_layer('conv3_block5_concat')([conv3_block4_concat,conv3_block5_2_conv])

  conv3_block6_0_bn = denseNet121.get_layer('conv3_block6_0_bn')(conv3_block5_concat)
  conv3_block6_0_relu = denseNet121.get_layer('conv3_block6_0_relu')(conv3_block6_0_bn)
  conv3_block6_1_conv = denseNet121.get_layer('conv3_block6_1_conv')(conv3_block6_0_relu)
  conv3_block6_1_bn = denseNet121.get_layer('conv3_block6_1_bn')(conv3_block6_1_conv)
  conv3_block6_1_relu = denseNet121.get_layer('conv3_block6_1_relu')(conv3_block6_1_bn)
  conv3_block6_2_conv = denseNet121.get_layer('conv3_block6_2_conv')(conv3_block6_1_relu)
  conv3_block6_concat = denseNet121.get_layer('conv3_block6_concat')([conv3_block5_concat,conv3_block6_2_conv])

  conv3_block7_0_bn = denseNet121.get_layer('conv3_block7_0_bn')(conv3_block6_concat)
  conv3_block7_0_relu = denseNet121.get_layer('conv3_block7_0_relu')(conv3_block7_0_bn)
  conv3_block7_1_conv = denseNet121.get_layer('conv3_block7_1_conv')(conv3_block7_0_relu)
  conv3_block7_1_bn = denseNet121.get_layer('conv3_block7_1_bn')(conv3_block7_1_conv)
  conv3_block7_1_relu = denseNet121.get_layer('conv3_block7_1_relu')(conv3_block7_1_bn)
  conv3_block7_2_conv = denseNet121.get_layer('conv3_block7_2_conv')(conv3_block7_1_relu)
  conv3_block7_concat = denseNet121.get_layer('conv3_block7_concat')([conv3_block6_concat,conv3_block7_2_conv])

  conv3_block8_0_bn = denseNet121.get_layer('conv3_block8_0_bn')(conv3_block7_concat)
  conv3_block8_0_relu = denseNet121.get_layer('conv3_block8_0_relu')(conv3_block8_0_bn)
  conv3_block8_1_conv = denseNet121.get_layer('conv3_block8_1_conv')(conv3_block8_0_relu)
  conv3_block8_1_bn = denseNet121.get_layer('conv3_block8_1_bn')(conv3_block8_1_conv)
  conv3_block8_1_relu = denseNet121.get_layer('conv3_block8_1_relu')(conv3_block8_1_bn)
  conv3_block8_2_conv = denseNet121.get_layer('conv3_block8_2_conv')(conv3_block8_1_relu)
  conv3_block8_concat = denseNet121.get_layer('conv3_block8_concat')([conv3_block7_concat,conv3_block8_2_conv])

  conv3_block9_0_bn = denseNet121.get_layer('conv3_block9_0_bn')(conv3_block8_concat)
  conv3_block9_0_relu = denseNet121.get_layer('conv3_block9_0_relu')(conv3_block9_0_bn)
  conv3_block9_1_conv = denseNet121.get_layer('conv3_block9_1_conv')(conv3_block9_0_relu)
  conv3_block9_1_bn = denseNet121.get_layer('conv3_block9_1_bn')(conv3_block9_1_conv)
  conv3_block9_1_relu = denseNet121.get_layer('conv3_block9_1_relu')(conv3_block9_1_bn)
  conv3_block9_2_conv = denseNet121.get_layer('conv3_block9_2_conv')(conv3_block9_1_relu)
  conv3_block9_concat = denseNet121.get_layer('conv3_block9_concat')([conv3_block8_concat,conv3_block9_2_conv])

  conv3_block10_0_bn = denseNet121.get_layer('conv3_block10_0_bn')(conv3_block9_concat)
  conv3_block10_0_relu = denseNet121.get_layer('conv3_block10_0_relu')(conv3_block10_0_bn)
  conv3_block10_1_conv = denseNet121.get_layer('conv3_block10_1_conv')(conv3_block10_0_relu)
  conv3_block10_1_bn = denseNet121.get_layer('conv3_block10_1_bn')(conv3_block10_1_conv)
  conv3_block10_1_relu = denseNet121.get_layer('conv3_block10_1_relu')(conv3_block10_1_bn)
  conv3_block10_2_conv = denseNet121.get_layer('conv3_block10_2_conv')(conv3_block10_1_relu)
  conv3_block10_concat = denseNet121.get_layer('conv3_block10_concat')([conv3_block9_concat,conv3_block10_2_conv])

  conv3_block11_0_bn = denseNet121.get_layer('conv3_block11_0_bn')(conv3_block10_concat)
  conv3_block11_0_relu = denseNet121.get_layer('conv3_block11_0_relu')(conv3_block11_0_bn)
  conv3_block11_1_conv = denseNet121.get_layer('conv3_block11_1_conv')(conv3_block11_0_relu)
  conv3_block11_1_bn = denseNet121.get_layer('conv3_block11_1_bn')(conv3_block11_1_conv)
  conv3_block11_1_relu = denseNet121.get_layer('conv3_block11_1_relu')(conv3_block11_1_bn)
  conv3_block11_2_conv = denseNet121.get_layer('conv3_block11_2_conv')(conv3_block11_1_relu)
  conv3_block11_concat = denseNet121.get_layer('conv3_block11_concat')([conv3_block10_concat,conv3_block11_2_conv])

  conv3_block12_0_bn = denseNet121.get_layer('conv3_block12_0_bn')(conv3_block11_concat)
  conv3_block12_0_relu = denseNet121.get_layer('conv3_block12_0_relu')(conv3_block12_0_bn)
  conv3_block12_1_conv = denseNet121.get_layer('conv3_block12_1_conv')(conv3_block12_0_relu)
  conv3_block12_1_bn = denseNet121.get_layer('conv3_block12_1_bn')(conv3_block12_1_conv)
  conv3_block12_1_relu = denseNet121.get_layer('conv3_block12_1_relu')(conv3_block12_1_bn)
  conv3_block12_2_conv = denseNet121.get_layer('conv3_block12_2_conv')(conv3_block12_1_relu)
  conv3_block12_concat = denseNet121.get_layer('conv3_block12_concat')([conv3_block11_concat,conv3_block12_2_conv])

  pool3_bn = denseNet121.get_layer('pool3_bn')(conv3_block12_concat)
  pool3_relu = denseNet121.get_layer('pool3_relu')(pool3_bn)
  pool3_conv = denseNet121.get_layer('pool3_conv')(pool3_relu)
  pool3_pool = denseNet121.get_layer('pool3_pool')(pool3_conv)
  #-------------------------------------------------------------------------

  conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3_pool)
  conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
  drop5 = Dropout(0.5)(conv5)

  up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
  merge6 = concatenate([pool3_conv,up6], axis = 3)
  conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
  conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

  up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
  merge7 = concatenate([pool2_conv,up7], axis = 3)
  conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
  conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

  up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
  merge8 = concatenate([conv1relu,up8], axis = 3)
  conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
  conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

  up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
  # merge9 = concatenate([conv1,up9], axis = 3)
  conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up9)
  conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

  model = Model(inputs, conv10)
  return model