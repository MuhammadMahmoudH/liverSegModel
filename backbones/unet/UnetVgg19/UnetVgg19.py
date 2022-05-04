def  UnetVgg19(inputs):
  vgg19 = VGG19(include_top=False,input_shape=(512,512,3),pooling=None)
  vgg19.trainable = False
  conv1 = vgg19.get_layer('block1_conv1')(inputs)
  conv1 = vgg19.get_layer('block1_conv2')(conv1)
  pool1 = vgg19.get_layer('block1_pool')(conv1)

  conv2 = vgg19.get_layer('block2_conv1')(pool1)
  conv2 = vgg19.get_layer('block2_conv2')(conv2)
  pool2 = vgg19.get_layer('block2_pool')(conv2)

  conv3 = vgg19.get_layer('block3_conv1')(pool2)
  conv3 = vgg19.get_layer('block3_conv2')(conv3)
  conv3 = vgg19.get_layer('block3_conv3')(conv3)
  conv3 = vgg19.get_layer('block3_conv4')(conv3)
  conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
  pool3 = vgg19.get_layer('block3_pool')(conv3)
  conv4 = vgg19.get_layer('block4_conv1')(pool3)
  conv4 = vgg19.get_layer('block4_conv2')(conv4)
  conv4 = vgg19.get_layer('block4_conv3')(conv4)
  conv4 = vgg19.get_layer('block4_conv4')(conv4)
  conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
  drop4 = Dropout(0.5)(conv4)
  pool4 = vgg19.get_layer('block4_pool')(drop4)

  conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
  conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
  drop5 = Dropout(0.5)(conv5)

  up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
  merge6 = concatenate([drop4,up6], axis = 3)
  conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
  conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

  up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
  merge7 = concatenate([conv3,up7], axis = 3)
  conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
  conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

  up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
  merge8 = concatenate([conv2,up8], axis = 3)
  conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
  conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

  up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
  merge9 = concatenate([conv1,up9], axis = 3)
  conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
  conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
  conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
  conv10 = Conv2D(3, 1, activation = 'softmax')(conv9)

  model = Model(inputs, conv10)
  return model
