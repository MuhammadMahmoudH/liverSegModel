def  UnetEfficientNetV2B0(inputs):
  efficientNetV2B0 = EfficientNetV2B0(include_top=False,input_shape=(512,512,3),pooling=None)
  efficientNetV2B0.trainable = False
  rescaling = efficientNetV2B0.get_layer('rescaling')(inputs)
  normalization = efficientNetV2B0.get_layer('normalization')(rescaling)
  stem_conv = efficientNetV2B0.get_layer('stem_conv')(normalization)
  stem_bn = efficientNetV2B0.get_layer('stem_bn')(stem_conv)
  stem_activation = efficientNetV2B0.get_layer('stem_activation')(stem_bn)
  block1a_project_conv = efficientNetV2B0.get_layer('block1a_project_conv')(stem_activation)
  block1a_project_bn = efficientNetV2B0.get_layer('block1a_project_bn')(block1a_project_conv)
  block1a_project_activation = efficientNetV2B0.get_layer('block1a_project_activation')(block1a_project_bn)
  #-------------------------------------------------------------------------
  block2a_expand_conv = efficientNetV2B0.get_layer('block2a_expand_conv')(block1a_project_activation)
  block2a_expand_bn = efficientNetV2B0.get_layer('block2a_expand_bn')(block2a_expand_conv)
  block2a_expand_activation = efficientNetV2B0.get_layer('block2a_expand_activation')(block2a_expand_bn)
  block2a_project_conv = efficientNetV2B0.get_layer('block2a_project_conv')(block2a_expand_activation)
  block2a_project_bn = efficientNetV2B0.get_layer('block2a_project_bn')(block2a_project_conv)
  block2b_expand_conv = efficientNetV2B0.get_layer('block2b_expand_conv')(block2a_project_bn)
  block2b_expand_bn = efficientNetV2B0.get_layer('block2b_expand_bn')(block2b_expand_conv)
  block2b_expand_activation = efficientNetV2B0.get_layer('block2b_expand_activation')(block2b_expand_bn)
  block2b_project_conv = efficientNetV2B0.get_layer('block2b_project_conv')(block2b_expand_activation)
  block2b_project_bn = efficientNetV2B0.get_layer('block2b_project_bn')(block2b_project_conv)
  block2b_add = efficientNetV2B0.get_layer('block2b_add')([block2b_project_bn,block2a_project_bn])
  #-------------------------------------------------------------------------
  block3a_expand_conv = efficientNetV2B0.get_layer('block3a_expand_conv')(block2b_add)
  block3a_expand_bn = efficientNetV2B0.get_layer('block3a_expand_bn')(block3a_expand_conv)
  block3a_expand_activation = efficientNetV2B0.get_layer('block3a_expand_activation')(block3a_expand_bn)
  block3a_project_conv = efficientNetV2B0.get_layer('block3a_project_conv')(block3a_expand_activation)
  block3a_project_bn = efficientNetV2B0.get_layer('block3a_project_bn')(block3a_project_conv)
  block3b_expand_conv = efficientNetV2B0.get_layer('block3b_expand_conv')(block3a_project_bn)
  block3b_expand_bn = efficientNetV2B0.get_layer('block3b_expand_bn')(block3b_expand_conv)
  block3b_expand_activation = efficientNetV2B0.get_layer('block3b_expand_activation')(block3b_expand_bn)
  block3b_project_conv = efficientNetV2B0.get_layer('block3b_project_conv')(block3b_expand_activation)
  block3b_project_bn = efficientNetV2B0.get_layer('block3b_project_bn')(block3b_project_conv)
  block3b_add = efficientNetV2B0.get_layer('block3b_add')([block3b_project_bn,block3a_project_bn])
  #-------------------------------------------------------------------------
  block4a_expand_conv = efficientNetV2B0.get_layer('block4a_expand_conv')(block3b_add)
  block4a_expand_bn = efficientNetV2B0.get_layer('block4a_expand_bn')(block4a_expand_conv)
  block4a_expand_activation = efficientNetV2B0.get_layer('block4a_expand_activation')(block4a_expand_bn)
  #-------------------------------------------------------------------------
  maxpool = MaxPool2D()(block4a_expand_activation)

  conv5 = Conv2D(576, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(maxpool)
  conv5 = Conv2D(576, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
  drop5 = Dropout(0.5)(conv5)

  up6 = Conv2D(384, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
  merge6 = concatenate([block4a_expand_activation,up6], axis = 3)
  conv6 = Conv2D(384, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
  conv6 = Conv2D(384, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

  up7 = Conv2D(192, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
  merge7 = concatenate([block2b_add,up7], axis = 3)
  conv7 = Conv2D(192, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
  conv7 = Conv2D(192, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

  up8 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
  merge8 = concatenate([block1a_project_activation,up8], axis = 3)
  conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
  conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

  up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
  merge9 = concatenate([normalization,up9], axis = 3)
  conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up9)
  conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

  model = Model(inputs, conv10)
  return model