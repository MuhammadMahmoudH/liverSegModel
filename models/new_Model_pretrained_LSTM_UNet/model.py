import numpy as np
from keras.layers import *
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.resnet_v2 import ResNet152V2


def preTrainedModels(inputs):
    # inputs = Input(inputs)
    model_vgg = VGG16(include_top=False, input_tensor=inputs, pooling=None)
    model_vgg.trainable = False
    vgg = Model(model_vgg.input, model_vgg.get_layer('block1_conv2').output)
    # print(vgg.summary())

    # model_vgg.summary()

    print(model_vgg.get_layer('block1_conv2').input_shape)

    model_resnet = ResNet152V2(include_top=False, input_tensor=inputs, pooling=None)
    model_resnet.trainable = False
    vgg = Model(model_vgg.input, model_vgg.get_layer('block1_conv2').output)
    vgg = vgg(inputs)
    return vgg, model_vgg

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

def unet():
    # inputs
    inputs = Input(shape=(512, 512, 3))
    num_classes = 1
    activation = 'dice_loss'

    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(inputs, 64)
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
    outputs = Conv2D(num_classes, 1, padding="same", activation=activation)(u9)

    # unet model with Keras Functional API
    unet_model = Model(inputs, outputs, name="U-Net")

    return unet_model

def resUNet():
    inputs = Input(shape=(512, 512, 3))
    num_classes = 1
    activation = 'dice_loss'
    vgg_, _ = preTrainedModels(inputs)
    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(vgg_, 64)
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
    outputs = Conv2D(num_classes, 1, padding="same", activation=activation)(u9)

    # unet model with Keras Functional API
    unet_model = Model(inputs, outputs, name="U-Net")

    return unet_model

    # history = unet_model.fit(train_loader,validation_data = test_loader,epochs=5)

def vggUNet():
    input_size = (512, 512, 3)
    num_classes = 1
    # activation = 'dice_loss'
    activation = 'sigmoid'
    inputs = Input(input_size)
    _, model_vgg = preTrainedModels(inputs)
    conv1 = model_vgg.get_layer('block1_conv1')(inputs)
    conv1 = model_vgg.get_layer('block1_conv2')(conv1)
    pool1 = model_vgg.get_layer('block1_pool')(conv1)

    conv2 = model_vgg.get_layer('block2_conv1')(pool1)
    conv2 = model_vgg.get_layer('block2_conv2')(conv2)
    pool2 = model_vgg.get_layer('block2_pool')(conv2)

    conv3 = model_vgg.get_layer('block3_conv1')(pool2)
    conv3 = model_vgg.get_layer('block3_conv2')(conv3)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = model_vgg.get_layer('block3_pool')(conv3)
    conv4 = model_vgg.get_layer('block4_conv1')(pool3)
    conv4 = model_vgg.get_layer('block4_conv2')(conv4)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = model_vgg.get_layer('block4_pool')(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(num_classes, 1, activation=activation)(conv9)

    model = Model(inputs, conv10)
    return model

def vggUNetLSTM(inpuuts_tensor):
    input_size = inpuuts_tensor
    num_classes = 1
    activation = 'sigmoid'
    N = input_size[0]
    inputs = Input(input_size)
    _, model_vgg = preTrainedModels(inputs)
    conv1 = model_vgg.get_layer('block1_conv1')(inputs)
    conv1 = model_vgg.get_layer('block1_conv2')(conv1)
    pool1 = model_vgg.get_layer('block1_pool')(conv1)

    conv2 = model_vgg.get_layer('block2_conv1')(pool1)
    conv2 = model_vgg.get_layer('block2_conv2')(conv2)
    pool2 = model_vgg.get_layer('block2_pool')(conv2)

    conv3 = model_vgg.get_layer('block3_conv1')(pool2)
    conv3 = model_vgg.get_layer('block3_conv2')(conv3)
    conv3 = model_vgg.get_layer('block3_conv3')(conv3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = model_vgg.get_layer('block3_pool')(conv3)
    # D1
    conv4 = model_vgg.get_layer('block4_conv1')(pool3)
    conv4 = model_vgg.get_layer('block4_conv2')(conv4)
    conv4 = model_vgg.get_layer('block4_conv3')(conv4)
    drop4 = Dropout(0.5)(conv4)
    # pool4 = model_vgg.get_layer('block4_pool')(drop4)

    up6 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(conv4)
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
    conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv9 = Conv2D(num_classes, 1, activation='sigmoid')(conv8)

    model = Model(inputs, conv9)

    return model
    # model.compile(optimizer=Adam(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# METRICS = [
#     'accuracy',
#     "mse",
#     # dice_coef,
#     # iou,
#     Recall(),
#     Precision()
# ]
#
# model.compile(optimizer=Adam(0.0001), loss="sparse_categorical_crossentropy", metrics="accuracy")
#
# checkpoint = ModelCheckpoint('weights.hdf5', monitor='val_accuracy', save_best_only=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', patience=5)
# earlystoppingmonitor = EarlyStopping(monitor='val_accuracy', patience=5)
# history = model.fit(train_loader, validation_data=test_loader, epochs=9,
#                     callbacks=[checkpoint, earlystoppingmonitor, reduce_lr])
#
# # using the model checkpoint
# best = load_model('weights.hdf5')
#
# best.evaluate(test_loader)
#
# model.save('/content/drive/MyDrive/final.h5')
#
# best = load_model('/content/drive/MyDrive/final.h5')
#
# rand = np.random.randint(0, 5000, 1)[0]
# x = os.listdir('/content/test_data/masks')[rand]
# x1 = os.listdir('/content/test_data/images')[rand]
# y = '/content/test_data/masks/'
# y1 = '/content/test_data/images/'
# read_images = imread(y1 + x1)
# read_mask = imread(y + x)
# plt.imshow(read_mask, cmap='gray')
#
# plt.imshow(read_images, cmap='gray')
#
# best = load_model('/content/drive/MyDrive/final.h5')
# # x = 'volume-129_slice_100.tiff'
# # x1 = 'volume-129_slice_100.jpg'
# # x = 'volume-130_slice_121.tiff'
# # x1 = 'volume-130_slice_121.jpg'
# x = 'volume-121_slice_214.tiff'
# x1 = 'volume-121_slice_214.jpg'
# y = '/content/test_data/masks/'
# y1 = '/content/test_data/images/'
# read_images = imread(y1 + x1)
# read_mask = imread(y + x)
# fig = plt.figure(figsize=(10, 10))
# fig.add_subplot(1, 2, 1)
# plt.imshow(read_images)
# fig.add_subplot(1, 2, 2)
# plt.imshow(read_mask)
# plt.show()
