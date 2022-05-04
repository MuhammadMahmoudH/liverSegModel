from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf
from keras.layers import  Flatten, Reshape, Input, Dropout, concatenate, Lambda, Add, AveragePooling2D
from keras.layers import BatchNormalization, Convolution2D, Conv2D, MaxPooling2D, UpSampling2D, Activation, ZeroPadding2D, Conv2DTranspose
from keras.models import Model
from keras.layers import Input, concatenate, ZeroPadding2D, Conv2D, MaxPooling2D, AveragePooling2D, \
    BatchNormalization, Activation, UpSampling2D
from keras.layers.merge import add, Concatenate
import keras.backend as K

def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)

def conv_block(input_tensor, filters, strides, d_rates):
    x = Conv2D(filters[0], kernel_size=1, dilation_rate=d_rates[0])(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters[1], kernel_size=3, strides=strides, padding='same', dilation_rate=d_rates[1])(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters[2], kernel_size=1, dilation_rate=d_rates[2])(x)
    x = BatchNormalization()(x)

    shortcut = Conv2D(filters[2], kernel_size=1, strides=strides)(input_tensor)
    shortcut = BatchNormalization()(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)

    return x

def identity_block(input_tensor, filters, d_rates):
    x = Conv2D(filters[0], kernel_size=1, dilation_rate=d_rates[0])(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters[1], kernel_size=3, padding='same', dilation_rate=d_rates[1])(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters[2], kernel_size=1, dilation_rate=d_rates[2])(x)
    x = BatchNormalization()(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)

    return x

def pyramid_pooling_block(input_tensor, bin_sizes):
    concat_list = [input_tensor]
    h = input_tensor.shape[1]
    w = input_tensor.shape[2]

    for bin_size in bin_sizes:
        x = AveragePooling2D(pool_size=(h//bin_size, w//bin_size), strides=(h//bin_size, w//bin_size))(input_tensor)
        x = Conv2D(512, kernel_size=1)(x)
        x = Lambda(lambda x: tf.image.resize(x, (h, w)))(x)

        concat_list.append(x)

    return concatenate(concat_list)


def pspnet50(num_classes, input_shape, lr_init, lr_decay):
    img_input = Input(input_shape)

    x = Conv2D(64, kernel_size=3, strides=(2, 2), padding='same')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, kernel_size=3, strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, kernel_size=3, strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    x = conv_block(x, filters=[64, 64, 256], strides=(1, 1), d_rates=[1, 1, 1])
    x = identity_block(x, filters=[64, 64, 256], d_rates=[1, 1, 1])
    x = identity_block(x, filters=[64, 64, 256], d_rates=[1, 1, 1])

    x = conv_block(x, filters=[128, 128, 512], strides=(2, 2), d_rates=[1, 1, 1])
    x = identity_block(x, filters=[128, 128, 512], d_rates=[1, 1, 1])
    x = identity_block(x, filters=[128, 128, 512], d_rates=[1, 1, 1])
    x = identity_block(x, filters=[128, 128, 512], d_rates=[1, 1, 1])

    x = conv_block(x, filters=[256, 256, 1024], strides=(1, 1), d_rates=[1, 2, 1])
    x = identity_block(x, filters=[256, 256, 1024], d_rates=[1, 2, 1])
    x = identity_block(x, filters=[256, 256, 1024], d_rates=[1, 2, 1])
    x = identity_block(x, filters=[256, 256, 1024], d_rates=[1, 2, 1])
    x = identity_block(x, filters=[256, 256, 1024], d_rates=[1, 2, 1])
    x = identity_block(x, filters=[256, 256, 1024], d_rates=[1, 2, 1])

    x = conv_block(x, filters=[512, 512, 2048], strides=(1, 1), d_rates=[1, 4, 1])
    x = identity_block(x, filters=[512, 512, 2048], d_rates=[1, 4, 1])
    x = identity_block(x, filters=[512, 512, 2048], d_rates=[1, 4, 1])

    x = pyramid_pooling_block(x, [1, 2, 3, 6])

    x = Conv2D(512, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    x = Conv2D(num_classes, kernel_size=1)(x)
    x = Conv2DTranspose(num_classes, kernel_size=(16, 16), strides=(8, 8), padding='same')(x)
    x = Activation('sigmoid')(x)

    model = Model(img_input, x)
    model.compile(optimizer=Adam(learning_rate=lr_init, decay=lr_decay),loss='binary_crossentropy',metrics=[dice_coef,'accuracy'])

    return model
# pspnet50_model = pspnet50(1,(512, 512, 3),0.0001,0.001)
# history = pspnet50_model.fit(train_loader,validation_data = test_loader,epochs=epochs)
# plot_acc(history)
# plot_loss(history)