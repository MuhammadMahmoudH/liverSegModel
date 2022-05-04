from keras.models import Model
from keras.layers import Flatten, Reshape, Input, Dropout, concatenate, Lambda, Add, AveragePooling2D
from keras.layers import BatchNormalization, Convolution2D, Conv2D, MaxPooling2D, UpSampling2D, Activation, \
    ZeroPadding2D, Conv2DTranspose


def SegNet(input_shape=(512, 512, 3), classes=1):
    img_input = Input(shape=input_shape)
    # Encoder
    x = img_input
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(256, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(512, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Decoder
    x = Conv2D(512, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(256, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(128, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(classes, (1, 1), padding="valid")(x)
    # x = Reshape((input_shape[0] * input_shape[1], classes))(x)
    x = Activation("sigmoid")(x)
    model = Model(img_input, x)

    return model

SegNet_model = SegNet(input_shape=(512, 512, 3), classes=1)
history = SegNet_model.fit(train_loader,validation_data = test_loader,epochs=epochs)
