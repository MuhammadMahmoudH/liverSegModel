import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import concatenate, UpSampling2D, MaxPooling2D, Dense, Conv2D, MaxPool2D, Dropout, Flatten, BatchNormalization, Input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model




def vgg16Model(vggSummary: bool = False, input_size = (255,255,3)):
    # input_shape = Input((512, 512, 3), name="vgg_base")
    # vgg16_base_model = VGG16(input_tensor=input_shape, include_top=False, weights='imagenet')
    """ Input """
    inputs = Input(input_size)

    """ Encoder """
    vgg16_base_model = VGG16(weights="imagenet", include_top=False, input_tensor=inputs)

    if (vggSummary):
        vgg16_base_model.summary()

    # vgg16_model = tf.keras.Sequential([
    #     vgg16_base_model,
    #     GlobalAveragePooling2D(),
    #     Dense(512, activation="relu"),
    #     BatchNormalization(),
    #     Dropout(0.6),
    #     Dense(256, activation="relu"),
    #     BatchNormalization(),
    #     Dropout(0.6),
    #     Dense(128, activation="relu"),
    #     BatchNormalization(),
    #     Dropout(0.4),
    #     Dense(64, activation="relu"),
    #     BatchNormalization(),
    #     Dropout(0.3),
    #     # GlobalAveragePooling2D(),
    #     Dense(1, activation="sigmoid")
    # ])

    # if (vggUnetSummary):
        # vgg16_model.summary()
    # return vgg16_base_model

def unet(pretrained_weights=None, input_size=(256, 256, 1)):

    inputs = vgg16Model(vggSummary=True, input_size = input_size)
    # print(input_size)
    # inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

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
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

# model = Model(vgg16_model)

    return model

# if __name__ == "__main__":
    # model = unet()