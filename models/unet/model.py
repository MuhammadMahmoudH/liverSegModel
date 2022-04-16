from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D,\
    BatchNormalization,Activation,Dropout,concatenate,Input
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
import numpy as np
import cv2
import os
from tensorflow import metrics
from tensorflow.keras import optimizers
import tensorflow as tf
import matplotlib.pyplot as plt
import glob

# Data preprocessing
def get_train_data():
    img_directory = "../../new_data/test"
    mask_directory = "../../new_data/test"
    # print(os.path.join(img_directory, "image/*"))
    images = []
    masks = []
    for img_path in sorted(glob.glob(os.path.join(img_directory, "image/*"))):
        print(img_path)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # image = cv2.resize(image, (256, 256))
        image=image/255
        images.append(image)

    for img_path in sorted(glob.glob(os.path.join(mask_directory, "mask/*"))):
        print(img_path)
        image = cv2.imread(img_path, 0)
        # image = cv2.resize(image, (256, 256))
        masks.append(image)
    print(len(masks))
    # converting list to numpy array
    images = np.array(images, dtype="float32")
    images = images / 255.0

    masks = np.array(masks, dtype="float32")
    masks[masks > 0.5] = 1
    masks[masks <= 0.5] = 0
    # expanding the array dimension
    masks = np.expand_dims(masks, axis=-1)

    print(images.shape)
    print(masks.shape)

    # spliting the data
    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

# Unet model
#function for two conolution
def conv2d_block(input_img,filter_size):
  #first convolution
  x=Conv2D(filter_size,(3,3),kernel_initializer="he_normal",padding="same")(input_img)
  x=BatchNormalization()(x)
  x=Activation("relu")(x)

  #second convolution
  x=Conv2D(filter_size,(3,3),kernel_initializer="he_normal",padding="same")(x)
  x=BatchNormalization()(x)
  x=Activation("relu")(x)
  return x


def UNet(input_img, n_filters=16):
    input_img = Input(input_img)
    # downsamping
    c1 = conv2d_block(input_img, n_filters * 1)
    p1 = MaxPooling2D(2, 2)(c1)
    d1 = Dropout(0.2)(p1)

    c2 = conv2d_block(d1, n_filters * 2)
    p2 = MaxPooling2D(2, 2)(c2)
    d2 = Dropout(0.2)(p2)

    c3 = conv2d_block(d2, n_filters * 4)
    p3 = MaxPooling2D(2, 2)(c3)
    d3 = Dropout(0.2)(p3)

    c4 = conv2d_block(d3, n_filters * 8)
    p4 = MaxPooling2D(2, 2)(c4)
    d4 = Dropout(0.2)(p4)

    c5 = conv2d_block(d4, n_filters * 16)
    # upsampling

    t6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding="same")(c5)
    j6 = concatenate([t6, c4])
    d6 = Dropout(0.2)(j6)
    c6 = conv2d_block(d6, n_filters * 8)

    t7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding="same")(c6)
    j7 = concatenate([t7, c3])
    d7 = Dropout(0.2)(j7)
    c7 = conv2d_block(d7, n_filters * 4)

    t8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding="same")(c7)
    j8 = concatenate([t8, c2])
    d8 = Dropout(0.2)(j8)
    c8 = conv2d_block(d8, n_filters * 2)

    t9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding="same")(c8)
    j9 = concatenate([t9, c1])
    d9 = Dropout(0.2)(j9)
    c9 = conv2d_block(d9, n_filters * 1)

    outputs = Conv2D(1, 1, activation="sigmoid")(c9)
    model = Model(inputs=input_img, outputs=outputs)
    model.summary()
    return model

# Creating function for dice coef and dice loss
def dice_coef(y_true,y_pred,smooth=1):
  intersection = tf.keras.backend.sum(y_true * y_pred, axis=[1,2,3])
  union = tf.keras.backend.sum(y_true, axis=[1,2,3]) + tf.keras.backend.sum(y_pred, axis=[1,2,3])
  return tf.keras.backend.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_coef_loss(y_true,y_pred):
  loss=dice_coef(y_true,y_pred)
  return -loss


#function to create the model
def get_model():
  input_img = Input((512,512, 3), name='img')
  model=UNet(input_img,n_filters=64)
  model.compile(optimizer=optimizers.Adam(1e-4), loss=dice_coef_loss, metrics=[
                                                                     dice_coef,
                                                                     metrics.Recall(),
                                                                     metrics.Precision()
                                                                     ])
  model.summary()
  return model


# if __name__ == "__main__":
#     # get train and validation data
#     X_train, X_val, y_train, y_val = get_train_data()
#
#     # print(len(X_train))
#
#     # get the model
#     model = get_model()
#     #
#     callback = [
#         EarlyStopping(monitor="val_loss", patience=10),
#         ModelCheckpoint('model.h5', monitor="val_loss", save_best_only=True, verbose=1),
#         ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, min_lr=0.0000001, verbose=1)
#         # CSVLogger()
#
#     ]
#     # fit the model
#     model.fit(X_train, y_train, batch_size=8, epochs=100, callbacks=callback, validation_data=(X_val, y_val))
# #
# execute this to run the program


