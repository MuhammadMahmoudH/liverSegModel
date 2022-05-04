import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten, BatchNormalization, Input
from tensorflow.keras.metrics import Recall, Precision

from metrics import dice_loss, dice_coef, iou
from tensorflow.keras.applications import ResNet50


def ResNet(train, validation):
    resnet_base_model = ResNet50(input_shape=(512,512,3), include_top=False, weights='imagenet')

    # resnet_base_model.summary()

    resnet_model = tf.keras.Sequential([
        resnet_base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation="relu"),
        BatchNormalization(),
        Dropout(0.6),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.6),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])

    resnet_model.summary()

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    METRICS = [
        'accuracy',
        'mse',
        dice_coef,
        iou,
        Recall(),
        Precision()
    ]

    resnet_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=METRICS)

#
    r = resnet_model.fit(train,
              epochs=10,
              validation_data=validation,
              # class_weight=class_weight,
              steps_per_epoch=100,
              validation_steps=25)

    return resnet_model

