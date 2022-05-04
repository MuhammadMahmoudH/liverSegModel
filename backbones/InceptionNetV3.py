from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten, BatchNormalization, Input
from tensorflow.keras.metrics import Recall, Precision

from metrics import dice_loss, dice_coef, iou
from tensorflow.keras.applications.inception_v3 import InceptionV3

def inception_base_model(inputs = (512, 512, 3)):
    inputs = Input(inputs)
    inception_base_model = InceptionV3(input_tensor=inputs,include_top=False,weights='imagenet')

    inception_base_model.summary()
    # return inception_base_model


    inception_model = tf.keras.Sequential([
        inception_base_model,
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

    return inception_model

# opt = tf.keras.optimizers.Adam(learning_rate=0.001)
# METRICS = [
#     'accuracy',
#     tf.keras.metrics.Precision(name='precision'),
#     tf.keras.metrics.Recall(name='recall')
# ]
# inception_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=METRICS)

# r = inception_model.fit(train,
#           epochs=10,
#           validation_data=validation,
#           class_weight=class_weight,
#           steps_per_epoch=100,
#           validation_steps=25)

# plt.figure(figsize=(12, 8))

# plt.subplot(2, 2, 1)
# plt.plot(r.history['loss'], label='Loss')
# plt.plot(r.history['val_loss'], label='Val_Loss')
# plt.legend()
# plt.title('Loss Evolution')

# plt.subplot(2, 2, 2)
# plt.plot(r.history['accuracy'], label='Accuracy')
# plt.plot(r.history['val_accuracy'], label='Val_Accuracy')
# plt.legend()
# plt.title('Accuracy Evolution')

# evaluation =inception_model.evaluate(test)
# print(f"Test Accuracy: {evaluation[1] * 100:.2f}%")

# evaluation = inception_model.evaluate(train)
# print(f"Train Accuracy: {evaluation[1] * 100:.2f}%")