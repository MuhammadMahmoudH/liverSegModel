import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.layers import Input
from metrics import dice_loss, dice_coef, iou
from time import time
from models.unet.model import UNet
from models.unet.model2 import preTrainedUNet
from models.unetplusplus.model import Vgg16UNetPlus, UNetPlus, vgg16, pretrained_model, hyperd_pretrained
from backbones.VGG16 import vgg16Model, unet
from backbones.ResNet import ResNet
from backbones.vgg19_unet import build_vgg19_unet
from models.new_Model_pretrained_LSTM_UNet.model import vggUNetLSTM
from models.new_Model_pretrained_LSTM_UNet.model import vggUNet as vggUNet
from vamethods import backbone, modelType
# from prepare_dcm.dataloader import gen_data_for_training
from models.deeplapv3.model import deeplabv3_plus
# from models.unetplusplus.model import Vgg16UNetPlus
from backbones.unet.unet.unet import unet
from backbones.unet.UnetDenseNet121.UnetDenseNet121 import UnetDenseNet121
from plotsAccLossModel import *

""" Global parameters """
H = 512
W = 512

""" Creating a directory """


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y


def load_data(path):
    x = sorted(glob(os.path.join(path, "image", "*png")))
    y = sorted(glob(os.path.join(path, "mask", "*png")))
    # print(x)
    return x, y


def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = x / 255.0
    x = x.astype(np.float32)
    return x


def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x


def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y


def tf_dataset(X, Y, batch=2):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(10)
    return dataset


if __name__ == "__main__":
    """ Seeding """
    np.random.seed(40)
    tf.random.set_seed(40)

    """ Models Parameters """

    modelType = modelType()

    # print(modelType)

    """ Directory for storing files """
    create_dir(f"files/{modelType}")

    """ Hyperparameters """
    batch_size = 2
    lr = 1e-4
    num_epochs = 200
    model_path = os.path.join(f"files/{modelType}", "liver_tumor_segmentation-{val_loss:.4f}.h5")
    csv_path = os.path.join(f"files/{modelType}", "data.csv")

    """ Dataset """
    # dataset_path = "new_data"
    dataset_path = "new_data_lits_without_aug"
    train_path = os.path.join(dataset_path, "train")
    valid_path = os.path.join(dataset_path, "test")

    train_x, train_y = load_data(train_path)
    train_x, train_y = shuffling(train_x, train_y)
    valid_x, valid_y = load_data(valid_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")

    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)

    model = deeplabv3_plus((H, W, 3))
    # input_img = Input((H, W, 3))

    # model = unet(input_img)

    # model = UnetDenseNet121()

    # model = UNet((H, W, 3),n_filters=64)
    # model = unet(input_size=(H, W, 3))
    # model = ResNet(train_dataset, valid_dataset)
    # model = preTrainedUNet(train_dataset, valid_dataset)
    # model = Vgg16UNetPlus()
    # model = UNetPlus()
    # model = pretrained_model((H, W, 3))
    # model = hyperd_pretrained()
    # print(model)
    # model = vggUNet()
    # model = vggUNetLSTM((H, W, 3))

    # len(gen_data_for_training())
    # train_gen, validation_gen = gen_data_for_training()
    # print(len(train_gen), len(validation_gen))
    # input_shape = (512, 512, 3)
    # model = build_vgg19_unet(input_shape)
    #
    # Print GPU Number
    from tensorflow.python.client import device_lib

    print(device_lib.list_local_devices())
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    model.summary()

    model.compile(loss=dice_loss, optimizer=Adam(lr),
                  metrics=["mse", "accuracy", dice_coef, iou, Recall(), Precision()])

    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        TensorBoard(log_dir="logs/" + modelType + "/{}".format(time())),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False),
    ]
    #
    # # to open tensorboard command tensorboard --logdir=logs/modelTypeName
    model._name = modelType
    #
    to_file = f"{modelType}.png"
    pltModel(model, to_file, True)
    to_file_visual = f"visual_{modelType}.png"
    #
    import visualkeras
    # # visualkeras.layered_view(model, draw_volume=False).show()  # display using your system viewer
    # # visualkeras.layered_view(model, to_file='visual_output.png', draw_volume=False)  # write to disk
    #
    # from tensorflow.python.keras.layers import Conv2DTranspose, BatchNormalization, Dense, Conv2D, Flatten, Dropout, \
    #     MaxPooling2D, ZeroPadding2D
    import visualkeras
    from tensorflow.keras import layers
    from collections import defaultdict

    colorMap = defaultdict(dict)
    colorMap[layers.InputLayer]['fill'] = '#a43858'
    colorMap[layers.Conv2D]['fill'] = '#00c28c'
    colorMap[layers.AveragePooling2D]['fill'] = '#ebbd52'
    colorMap[layers.BatchNormalization]['fill'] = '#118ab2'
    colorMap[layers.Activation]['fill'] = '#002738'
    colorMap[layers.Concatenate]['fill'] = '#a48439'
    colorMap[layers.Conv2DTranspose]['fill'] = '#89525f'
    # colorMap[layers.DepthwiseConv2D]['fill'] = 'grey'
    # colorMap[layers.GlobalAveragePooling2D]['fill'] = 'black'
    # colorMap[layers.Add]['fill'] = 'crimson'

    # from keras_sequential_ascii import keras2ascii
    #
    # keras2ascii(model)

    # print(color_map)

    visualkeras.layered_view(model, color_map=colorMap,
                             to_file=to_file_visual,
                             # draw_volume=False,
                             shade_step=20,
                             spacing=100,
                             # legend=True,
                             # type_ignore=[ZeroPadding2D, Dropout, Flatten]
                             ).show()  # write and show

    is_fitting = False

    if (is_fitting):
        model.fit(
            train_dataset,
            epochs=num_epochs,
            validation_data=valid_dataset,
            callbacks=callbacks
        )

    # model.fit(
    #     train_gen,
    #     epochs=num_epochs,
    #     validation_data=validation_gen,
    #     callbacks=callbacks
    # )
