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
from models.unetplusplus.model import Vgg16UNetPlus
from backbones.VGG16 import vgg16Model, unet
from backbones.ResNet import ResNet
from backbones.vgg19_unet import build_vgg19_unet
from models.new_Model_pretrained_LSTM_UNet.model import vggUNetLSTM



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
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Models Parameters """
    backbones = {
        1: 'ResNet',
        2: 'VGG16',
        3: 'VGG19',
        4: 'InceptionNetV3',
        5: 'DenseNet121'
    }

    modelType = {
        1: 'unet',
        2: 'deeplabv3_plus',
        3: 'unet++',
        4: 'LSTM',
        5: 'BCDU_net_D3',
        6: 'R-cnn',
        7: 'pretrained_LSTM_UNet',
    }
    modelType = modelType[7]


    """ Directory for storing files """
    create_dir("files/" + modelType)

    """ Hyperparameters """
    batch_size = 2
    lr = 1e-4
    num_epochs = 20
    model_path = os.path.join("files/" + modelType, "model.h5")
    csv_path = os.path.join("files/" + modelType, "data.csv")

    """ Dataset """
    dataset_path = "new_data"
    train_path = os.path.join(dataset_path, "train")
    valid_path = os.path.join(dataset_path, "test")

    train_x, train_y = load_data(train_path)
    train_x, train_y = shuffling(train_x, train_y)
    valid_x, valid_y = load_data(valid_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")

    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)


    # model = deeplabv3_plus((H, W, 3))
    # input_img =Input((H, W, 3))
    # model = UNet((H, W, 3),n_filters=64)
    # model = unet(input_size=(H, W, 3))
    # model = ResNet(train_dataset, valid_dataset)
    # model = preTrainedUNet(train_dataset, valid_dataset)
    # model = Vgg16UNetPlus()
    model = vggUNetLSTM((H, W, 3))
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

    model.compile(loss=dice_loss, optimizer=Adam(lr),
                  metrics=["mse", "accuracy", dice_coef, iou, Recall(), Precision()])

    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        TensorBoard(log_dir='logs/' + modelType + '/{}'.format(time())),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False),
    ]

    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        callbacks=callbacks
    )
