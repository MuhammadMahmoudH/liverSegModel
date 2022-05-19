
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from metrics import dice_loss, dice_coef, iou
from train import create_dir
from vamethods import backbone, modelType
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay





""" Global parameters """
H = 512
W = 512

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(40)
    tf.random.set_seed(40)

    # modelType = modelType()
    # modelType = 'pretrained_LSTM_UNet'
    modelType = 'unetplusplus_vgg16_lits_dcm'

    """ Directory for storing files """
    create_dir(f"test_images/{modelType}/mask")

    """ Loading model """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model(f"files/{modelType}/liver_tumor_segmentation-0.6020.h5")

    """ Load the dataset """
    data_x = glob("test_images/image/*")
    i = 0
    for path in tqdm(data_x, total=len(data_x)):
        """ Extracting name """
        i += 1
        name = path.split("\\")[-1].split(".")[0]

        """ Reading the image """
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        h, w, _ = image.shape
        x = cv2.resize(image, (W, H))
        x = x/255.0
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=0)

        """ Prediction """
        y = model.predict(x)[0]
        y = cv2.resize(y, (w, h))
        y = np.expand_dims(y, axis=-1)

        """ Save the image """
        masked_image = image * y
        line = np.ones((h, 10, 3)) * 128
        cat_images = np.concatenate([image, line, masked_image], axis=1)
        print(i)
        cv2.imwrite(f"test_images/{modelType}/mask/{i}.png", cat_images)

        # Predict
        # y_prediction = y
        # y_prediction = np.argmax(y_prediction, axis=1)
        # y_test = np.argmax(y, axis=1)
        # # Create confusion matrix and normalizes it over predicted (columns)
        # result = confusion_matrix(y_test, y_prediction, normalize='pred')
        # disp = ConfusionMatrixDisplay(confusion_matrix=result)
        # disp.plot(cmap=plt.cm.Blues)
        # plt.show()
        # print(result)
