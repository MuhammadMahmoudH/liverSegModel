
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from metrics import dice_loss, dice_coef, iou
from train import load_data
from vamethods import backbone, modelType
# from sklearn.metrics import confusion_matrix



""" Global parameters """
H = 512
W = 512

""" Creating a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_results(image, mask, y_pred, save_image_path):
    ## i - m - yp - yp*i
    line = np.ones((H, 10, 3)) * 128

    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    mask = mask * 255

    y_pred = np.expand_dims(y_pred, axis=-1)    ## (512, 512, 1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)  ## (512, 512, 3)

    masked_image = image * y_pred
    y_pred = y_pred * 255

    # cat_images = np.concatenate([image, line, mask, line, y_pred, line, masked_image], axis=1)
    cat_images = np.concatenate([image, line, y_pred, line, masked_image], axis=1)
    print(save_image_path)
    cv2.imwrite(save_image_path, cat_images)

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ select model type """
    modelType = modelType()
    # modelType = 'unetplusplus_vgg16_lits'

    """ Directory for storing files """
    create_dir(f"results/{modelType}")

    """ Loading model """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model(f"files/{modelType}/model.h5")

    """ Load the dataset """
    dataset_path = "new_data1"
    valid_path = os.path.join(dataset_path, "test")
    test_x, test_y = load_data(valid_path)
    print(f"Test: {len(test_x)} - {len(test_y)}")

    """ Evaluation and Prediction """
    SCORE = []
    i = 0
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        i+=1
        """ Extract the name """
        name = x.split("/")[-1].split(".")[0]

        """ Reading the image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        x = image/255.0
        x = np.expand_dims(x, axis=0)

        """ Reading the mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)

        """ Prediction """
        y_pred = model.predict(x)[0]
        y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = y_pred > 0.5
        y_pred = y_pred.astype(np.int32)

        """ Saving the prediction """
        save_image_path = f"results/{modelType}/{i}.png"
        save_results(image, mask, y_pred, save_image_path)

        """ Flatten the array """
        mask = mask.flatten()
        y_pred = y_pred.flatten()

        """ Calculating the metrics values """
        acc_value = accuracy_score(mask, y_pred)
        f1_value = f1_score(mask, y_pred, labels=[0, 1], average="micro")
        jac_value = jaccard_score(mask, y_pred, labels=[0, 1], average="micro")
        recall_value = recall_score(mask, y_pred, labels=[0, 1], average="micro")
        precision_value = precision_score(mask, y_pred, labels=[0, 1], average="micro")
        confusion_value = confusion_matrix(mask, y_pred, labels=[1,0], normalize='pred')
        SCORE.append([name, acc_value, f1_value, jac_value, recall_value, precision_value, confusion_value])

    """ Metrics values """
    score = [s[1:]for s in SCORE]
    score = np.mean(score, axis=0)
    print(f"Accuracy: {score[0]:0.5f}")
    print(f"F1: {score[1]:0.5f}")
    print(f"Jaccard: {score[2]:0.5f}")
    print(f"Recall: {score[3]:0.5f}")
    print(f"Precision: {score[4]:0.5f}")
    # x= list(map('{:.2f}%'.format, score[5]))
    print(score[5])
    disp = ConfusionMatrixDisplay(confusion_matrix=score[5])
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig(f"files/{modelType}/confusionMatrix.png")
    plt.show()
    df = pd.DataFrame(SCORE, columns=["Image", "Accuracy", "F1", "Jaccard", "Recall", "Precision", 'confusion'])
    df.to_csv(f"files/{modelType}/score.csv")
