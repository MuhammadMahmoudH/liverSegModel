# import Libraries

import os
import requests
import io
import random
import glob
import zipfile
import shutil
import csv
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL.Image import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Conv2DTranspose, concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

# Delete inputs folder and masks
# shutil.rmtree('dataset/livertumor_masks')
# shutil.rmtree('dataset/inputs')
# shutil.rmtree('dataset')
# !mkdir dataset

# Data variables defined

dataset_folder = 'dataset'
inputs_folder = os.path.join(dataset_folder, 'inputs')
livertumor_masks_folder = os.path.join(dataset_folder, 'livertumor_masks')


def process_ircad_database(database_id, inputs_folder, livertumor_masks_folder, overwrite=False):
    # get dataset folder name
    # database_name = database_url.split('/')[-1].split(".zip")[0]
    database_name = database_id

    # Download the dataset if doesn't exist and extract to appropriate folder
    database_folder = os.path.join(dataset_folder, database_name)
    print(database_folder)

    if not os.path.exists(database_folder):
        os.makedirs(database_folder)

    # Folders form input and livertumor masks dicom
    # label_dicom_folder = os.path.join(database_folder, 'PATIENT_DICOM')
    label_dicom_folder = os.path.join(database_folder , 'MASKS_DICOM', 'liver')
    livertumor_mask_dicom_folder = os.path.join(database_folder, 'MASKS_DICOM', 'livertumor')

    # Select all input and mask dicom files
    label_dicom_paths = glob.glob(os.path.join(label_dicom_folder, '*'))
    livertumor_mask_dicom_paths = glob.glob(os.path.join(livertumor_mask_dicom_folder, '*'))

    n_label = len(label_dicom_paths)
    n_livertumor_masks = len(livertumor_mask_dicom_paths)
    print("Label files: {0}".format(n_label))
    print("livertumor mask files: {0}".format(n_livertumor_masks))
    print(50 * '-')

    # Check if we have files and both input and mask folder have the same number of files
    if (len(label_dicom_paths) != len(livertumor_mask_dicom_paths) and len(label_dicom_paths) > 0
            and len(livertumor_mask_dicom_paths) > 0):
        return False
    print(len(label_dicom_paths) == len(livertumor_mask_dicom_paths) and len(label_dicom_paths) > 0 and len(
        livertumor_mask_dicom_paths) > 0)

    # print(len(label_dicom_paths), len(livertumor_mask_dicom_paths) , len(label_dicom_paths) , len(livertumor_mask_dicom_paths))

    # Move files to appropriate folder
    current_index = 1

    # Update the current_index with the last file index on the folder
    if (overwrite == False):
        # get dataset inputs dicom
        input_dicom_paths = glob.glob(os.path.join(inputs_folder, '*.dcm'))

        if (len(input_dicom_paths) == 0):
            current_index = 1
        else:
            # print([int(x.split('/')[-1].split(".dcm")[0][5:10]) for x in input_dicom_paths])

            # Grab the file names and extract just the number so input00012.dicom -> 12
            indices = [int(x.split('/')[-1].split(".dcm")[0][5:10]) for x in input_dicom_paths]
            indices.sort()
            last_index = indices[-1]

            current_index = last_index + 1

    print("Moving files starting with index: {0}".format(current_index))
    print(50 * '-')

    for label_dicom_path, livertumor_mask_dicom_path in zip(label_dicom_paths, livertumor_mask_dicom_paths):
        # assert(label_dicom_path.split("\\")[-1] == livertumor_mask_dicom_path.split("\\")[-1])
        new_input_dicom_path = os.path.join(inputs_folder, 'input' + str(current_index).zfill(5) + ".dcm")
        new_livertumor_mask_dicom_path = os.path.join(livertumor_masks_folder,
                                                      'mask' + str(current_index).zfill(5) + ".dcm")

        shutil.copy(label_dicom_path, new_input_dicom_path)
        shutil.copy(livertumor_mask_dicom_path, new_livertumor_mask_dicom_path)

        current_index += 1

    print("Moving process complete")
    print(50 * '-')

    # Delete temporaty database folder after we exctracted files we wantes
    shutil.rmtree(database_folder)

    print("Database folder excluded")
    print(50 * '-')


def dir_create(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def data_preprocessing_creation():
    # Create Dataset folder
    dir_create(dataset_folder)

    # Create Input DICOM folder
    dir_create(inputs_folder)

    # Create liver tumor Masks folder
    dir_create(livertumor_masks_folder)

    # dataset folders ids and datasets dicom folders name
    database_ids = [
        "3Dircadb1.1",
        "3Dircadb1.2",
        "3Dircadb1.3",
        "3Dircadb1.4",
        "3Dircadb1.5",
        "3Dircadb1.6",
        "3Dircadb1.7",
        "3Dircadb1.8",
        "3Dircadb1.9",
        "3Dircadb1.10",
        "3Dircadb1.11",
        "3Dircadb1.12",
        "3Dircadb1.13",
        "3Dircadb1.14",
        "3Dircadb1.15",
        "3Dircadb1.16",
        "3Dircadb1.17",
        "3Dircadb1.18",
        "3Dircadb1.19",
        "3Dircadb1.20",
    ]

    for database_id in database_ids:
        process_ircad_database(database_id, inputs_folder, livertumor_masks_folder)


# Load data
############################################################################################

# Load a DICOM file and convert it to [0-255] gray scale
# For IRC  Dataset minimum value = -2048 and maximum value = 3247
def load_dicom(dicom_path):
    # Load dicom file
    dicom = pydicom.read_file(dicom_path)

    # Select only pixel image data
    pixel_data = dicom.pixel_array

    # Normalize data using slope and interceptallow to transform the pixel values to Hounsfield Units
    intercept = dicom.RescaleIntercept
    slope = dicom.RescaleSlope

    pixel_data = pixel_data * slope + intercept

    image = np.zeros(pixel_data.shape)

    # Normalize image from 16 bits int to 8 bits unsigned int (JPG)
    image = (pixel_data + 65535)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    image[image > 255] = 255
    image[image < 1] = 0

    return image.astype('uint8')


# Data binding
def dBinding():
    # DICOM image files
    input_files = glob.glob(os.path.join(inputs_folder, '*.dcm'))
    livertumor_mask_files = glob.glob(os.path.join(livertumor_masks_folder, '*.dcm'))

    # Sort names to avoid connecting wrong numbered files
    input_files.sort()
    livertumor_mask_files.sort()

    assert (len(input_files) == len(livertumor_mask_files))

    # Our dataset is created using the tuple (input image file, ground truth image file)
    data = []
    for input_image, livertumor_mask_image in zip(input_files, livertumor_mask_files):
        data.append((input_image, livertumor_mask_image))

        # Plot 5 data samples from dataset

        n_samples = 5

        # print(input_image, livertumor_mask_image)

        # for i in range(n_samples):
        #     # define the size of images
        #     f, (ax1, ax2) = plt.subplots(1, 2)
        #     f.set_figwidth(10)
        #
        #     # randomly select a sample
        #     idx = np.random.randint(0, len(data))
        #     input_image_path, livertumor_mask_image_path = data[idx]
        #
        #     input_image = load_dicom(input_image_path)
        #     livertumor_mask_image = load_dicom(livertumor_mask_image_path)
        #
        #     ax1.imshow(input_image, cmap='gray')
        #     ax1.set_title('Input Image # {0}'.format(idx))
        #
        #     ax2.imshow(livertumor_mask_image, cmap='gray')
        #     ax2.set_title('liver tumor Mask Image # {0}'.format(idx))

    return data


############################################################################################
# Data Augmentation
############################################################################################

#############################
# Resize Image
#############################
class Resize(object):

    def __init__(self, input_size, output_size):

        assert isinstance(input_size, (int, tuple))
        assert isinstance(output_size, (int, tuple))

        self.input_size = input_size
        self.output_size = output_size

    def __call__(self, input_image, mask_image):

        h_in, w_in = input_image.shape[:2]
        h_out, w_out = mask_image.shape[:2]

        if isinstance(self.input_size, int):
            if h_in > w_in:
                new_h_in, new_w_in = self.input_size * h_in / w_in, self.input_size
            else:
                new_h_in, new_w_in = self.input_size, self.input_size * w_in / h_in
        else:
            new_h_in, new_w_in = self.input_size

        if isinstance(self.output_size, int):
            if h_out > w_out:
                new_h_out, new_w_out = self.output_size * h_out / w_out, self.output_size
            else:
                new_h_out, new_w_out = self.output_size, self.output_size * w_out / h_out
        else:
            new_h_out, new_w_out = self.output_size

        new_h_in, new_w_in = int(new_h_in), int(new_w_in)
        new_h_out, new_w_out = int(new_h_out), int(new_w_out)

        input_image = cv2.resize(input_image, (new_w_in, new_h_in))
        mask_image = cv2.resize(mask_image, (new_w_out, new_h_out))

        return input_image, mask_image


#############################
# Translate Image
#############################
class RandomTranslation(object):

    def __init__(self, ratio=0.4, background_color=(0), prob=0.5):
        self.background_color = background_color
        self.ratio = ratio
        self.prob = prob

    def __call__(self, input_image, mask_image):
        if random.uniform(0, 1) <= self.prob:
            img_h, img_w = input_image.shape

            x = int(np.random.uniform(-self.ratio, self.ratio) * img_w)
            y = int(np.random.uniform(-self.ratio, self.ratio) * img_h)

            M = np.float32([[1, 0, x],
                            [0, 1, y]])

            input_image_translated = cv2.warpAffine(input_image, M, (img_w, img_h), borderValue=self.background_color)
            imask_image_translated = cv2.warpAffine(mask_image, M, (img_w, img_h), borderValue=self.background_color)

            return input_image_translated, imask_image_translated

        return input_image, mask_image


#############################
# Scale Image
#############################
class RandomScale(object):

    def __init__(self, lower=0.4, upper=1.4, background_color=(0), prob=0.5):

        self.background_color = background_color
        self.lower = lower
        self.upper = upper
        self.prob = prob

    def __call__(self, input_image, mask_image):

        if random.uniform(0, 1) <= self.prob:

            input_img_h, input_img_w = input_image.shape
            mask_img_h, mask_img_w = mask_image.shape

            # Create canvas with random ration between lower and upper
            ratio = random.uniform(self.lower, self.upper)

            scale_x = ratio
            scale_y = ratio

            # Scale the image
            scaled_input_image = cv2.resize(input_image.astype('float32'), (0, 0), fx=scale_x, fy=scale_y)
            scaled_mask_image = cv2.resize(mask_image.astype('float32'), (0, 0), fx=scale_x, fy=scale_y)

            top = 0
            left = 0

            if ratio < 1:

                # Input image
                background = np.zeros((input_img_h, input_img_w), dtype=np.uint8)

                background[:, :] = self.background_color

                y_lim = int(min(scale_x, 1) * input_img_h)
                x_lim = int(min(scale_y, 1) * input_img_w)

                top = (input_img_h - y_lim) // 2
                left = (input_img_w - x_lim) // 2

                background[top:y_lim + top, left:x_lim + left] = scaled_input_image[:y_lim, :x_lim]

                scaled_input_image = background

                # Mask image
                background = np.zeros((mask_img_h, mask_img_w), dtype=np.uint8)

                background[:, :] = self.background_color

                y_lim = int(min(scale_x, 1) * mask_img_h)
                x_lim = int(min(scale_y, 1) * mask_img_w)

                top = (mask_img_h - y_lim) // 2
                left = (mask_img_w - x_lim) // 2

                background[top:y_lim + top, left:x_lim + left] = scaled_mask_image[:y_lim, :x_lim]

                scaled_mask_image = background

            else:

                top = (scaled_input_image.shape[0] - input_img_h) // 2
                left = (scaled_input_image.shape[1] - input_img_w) // 2

                scaled_input_image = scaled_input_image[top:input_img_h + top, left:input_img_w + left]

                top = (scaled_mask_image.shape[0] - mask_img_h) // 2
                left = (scaled_mask_image.shape[1] - mask_img_w) // 2

                scaled_mask_image = scaled_mask_image[top:mask_img_h + top, left:mask_img_w + left]

            return scaled_input_image, scaled_mask_image

        return input_image, mask_image


#############################
# Flip image
#############################
class RandomFlip(object):

    def __init__(self, prob=0.5):

        self.prob = prob

    def __call__(self, input_image, mask_image):

        if random.uniform(0, 1) <= self.prob:

            # Get image shape
            h, w = input_image.shape[:2]

            # Flip image
            input_image = input_image[:, ::-1]

            # Get image shape
            h, w = mask_image.shape[:2]

            # Flip image
            mask_image = mask_image[:, ::-1]

            # Random flip horizontally
            if random.uniform(0, 1) <= 0.5:
                input_image = np.flip(input_image)
                mask_image = np.flip(mask_image)

            return input_image, mask_image

        return input_image, mask_image


#############################
# Change image brightness
#############################
class RandomBrightness():

    def __init__(self, lower=-25, upper=25, prob=0.5):
        self.lower = lower
        self.upper = upper
        self.prob = prob

    def __call__(self, input_image, mask_image):
        if random.uniform(0, 1) <= self.prob:
            amount = int(random.uniform(self.lower, self.upper))

            input_image = np.clip(input_image + amount, 0, 255).astype("uint8")

            return input_image, mask_image

        return input_image, mask_image


#############################
# Change image contrast
#############################
class RandomContrast():

    def __init__(self, lower=0.5, upper=1.5, prob=0.5):
        self.lower = lower
        self.upper = upper
        self.prob = prob

    def __call__(self, input_image, mask_image):
        if random.uniform(0, 1) <= self.prob:
            amount = random.uniform(self.lower, self.upper)

            input_image = np.clip(input_image * amount, 0, 255).astype("uint8")

            return input_image, mask_image

        return input_image, mask_image


#############################
# Gaussian Noise
#############################
class GaussianNoise(object):

    def __init__(self, mean=0.0, var=0.1, prob=0.5):
        self.mean = mean
        self.var = var
        self.prob = prob

    def __call__(self, input_image, mask_image):
        if random.uniform(0, 1) <= self.prob:
            # Get image shape
            h, w = input_image.shape[:2]

            sigma = self.var ** 0.5

            gauss = np.random.normal(self.mean, sigma, (w, h))
            gauss = gauss.reshape(w, h)

            noise_input_image = input_image + gauss

            return noise_input_image, mask_image

        return input_image, mask_image


#############################
# Normalize
#############################
class Normalize(object):
    """Normalize the color range to [0,1]."""

    def __call__(self, input_image, mask_image):
        return input_image / 255, mask_image / 255

    ############################################################################################


# Test Data Augmentation
###########################################################################################
def plot_transformation(transformation, n_samples=3, normalize=False):
    data = dBinding
    for i in range(n_samples):

        # define the size of images
        f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
        f.set_figwidth(14)

        # randomly select a sample
        idx = np.random.randint(0, len(data))
        input_image_path, livertumor_mask_image_path = data[idx]

        input_image = load_dicom(input_image_path)
        livertumor_mask_image = load_dicom(livertumor_mask_image_path)

        if normalize:
            norm = Normalize()
            input_image, livertumor_mask_image = norm(input_image, livertumor_mask_image)

        new_input_image, new_livertumor_mask_image = transformation(input_image, livertumor_mask_image)

        ax1.imshow(input_image, cmap='gray')
        ax1.set_title('Original Input')

        ax2.imshow(new_input_image, cmap='gray')
        ax2.set_title(type(transformation).__name__)

        ax3.imshow(livertumor_mask_image, cmap='gray')
        ax3.set_title('Original livertumor Mask')

        ax4.imshow(new_livertumor_mask_image, cmap='gray')
        ax4.set_title(type(transformation).__name__)

        plt.show()

def ploting_Translation():

    ##########################
    # Resize Test
    ##########################
    resize = Resize((512, 512), (512, 512))
    plot_transformation(resize)

    ##########################
    # Random Translation
    ##########################
    translation = RandomTranslation(ratio=0.2, prob=1.0)
    plot_transformation(translation)

    ##########################
    # Random Scale
    ##########################
    scale = RandomScale(prob=1.0)
    plot_transformation(scale)

    ##########################
    # Random Brightness Test
    ##########################
    bright = RandomBrightness(prob=1.0)
    plot_transformation(bright)

    ##########################
    # Random Contrast Test
    ##########################
    contrast = RandomContrast(prob=1.0)
    plot_transformation(contrast)

    ##########################
    # Gaussian Noise Test
    ##########################
    noise = GaussianNoise(mean=0.0, var=0.001, prob=1.0)
    plot_transformation(noise, normalize=True)

    ##########################
    # Normalize Test
    ##########################
    normalize = Normalize()
    plot_transformation(normalize)

###########################################################################################
# Create Dataset
###########################################################################################

# Create X, y tuple from image_path, key_pts tuple

def createXy(data, transformations=None):
    input_image_path, livertumor_mask_image_path = data

    input_image = load_dicom(input_image_path)
    livertumor_mask_image = load_dicom(livertumor_mask_image_path)

    # Apply transformations for the tuple (image, labels, boxes)
    if transformations:
        for t in transformations:
            input_image, livertumor_mask_image = t(input_image, livertumor_mask_image)

    input_image = np.expand_dims(input_image, axis=-1)
    livertumor_mask_image = np.expand_dims(livertumor_mask_image, axis=-1)

    return input_image, livertumor_mask_image


# Generator for using with model
def generator(data, transformations=None, batch_size=4, shuffle_data=True):
    n_samples = len(data)

    # Loop forever for the generator
    while 1:

        if shuffle_data:
            data = shuffle(data)

        for offset in range(0, n_samples, batch_size):

            batch_samples = data[offset:offset + batch_size]

            X = []
            y = []

            for sample_data in batch_samples:
                image, target = createXy(sample_data, transformations)

                X.append(image)
                y.append(target)

            X = np.asarray(X).astype('float32')
            y = np.asarray(y).astype('float32')

            yield (shuffle(X, y))


# ############################################################################################
# # Unet Model
# ############################################################################################
# def unet_encoder(inputs, filters, block_id):
#     x = Conv2D(filters, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
#                name='block_' + str(block_id) + '_unet_encoder_conv2d_1')(inputs)
#     x = BatchNormalization(name='block_' + str(block_id) + '_unet_encoder_conv_batch_1')(x)
#     x = Activation('relu', name='block_' + str(block_id) + '_unet_encoder_relu_1')(x)
#
#     x = Conv2D(filters, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
#                name='block_' + str(block_id) + '_unet_encoder_conv2d_2')(x)
#     x = BatchNormalization(name='block_' + str(block_id) + '_unet_encoder_conv_batch_2')(x)
#     x = Activation('relu', name='block_' + str(block_id) + '_unet_encoder_relu_2')(x)
#
#     return x
#
#
# def unet_encoder_pool(inputs, filters, block_id):
#     x = unet_encoder(inputs, filters, block_id)
#     x = MaxPooling2D(pool_size=(2, 2), name='block_' + str(block_id) + '_unet_pooling')(x)
#
#     return x
#
#
# def unet_decoder(inputs_a, inputs_b, filters, block_id):
#     x = Conv2DTranspose(filters, kernel_size=(2, 2), strides=(2, 2), padding='same')(inputs_a)
#
#     x = concatenate([x, inputs_b], axis=3)
#
#     x = Conv2D(filters, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
#                name='block_' + str(block_id) + '_unet_decoder_conv2d_1')(x)
#     x = BatchNormalization(name='block_' + str(block_id) + '_unet_decoder_conv_batch_1')(x)
#     x = Activation('relu', name='block_' + str(block_id) + '_unet_decoder_relu_1')(x)
#
#     x = Conv2D(filters, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
#                name='block_' + str(block_id) + '_unet_decoder_conv2d_2')(x)
#     x = BatchNormalization(name='block_' + str(block_id) + '_unet_decoder_conv_batch_2')(x)
#     x = Activation('relu', name='block_' + str(block_id) + '_unet_decoder_relu_2')(x)
#
#     return x
#
#
# def UNet(input_shape=(512, 512, 1)):
#     Image = Input(shape=input_shape)
#
#     encoder = unet_encoder_pool(Image, filters=32, block_id=0)
#     encoder = unet_encoder_pool(encoder, filters=64, block_id=1)
#     encoder = unet_encoder_pool(encoder, filters=128, block_id=2)
#     encoder = unet_encoder_pool(encoder, filters=256, block_id=3)
#     encoder = unet_encoder(encoder, filters=512, block_id=4)
#
#     encoder_model = Model(inputs=Image, outputs=encoder)
#
#     tensor1 = encoder_model.get_layer('block_3_unet_encoder_relu_2').output
#     tensor2 = encoder_model.get_layer('block_2_unet_encoder_relu_2').output
#     tensor3 = encoder_model.get_layer('block_1_unet_encoder_relu_2').output
#     tensor4 = encoder_model.get_layer('block_0_unet_encoder_relu_2').output
#
#     decoder = unet_decoder(encoder, tensor1, filters=256, block_id=4)
#     decoder = unet_decoder(decoder, tensor2, filters=128, block_id=5)
#     decoder = unet_decoder(decoder, tensor3, filters=64, block_id=6)
#     decoder = unet_decoder(decoder, tensor4, filters=32, block_id=7)
#
#     decoder = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), activation='sigmoid')(decoder)
#
#     model = Model(inputs=Image, outputs=decoder)
#
#     return model
#
#
# ############################################################################################
# # Unet Loss
# ############################################################################################
# class UNetLoss():
#
#     def __init__(self, smooth=1):
#         self.smooth = smooth
#
#     def calculate_loss(self, y_true, y_pred):
#         y_true_f = K.flatten(y_true)
#         y_pred_f = K.flatten(y_pred)
#
#         intersection = K.sum(y_true_f * y_pred_f)
#         coefficient = (2. * intersection + self.smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + self.smooth)
#         loss = 1 - coefficient
#         return loss


def data_for_splitting():
    data = dBinding()
    train_data, test_valid_data = train_test_split(data, test_size=0.20, random_state=42)
    test_data, valid_data = train_test_split(test_valid_data, test_size=0.50, random_state=42)

    return train_data, test_data, valid_data
    #


def save_dicom_images_to_png(train_data, test_data, valid_data, image_data_path='dataset/data'):

    pure_data = {'train': train_data, 'test': test_data, 'validate': valid_data}

    for key, value in tqdm(pure_data.items(), total=len(pure_data.items())):
        print('data of {0} data length: {1}'.format(key, len(value)))

        for x, y in value:
            nameTrain = os.path.basename(x).split("/")[-1].split(".dcm")[0]
            nameMask = os.path.basename(y).split("/")[-1].split(".dcm")[0]
            #
            x = load_dicom(x)
            y = load_dicom(y)

            save_path = f'{image_data_path}/{key}'

            # # create directory path if not existing
            dir_create(os.path.join(save_path, "images"))
            dir_create(os.path.join(save_path, "masks"))
            #
            # # print(name)
            tmp_image_name = f"{nameTrain}.png"
            tmp_mask_name = f"{nameMask}.png"
            #
            image_path = os.path.join(save_path, "images", tmp_image_name)
            mask_path = os.path.join(save_path, "masks", tmp_mask_name)

            # print(image_path)
            # print(mask_path)

            # write Training and masks images
            if not cv2.imwrite(image_path, x):
                raise Exception("Could not write image")

            if not cv2.imwrite(mask_path, y):
                raise Exception("Could not write mask")

            # print(image_path, mask_path)
            #
            # x = x.pixel_array.astype(float)
            # y = y.pixel_array.astype(float)
            #
            # x = (np.maximum(x, 0) / x.max()) * 255.0
            # y = (np.maximum(y, 0) / y.max()) * 255.0
            # #
            # x = np.uint8(x)
            # final_image = Image.fromarray(x)
            # final_image.show()

            # print(x)

    # print('Size of train data: {0}'.format(len(train_data)))
    # print('Size of test data: {0}'.format(len(test_data)))
    # print('Size of valid data: {0}'.format(len(valid_data)))

    # train_generator = generator(train_data, train_transformations, batch_size)
    # test_generator = generator(test_data, test_transformations, batch_size)

    # # callbacks
    # model_path = 'saved_models'
    #
    # model = UNet()
    # model.summary()
    #
    # # Create loss function
    # loss = UNetLoss()
    #
    # # Create Optimizer
    # optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #
    # # Compile model for training
    # model.compile(optimizer, loss=loss.calculate_loss, metrics=['accuracy'])
    #
    # # File were the best model will be saved during checkpoint
    # model_file = os.path.join(model_path, 'livertumor_segmentation-{val_loss:.4f}.h5')
    #
    # # Check point for saving the best model
    # check_pointer = ModelCheckpoint(model_file, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    #
    # # Logger to store loss on a csv file
    # csv_logger = CSVLogger(filename='livertumor_segmentation.csv', separator=',', append=True)

    # return train_generator, test_generator

def save_augmented_images_to_png(train_data, test_data, valid_data, image_data_path='dataset/data'):

    pure_data = {'train': train_data, 'test': test_data, 'validate': valid_data}

    for key, value in tqdm(pure_data.items(), total=len(pure_data.items())):
        print('data of {0} data'.format(key))
        index = 0
        for x, y in value:
            # nameTrain = os.path.basename(x).split("/")[-1].split(".dcm")[0]
            # nameMask = os.path.basename(y).split("/")[-1].split(".dcm")[0]
            # #
            # x = load_dicom(x)
            # y = load_dicom(y)
            index +=1
            save_path = f'{image_data_path}/{key}'

            # # create directory path if not existing
            dir_create(os.path.join(save_path, "images"))
            dir_create(os.path.join(save_path, "masks"))
            #
            # # print(name)
            tmp_image_name = f"image{index}.png"
            tmp_mask_name = f"mask{index}.png"
            #
            image_path = os.path.join(save_path, "images", tmp_image_name)
            mask_path = os.path.join(save_path, "masks", tmp_mask_name)

            # print(image_path)
            # print(mask_path)

            # write Training and masks images
            if not cv2.imwrite(image_path, x):
                raise Exception("Could not write image")

            if not cv2.imwrite(mask_path, y):
                raise Exception("Could not write mask")

            # print(image_path, mask_path)
            #
            # x = x.pixel_array.astype(float)
            # y = y.pixel_array.astype(float)
            #
            # x = (np.maximum(x, 0) / x.max()) * 255.0
            # y = (np.maximum(y, 0) / y.max()) * 255.0
            # #
            # x = np.uint8(x)
            # final_image = Image.fromarray(x)
            # final_image.show()

            # print(x)

    # print('Size of train data: {0}'.format(len(train_data)))
    # print('Size of test data: {0}'.format(len(test_data)))
    # print('Size of valid data: {0}'.format(len(valid_data)))

    # train_generator = generator(train_data, train_transformations, batch_size)
    # test_generator = generator(test_data, test_transformations, batch_size)

    # # callbacks
    # model_path = 'saved_models'
    #
    # model = UNet()
    # model.summary()
    #
    # # Create loss function
    # loss = UNetLoss()
    #
    # # Create Optimizer
    # optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #
    # # Compile model for training
    # model.compile(optimizer, loss=loss.calculate_loss, metrics=['accuracy'])
    #
    # # File were the best model will be saved during checkpoint
    # model_file = os.path.join(model_path, 'livertumor_segmentation-{val_loss:.4f}.h5')
    #
    # # Check point for saving the best model
    # check_pointer = ModelCheckpoint(model_file, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    #
    # # Logger to store loss on a csv file
    # csv_logger = CSVLogger(filename='livertumor_segmentation.csv', separator=',', append=True)

    # return train_generator, test_generator


def augment_data(train_data, test_data, valid_data):
    # Data augmentation
    train_transformations = [
        RandomBrightness(prob=0.5),
        RandomContrast(prob=0.5),
        RandomTranslation(ratio=0.2, prob=0.5),
        RandomScale(lower=0.8, upper=1.2, prob=0.5),
        RandomFlip(prob=0.5),
        Resize((512, 512), (512, 512)),
        Normalize()
    ]

    test_transformations = [
        RandomBrightness(prob=0.5),
        RandomContrast(prob=0.5),
        RandomTranslation(ratio=0.2, prob=0.5),
        RandomScale(lower=0.8, upper=1.2, prob=0.5),
        RandomFlip(prob=0.5),
        Resize((512, 512), (512, 512)),
        Normalize()
    ]

    valid_transformations = [
        RandomBrightness(prob=0.5),
        RandomContrast(prob=0.5),
        RandomTranslation(ratio=0.2, prob=0.5),
        RandomScale(lower=0.8, upper=1.2, prob=0.5),
        RandomFlip(prob=0.5),
        Resize((512, 512), (512, 512)),
        Normalize()
    ]

    # Hyper parameters
    epochs = 100
    batch_size = 2
    learning_rate = 0.001
    weight_decay = 5e-4
    momentum = .9

    train_generator = generator(train_data, train_transformations, batch_size)
    test_generator = generator(test_data, test_transformations, batch_size)
    validate_generator = generator(valid_data, valid_transformations, batch_size)

    for x, y in train_generator:
        print(y)

    # return train_generator, test_generator, validate_generator

def save_augmented_images_to_png_2dir():
    data = dBinding()
    for x, y in tqdm(data, total=len(data)):
        nameTrain = os.path.basename(x).split("/")[-1].split(".dcm")[0]
        nameMask = os.path.basename(y).split("/")[-1].split(".dcm")[0]
        #
        x = load_dicom(x)
        y = load_dicom(y)

        save_path = f'dataset'

        # # create directory path if not existing
        dir_create(os.path.join(save_path, "images"))
        dir_create(os.path.join(save_path, "masks"))
        #
        # # print(name)
        tmp_image_name = f"{nameTrain}.jpg"
        tmp_mask_name = f"{nameMask}.png"
        #
        image_path = os.path.join(save_path, "images", tmp_image_name)
        mask_path = os.path.join(save_path, "masks", tmp_mask_name)

        # print(image_path)
        # print(mask_path)

        # write Training and masks images
        if not cv2.imwrite(image_path, x):
            raise Exception("Could not write image")

        if not cv2.imwrite(mask_path, y):
            raise Exception("Could not write mask")


if __name__ == "__main__":
    data_preprocessing_creation()
    save_augmented_images_to_png_2dir()
    # train_data, test_data, valid_data = data_for_splitting()
    # save_dicom_images_to_png(train_data, test_data, valid_data, 'dataset/data')

    # train_generator, test_generator, validate_generator = augment_data(train_data, test_data, valid_data)
    # # save_augmented_images_to_png(train_generator, test_generator, validate_generator, 'dataset/data_augmented')