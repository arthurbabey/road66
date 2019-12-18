"""
Train the final model, generate predictions and create an AIcrowd submission.
"""

import argparse
import keras
import os,sys
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg

from skimage import rotation, resize
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from scripts/model import unet
from scripts/helpers_unet import *
from googleDriveFileDownloader import googleDriveFileDownloader



SUBMISSION_PATH = 'submission/new_submission.csv'
TRAINSET_PATH = 'datasets/training/'
TESTSET_PATH = 'datasets/test_set_images'

IMG_SIZE = 400
INPUT_SIZE = (IMG_SIZE, IMG_SIZE, 3)
BATCH_SIZE = 1
EPOCHS, STEP_PER_EPOCH = 100, 600
FOREGROUND_THRESHOLD = 0.25

a = googleDriveFileDownloader()
pretrain_model = a.downloadFile("https://drive.google.com/uc?id=1O4x8rwGJAh8gRo8sjm0kuKFf6vCEm93G&export=download")


"""
Setting seed for reproducibility
"""

SEED = 2019
np.random.seed(seed)
tf.set_random_seed(seed)


def run(train = False):

    """
    Create best AIcrowd submission from either a pretrained model or by training the model from scratch
    """
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    if sess:
        print('working on gpu')

    model = unet(input_size = INPUT_SIZE)

    if train:
        model = training(model)

    else:
        model.load_weights('best_model/UNcusto6_1200plus.h5')
        model.load_weights(pretrain_model)


    imgs_test = load_testset(path = TESTSET_PATH)

    test_image_unet_submission(imgs_test, filename = SUBMISSION_PATH, foreground_threshold = FOREGROUND_THRESHOLD)

    print('Done !'')
    print('CSV submission created in: {}'.format(SUBMISSION_PATH))



def training(model, test_size = 0.2):
    """
    Train model from scratch.
    """

    imgs, gt_imgs = rotate_images(load_trainset(path = 'TRAINSET_PATH'), [ 15, 30, 45, 60, 75])
    n = len(imgs)

    Xr, Yr = resize_image(imgs, gt_imgs, IMG_SIZE)
    Yr = np.reshape(Yr, (n, IMG_SIZE, IMG_SIZE, 1))
    x_train, x_val, y_train, y_val = train_test_split(Xr, Yr, test_size=test_size,  random_state=SEED)

    data_gen_args = dict(rotation_range=0.,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=15,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,)


    image_datagen = ImageDataGenerator(**data_gen_args, fill_mode ='reflect')
    train_gen = XYaugmentGenerator(x_train, y_train, image_datagen, seed = SEED, batch_size = BATCH_SIZE)

    model_filename = 'train_from_scratch.h5'

    cp = ModelCheckpoint(model_filename, verbose=1, monitor='val_loss', save_best_only=True)
    lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, verbose=1, mode='min', min_lr= 1e-8)
    es = EarlyStopping(monitor = 'val_loss', patience = 25, mode = 'min')

    model.compile(loss='binary_crossentropy',
                  optimizer= Adam(lr = 1e-4),
                  metrics=['binary_accuracy'])

    histry = model.fit_generator(
        train_gen,
        steps_per_epoch=STEP_PER_EPOCH,
        epochs=EPOCHS,
        validation_data=(x_val, y_val),
        callbacks=[cp, lr, es]

    return model


def parse_args():
    """
    Parse command line flags.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', action='store_true', default=False, dest='train', help='Train model from scratch')
    results = parser.parse_args()

    return {'train': results.train}

if __name__ == '__main__':
    args = parse_args()
    run(args['train'])
