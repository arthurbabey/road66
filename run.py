"""
Train the final model, generate predictions and create an AIcrowd submission.
"""

import argparse
import keras
import os,sys
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg


from PIL import Image
from sklearn.model_selection import train_test_split
from scripts.unet import unet
from scripts.helpers_unet import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import backend as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


SUBMISSION_PATH = 'submission/new_submission.csv'
TRAINSET_PATH = 'datasets/training/'
TESTSET_PATH = 'datasets/test_set_images'

IMG_SIZE = 400
INPUT_SIZE = (IMG_SIZE, IMG_SIZE, 3)
BATCH_SIZE = 1
EPOCHS, STEP_PER_EPOCH = 100, 600
FOREGROUND_THRESHOLD = 0.25


"""
Setting seed for reproducibility
"""

SEED = 2019
np.random.seed(SEED)
tf.compat.v1.set_random_seed(SEED)


def run(train = False):

    """
    Create best AIcrowd submission from either a pretrained model or by training the model from scratch
    """
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    warnings.filterwarnings("ignore")


    if sess:
        print('Working on gpu')

    if train:
        model = training()

    else:
        print('***************')
        model = unet(input_size = INPUT_SIZE)
        print('*************** MODEL VA LOAD')
        model.load_weights('best_model/lastrun.h5')
        print('Model loaded ! ')


    imgs_test = load_testset(path = TESTSET_PATH)
    print('*******  tttttt ********')

    test_image_unet_submission(imgs_test, model = model, filename = SUBMISSION_PATH, foreground_threshold = FOREGROUND_THRESHOLD)

    print('Done !')
    print('CSV submission created in: {}'.format(SUBMISSION_PATH))



def training(test_size = 0.2):
    """
    Train model from scratch.
    """
    model = unet(input_size = INPUT_SIZE)

    imgs, gt_imgs = load_trainset(path = TRAINSET_PATH)
    imgs, gt_imgs = rotate_images(imgs, gt_imgs, [ 15, 30, 45, 60, 75])
    n = len(imgs)

    Xr, Yr = resize_image(imgs, gt_imgs, IMG_SIZE)
    Yr = np.reshape(Yr, (n, IMG_SIZE, IMG_SIZE, 1))
    x_train, x_val, y_train, y_val = train_test_split(Xr, Yr, test_size=test_size,  random_state=SEED)

    print('TRAIN TEST SPLIT DONE')

    data_gen_args = dict(
            width_shift_range=0.05,
            height_shift_range=0.05,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,)


    image_datagen = ImageDataGenerator(**data_gen_args, fill_mode ='reflect')
    train_gen = XYaugmentGenerator(x_train, y_train, image_datagen, seed = SEED, batch_size = BATCH_SIZE)

    print('DATA AUGMENTATION DONE')


    model_filename = 'train_from_scratch.h5'

    cp = ModelCheckpoint(model_filename, verbose=1, monitor='val_loss', save_best_only=True)
    lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, verbose=1, mode='min', min_lr= 1e-8)
    es = EarlyStopping(monitor = 'val_loss', patience = 25, mode = 'min')

    model.compile(loss='binary_crossentropy',
                  optimizer= Adam(lr = 1e-4),
                  metrics=['binary_accuracy'])

    history = model.fit_generator(
        train_gen,
        steps_per_epoch=STEP_PER_EPOCH,
        epochs=EPOCHS,
        validation_data=(x_val, y_val),
        callbacks=[cp, lr, es])

    return model

def XYaugmentGenerator(X1, y, gen, seed = SEED, batch_size = BATCH_SIZE):
    genX1 = gen.flow(X1, y, batch_size=batch_size, seed=seed)
    genX2 = gen.flow(y, X1, batch_size=batch_size, seed=seed)
    while True:
        X1i = genX1.next()
        X2i = genX2.next()

        yield X1i[0], X2i[0]


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
