import keras
import numpy as np
import tensorflow as tf
import os,sys
import sys
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.pyplot as plt

from PIL import Image
from skimage.transform import rotate
from skimage.transform import resize
from sklearn.model_selection import train_test_split




def load_image(infilename):
    data = mpimg.imread(infilename)
    return data


# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

# Convert array of labels to an image

def label_to_img(imgwidth, imgheight, w, h, labels):
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            im[j:j+w, i:i+h] = labels[idx]
            idx = idx + 1
    return im

def value_to_class(v, foreground_threshold=0.25):

    df = np.sum(v)
    if df > foreground_threshold:
        return 1
    else:
        return 0

def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def rotate_images(X, Y , degrees):
    """
    increase the number of data
    by adding rotations of the base data

    """

    X = np.array(X)
    Y = np.array(Y)
    rotimg = np.zeros(X.shape)
    rotgtimg = np.zeros(Y.shape)

    Xtemp = X
    Ytemp = Y

    #rotate all images by degree and add them to the data vector
    for degree in degrees:
        for i in range(len(Xtemp)):
            rotimg[i] = rotate(Xtemp[i], degree, resize=False, mode='reflect')
            rotgtimg[i] = rotate(Ytemp[i], degree, resize=False, mode='reflect')
        X = np.concatenate([X,rotimg])
        Y = np.concatenate([Y,rotgtimg])

    return X,Y

def resize_image(X, Y, size = 512):

    X = np.asarray(X)
    Y = np.asarray(Y)

    Xresize = np.asarray([resize(X[i], (size,size), mode = 'reflect') for i in range(X.shape[0])])
    Yresize = np.asarray([resize(Y[i], (size,size), mode = 'reflect') for i in range(X.shape[0])])


    return Xresize, Yresize


def create_submission(y_pred, filename = 'filename', patch_size = 16, img_size = 608):
    """
    Create a submission (csv format) for AIcrowd
    from given prediction

    """
    n = img_size // patch_size
    y_pred = np.reshape(y_pred, (-1, n, n))

    with open(filename, 'w') as f:
        f.write('id,prediction\n')
        for i in range(y_pred.shape[0]):
            img = y_pred[i]
            for j in range(img.shape[0]):
                for k in range(img.shape[1]):
                    name = '{:03d}_{}_{},{}'.format(i + 1, j * patch_size, k * patch_size, int(img[j,k]))
                    f.write(name + '\n')


def balance_data(x_train, y_train):
    c0 = 0  # bgrd
    c1 = 0  # road
    for i in range(len(y_train)):
        if y_train[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

    print('Balancing training data...')
    min_c = min(c0, c1)
    idx0 = [i for i, j in enumerate(y_train) if j[0] == 1]
    idx1 = [i for i, j in enumerate(y_train) if j[1] == 1]
    new_indices = idx0[0:min_c] + idx1[0:min_c]
    print(len(new_indices))
    print(x_train.shape)
    x_train = x_train[new_indices, :, :, :]
    y_train = y_train[new_indices]

    train_size = y_train.shape[0]

    c0 = 0
    c1 = 0
    for i in range(len(y_train)):
        if y_train[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

    return x_train, y_train


def img_crop(im, w, h, border = 0, step = 16):
    """
    Return the patches list of an image.
    """
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    if border != 0:
        im_r = np.pad(im[:,:,0], ((border, border), (border, border)), 'reflect')
        im_g = np.pad(im[:,:,1], ((border, border), (border, border)), 'reflect')
        im_b = np.pad(im[:,:,2], ((border, border), (border, border)), 'reflect')
        im = np.dstack((im_r, im_g, im_b))
    for i in range(0,imgheight,step):
        for j in range(0,imgwidth,step):
            if is_2d:
                im_patch = im[j:j+w+2*border, i:i+h+2*border]
            else:
                im_patch = im[j:j+w+2*border, i:i+h+2*border, :]
            list_patches.append(im_patch)
    return list_patches


def load_trainset(path = 'Data/test_set_images'):

    # Loaded a set of images
    root_dir = path

    image_dir = root_dir + "images/"
    files = os.listdir(image_dir)
    n = len(files) # Use all images
    print("Loading " + str(n) + " testing images")
    imgs = [load_image(image_dir + files[i]) for i in range(100)]

    gt_dir = root_dir + "groundtruth/"
    print("Loading " + str(n) + " groundtruth")
    gt_imgs = [load_image(gt_dir + files[i]) for i in range(100)]

    return imgs, gt_imgs


def load_testset(path = 'Data/test_set_images'):


  root_testdir = path
  test_names = os.listdir(root_testdir)

  num_test = len(test_names)
  order = [int(test_names[i].split("_")[1]) for i in range(num_test)]
  index = np.argsort(order)

  # Load image and reorder them
  imgs_test = [load_image(os.path.join(root_testdir, test_names[i], test_names[i]) + ".png")
               for i in range(num_test)]
  imgs_test = [imgs_test[i] for i in index]

  return imgs_test


def test_image_unet_submission(imgs_test, model, size = 400, foreground_threshold = 0.25, filename = 'submission.csv'):


  img1 = []
  img2 = []
  img3 = []
  img4 = []

  shift = 608 - size


  for img in imgs_test:
    img = img[0:size, 0:size, :]
    img1.append(img)

  for img in imgs_test:
    img = img[shift:608, 0:size, :]
    img2.append(img)

  for img in imgs_test:
    img = img[0:size, shift:608, :]
    img3.append(img)

  for img in imgs_test:
    img = img[shift:608, shift:608, :]
    img4.append(img)

  img_pred1 = model.predict(np.asarray(img1), batch_size = 1, verbose = 1)
  img_pred2 = model.predict(np.asarray(img2), batch_size = 1, verbose = 1)
  img_pred3 = model.predict(np.asarray(img3), batch_size = 1, verbose = 1)
  img_pred4 = model.predict(np.asarray(img4), batch_size = 1, verbose = 1)


  img_pred1[img_pred1 <= 0.557] = 0
  img_pred1[img_pred1 > 0.557] = 1

  img_pred2[img_pred2 <= 0.557] = 0
  img_pred2[img_pred2 > 0.557] = 1

  img_pred3[img_pred3 <= 0.557] = 0
  img_pred3[img_pred3 > 0.557] = 1

  img_pred4[img_pred4 <= 0.557] = 0
  img_pred4[img_pred4 > 0.557] = 1

  img1 = np.asarray(img1)
  img2 = np.asarray(img2)
  img3 = np.asarray(img3)
  img4 = np.asarray(img4)

  list_m1 = []
  list_m2 = []
  list_merge = []

  for i in range(50):
    m1 = np.concatenate((img_pred1[i, 0:shift, :, :], img_pred2[i, :, :, :]), axis = 0)
    list_m1.append(m1)
    m2 = np.concatenate((img_pred3[i, 0:shift, :, :], img_pred4[i, :, :, :]), axis = 0)
    list_m2.append(m2)

  list_m1 = np.asarray(list_m1)
  list_m2 = np.asarray(list_m2)

  for i in range(50):
    merge = np.concatenate((list_m1[i,:,:,:], list_m2[i, :, (size - shift):size, :]), axis=1)
    list_merge.append(merge)


  y_pred = np.asarray(list_merge)

  pred_patch = [img_crop(y_pred[i], 16, 16) for i in range(y_pred.shape[0])]
  pred_patch = np.asarray([pred_patch[i][j] for i in range(len(pred_patch)) for j in range(len(pred_patch[i]))])
  pred_patch = np.asarray([value_to_class(np.mean(pred_patch[i]), foreground_threshold=foreground_threshold) for i in range(pred_patch.shape[0])])

  create_submission(pred_patch, filename)
