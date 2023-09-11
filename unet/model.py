"""
U-Net Code References:
    pixel-weighted cross-entropy
        https://github.com/keras-team/keras/issues/6261
    unet-weight-map computation:
        https://stackoverflow.com/questions/50255438/pixel-wise-loss-weight-for-image-segmentation-in-keras
        https://arxiv.org/pdf/1505.04597.pdf
    U-Net construction:
        https://stackoverflow.com/questions/58134005/keras-u-net-weighted-loss-implementation
        UNET-TGS: https://medium.com/@harshall.lamba/understanding-semantic-segmentation-with-unet-6be4f42d4b47 

Data Loading Reference:
    Mask R-CNN
    Copyright (c) 2017 Matterport, Inc.
    Licensed under the MIT License (see LICENSE for details)
    Written by Waleed Abdulla
    Modifications by Roland S. Zimmermann, Julien Siems

Additional sources:
    mrcnn_mask_edge_loss_graph loss function
        Copyright (c) 2018/2019 Roland S. Zimmermann, Julien Siems
        Licensed under the MIT License (see LICENSE for details)

"""



import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
from scipy.ndimage.morphology import distance_transform_edt
from skimage import draw
import tensorflow as tf
from tensorflow import keras
from keras.models import Model, load_model
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, Concatenate, MaxPool2D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, AveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers.merge import concatenate, add, multiply
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
from keras.applications import vgg16
import cv2
import glob
import json
import skimage.draw

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass

try:
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
except:
    pass


_epsilon = tf.convert_to_tensor(K.epsilon(), np.float32)

class CheckpointSaver(keras.callbacks.Callback):
    
    def __init__(self, measurement='loss', name='unet', settings={}):
        self.measurement = measurement
        self.name = name
        self.dataset = settings['dataset']
        
    def on_train_begin(self, logs={}):
        self.best_val_loss = 999999
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.val_losses.append(logs.get('val_{}'.format(self.measurement)))
        mean_val_loss = np.mean(np.array(self.val_losses[-1]))
        if mean_val_loss < self.best_val_loss:
            self.model.save("{}/{}/{}_checkpoint.h5".format(self.dataset, self.name, self.name))
            self.best_val_loss = mean_val_loss
            print("saving checkpoint :{}".format(mean_val_loss) )


def create_augmentation():
    aug = ImageDataGenerator(
        zca_whitening=False,
        width_shift_range=0.,
        height_shift_range=0., #20,
        zoom_range=0.05,
        rotation_range=10,
        horizontal_flip=True,
        vertical_flip=False
        )
    return aug


def get_augmentation_set():
    img_aug = create_augmentation()
    mask_aug = create_augmentation()
    wt_aug = create_augmentation() 
    return [img_aug, mask_aug, wt_aug]


def pixel_weighted_cross_entropy(weights, targets, predictions):
    loss_val = tf.keras.losses.binary_crossentropy(targets, predictions)
    weighted_loss_val = weights * loss_val
    return K.mean(weighted_loss_val)


def mrcnn_mask_edge_loss_graph(y_pred, y_true):
    """
    mrcnn_mask_edge_loss_graph loss function
    Copyright (c) 2018/2019 Roland S. Zimmermann, Julien Siems
    """
    edge_filters = ["sobel-y"]
    norm = "l2"
    weight_factor = 2.0

    # sobel kernels
    sobel_x_kernel = tf.reshape(tf.constant([[1, 2, 1],
                                             [0, 0, 0],
                                             [-1, -2, -1]], dtype=tf.float32),
                                shape=[3, 3, 1, 1], name='sobel_x_kernel')
    sobel_y_kernel = tf.reshape(tf.constant([[1, 0, -1],
                                             [2, 0, -2],
                                             [1, 0, -1]], dtype=tf.float32),
                                shape=[3, 3, 1, 1], name='sobel_y_kernel')

    # prewitt kernels
    prewitt_x_kernel = tf.reshape(tf.constant([[1, 0, -1],
                                             [1, 0, -1],
                                             [1, 0, -1]], dtype=tf.float32),
                                shape=[3, 3, 1, 1], name='prewitt_x_kernel')
    prewitt_y_kernel = tf.reshape(tf.constant([[1, 1, 1],
                                             [0, 0, 0],
                                             [-1, -1, -1]], dtype=tf.float32),
                                shape=[3, 3, 1, 1], name='prewitt_y_kernel')

    # prewitt kernels
    kayyali_senw_kernel = tf.reshape(tf.constant([[6, 0, -6],
                                             [0, 0, -0],
                                             [-6, 0, 6]], dtype=tf.float32),
                                shape=[3, 3, 1, 1], name='kayyali_senw_kernel')
    kayyali_nesw_kernel = tf.reshape(tf.constant([[-6, 0, 6],
                                             [0, 0, 0],
                                             [6, 0, -6]], dtype=tf.float32),
                                shape=[3, 3, 1, 1], name='kayyali_nesw_kernel')

    # roberts kernels
    roberts_x_kernel = tf.reshape(tf.constant([[1, 0],
                                              [0, -1]], dtype=tf.float32),
                                shape=[2, 2, 1, 1], name='roberts_x_kernel')
    roberts_y_kernel = tf.reshape(tf.constant([[0, -1],
                                               [1, 0]], dtype=tf.float32),
                                shape=[2, 2, 1, 1], name='roberts_y_kernel')

    # laplace kernel
    laplacian_kernel = tf.reshape(tf.constant([[1, 1, 1],
                                               [1, -8, 1],
                                               [1, 1, 1]], dtype=tf.float32),
                                  shape=[3, 3, 1, 1], name='laplacian_kernel')

    gaussian_kernel = tf.reshape(tf.constant([[0.077847, 0.123317, 0.077847],
                                              [0.123317, 0.195346, 0.1233179],
                                              [0.077847, 0.123317, 0.077847]], dtype=tf.float32),
                                 shape=[3, 3, 1, 1], name='gaussian_kernel')

    filter_map = {
        "sobel-x": sobel_x_kernel,
        "sobel-y": sobel_y_kernel,
        "roberts-x": roberts_x_kernel,
        "roberts-y": roberts_y_kernel,
        "prewitt-x": prewitt_x_kernel,
        "prewitt-y": prewitt_y_kernel,
        "kayyali-senw": kayyali_senw_kernel,
        "kayyali-nesw": kayyali_nesw_kernel,        
        "laplace": laplacian_kernel
    }

    lp_norm_map = {
        "l1": 1,
        "l2": 2,
        "l3": 3,
        "l4": 4,
        "l5": 5
    }

    if norm not in lp_norm_map:
        raise ValueError("The `norm` '{0}' is not supported. Supported values are: [l1...l5]".format(norm))

    edge_filters = tf.concat([filter_map[x] for x in edge_filters], axis=-1)

    # Add one channel to masks
    # y_pred = tf.expand_dims(y_pred, -1, name='y_pred')
    # y_true = tf.expand_dims(y_true, -1, name='y_true')

    smoothing_predictions = True
    if smoothing_predictions:
        # First filter with gaussian to smooth edges of predictions
        y_pred = tf.nn.conv2d(input=y_pred, filter=gaussian_kernel, strides=[1, 1, 1, 1], padding='SAME')

    y_pred_edges = tf.nn.conv2d(input=y_pred, filter=edge_filters, strides=[1, 1, 1, 1], padding='SAME')

    smoothing_gt = False
    if smoothing_gt:
        # First filter with gaussian to smooth edges of groundtruth
        y_true = tf.nn.conv2d(input=y_true, filter=gaussian_kernel, strides=[1, 1, 1, 1], padding='SAME')
    y_true_edges = tf.nn.conv2d(input=y_true, filter=edge_filters, strides=[1, 1, 1, 1], padding='SAME')

    def append_magnitude(edges, name=None):
        magnitude = tf.expand_dims(tf.sqrt(edges[:, :, :, 0] ** 2 + edges[:, :, :, 1] ** 2), axis=-1)
        return tf.concat([edges, magnitude], axis=-1, name=name)

    def lp_loss(y_true, y_pred, p):
        return K.mean(K.pow(K.abs(y_pred - y_true), p), axis=-1)

    def smoothness_loss(y_true, y_pred, p):
        weight_smoothness = K.exp(-K.abs(y_true))
        smoothness = y_pred * weight_smoothness
        smoothness = smoothness[:, :, :, 0] + smoothness[:, :, :, 1]
        return K.mean(K.pow(K.abs(smoothness), p))

    # calculate the edge agreement loss per pixel
    pixel_wise_edge_loss = K.switch(tf.size(y_true_edges) > 0,
                                    lp_loss(y_true=y_true_edges, y_pred=y_pred_edges, p=lp_norm_map[norm]),
                                    tf.constant(0.0))

    # multiply the pixelwise edge agreement loss with a scalar factor
    pixel_wise_edge_loss = weight_factor * pixel_wise_edge_loss

    weight_entropy = False
    if weight_entropy:
        pixel_wise_cross_entropy_loss = K.switch(tf.size(y_true) > 0,
                                                 K.binary_crossentropy(target=y_true, output=y_pred),
                                                 tf.constant(0.0))

        weighted_cross_entropy_loss = tf.squeeze(pixel_wise_cross_entropy_loss) * tf.exp(pixel_wise_edge_loss / 16)
        weighted_cross_entropy_loss = K.mean(weighted_cross_entropy_loss) / 1

        return weighted_cross_entropy_loss
    else:
        # return the mean of the pixelwise edge agreement loss
        edge_loss = K.mean(pixel_wise_edge_loss)
        return edge_loss


def get_weighted_unet_with_vgg(input_shape, n_filters = 4, dropout = 0.1, batchnorm = True, is_training=True):
    ip = Input(input_shape, name='img')
    weight_ip = Input(shape=(input_shape[0],input_shape[1],1), name='weighted_ip')

    encoder = vgg16.VGG16(input_tensor=ip, include_top=False, input_shape=input_shape)
    z = encoder.output
    encoder_output = z

    #---------------------------------
    conv5 = encoder_output
    conv4 = encoder.get_layer('block5_conv3').output
    conv3 = encoder.get_layer('block4_conv3').output
    conv2 = encoder.get_layer('block3_conv3').output
    conv1 = encoder.get_layer('block2_conv2').output
    conv0 = encoder.get_layer('block1_conv2').output
    
    model_ = Conv2DTranspose(256, (3,3), strides=(2,2), kernel_initializer='he_normal', padding='same')(conv5)
    concat1_ = Concatenate()([model_, conv4])
    model_ = Conv2D(512, 3, activation='relu', strides=1, kernel_initializer='he_normal', padding='same')(concat1_)
    
    up6 = Conv2DTranspose(64, 2, strides=2, kernel_initializer='he_normal', padding='same')(model_)
    conv6 = Concatenate()([up6, conv3])
    conv6 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = Dropout(0.4)(conv6)

    up7 = Conv2DTranspose(48, 2, strides=2, kernel_initializer='he_normal', padding='same')(conv6)
    conv7 = Concatenate()([up7, conv2])
    conv7 = Conv2D(48, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = Conv2D(48, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = Dropout(0.3)(conv7)

    up8 = Conv2DTranspose(32, 2, strides=2, kernel_initializer='he_normal', padding='same')(conv7)
    conv8 = Concatenate()([up8, conv1])
    conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = Dropout(0.2)(conv8)

    #working with 16 vggfullring
    up9 = Conv2DTranspose(16, 2, strides=2, kernel_initializer='he_normal', padding='same')(conv8)
    conv9 = Concatenate()([up9, conv0])
    conv9 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Dropout(0.1)(conv9)
    
    #--------------------
    c10 = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer='he_normal', name="output_sigmoid_layer")(conv9)
    targets   = Input(shape=(input_shape[0],input_shape[1],1) )
    model = Model(inputs=[ip, weight_ip, targets], outputs=[c10])  
    px_loss = pixel_weighted_cross_entropy(weight_ip, targets, c10)
    model.add_loss(px_loss)

    return model



def get_weighted_unet(input_shape, n_filters = 4, dropout = 0.1, batchnorm = True, is_training=True):
    ip = Input(input_shape, name='img')
    weight_ip = Input(shape= (input_shape[0], input_shape[1],1), name='weighted_ip')

    # adding the layers
    conv1 = Conv2D(n_filters * 1, 3, activation='relu', padding='same', kernel_initializer='he_normal')(ip)
    conv1 = Conv2D(n_filters * 1, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = Dropout(0.1)(conv1)
    mpool1 = MaxPool2D()(conv1)

    conv2 = Conv2D(n_filters * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mpool1)
    conv2 = Conv2D(n_filters * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = Dropout(0.2)(conv2)
    mpool2 = MaxPool2D()(conv2)

    conv3 = Conv2D(n_filters * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mpool2)
    conv3 = Conv2D(n_filters * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = Dropout(0.3)(conv3)
    mpool3 = MaxPool2D()(conv3)

    conv4 = Conv2D(n_filters * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mpool3)
    conv4 = Conv2D(n_filters * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = Dropout(0.4)(conv4)
    mpool4 = MaxPool2D()(conv4)

    conv5 = Conv2D(n_filters * 16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mpool4)
    conv5 = Conv2D(n_filters * 16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = Dropout(0.5)(conv5)

    up6 = Conv2DTranspose(n_filters * 8, 2, strides=2, kernel_initializer='he_normal', padding='same')(conv5)
    conv6 = Concatenate()([up6, conv4])
    conv6 = Conv2D(n_filters * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = Conv2D(n_filters * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = Dropout(0.4)(conv6)

    up7 = Conv2DTranspose(n_filters * 4, 2, strides=2, kernel_initializer='he_normal', padding='same')(conv6)
    conv7 = Concatenate()([up7, conv3])
    conv7 = Conv2D(n_filters * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = Conv2D(n_filters * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = Dropout(0.3)(conv7)

    up8 = Conv2DTranspose(n_filters * 2, 2, strides=2, kernel_initializer='he_normal', padding='same')(conv7)
    conv8 = Concatenate()([up8, conv2])
    conv8 = Conv2D(n_filters * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = Conv2D(n_filters * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = Dropout(0.2)(conv8)

    up9 = Conv2DTranspose(n_filters * 1, 2, strides=2, kernel_initializer='he_normal', padding='same')(conv8)
    conv9 = Concatenate()([up9, conv1])
    conv9 = Conv2D(n_filters * 1, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(n_filters * 1, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Dropout(0.1)(conv9)

    c10 = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer='he_normal', name="output_sigmoid_layer")(conv9)


    targets   = Input(shape=(input_shape[0], input_shape[1],1) )
    model = Model(inputs=[ip, weight_ip, targets], outputs=[c10])
    model.add_loss(pixel_weighted_cross_entropy(weight_ip, targets, c10))

    return model



def get_weighted_unet_edge(input_shape, n_filters = 4, dropout = 0.1, batchnorm = True, is_training=True):
    ip = Input(input_shape, name='img')
    weight_ip = Input(shape= (input_shape[0], input_shape[1],1), name='weighted_ip')

    # adding the layers
    conv1 = Conv2D(n_filters * 1, 3, activation='relu', padding='same', kernel_initializer='he_normal')(ip)
    conv1 = Conv2D(n_filters * 1, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = Dropout(0.1)(conv1)
    mpool1 = MaxPool2D()(conv1)

    conv2 = Conv2D(n_filters * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mpool1)
    conv2 = Conv2D(n_filters * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = Dropout(0.2)(conv2)
    mpool2 = MaxPool2D()(conv2)

    conv3 = Conv2D(n_filters * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mpool2)
    conv3 = Conv2D(n_filters * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = Dropout(0.3)(conv3)
    mpool3 = MaxPool2D()(conv3)

    conv4 = Conv2D(n_filters * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mpool3)
    conv4 = Conv2D(n_filters * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = Dropout(0.4)(conv4)
    mpool4 = MaxPool2D()(conv4)

    conv5 = Conv2D(n_filters * 16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mpool4)
    conv5 = Conv2D(n_filters * 16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = Dropout(0.5)(conv5)

    up6 = Conv2DTranspose(n_filters * 8, 2, strides=2, kernel_initializer='he_normal', padding='same')(conv5)
    conv6 = Concatenate()([up6, conv4])
    conv6 = Conv2D(n_filters * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = Conv2D(n_filters * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = Dropout(0.4)(conv6)

    up7 = Conv2DTranspose(n_filters * 4, 2, strides=2, kernel_initializer='he_normal', padding='same')(conv6)
    conv7 = Concatenate()([up7, conv3])
    conv7 = Conv2D(n_filters * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = Conv2D(n_filters * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = Dropout(0.3)(conv7)

    up8 = Conv2DTranspose(n_filters * 2, 2, strides=2, kernel_initializer='he_normal', padding='same')(conv7)
    conv8 = Concatenate()([up8, conv2])
    conv8 = Conv2D(n_filters * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = Conv2D(n_filters * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = Dropout(0.2)(conv8)

    up9 = Conv2DTranspose(n_filters * 1, 2, strides=2, kernel_initializer='he_normal', padding='same')(conv8)
    conv9 = Concatenate()([up9, conv1])
    conv9 = Conv2D(n_filters * 1, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(n_filters * 1, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Dropout(0.1)(conv9)

    c10 = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer='he_normal', name="output_sigmoid_layer")(conv9)
    targets   = Input(shape=(input_shape[0], input_shape[1],1) )
    model = Model(inputs=[ip, weight_ip, targets], outputs=[c10])
    model.add_loss(mrcnn_mask_edge_loss_graph(c10, targets))
    return model


def get_weighted_unet_both(input_shape, n_filters = 4, dropout = 0.1, batchnorm = True, is_training=True):
    ip = Input(input_shape, name='img')
    weight_ip = Input(shape= (input_shape[0], input_shape[1],1), name='weighted_ip')

    # adding the layers
    conv1 = Conv2D(n_filters * 1, 3, activation='relu', padding='same', kernel_initializer='he_normal')(ip)
    conv1 = Conv2D(n_filters * 1, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = Dropout(0.1)(conv1)
    mpool1 = MaxPool2D()(conv1)

    conv2 = Conv2D(n_filters * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mpool1)
    conv2 = Conv2D(n_filters * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = Dropout(0.2)(conv2)
    mpool2 = MaxPool2D()(conv2)

    conv3 = Conv2D(n_filters * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mpool2)
    conv3 = Conv2D(n_filters * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = Dropout(0.3)(conv3)
    mpool3 = MaxPool2D()(conv3)

    conv4 = Conv2D(n_filters * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mpool3)
    conv4 = Conv2D(n_filters * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = Dropout(0.4)(conv4)
    mpool4 = MaxPool2D()(conv4)

    conv5 = Conv2D(n_filters * 16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mpool4)
    conv5 = Conv2D(n_filters * 16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = Dropout(0.5)(conv5)

    up6 = Conv2DTranspose(n_filters * 8, 2, strides=2, kernel_initializer='he_normal', padding='same')(conv5)
    conv6 = Concatenate()([up6, conv4])
    conv6 = Conv2D(n_filters * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = Conv2D(n_filters * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = Dropout(0.4)(conv6)

    up7 = Conv2DTranspose(n_filters * 4, 2, strides=2, kernel_initializer='he_normal', padding='same')(conv6)
    conv7 = Concatenate()([up7, conv3])
    conv7 = Conv2D(n_filters * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = Conv2D(n_filters * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = Dropout(0.3)(conv7)

    up8 = Conv2DTranspose(n_filters * 2, 2, strides=2, kernel_initializer='he_normal', padding='same')(conv7)
    conv8 = Concatenate()([up8, conv2])
    conv8 = Conv2D(n_filters * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = Conv2D(n_filters * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = Dropout(0.2)(conv8)

    up9 = Conv2DTranspose(n_filters * 1, 2, strides=2, kernel_initializer='he_normal', padding='same')(conv8)
    conv9 = Concatenate()([up9, conv1])
    conv9 = Conv2D(n_filters * 1, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(n_filters * 1, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Dropout(0.1)(conv9)

    c10 = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer='he_normal', name="output_sigmoid_layer")(conv9)


    targets   = Input(shape=(input_shape[0], input_shape[1],1) )
    model = Model(inputs=[ip, weight_ip, targets], outputs=[c10])
    model.add_loss(mrcnn_mask_edge_loss_graph(c10, targets))
    model.add_loss(pixel_weighted_cross_entropy(weight_ip, targets, c10))
    return model


def get_weighted_unet_with_vgg_edge(input_shape, n_filters = 4, dropout = 0.1, batchnorm = True, is_training=True):
    ip = Input(input_shape, name='img')
    weight_ip = Input(shape=(input_shape[0],input_shape[1],1), name='weighted_ip')

    encoder = vgg16.VGG16(input_tensor=ip, include_top=False, input_shape=input_shape)
    z = encoder.output
    encoder_output = z

    #---------------------------------
    conv5 = encoder_output
    conv4 = encoder.get_layer('block5_conv3').output
    conv3 = encoder.get_layer('block4_conv3').output
    conv2 = encoder.get_layer('block3_conv3').output
    conv1 = encoder.get_layer('block2_conv2').output
    conv0 = encoder.get_layer('block1_conv2').output
    
    model_ = Conv2DTranspose(256, (3,3), strides=(2,2), kernel_initializer='he_normal', padding='same')(conv5)
    concat1_ = Concatenate()([model_, conv4])
    model_ = Conv2D(512, 3, activation='relu', strides=1, kernel_initializer='he_normal', padding='same')(concat1_)

    up6 = Conv2DTranspose(64, 2, strides=2, kernel_initializer='he_normal', padding='same')(model_)
    conv6 = Concatenate()([up6, conv3])
    conv6 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = Dropout(0.4)(conv6)

    up7 = Conv2DTranspose(48, 2, strides=2, kernel_initializer='he_normal', padding='same')(conv6)
    conv7 = Concatenate()([up7, conv2])
    conv7 = Conv2D(48, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = Conv2D(48, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = Dropout(0.3)(conv7)

    up8 = Conv2DTranspose(32, 2, strides=2, kernel_initializer='he_normal', padding='same')(conv7)
    conv8 = Concatenate()([up8, conv1])
    conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = Dropout(0.2)(conv8)

    #working with 16 vggfullring
    up9 = Conv2DTranspose(16, 2, strides=2, kernel_initializer='he_normal', padding='same')(conv8)
    conv9 = Concatenate()([up9, conv0])
    conv9 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Dropout(0.1)(conv9)
    
    #--------------------
    c10 = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer='he_normal', name="output_sigmoid_layer")(conv9)
    targets   = Input(shape=(input_shape[0],input_shape[1],1) )
    model = Model(inputs=[ip, weight_ip, targets], outputs=[c10])
    model.add_loss(mrcnn_mask_edge_loss_graph(c10, targets))

    return model


def get_weighted_unet_with_vgg_both(input_shape, n_filters = 4, dropout = 0.1, batchnorm = True, is_training=True):
    ip = Input(input_shape, name='img')
    weight_ip = Input(shape=(input_shape[0],input_shape[1],1), name='weighted_ip')

    encoder = vgg16.VGG16(input_tensor=ip, include_top=False, input_shape=input_shape)
    z = encoder.output
    encoder_output = z

    #---------------------------------
    conv5 = encoder_output
    conv4 = encoder.get_layer('block5_conv3').output
    conv3 = encoder.get_layer('block4_conv3').output
    conv2 = encoder.get_layer('block3_conv3').output
    conv1 = encoder.get_layer('block2_conv2').output
    conv0 = encoder.get_layer('block1_conv2').output
    
    model_ = Conv2DTranspose(256, (3,3), strides=(2,2), kernel_initializer='he_normal', padding='same')(conv5)
    concat1_ = Concatenate()([model_, conv4])
    model_ = Conv2D(512, 3, activation='relu', strides=1, kernel_initializer='he_normal', padding='same')(concat1_)
    
    up6 = Conv2DTranspose(64, 2, strides=2, kernel_initializer='he_normal', padding='same')(model_)
    conv6 = Concatenate()([up6, conv3])
    conv6 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = Dropout(0.4)(conv6)

    up7 = Conv2DTranspose(48, 2, strides=2, kernel_initializer='he_normal', padding='same')(conv6)
    conv7 = Concatenate()([up7, conv2])
    conv7 = Conv2D(48, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = Conv2D(48, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = Dropout(0.3)(conv7)

    up8 = Conv2DTranspose(32, 2, strides=2, kernel_initializer='he_normal', padding='same')(conv7)
    conv8 = Concatenate()([up8, conv1])
    conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = Dropout(0.2)(conv8)

    #working with 16 vggfullring
    up9 = Conv2DTranspose(16, 2, strides=2, kernel_initializer='he_normal', padding='same')(conv8)
    conv9 = Concatenate()([up9, conv0])
    conv9 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Dropout(0.1)(conv9)
    
    #--------------------
    c10 = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer='he_normal', name="output_sigmoid_layer")(conv9)
    targets   = Input(shape=(input_shape[0],input_shape[1],1) )
    model = Model(inputs=[ip, weight_ip, targets], outputs=[c10])
    model.add_loss(mrcnn_mask_edge_loss_graph(c10, targets))
    model.add_loss(pixel_weighted_cross_entropy(weight_ip, targets, c10))

    return model
