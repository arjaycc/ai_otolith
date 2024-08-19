"""
U-Net Code References:
    pixel-weighted cross-entropy
        https://github.com/keras-team/keras/issues/6261
    unet-weight-map computation:
        https://stackoverflow.com/questions/50255438/pixel-wise-loss-weight-for-image-segmentation-in-keras
        https://arxiv.org/pdf/1505.04597.pdf
    U-Net construction:
        https://stackoverflow.com/questions/58134005/keras-u-net-weighted-loss-implementation (->> which points to 
            another reference (https://jaidevd.com/posts/weighted-loss-functions-for-instance-segmentation/)
        UNET-TGS: https://medium.com/@harshall.lamba/understanding-semantic-segmentation-with-unet-6be4f42d4b47 
        https://github.com/nikhilroxtomar/Unet-with-Pretrained-Encoder/tree/master
        https://www.kaggle.com/code/aithammadiabdellatif/vgg16-u-net
        https://www.kaggle.com/code/basu369victor/transferlearning-and-unet-to-segment-rocks-on-moon
        https://www.kaggle.com/code/mistag/train-keras-u-net-mobilenetv2

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
import sys
import random
import pandas as pd
import numpy as np
import json
import re
import math
import glob
import cv2
import tensorflow as tf
from tensorflow import keras
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras import optimizers
from skimage import img_as_ubyte

from .unet_aux import *

ROOT_DIR = os.path.abspath("../")
print(ROOT_DIR)
sys.path.append(ROOT_DIR)
from unet import model as modellib
from mrcnn import utils

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



def train(name='unet', data_params={}, unet_params={}, train_params={}, settings={}):
    random.seed(101)
    
    idr = settings['idr']
    HYPERPARAMS_EXPERIMENT = False
    if idr == 999:
        HYPERPARAMS_EXPERIMENT = True
    #try:
    #    NORMALIZE = data_params['normalize']
    #except:
    #    NORMALIZE = True

    try:
        AUGMENT = data_params['augment']
    except:
        AUGMENT = True

    try:
        FULL_RING = train_params['full_ring']
    except:
        FULL_RING = False


    unet_model = modellib.UNetModel(mode="training")
    unet_model.get_unet(settings, data_params, unet_params)
    (_x, _y,CHANNELS) = unet_model.input_shape
    
    
    raw_X_tr = []
    raw_y_tr = []
    raw_X_val = []
    raw_y_val = []

    data_limit = 99999
    epoch_limit = 200

    config = OtolithConfig()


    dataset_limit = int(settings['new_items'])
    epoch_limit = 40

    dataset = OtolithDataset()
    dataset.load_otolith_data('{}/train_{}_{}/'.format(settings['dataset'], 
                            settings['split_name'] , 0), settings=settings)
    dataset.prepare()
    val = OtolithDataset()
    val.load_otolith_data('{}/train_{}_{}/'.format('datasets_north', 'randsub', '0'), settings=settings)
    val.prepare()

    rehearsal_set = OtolithDataset()
    rehearsal_set.load_otolith_data('{}/train_{}_{}/'.format('datasets_north', 'randsub', '0'), settings=settings)
    rehearsal_set.prepare()

    image_ids = rehearsal_set.image_ids[:int(settings['old_items'])] # [:132] 
    for idx, image_id in enumerate(image_ids):
        image = rehearsal_set.load_image(image_id)
        
        mask, num_masks = rehearsal_set.load_mask(image_id)
        image, window, scale, padding = val.resize_image(
        image, 
        min_dim=config.IMAGE_MIN_DIM,
        max_dim=config.IMAGE_MAX_DIM,
        padding=True
        )
        
        mask = rehearsal_set.resize_mask(mask, scale, padding)    
        wt_map = np.zeros([config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM], dtype=np.uint8)
        wt_map[:,:] = mask[:,:,0]
        weights = rehearsal_set.load_weights(wt_map, unet_params)
        raw_X_tr.append( image )
        raw_y_tr.append( (mask, weights) )   


     
    image_ids = dataset.image_ids[:dataset_limit] 
    for idx, image_id in enumerate(image_ids):
        image = dataset.load_image(image_id)
        
        mask, num_masks = dataset.load_mask(image_id)
        image, window, scale, padding = dataset.resize_image(
        image, 
        min_dim=config.IMAGE_MIN_DIM,
        max_dim=config.IMAGE_MAX_DIM,
        padding=True
        )
        mask = dataset.resize_mask(mask, scale, padding)        
        wt_map = np.zeros([config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM], dtype=np.uint8)
        wt_map[:,:] = mask[:,:,0]
        weights = dataset.load_weights(wt_map, unet_params)
 
        raw_X_tr.append( image )
        raw_y_tr.append( (mask, weights) )   
        
        print("train1 data done {}".format(idx))


    image_ids = val.image_ids
    for idx, image_id in enumerate(image_ids):
        image = val.load_image(image_id)
        mask, num_masks = val.load_mask(image_id)
        image, window, scale, padding = val.resize_image(
        image, 
        min_dim=config.IMAGE_MIN_DIM,
        max_dim=config.IMAGE_MAX_DIM,
        padding=True
        )
        
        mask = val.resize_mask(mask, scale, padding)
        wt_map = np.zeros([config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM], dtype=np.uint8)
        wt_map[:,:] = mask[:,:,0]
        weights = val.load_weights(wt_map, unet_params)
 
        raw_X_val.append( image )
        raw_y_val.append( (mask, weights) )  
        
        print("val data done {}".format(idx))

    train_array = []
    for idx, image in enumerate(raw_X_tr):
        if CHANNELS == 3:
            img_gray = skimage.color.rgb2gray(image)
            img = skimage.color.gray2rgb(img_gray)
        else:
            img = skimage.color.rgb2gray(image)

        msk = raw_y_tr[idx][0]
        wt = raw_y_tr[idx][1]
        train_array.append([img, msk, wt, 1])


    val_array = []
    for idx, image in enumerate(raw_X_val):
        if CHANNELS == 3:
            img_gray = skimage.color.rgb2gray(image)
            img = skimage.color.gray2rgb(img_gray)
        else:
            img = skimage.color.rgb2gray(image)
        
        msk = raw_y_val[idx][0]
        wt = raw_y_val[idx][1]
        val_array.append([img, msk, wt, 0])
    
    train_augmenters = []
    val_augmenters = []
    if AUGMENT:
        train_augmenters = modellib.get_augmentation_set()
        val_augmenters = modellib.get_augmentation_set()

    train_gen = data_generator(train_array, config, augmentations=train_augmenters, batch_size=1, channels=CHANNELS)
    val_gen = data_generator(val_array, config, augmentations=val_augmenters, batch_size=1, validation=True, channels=CHANNELS)

    
    adam = optimizers.Adam(lr=0.0004, decay=0.0)
    unet_model.compile(optimizer=adam, metrics=['accuracy', 'mse', 'mape'])

    print("training")
    #earlystop = EarlyStopping(patience=100)
    checkpoint_save = modellib.RehearsalPlotSaver(measurement='loss', name=name, settings=settings)
    unet_model.fit(train_gen, val_gen, val_steps=len(val_array), steps_per_epoch=50, epochs=epoch_limit, verbose=1, callbacks=[checkpoint_save])
    
    save_dir = "{}/{}/{}.h5".format(settings['dataset'], name, name)
    unet_model.save(save_dir)
    with open('{}/{}/setting.json'.format(settings['dataset'], name), 'w') as fbase:
        json.dump(settings, fbase)


