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
    
    if HYPERPARAMS_EXPERIMENT:
        if FULL_RING:

            dataset = OtolithDataset()
            dataset.load_otolith_data('{}/train_paramfull_999/'.format(settings['dataset']), settings=settings)
            dataset.prepare()

            val = OtolithDataset(mode="flip")
            val.load_otolith_data('{}/valid_paramfull_999/'.format(settings['dataset']), settings=settings )
            val.prepare()
        else:
            dataset = OtolithDataset()
            dataset.load_otolith_data('{}/train_param_999/'.format(settings['dataset']), settings=settings )
            dataset.prepare()

            val = OtolithDataset(mode="flip")
            val.load_otolith_data('{}/valid_param_999/'.format(settings['dataset']), settings=settings)
            val.prepare()
    else:
        dataset = OtolithDataset()
        dataset.load_otolith_data('{}/train_{}_{}/'.format(settings['dataset'], 
                                settings['split_name'] , idr), settings=settings)
        dataset.prepare()

        val = OtolithDataset(mode="flip")
        val.load_otolith_data('{}/valid_{}_{}/'.format(settings['dataset'], 
                                settings['split_name'], idr), settings=settings)
        val.prepare()

    config = OtolithConfig()
     
    raw_X_tr = []
    raw_y_tr = [] 
    image_ids = dataset.image_ids
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
        
        print("train data done {}".format(idx))

    raw_X_val = []
    raw_y_val = []
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
    checkpoint_save = modellib.CheckpointSaver(measurement='loss', name=name, settings=settings)
    unet_model.fit(train_gen, val_gen, val_steps=len(val_array), steps_per_epoch=50, epochs=200, verbose=1, callbacks=[checkpoint_save])
    
    save_dir = "{}/{}/{}.h5".format(settings['dataset'], name, name)
    unet_model.save(save_dir)
    with open('{}/{}/setting.json'.format(settings['dataset'], name), 'w') as fbase:
        json.dump(settings, fbase)


def evaluate(name='unet', full_ring_type=False, data_params={}, settings={}):

    random.seed(101)
    idr = settings['idr']
    domain = settings['dataset']
    
    HYPERPARAMS_EXPERIMENT = False
    if idr == 999:
        HYPERPARAMS_EXPERIMENT = True
    #try:
    #    normalize = data_params['normalize']
    #except:
    #    normalize = True
    unet_model = modellib.UNetModel(mode="testing")
    if 'source_dataset' in settings:
        model_dir = '{}/{}/{}{}.h5'.format(settings['source_dataset'], name, name, settings['checkpoint'])
    else:
        model_dir = '{}/{}/{}{}.h5'.format(settings['dataset'], name, name, settings['checkpoint'])

    (_x, _y, CHANNELS) = unet_model.load(model_dir, settings, data_params, {})


    config = InferenceConfig()
    all_test = []
    if HYPERPARAMS_EXPERIMENT or settings['search_mode']:
        test_ds = InferenceDataset()
        if full_ring_type:
            test_ds.load_otolith_data('{}/images/'.format(domain), 2, exclude="train_paramfull_999", settings=settings)
        else:
            test_ds.load_otolith_data('{}/images/'.format(domain), 2, exclude="train_param_999", settings=settings)
        test_ds.prepare()

    else:
        test_ds = InferenceDataset()
        if domain == 'datasets_baltic':
            test_ds.load_otolith_data('{}/images'.format(domain), 1, exclude="train_{}_{}".format(settings['split_name'], idr), settings=settings)
            #for image_id in test_ds.image_ids:
            #    all_test.append([image_id, test_ds] )
        else:
            test_ds.load_otolith_data('{}/images/'.format(domain), 2, exclude="train_{}_{}".format(settings['split_name'], idr), settings=settings)
        
        test_ds.prepare()

    if domain == 'datasets_baltic':
        with open('datasets_baltic/all_data_map.json') as fson:
            data_map = json.load(fson)

    exact_count = 0
    offbyone_count = 0

    exact_reading = 0
    offbyone_reading = 0
    
    pred_lines = []
    count_lines = []

    for item_idx, image_id in enumerate(test_ds.image_ids):
         
        image = test_ds.load_image(image_id)
        multi_masks, class_ids = test_ds.load_mask(image_id)

        info = test_ds.image_info[image_id]
        image_path = info['path']
        fname = image_path.replace('\\','/').split('/')[-1]

        if domain == 'datasets_north':
            nucleus_mask = multi_masks[1]
            manual_age = int(fname.split('_age_')[1].split('.')[0])
        else:
            manual_age = int(data_map[fname])
        
        whole_mask = multi_masks[0]
        img_new  = image.copy()
        mask_new = whole_mask.copy()

        if 'brighten' in settings:
            img_new = modify_image(img_new, mask_new, settings)
                         
        image, window, scale, padding = test_ds.resize_image(
            img_new, 
            min_dim=config.IMAGE_MIN_DIM,
            max_dim=config.IMAGE_MAX_DIM,
            padding=True
        )
        
        if domain == 'datasets_north':
            nucleus_mask = test_ds.resize_mask(nucleus_mask, scale, padding)
        else:
            nucleus_mask = None
            whole_mask = test_ds.resize_mask(mask_new, scale, padding)
            
        cx, cy = get_center(whole_mask, nucleus_mask)
        
        img_gray = skimage.color.rgb2gray(image)
        img = skimage.color.gray2rgb(img_gray)
        if CHANNELS == 1:
            Z_test =  np.zeros( ( 1, config.IMAGE_MAX_DIM,config.IMAGE_MAX_DIM,1), dtype=np.float32)
            Z_test[0,:,:,0] = img[:,:,0]
        else:
            Z_test =  np.zeros( ( 1, config.IMAGE_MIN_DIM,config.IMAGE_MIN_DIM,3), dtype=np.float32)
            Z_test[0,:,:,:] = img

        preds_test = unet_model.model.predict(Z_test[0:1], verbose=1)
        preds_test_t = (preds_test > 0.50).astype(np.uint8)
            
        item =  preds_test_t[0].squeeze()
        ai_reading = count_prediction_output(domain, item, cx, cy)
 
        print("Prediction ======= {}".format(ai_reading))
        pred_lines.append("{},{},{}".format(fname, ai_reading, manual_age))

        if abs(ai_reading - manual_age) < 1:
            exact_reading += 1 
        if abs(ai_reading - manual_age) < 2:
            offbyone_reading += 1 

    if 'brighten' in settings:
        output_file = "{}/{}/unet_br_{}_{}_of_{}.txt".format(domain, name, exact_reading, offbyone_reading, len(test_ds.image_ids) )
    else:
        output_file = "{}/{}/unet_rd_{}_{}_of_{}.txt".format(domain, name, exact_reading, offbyone_reading, len(test_ds.image_ids) )
    with open(output_file, 'w') as fout:
        fout.write("\n".join(pred_lines))
