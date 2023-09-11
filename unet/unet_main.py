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
    try:
        CHANNELS = data_params['channels']
    except:
        CHANNELS = 1
    try:
        NORMALIZE = data_params['normalize']
    except:
        NORMALIZE = True

    try:
        AUGMENT = data_params['augment']
    except:
        AUGMENT = True

    try:
        TRANSFER = data_params['transfer']
    except:
        TRANSFER = False

    try:
        FULL_RING = train_params['full_ring']
    except:
        FULL_RING = False

    try:
        LOSS_FUNCTION = unet_params['loss_function']
    except:
        LOSS_FUNCTION = 'weighted'

    if FULL_RING:

        dataset = OtolithDataset()
        dataset.load_otolith_data('{}/train_paramfull_999/'.format(settings['dataset']), settings=settings)
        dataset.prepare()

        val = OtolithDataset(mode="flip")
        val.load_otolith_data('{}/valid_paramfull_999/'.format(settings['dataset']), settings=settings )
        val.prepare()
    else:
        dataset = OtolithDataset()
        if idr == 999:
            dataset.load_otolith_data('{}/train_param_999/'.format(settings['dataset']), settings=settings )
        else:
            dataset.load_otolith_data('{}/train_{}_{}/'.format(settings['dataset'], settings['split_name'] , idr), settings=settings)
        dataset.prepare()

        val = OtolithDataset(mode="flip")
        if idr == 999:
            val.load_otolith_data('{}/valid_param_999/'.format(settings['dataset']), settings=settings)
        else:
            val.load_otolith_data('{}/valid_{}_{}/'.format(settings['dataset'], settings['split_name'], idr), settings=settings)
        val.prepare()

    config = OtolithConfig()
    
    
    all_x= []
    all_y = []
    
    raw_X_tr = []
    raw_y_tr = []
    
    
    all_weights = []
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

        all_x.append( image )
        all_y.append( (mask, weights) )   
        
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

        all_x.append( image )
        all_y.append( (mask, weights) )   
        
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

    
    input_shape = (512, 512, CHANNELS)
    adam = optimizers.Adam(lr=0.0004, decay=0.0)
    if TRANSFER:
        if LOSS_FUNCTION == 'edge':
            model = modellib.get_weighted_unet_with_vgg_edge(input_shape, n_filters=4, dropout=0.15, batchnorm=True)
        elif LOSS_FUNCTION == 'weighted':
            model = modellib.get_weighted_unet_with_vgg(input_shape, n_filters=4, dropout=0.15, batchnorm=True)
        else:
            model = modellib.get_weighted_unet_with_vgg_both(input_shape, n_filters=4, dropout=0.15, batchnorm=True)
    else:
        if LOSS_FUNCTION == 'edge':
            model = modellib.get_weighted_unet_edge(input_shape, n_filters=4, dropout=0.15, batchnorm=True)
        elif LOSS_FUNCTION == 'weighted':
            model = modellib.get_weighted_unet(input_shape, n_filters=4, dropout=0.15, batchnorm=True)
        else:
            model = modellib.get_weighted_unet_both(input_shape, n_filters=4, dropout=0.15, batchnorm=True)


    model.compile(optimizer=adam, metrics=['accuracy', 'mse', 'mape']) # loss=my_loss
    for layer in model.layers:
        layer.trainable = True

    print("training")
    #earlystop = EarlyStopping(patience=100)
    checkpoint_save = modellib.CheckpointSaver(measurement='loss', name=name, settings=settings)
    results = model.fit_generator(train_gen, validation_data=val_gen, validation_steps=len(val_array), steps_per_epoch=50, epochs=200, verbose=1, callbacks=[checkpoint_save])
    
    model.save("{}/{}/{}.h5".format(settings['dataset'], name, name))


def evaluate(name='unet', full_ring_type=False, data_params={}, settings={}):

    random.seed(101)
    idr = settings['idr']
    domain = settings['dataset']
    
    try:
        channels = data_params['channels']
    except:
        channels = 1

    try:
        normalize = data_params['normalize']
    except:
        normalize = True
    
    if 'source_dataset' in settings:
        model_annulus = load_model('{}/{}/{}{}.h5'.format(settings['source_dataset'], name, name, settings['checkpoint']), compile=False)
    else:
        model_annulus = load_model('{}/{}/{}{}.h5'.format(settings['dataset'], name, name, settings['checkpoint']), compile=False)


    config = InferenceConfig()
    valid = InferenceDataset()

    all_valid = []
    if settings['search_mode']:
        if full_ring_type:
            valid.load_otolith_data('{}/images/'.format(domain), 2, exclude="train_paramfull_999", settings=settings)
        else:
            valid.load_otolith_data('{}/images/'.format(domain), 2, exclude="train_param_999", settings=settings)
        valid.prepare()

    else:
        #BALTIC TEMP
        if domain == 'datasets_baltic':
            valid = InferenceDataset()
            valid.load_otolith_data('{}/images'.format(domain), 1, exclude="train_{}_{}".format(settings['split_name'], idr), settings=settings)
            valid.prepare()
            for image_id in valid.image_ids:
                all_valid.append([image_id, valid] )
        else:
            valid.load_otolith_data('{}/images/'.format(domain), 2, exclude="train_{}_{}".format(settings['split_name'], idr), settings=settings)
            valid.prepare()
            for image_id in valid.image_ids:
                all_valid.append([image_id, valid] )

    if domain == 'datasets_baltic':
        with open('datasets_baltic/all_data_map.json') as fson:
            data_map = json.load(fson)
        with open('datasets_baltic/all_bounds_1155.json') as fson:
            all_bounds_1155 = json.load(fson)

    exact_count = 0
    offbyone_count = 0

    exact_reading = 0
    offbyone_reading = 0
    
    pred_lines = []
    count_lines = []

    for item_idx, validation_data in enumerate(all_valid):
        
        image_id = validation_data[0]
        valid = validation_data[1]
        
        image = valid.load_image(image_id)
        multi_masks, class_ids = valid.load_mask(image_id)

        info = valid.image_info[image_id]
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
                         
        image, window, scale, padding = valid.resize_image(
            img_new, 
            min_dim=config.IMAGE_MIN_DIM,
            max_dim=config.IMAGE_MAX_DIM,
            padding=True
        )
        
        if domain == 'datasets_north':
            nucleus_mask = valid.resize_mask(nucleus_mask, scale, padding)
        else:
            nucleus_mask = None
            whole_mask = valid.resize_mask(mask_new, scale, padding)
            
        cx, cy = get_center(whole_mask, nucleus_mask)
        
        img_gray = skimage.color.rgb2gray(image)
        img = skimage.color.gray2rgb(img_gray)
        if channels == 1:
            Z_val =  np.zeros( ( 1, config.IMAGE_MAX_DIM,config.IMAGE_MAX_DIM,1), dtype=np.float32)
            Z_val[0,:,:,0] = img[:,:,0]
        else:
            Z_val =  np.zeros( ( 1, config.IMAGE_MIN_DIM,config.IMAGE_MIN_DIM,3), dtype=np.float32)
            Z_val[0,:,:,:] = img

        preds_test = model_annulus.predict(Z_val[0:1], verbose=1)
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
        output_file = "{}/{}/zntbn_{}_{}_of_{}.txt".format(domain, name, exact_reading, offbyone_reading, len(all_valid) )
    else:
        output_file = "{}/{}/zntrd_{}_{}_of_{}.txt".format(domain, name, exact_reading, offbyone_reading, len(all_valid) )
    with open(output_file, 'w') as fout:
        fout.write("\n".join(pred_lines))
