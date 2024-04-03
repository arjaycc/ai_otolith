"""
Main Code Reference:
    Mask R-CNN
    Copyright (c) 2017 Matterport, Inc.
    Licensed under the MIT License (see LICENSE for details)
    Written by Waleed Abdulla
    
    Modifications
    Copyright (c) 2018/2019 Roland S. Zimmermann, Julien Siems
    Licensed under the MIT License (see LICENSE for details)
"""

import tensorflow as tf
import json
from skimage.morphology import label
from scipy.ndimage.morphology import distance_transform_edt
import numpy as np
import glob
import skimage.draw
import os
import re
import cv2
import sys

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)

from mrcnn import utils
from mrcnn import config
from mrcnn import model as modellib

from .mrcnn_aux import *


def train(name='mrcnn', data_params={}, edge_params={}, train_params={}, settings={}):
    
    if settings:
        mode = settings['run_type']
        idr = settings['idr']
        domain = settings['dataset']
    else:
        mode = 'test'
        idr = 999
        domain = 'datasets_north'
    
    config = OtolithConfig()
    
    config.USE_MINI_MASK = data_params["mini_mask"]
    config.IMAGE_MIN_DIM = data_params["img_size"]
    config.IMAGE_MAX_DIM = data_params["img_size"]
    
    config.RPN_NMS_THRESHOLD = train_params["rpn_nms"]
    config.DETECTION_MIN_CONFIDENCE = train_params["detection_confidence"]
    config.DETECTION_NMS_THRESHOLD = train_params["detection_nms"]
    
    config.EDGE_LOSS_SMOOTHING = edge_params["smoothing"]
    config.EDGE_LOSS_FILTERS = edge_params["edge_loss"]
    config.EDGE_LOSS_WEIGHT_FACTOR = edge_params["weight_factor"]
    
    
    if mode == 'both' or mode == 'train':
        dataset = OtolithDataset()
        if idr == 999:
            dataset.load_otolith_data('{}/train_param_{}/'.format(domain, idr), settings=settings)
        else:
            dataset.load_otolith_data('{}/train_{}_{}/'.format(domain,settings['split_name'], idr), settings=settings)
        dataset.prepare()
        print(len(dataset.image_ids))

        val = OtolithDataset(mode="flip")
        if idr == 999:
            val.load_otolith_data('{}/valid_param_{}/'.format(domain, idr), settings=settings )
        else:
            val.load_otolith_data('{}/valid_{}_{}/'.format(domain, settings['split_name'], idr), settings=settings)
        val.prepare()

    if mode == "both" or mode == 'train':
        model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir='{}/{}/'.format(domain, name))
    
    
    ROOT_DIR = os.getcwd()
    COCO_MODEL_PATH = ''
    if settings['base'] == 'coco':
        COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    elif settings['base'] == 'north':
        if settings['continual'] == 0:
            COCO_MODEL_PATH = '{}/{}/mrcnn_checkpoint.h5'.format(domain, "mrcnn_randsub{}run1_6".format(settings['base_id'])) # fixed for baltic
        else:
            if idr > 0:
                prev_id = idr - 1
                #COCO_MODEL_PATH = '{}/{}/mrcnn_checkpoint.h5'.format(domain, "mrcnn_in{}normalbasednorth{}run1_2".format(settings['base_id'], prev_id))
                COCO_MODEL_PATH = '{}/{}/mrcnn_checkpoint.h5'.format(domain, "mrcnn_{}{}run1_2".format(settings['run_label'], prev_id))
            else:
                COCO_MODEL_PATH = '{}/{}/mrcnn_checkpoint.h5'.format(domain, "mrcnn_randsub{}run1_6".format(settings['base_id'])) # fixed for baltic
    elif settings['base'] == 'baltic':
        COCO_MODEL_PATH = '{}/{}/mrcnn_checkpoint.h5'.format(domain, "mrcnn_randfold{}run1_2".format(settings['base_id']) )
    elif settings['base'] == 'none':
        COCO_MODEL_PATH = ''

    #prev_id = idr - 1
    #COCO_MODEL_PATH = '{}/{}/mrcnn_checkpoint.h5'.format(domain, "mrcnn_in0reloadbasednorth{}run1_2".format(prev_id))
    #COCO_MODEL_PATH = '{}/{}/mrcnn_checkpoint.h5'.format(domain, "mrcnn_randsub1run1_6")
    #COCO_MODEL_PATH = ''
    #COCO_MODEL_PATH = '{}/{}/mrcnn_checkpoint.h5'.format(domain, "mrcnn_randfold1run1_2")
    if settings['base'] != 'none':
        if mode == "both" or mode == 'train':
            model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
    
    with tf.device('/gpu:0'):
        if mode == "both" or mode == 'train':
            model.train(dataset, val, 
                    learning_rate=config.LEARNING_RATE, 
                    epochs=100, 
                    layers='heads')
            model.train(dataset, val, 
                    learning_rate=config.LEARNING_RATE/10.0, 
                    epochs=200, 
                    layers='all')
            model.keras_model.save_weights('{}/{}/mrcnn_last.h5'.format(domain, name))

            with open('{}/{}/setting.json'.format(settings['dataset'], name), 'w') as fbase:
                json.dump(settings, fbase)

        with open('{}/{}/stat.txt'.format(domain, name), 'w') as fbase:
            fbase.write('cpath: {}'.format(COCO_MODEL_PATH))
        if mode == 'train':
            return

        inference_config = InferenceConfig()
        inference_config.RPN_NMS_THRESHOLD = train_params["rpn_nms"]
        inference_config.DETECTION_MIN_CONFIDENCE = train_params["detection_confidence"]
        inference_config.DETECTION_NMS_THRESHOLD = train_params["detection_nms"]
        model = modellib.MaskRCNN(mode="inference", 
                                  config=inference_config,
                                  model_dir='{}/{}/'.format(domain, name))

        if 'source_dataset' in settings:
            OTO_MODEL_PATH = os.path.join(ROOT_DIR, '{}/{}/mrcnn_checkpoint.h5'.format(settings['source_dataset'], name) )
        else:
            OTO_MODEL_PATH = os.path.join(ROOT_DIR, '{}/{}/mrcnn_checkpoint.h5'.format(domain, name) )

        model.load_weights(OTO_MODEL_PATH, by_name=True)

        all_valid = []
        if domain == 'datasets_north':
            valid = InferenceDataset()
            if idr == 999:
                valid.load_otolith_data('{}/images/'.format(domain), 2, exclude="train_param_{}".format(idr), settings=settings)
            else:
                valid.load_otolith_data('{}/images/'.format(domain), 2, exclude="train_{}_{}".format(settings['split_name'], idr), settings=settings)
            valid.prepare()
            
            for image_id in valid.image_ids:
                all_valid.append([image_id, valid] )
        else:
            valid = InferenceDataset()
            valid.load_otolith_data('{}/images'.format(domain), 1, exclude="train_{}_{}".format(settings['split_name'], idr), settings=settings)
            valid.prepare()
            for image_id in valid.image_ids:
                all_valid.append([image_id, valid] )

        if domain == 'datasets_baltic':
            with open('datasets_baltic/all_data_map.json') as fson:
                data_map = json.load(fson)
            with open('datasets_baltic/all_bounds_1155.json') as fson:
                all_bounds_1155 = json.load(fson)

        print("evaluating")
        exact_reading = 0
        offbyone_reading = 0
        
        pred_lines = []
        for item_idx, validation_data in enumerate(all_valid):
            image_id = validation_data[0]
            valid = validation_data[1]
            
            image = valid.load_image(image_id)
            multi_masks, class_ids = valid.load_mask(image_id)
            
            info = valid.image_info[image_id]
            image_path = info['path']
            fname = image_path.replace('\\','/').split('/')[-1]
        
            whole_mask = multi_masks[0]
            if domain == 'datasets_north':
                nucleus_mask = multi_masks[1]
                manual_age = int(fname.split('_age_')[1].split('.')[0])
            else:
                manual_age = int(data_map[fname])
               
            img_new  = image.copy()
            mask_new = whole_mask.copy()

            if 'brighten' in settings:
                img_new = modify_image(img_new, mask_new, settings)

            original_image, window, scale, padding, _ = utils.resize_image(
                img_new, 
                min_dim=inference_config.IMAGE_MIN_DIM,
                max_dim=inference_config.IMAGE_MAX_DIM,
            )

            if domain == 'datasets_north':
                nucleus_mask = utils.resize_mask(nucleus_mask, scale, padding)
                whole_mask = None
            else:
                whole_mask = utils.resize_mask(mask_new, scale, padding)
                nucleus_mask = None
            cx, cy = get_center(whole_mask, nucleus_mask)
                
            results = model.detect([original_image], verbose=1)
            ai_reading = count_detected_masks(results, cx, cy)

            if abs(ai_reading - manual_age) < 1:
                exact_reading += 1 
            if abs(ai_reading - manual_age) < 2:
                offbyone_reading += 1 
            pred_lines.append("{},{},{}".format(fname, ai_reading, manual_age))

        if 'brighten' in settings:
            output_file = "{}/{}/mrcnn_br_{}_{}_of_{}.txt".format(settings['dataset'], name, exact_reading, offbyone_reading, len(all_valid) )
        else:
            output_file = "{}/{}/mrcnn_rd_{}_{}_of_{}.txt".format(settings['dataset'], name, exact_reading, offbyone_reading, len(all_valid) )
        with open(output_file, 'w') as fout:
            fout.write("\n".join(pred_lines))
