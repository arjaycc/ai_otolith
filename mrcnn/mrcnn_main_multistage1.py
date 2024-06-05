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
from sklearn.metrics import jaccard_score, f1_score, accuracy_score, precision_score, recall_score

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
    if 'base' in settings:
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
    else:
        COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
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

        exact_reading = 0
        offbyone_reading = 0 
        pred_lines = []

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


        ff = glob.glob("{}/images_remain/*.png".format(domain))
        age_distances = {}
        for imgfile in ff:
            print(imgfile)
            image_name = imgfile.replace("\\", "/").split("/")[-1]
            imgraw = skimage.io.imread("{}/images_remain/{}".format(domain,image_name))

            sq_img, window, scale, padding, _ = utils.resize_image(
                imgraw, 
                min_dim=inference_config.IMAGE_MIN_DIM,
                max_dim=inference_config.IMAGE_MAX_DIM,
            )
            sqmaskraw = utils.resize_mask(maskraw, scale, padding)
            sqmaskraw = sqmaskraw[sqmaskraw>=1].astype(np.uint8)

            print(image_name)
            img_gray = skimage.color.rgb2gray(sq_img)
            img = skimage.color.gray2rgb(img_gray)
            if CHANNELS == 1:
                Z_test =  np.zeros( ( 1, config.IMAGE_MAX_DIM,config.IMAGE_MAX_DIM,1), dtype=np.float32)
                Z_test[0,:,:,0] = img[:,:,0]
            else:
                Z_test =  np.zeros( ( 1, config.IMAGE_MIN_DIM,config.IMAGE_MIN_DIM,3), dtype=np.float32)
                Z_test[0,:,:,:] = img

            preds_test = unet_model.model.predict(Z_test[0:1], verbose=1)
            preds_test_t = (preds_test > 0.50).astype(np.uint8)
            print(preds_test.shape)
            ypred = preds_test_t[0].squeeze()
            ythresh = np.zeros([ypred.shape[0], ypred.shape[1],1], dtype=np.uint8)
            ythresh[:,:,0] = ypred[:,:]
            whole_mask = cv2.dilate(ythresh, np.ones( (4,4), np.uint8 ), iterations=10)
            ncontours, hierarchy = cv2.findContours(whole_mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            whole_ctr = max(ncontours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(whole_ctr)
            with open("{}/{}/output/bbox_{}.json".format(domain, name, image_name), "w") as fout:
                json.dump([x,y,w,h], fout, indent=4)
