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

        test_ds = InferenceDataset()

        if settings['mask_score'] == 'outer':
            test_ds.load_otolith_data('{}/images_raw'.format(domain), 1, exclude="train_{}_{}".format(settings['split_name'], idr), settings=settings)
        elif settings['mask_score'] == 'nucleus':
            nucleus_annotations = {}
            if 'mask_score' in settings and settings['mask_score'] == 'nucleus':
                json_files = glob.glob('{}/images/1_core/*.json'.format(domain))
                annotations = {}
                for json_file in json_files:
                    _annotations = json.load(open(json_file))
                    annotations.update(_annotations)
                annotations = list(annotations.values())
                annotations = [a for a in annotations if a['regions']]
                for a in annotations:
                    nucleus_annotations[a['filename']] = a
            
            if domain == 'datasets_baltic':
                test_ds.load_otolith_data('{}/images'.format(domain), 1, exclude="train_{}_{}".format(settings['split_name'], idr), settings=settings)
            else:
                test_ds.load_otolith_data('{}/images/'.format(domain), 2, exclude="train_{}_{}".format(settings['split_name'], idr), settings=settings)
        test_ds.prepare()


        if domain == 'datasets_baltic':
            with open('datasets_baltic/all_data_map.json') as fson:
                data_map = json.load(fson)


        print("evaluating")
        exact_reading = 0
        offbyone_reading = 0
        
        pred_lines = []
        for item_idx, image_id in enumerate(test_ds.image_ids):

            
            image = test_ds.load_image(image_id)
            multi_masks, class_ids = test_ds.load_mask(image_id)
            
            info = test_ds.image_info[image_id]
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
#                 whole_mask = None
            else:
                nucleus_mask = None
            whole_mask = utils.resize_mask(mask_new, scale, padding)
                
            results = model.detect([original_image], verbose=1)

            if settings['mask_score'] == 'nucleus':
                a = nucleus_annotations[fname]
                polygons = [r['shape_attributes'] for r in a['regions']]
                target = np.zeros([img_new.shape[0], img_new.shape[1], 1], dtype=np.uint8)
                for idx, poly in enumerate(polygons):
                    rr, cc = skimage.draw.polygon(poly['all_points_y'], poly['all_points_x'])
                    target[rr, cc, 0] = 1
                target = utils.resize_mask(target, scale, padding)
            else:
                target = whole_mask
            target = target.astype(np.uint8)
            target = target.squeeze()
            score_info = compute_score(target, results[0], settings['mask_score'])
            #pred_lines.append("{},{},{}".format(fname, score_info['dice_score'], score_info['f1_score']))
            pred_lines.append("{},{},{},{},{},{}".format(
                                    fname, 
                                    score_info['accuracy_score'], 
                                    score_info['recall_score'],
                                    score_info['precision_score'],
                                    score_info['jaccard_score'],
                                    score_info['f1_score'],
                                )
                            )


        output_file = "{}/{}/mrcnn_scores_main.txt".format(domain, name )     
        with open(output_file, 'w') as fout:
            fout.write("\n".join(pred_lines))
            
            
def get_main_detection(ypred, region):
    boxes = ypred['rois']
    scores = ypred['scores']
    masks = ypred['masks']
    if region == 'nucleus':
        max_score = 0
        max_score_idx = 0 
        for i in range(boxes.shape[0]):
            val = scores[i]
            if val > max_score:
                max_score = val
                max_score_idx = i
                print("max_scoreeee: ", max_score)
        return max_score_idx
    else:
        samp = masks[:,:,0]
        h,w = samp.shape[:2]
        midx = int(w/2.0)
        midy = int(h/2.0)
        print(samp.shape)
        print(midx, midy)
        nearest_dist = 99999
        nearest_idx = 0
        for i in range(boxes.shape[0]):
            y1, x1, y2, x2 = boxes[i]
            print(boxes[i])
            bx = int((x1 + x2)/2.0)
            by = int((y1 + y2)/2.0)
            dist = math.hypot(bx-midx, by-midy)
            print(dist)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_idx = i
        return nearest_idx

 
def compute_score(ytrue, ypred, region):
    main_idx = get_main_detection(ypred, region)
    boxes = ypred['rois']
    scores = ypred['scores']
    masks = ypred['masks']
    try:
        new_mask = masks[:,:,main_idx]
    except:
        return {'jaccard_score': 0.0, 'f1_score': 0.0, 'accuracy_score': 0.0, 'recall_score': 0.0, 'precision_score': 0.0}
    new_mask = new_mask.astype(np.uint8)
    new_mask = new_mask.squeeze()
    
    print(ytrue.shape)
    print(new_mask.shape)
    print("-----")
    print(np.max(ytrue))
    print(np.max(new_mask))
    print(">>>>>")
    print(np.min(ytrue))
    print(np.min(new_mask))
    print("<<<<<")
    
#     contours, hierarchy = cv2.findContours(ypred.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
#     if len(contours) == 0:
#         return {'dice_score': 0.0, 'f1_score': 0.0}
#     main_contour = max(contours, key=cv2.contourArea)
    
#     new_mask = np.zeros([ypred.shape[0], ypred.shape[1], 1], dtype=np.uint8)
#     cv2.drawContours(new_mask, [main_contour], -1, (1), -1)
#     ypred = new_mask.squeeze()
    
    info = {
        'jaccard_score': jaccard_score(ytrue, new_mask, average='micro'),
        'f1_score': f1_score(ytrue, new_mask, average='micro'),
        'accuracy_score': accuracy_score(ytrue, new_mask, normalize=True),
        'precision_score': precision_score(ytrue, new_mask, average='micro'),
        'recall_score': recall_score(ytrue, new_mask, average='micro')
    }
    print(info)
    return info

 
