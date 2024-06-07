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
            with open("{}/{}/output/boundingbox_{}.json".format(domain, settings['input_run1'], image_name)) as fin:
                bbox = json.load(fin)
                print(bbox)
                x,y,w,h = [int(bb) for bb in bbox]
                ofs = 50
                imgraw = imgraw[max([y-ofs,0]):min([y+h+ofs, imgraw.shape[0]]), max([x-ofs,0]):min([x+w+ofs,imgraw.shape[1]])]

            sq_img, window, scale, padding, _ = utils.resize_image(
                imgraw, 
                min_dim=inference_config.IMAGE_MIN_DIM,
                max_dim=inference_config.IMAGE_MAX_DIM,
            )
            print(image_name)

            results = model.detect([sq_img], verbose=1)
            r = results[0]
            main_idx = get_main_detection(r, "nucleus")
            boxes = r['rois']
            scores = r['scores']
            masks = r['masks']
            print(scores)
            try:
                new_mask = masks[:,:,main_idx]
            except:
                print("except---------------")
                coordx, coordy = (int(config.IMAGE_MAX_DIM/2.0), int(config.IMAGE_MAX_DIM/2.0))
                with open("{}/{}/output/nucleus_{}.json".format(domain, name, image_name), "w") as fout:
                    json.dump([coordx, coordy], fout, indent=4)
                continue

            new_mask = new_mask.astype(np.uint8)
            new_mask = new_mask.squeeze()
            ypred = new_mask.copy()
            pthresh = np.zeros([ypred.shape[0], ypred.shape[1],1], dtype=np.uint8)
            pthresh[:,:,0] = ypred[:,:]
            pcontours, hierarchy = cv2.findContours(pthresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            pred_contour = max(pcontours, key=cv2.contourArea)
            nr_M = cv2.moments(pred_contour)
            coordx = int(nr_M["m10"] / nr_M["m00"])
            coordy = int(nr_M["m01"] / nr_M["m00"])
            with open("{}/{}/output/nucleus_{}.json".format(domain, name, image_name), "w") as fout:
                json.dump([coordx, coordy], fout, indent=4)

                
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