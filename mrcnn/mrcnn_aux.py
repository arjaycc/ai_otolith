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

import sys
import os
import math
import random
import numpy as np
import tensorflow as tf
import scipy
import skimage.color
import skimage.io
import skimage.transform
import urllib.request
import shutil
import warnings
import cv2
import glob
import json
import skimage.draw
from scipy.ndimage.morphology import distance_transform_edt
from skimage.measure import find_contours
from skimage import img_as_ubyte 

COCO_MODEL_URL = "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5"
ROOT_DIR = os.path.abspath("../")

sys.path.append(ROOT_DIR)
from mrcnn import utils
from mrcnn import config


def ellipse_center(wh_cts):
    ellipse = cv2.fitEllipse(wh_cts)
    (ex,ey), (d1,d2), angle = ellipse
    
    min_angle = -(180-angle)
    all_r = []
    r_test  = 1
    for item in wh_cts:
        r_test = np.sqrt(    (item[0][0]-ex)  **2 + (item[0][1]-ey)**2)
        theta_test = np.arctan2( (item[0][1]-ey), (item[0][0]-ex) )
        if theta_test - math.radians(min_angle) < 0.01 and theta_test - math.radians(min_angle) > 0.005:
            print(r_test)
            break

    rlen = r_test*0.3 #0.4585
    cx = ex + math.cos(math.radians(min_angle))*rlen
    cy = ey + math.sin(math.radians(min_angle))*rlen

    return cx, cy


def get_center(whole_mask, nucleus_mask):
    if nucleus_mask is not None:
        nr_thresh = np.zeros([nucleus_mask.shape[0], nucleus_mask.shape[1],1], dtype=np.uint8)
        nr_thresh[:,:,0] = nucleus_mask[:,:,0]
        nr_contours, hierarchy = cv2.findContours(nr_thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        nr_cts = max(nr_contours, key=cv2.contourArea)
        nr_M = cv2.moments(nr_cts)
        cx = int(nr_M["m10"] / nr_M["m00"])
        cy = int(nr_M["m01"] / nr_M["m00"])
    else:
        wh_thresh = np.zeros([whole_mask.shape[0], whole_mask.shape[1],1], dtype=np.uint8)
        wh_thresh[:,:,0] = whole_mask[:,:,0]
        wh_contours, hierarchy = cv2.findContours(wh_thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        wh_cts = max(wh_contours, key=cv2.contourArea)
        cx, cy = ellipse_center(wh_cts)
    return cx, cy


def modify_image(img_new, mask_new, settings):       
    nwh_thresh = np.zeros([mask_new.shape[0], mask_new.shape[1],1], dtype=np.uint8)
    nwh_thresh = mask_new[:,:,0]
    nwh_contours, h = cv2.findContours(nwh_thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    nwh_cts = max(nwh_contours, key=cv2.contourArea)

    test_mask = np.ones(mask_new.shape[:2], dtype="uint8") * 255
    cv2.drawContours(test_mask, [nwh_cts], -1, 0, -1)

    mapFg = cv2.erode(test_mask, np.ones( (5,5), np.uint8), iterations=10)
    test_img_mask = cv2.bitwise_not(mapFg)
    
    imagec = img_as_ubyte(img_new.copy())
    image_res = cv2.bitwise_and(imagec, imagec, mask=test_img_mask)
    if settings['dataset'] == 'datasets_north':
        img_new = cv2.convertScaleAbs(img_as_ubyte(image_res), alpha=1.5, beta=10)
    else:
        img_new = cv2.convertScaleAbs(img_as_ubyte(image_res), alpha=2, beta=20)
    return img_new


def count_detected_masks(results, cx, cy):
    r = results[0]

    contours = []
    for box in r['rois']:
        y1, x1, y2, x2 = box
        midx = int((x1+x2)/2.0)
        midy = int((y1+y2)/2.0)
        contours.append(box)

    hlist = []
    alist = []
    for c in contours:
        y1, x1, y2, x2 = c
        hull = [(x1,y1), (x2,y2)]
        angles = []
        for item in hull:
            x = item[0]
            y = item[1]
            test = np.arctan2(y-cy,x-cx)*180/np.pi
            if test >= 0:
                angle = (90+test)%360
            else:
                angle = (90+(360+test))%360
            angles.append([x,y,angle])
        angles = sorted(angles, key=lambda x: x[2])
        alist.append(angles)
        hlist.append(c)

    clist = []
    for idx,contour in enumerate(contours):
        y1, x1, y2, x2 = contour
        midx = int((x1+x2)/2.0)
        midy = int((y1+y2)/2.0)
        dist = ( (midx-cx)**2 + (midy-cy)**2 ) ** 0.5
        clist.append([contour,dist, idx])

    sorted_c = sorted(clist, key=lambda kx: kx[1], reverse=True)

    labels = []
    for c in sorted_c:
        idx = c[2]
        dist = c[1]
        val = 1
        key = 0
        for label in labels:
            intersect, start_angle, end_angle = check_angle_intersection(label, c, alist)
            if intersect:
                val = label[0] + 1
                key = label[1]
        labels.append([val, idx, dist, c  ])
    label_list = [l[0:2] for l in labels]

    try:
        sorted_labels = sorted(label_list, key=lambda x: x[0])
        ai_reading = sorted_labels[-1][0]
    except:
        ai_reading = 0
    return ai_reading


def filter_contour(cont, st, en, cx, cy):
    new_contour = []
    for item in cont:
        
        x, y = item[0]
        test = np.arctan2(y-cy,x-cx)*180/np.pi
        if test >= 0:
            an = (90+test)%360
        else:
            an = (90+(360+test))%360
        if an > st and an < en:
            new_contour.append(item)
    return np.array(new_contour)


def check_angle_intersection(label, c, angles):
    idx_label = label[1]
    idx_c = c[2]
    angle_label = angles[idx_label]
    angle_label_start = angle_label[0][2]
    angle_label_end = angle_label[-1][2]
    
    angle_c = angles[idx_c]
    angle_c_start = angle_c[0][2]
    angle_c_end = angle_c[-1][2]
    intersect = False
    start_angle = None
    end_angle = None
    if angle_label_end < angle_c_end:
        if angle_label_end > angle_c_start:
            intersect = True
            if angle_label_start > angle_c_start:
                start_angle = angle_label_start
                end_angle = angle_label_end
            else:
                start_angle = angle_c_start
                end_angle = angle_label_end

    if angle_c_end < angle_label_end:
        if angle_c_end > angle_label_start:
            intersect = True
            if angle_label_start > angle_c_start:
                start_angle = angle_label_start
                end_angle = angle_c_end
            else:
                start_angle = angle_c_start
                end_angle = angle_c_end
    return intersect, start_angle, end_angle


class OtolithConfig(config.Config):
    """
    Config Base Class + Modifications
    Copyright (c) 2017 Matterport, Inc.
    Copyright (c) 2018/2019 Roland S. Zimmermann, Julien Siems
    """
    NAME = "otolith"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1

    STEPS_PER_EPOCH = 50
    DETECTION_MIN_CONFIDENCE = 0.85 #0.7
    DETECTION_NMS_THRESHOLD = 0.2
    VALIDATION_STEPS = 1 
    #-----
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    
    EDGE_LOSS_NORM = "l2"
    EDGE_LOSS_WEIGHT_FACTOR = 2.0 #2.0
    EDGE_LOSS_WEIGHT_ENTROPY = True
    
#     RUN_NAME = args.run_name
    EDGE_LOSS_SMOOTHING_GT = True
    EDGE_LOSS_SMOOTHING_PREDICTIONS = True
    EDGE_LOSS_FILTERS = ["sobel-y"] #sobel-y

    USE_MINI_MASK = False

    
class InferenceConfig(OtolithConfig):
    """
    Config Base Class + Modifications
    Copyright (c) 2017 Matterport, Inc.
    Copyright (c) 2018/2019 Roland S. Zimmermann, Julien Siems
    """
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


with open('datasets_baltic/all_data_map.json') as fson:
    data_map = json.load(fson)

    
class OtolithDataset(utils.Dataset):
    """
    Dataset Loader Base Class
    Copyright (c) 2017 Matterport, Inc.
    """
    
    def __init__(self, mode="train", **kwargs):
        super(OtolithDataset, self).__init__(**kwargs)
        self.mode = mode
    
    def load_otolith_data(self, data_dir, settings={}):
        self.add_class('annulus', 1, 'annulus')
        json_files = glob.glob('{}/*.json'.format(data_dir))
        annotations = {}
        for json_file in json_files:
            _annotations = json.load(open(json_file))
            annotations.update(_annotations)
        annotations = list(annotations.values())
        annotations = [a for a in annotations if a['regions']]
        
        for a in annotations:
            if settings['dataset'] == 'datasets_north':
                if 'age_limit' in settings and int(a["filename"].split("_age_")[1].split(".")[0]) > settings['age_limit']:
                    continue
            else:
                if 'age_limit' in settings and int(data_map[a["filename"]]) > settings["age_limit"]:
                    continue
            poly = [r['shape_attributes'] for r in a['regions']]
            
            img_path = '{}/{}'.format(data_dir, a['filename'])
            try:
                img = skimage.io.imread(img_path)
            except:
                continue
            h, w = img.shape[:2]
            self.add_image('annulus', image_id=a['filename'], path=img_path, width=w, height=h, polygons=poly)

            
    def load_mask(self, img_id):
        info = self.image_info[img_id]
        
        mask = np.zeros([info['height'], info['width'], len(info['polygons'])], dtype=np.uint8)
        
        for idx, poly in enumerate(info['polygons']):
            rr, cc = skimage.draw.polygon(poly['all_points_y'], poly['all_points_x'])
            mask[rr, cc, idx] = 1.0
            
        if self.mode == 'flip':
            mask = np.fliplr(mask)
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)
        

class InferenceDataset(utils.Dataset):
    """
    Dataset Loader Base Class
    Copyright (c) 2017 Matterport, Inc.
    """
    
    def load_otolith_data(self, data_dir, num_annotations=1, exclude=None, settings={}):
        exclude_train = {}
        if exclude is not None:
            exclude_train = {}
            training = glob.glob("{}/{}/*.png".format(settings['dataset'], exclude))
            for item in training:
                strs = item.replace("\\", "/").split("/")
                exclude_train[strs[-1]] = 1
        
        target = None
        if 'target_species' in settings:
            target = settings['target_species']
        sp_map = {}
        if target is not None:
            with open("species_map.json") as fj:
                sp_map = json.load(fj)
        image_set = {}
        poly_set = {}
        for i in range(num_annotations):
            self.add_class('annulus_{}'.format(i), i+1, 'annulus_{}'.format(i))
            json_files = glob.glob('{}/{}/*.json'.format(data_dir, i))
            annotations = {}
            for json_file in json_files:
                _annotations = json.load(open(json_file))
                annotations.update(_annotations)
            annotations = list(annotations.values())
            annotations = [a for a in annotations if a['regions']]
            
            for a in annotations:
                if settings['dataset'] == 'datasets_north': 
                    if 'age_limit' in settings and int(a["filename"].split("_age_")[1].split(".")[0]) <= settings['age_limit']:
                        continue
                else:
                    if 'age_limit' in settings and int(data_map[a["filename"]]) <= settings["age_limit"]:
                        continue
                if a["filename"] in exclude_train:
                    if settings['dataset'] == 'datasets_north':
                        if 'age_limit' in settings and int(a["filename"].split("_age_")[1].split(".")[0]) > settings['age_limit']:
                            pass
                        else:
                            continue
                    else:
                        if 'age_limit' in settings and int(data_map[a["filename"]]) > settings["age_limit"]:
                            pass
                        else:
                            continue
                if a["filename"] in sp_map and sp_map[a["filename"]] != target:
                    continue
                img_path = '{}/{}'.format(data_dir, a['filename'])
                try:
                    img = skimage.io.imread(img_path)
                except:
                    continue
                poly = [r['shape_attributes'] for r in a['regions']]
                if a['filename'] not in image_set:
                    h, w = img.shape[:2]
                    image_set[a['filename']] = (img_path, w, h)
                    poly_set[a['filename']] = [poly]
                else:
                    poly_set[a['filename']].append(poly)
                    
        for k, v in image_set.items():
            polygons = poly_set[k]
            self.add_image('annulus', image_id=k, path=v[0], width=v[1], height=v[2], polygons=polygons)

            
    def load_mask(self, img_id):
        info = self.image_info[img_id]
        print('{}::{}'.format(img_id, info['path']) )
        
        polygons = info['polygons']
        all_masks = []
        for cls, polygon in enumerate(polygons):
            mask = np.zeros([info['height'], info['width'], 1], dtype=np.uint8)
            
            for idx, poly in enumerate(polygon):
                rr, cc = skimage.draw.polygon(poly['all_points_y'], poly['all_points_x'])
                mask[rr, cc, 0] = 1.0
            all_masks.append(mask)
        return all_masks, np.ones([mask.shape[-1]], dtype=np.int32)
        
    
