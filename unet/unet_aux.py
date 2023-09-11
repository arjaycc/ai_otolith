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
import re
import glob
import json
import skimage.draw
from scipy.ndimage.morphology import distance_transform_edt
import logging
import itertools
import colorsys
import numpy as np
from skimage.measure import find_contours
from skimage.morphology import label
from skimage import img_as_ubyte
from skimage.transform import resize

ROOT_DIR = os.path.abspath("../")

print(ROOT_DIR)
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


def sort_contours(domain, mask_item, cx, cy):
    contours, hierarchy = cv2.findContours(mask_item.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    #contours = sorted(contours,key=cv2.contourArea, reverse=False)
    print(len(contours))
    if domain == 'datasets_north':
        contours = [c for c in contours if cv2.contourArea(c) > 50 ]
    else:
        contours = [c for c in contours if cv2.contourArea(c) > 150 ]
    #print("len1", len(contours))
    hlist = []
    anlist = []
    for c in contours:
        hull = cv2.convexHull(c)
        angles_list = []
        for item in hull:
            ptx = item[0][0]
            pty = item[0][1]
            test = np.arctan2(pty-cy,ptx-cx)*180/np.pi
            if test >= 0:
                angle_val = (90+test)%360
            else:
                angle_val = (90+(360+test))%360
            angles_list.append([ptx,pty,angle_val])
        angles_list = sorted(angles_list, key=lambda x: x[2])
        anlist.append(angles_list)
        hlist.append(hull)

    clist = []
    for idx,contour in enumerate(contours):
        rx,ry,rw,rh = cv2.boundingRect(contour)
        M = cv2.moments(contour)
        x2 = int(M["m10"] / M["m00"])
        y2 = int(M["m01"] / M["m00"])
        dist = math.hypot(x2-cx,y2-cy)
        clist.append([contour,dist, idx, x2, y2])
    print("len2", len(clist))
    #print("len3", len(anlist))
    sorted_c = sorted(clist, key=lambda x: x[1], reverse=True)
    return sorted_c, anlist


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


def count_prediction_output(domain, mask_item, cx, cy):
    contours, hierarchy = cv2.findContours(mask_item.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours,key=cv2.contourArea, reverse=False)

    if domain == 'datasets_north':
        contours = [c for c in contours if cv2.contourArea(c) > 50 ]
    else:
        contours = [c for c in contours if cv2.contourArea(c) > 150 ]

    sorted_c, alist = sort_contours(domain, mask_item, cx, cy )
    label_list = []
    for c in sorted_c:
        idx = c[2]
        xpos = c[3]
        ypos = c[4]
        if domain == 'datasets_north':
            if xpos > cx:
                continue
        dist = c[1]
        val = 1
        key = 0
        for label_item in label_list:
            intersect, start_angle, end_angle = check_angle_intersection(label_item, c, alist)
            if intersect:
                val = label_item[0] + 1
                key = label_item[1]    
        label_list.append([val, idx, dist, c[0]])

    sorted_labels = sorted(label_list, key=lambda x: x[0])
    try:
        ai_reading = sorted_labels[-1][0]
    except:
        ai_reading = 0
    return ai_reading



def unet_weight_map(y, wc=None, w0 = 10, sigma = 5):
    """
    Sources:
    https://stackoverflow.com/questions/50255438/pixel-wise-loss-weight-for-image-segmentation-in-keras
    https://arxiv.org/pdf/1505.04597.pdf
    """
    from skimage.morphology import label
    from scipy.ndimage.morphology import distance_transform_edt
    labels = label(y)
    no_labels = labels == 0
    label_ids = sorted(np.unique(labels))[1:]

    if len(label_ids) > 1:
        distances = np.zeros((y.shape[0], y.shape[1], len(label_ids)))

        for i, label_id in enumerate(label_ids):
            distances[:,:,i] = distance_transform_edt(labels != label_id)

        distances = np.sort(distances, axis=2)
        d1 = distances[:,:,0]
        d2 = distances[:,:,1]
        w = w0 * np.exp(-1/2*((d1 + d2) / sigma)**2) * no_labels
    else:
        w = np.zeros_like(y)
    if wc:
        class_weights = np.zeros_like(y)
        for k, v in wc.items():
            class_weights[y == k] = v
        w = w + class_weights
    return w


class OtolithConfig(config.Config):
    """
    Config Base Class
    Copyright (c) 2017 Matterport, Inc.
    """
    NAME = "otolith"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1
    STEPS_PER_EPOCH = 10
    DETECTION_MIN_CONFIDENCE = 0.8
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    
class InferenceConfig(OtolithConfig):
    """
    Config Base Class
    Copyright (c) 2017 Matterport, Inc.
    """
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    
with open('datasets_baltic/all_data_map.json') as fson:
    data_map = json.load(fson)
    
class OtolithDataset(utils.Dataset):
    """
    Dataset Base Class and functions
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
            try:
                a_filename = '{}_a.jpg'.format( name_map[a['filename']] )
                img_path = '{}/{}'.format(data_dir, a_filename)
                img = skimage.io.imread(img_path)
            except:
                # raise
                try:
                    img_path = '{}/{}'.format(data_dir, a['filename'])
                    img = skimage.io.imread(img_path)
                except:
                    continue
            h, w = img.shape[:2]
            self.add_image('annulus', image_id=a['filename'], path=img_path, width=w, height=h, polygons=poly)

            
    def load_mask(self, img_id):
        info = self.image_info[img_id]
        mask = np.zeros([info['height'], info['width'], 1], dtype=np.uint8)
        for idx, poly in enumerate(info['polygons']):
            rr, cc = skimage.draw.polygon(poly['all_points_y'], poly['all_points_x'])
            mask[rr, cc, 0] = 1 

        if self.mode == 'flip':
            mask = np.fliplr(mask)
        return mask,  len(info['polygons'])

    
    def load_image(self, image_id):
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        return image
    
    def resize_image(self, image, min_dim=None, max_dim=None, padding=False):
        # Default window (y1, x1, y2, x2) and default scale == 1.
        h, w = image.shape[:2]
        window = (0, 0, h, w)
        scale = 1

        # Scale?
        if min_dim:
            # Scale up but not down
            scale = max(1, min_dim / min(h, w))
        # Does it exceed max dim?
        if max_dim:
            image_max = max(h, w)
            if round(image_max * scale) > max_dim:
                scale = max_dim / image_max
        # Resize image and mask
        if scale != 1:
            image = resize(
                image, (round(h * scale), round(w * scale)))
        # Need padding?
        if padding:
            # Get new height and width
            h, w = image.shape[:2]
            top_pad = (max_dim - h) // 2
            bottom_pad = max_dim - h - top_pad
            left_pad = (max_dim - w) // 2
            right_pad = max_dim - w - left_pad
            padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
            image = np.pad(image, padding, mode='constant', constant_values=0)
            window = (top_pad, left_pad, h + top_pad, w + left_pad)
        return image, window, scale, padding


    def resize_mask(self, mask, scale, padding):
        h, w = mask.shape[:2]
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
        mask = np.pad(mask, padding, mode='constant', constant_values=0)
        return mask
    

    def load_weights(self, wt_mapx, unet_params={}):
        wc = {
            0: 1, # background
            1: 5  # objects
        }

        if bool(unet_params):
            try:
                w0 = unet_params['w0']
            except:
                w0 = 10
            try:
                sigma = unet_params['sigma']
            except:
                sigma = 5
            weights = unet_weight_map(wt_mapx, wc, w0, sigma)
        else:
            weights = unet_weight_map(wt_mapx, wc)
        return weights


class InferenceDataset(utils.Dataset):
    """
    Dataset Base Class and functions
    Copyright (c) 2017 Matterport, Inc.
    """
    
    def load_otolith_data(self, data_dir, num_annotations=1, exclude=None, settings={}):
        exclude_train = {} 
        if exclude is not None:
            exclude_train = {}
            training_files = glob.glob("{}/{}/*.png".format(settings['dataset'], exclude))
            for item in training_files:
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
            
            print(len(annotations))
            count = 0
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
                    
                count += 1
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
            print("Count == ", count)
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

    def load_image(self, image_id):
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        return image
    
    
    def resize_image(self, image, min_dim=None, max_dim=None, padding=False):
        # Default window (y1, x1, y2, x2) and default scale == 1.
        h, w = image.shape[:2]
        window = (0, 0, h, w)
        scale = 1

        # Scale?
        if min_dim:
            # Scale up but not down
            scale = max(1, min_dim / min(h, w))
        # Does it exceed max dim?
        if max_dim:
            image_max = max(h, w)
            if round(image_max * scale) > max_dim:
                scale = max_dim / image_max
        # Resize image and mask
        if scale != 1:
            image = resize(
                image, (round(h * scale), round(w * scale)))
        # Need padding?
        if padding:
            # Get new height and width
            h, w = image.shape[:2]
            top_pad = (max_dim - h) // 2
            bottom_pad = max_dim - h - top_pad
            left_pad = (max_dim - w) // 2
            right_pad = max_dim - w - left_pad
            padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
            image = np.pad(image, padding, mode='constant', constant_values=0)
            window = (top_pad, left_pad, h + top_pad, w + left_pad)
        return image, window, scale, padding


    def resize_mask(self, mask, scale, padding):
        h, w = mask.shape[:2]
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
        mask = np.pad(mask, padding, mode='constant', constant_values=0)
        return mask

    
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


def load_image_gt_from_array(dataset_array, config, image_id):

    image = dataset_array[image_id][0]
    mask = dataset_array[image_id][1]
    bbox = dataset_array[image_id][2]
    age = dataset_array[image_id][3]

    return image, 0, age, bbox, mask

    

def data_generator(dataset_array, config, augmentations=[], batch_size=20, validation=False, channels=3):
    """
    data generator adapted from Matterport Mask RCNN implementation modified for U-Net
    """

    import time
    b = 0  # batch item index
    image_index = -1
    image_ids = np.arange(len(dataset_array)) #np.copy(dataset.image_ids)
    error_count = 0

    shuffle = True
    # Keras requires a generator to run indefinately.
    while True:
#         try:
        # Increment index to pick next image. Shuffle if at the start of an epoch.
        image_index = (image_index + 1) % len(image_ids)
        if shuffle and image_index == 0:
            np.random.shuffle(image_ids)

        # Get GT bounding boxes and masks for image.
        image_id = image_ids[image_index]
        image, image_meta, age, gt_boxes, gt_masks = \
            load_image_gt_from_array(dataset_array, config, image_id)

        # Skip images that have no instances. This can happen in cases
        # where we train on a subset of classes and the image doesn't
        # have any of the classes we care about.
#             if not np.any(gt_class_ids > 0):
#                 continue

        # Init batch arrays
        if b == 0:
            batch_images = np.zeros(
                (batch_size, config.IMAGE_MIN_DIM,config.IMAGE_MIN_DIM, channels), dtype=np.float32)
            batch_gt_masks = np.zeros(
                (batch_size, gt_masks.shape[0], gt_masks.shape[1],1), dtype=gt_masks.dtype)
            batch_wts = np.zeros(
                (batch_size, gt_masks.shape[0], gt_masks.shape[1], 1), dtype=np.float32)
            batch_age = np.zeros(
                (batch_size, 1), dtype=np.float32)
            
        img = image
        if channels == 3:
            batch_images[b,:,:,:] = img
        else:
            batch_images[b,:,:,0] = img
        batch_gt_masks[b, :, :, 0] = gt_masks[:,:,0]
        batch_wts[b, :, :,0] = gt_boxes[:,:]
        batch_age[b, 0] = age
        b += 1

        # Batch full?
        if b >= batch_size:
            if validation:
                outputs = []
                new_wts = np.zeros(
                    (batch_size, gt_masks.shape[0], gt_masks.shape[1]), dtype=np.float32)
                new_wts[:,:,:] = batch_wts[:,:,:,0]
                
                inputs = [batch_images, batch_wts, batch_gt_masks]
            else:
                inputs = [batch_images, batch_wts, batch_gt_masks]
                outputs = []
                if len(augmentations)>0:
                    img_aug, mask_aug, wt_aug =  augmentations
                    cur_time = int(time.time())

                    (b_images, labels_img) = next(img_aug.flow(
                                batch_images,  
                                batch_age, 
                                batch_size=batch_size, 
                                shuffle=False,
                                seed=cur_time,
                            )
                        )
       
                    (b_masks, labels_mask) = next(mask_aug.flow(
                                batch_gt_masks, 
                                batch_age, 
                                batch_size=batch_size, 
                                shuffle=False,
                                seed=cur_time,
                            )
                        )
                    (b_wts, labels_wt) = next(wt_aug.flow(
                                batch_wts, 
                                batch_age, 
                                batch_size=batch_size, 
                                shuffle=False,
                                seed=cur_time,
                            )
                        )


                    inputs = [b_images, b_wts, b_masks]

            b = 0
            yield inputs, outputs

        
### FOR VISUALIZATION, by Matterport
        
def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors



def print_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]

    for i in range(N):
        y1, x1, y2, x2 = boxes[i]
        cv2.putText(image, str(captions[i]), (int(x1),int(y1) ),  cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0) )
    
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            x = random.randint(x1, (x1 + x2) // 2)
            caption = "{} {}".format(label, score) if score else label
        else:
            caption = captions[i]

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

    skimage.io.imsave(ax, masked_image.astype(np.uint8))
    return masked_image #plt.gcf()
