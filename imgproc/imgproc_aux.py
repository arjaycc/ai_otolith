"""
Some sources:
    Peakdetection: https://gist.github.com/antiface/7177333
    Polar transform: https://stackoverflow.com/questions/3798333/image-information-along-a-polar-coordinate-system
"""

import numpy as np
import cv2
import glob
import json
import skimage.draw
import os
import sys
import random
import math
from skimage.morphology import label
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
from numpy import NaN, Inf, arange, isscalar, asarray, array
from scipy.ndimage.morphology import distance_transform_edt
from scipy.signal import find_peaks, detrend
from scipy import signal
from scipy.interpolate import interp1d
from skimage.util import img_as_ubyte
from skimage.morphology import disk
from skimage import exposure
from skimage.filters import rank
from skimage.transform import resize
from scipy import interpolate

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)

from mrcnn import utils
from mrcnn import config


def get_center_cts(whole_mask, nucleus_mask, settings):

    wh_thresh = np.zeros([whole_mask.shape[0], whole_mask.shape[1],1], dtype=np.uint8)
    wh_thresh[:,:,0] = whole_mask[:,:,0]

    wh_contours, hierarchy = cv2.findContours(wh_thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    wh_cts = max(wh_contours, key=cv2.contourArea)
    if nucleus_mask is not None:
        nr_thresh = np.zeros([nucleus_mask.shape[0], nucleus_mask.shape[1],1], dtype=np.uint8)
        nr_thresh[:,:,0] = nucleus_mask[:,:,0]

        nr_contours, hierarchy = cv2.findContours(nr_thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        nr_cts = max(nr_contours, key=cv2.contourArea)
        nr_M = cv2.moments(nr_cts)
        cx = int(nr_M["m10"] / nr_M["m00"])
        cy = int(nr_M["m01"] / nr_M["m00"])
    else:
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

    return wh_cts, cx, cy


def create_transect(og_gray, ncontour, max_left_angle, arc_width, offset, cx, cy):
    included_left_points_down = []
    included_left_dist_down = []
    included_left_theta_down = []
    for item in ncontour:
        r_test = np.sqrt(    (item[0][0]-cx)  **2 + (item[0][1]-cy)**2)
        theta_test = np.arctan2( (item[0][1]-cy), (item[0][0]-cx) )

        if theta_test > max_left_angle-arc_width  and theta_test < max_left_angle:
            included_left_points_down.append( (item[0][0], item[0][1]) )
            included_left_dist_down.append(r_test)
            included_left_theta_down.append(theta_test)

    x_dist = np.arange(len(included_left_dist_down))
    y_dist = np.array(included_left_dist_down)
    f_dist = interpolate.interp1d(x_dist,y_dist)
    new_x_dist = np.linspace(0,len(included_left_dist_down)-1,num=224)
    included_left_dist_down = f_dist(new_x_dist)

    x_theta = np.arange(len(included_left_theta_down))
    y_theta = np.array(included_left_theta_down)
    f_theta = interpolate.interp1d(x_theta,y_theta)
    new_x_theta = np.linspace(0,len(included_left_theta_down)-1,num=224)
    included_left_theta_down = f_theta(new_x_theta)

    image_transect_col = np.zeros([224, len(included_left_theta_down), 3], dtype=np.uint8)
    image_transect_down = np.zeros([224, len(included_left_theta_down)], dtype=np.uint8)

    for i in range(224):
        pct = (i+1)/224
        up_vals = []
        col_vals = []
        down_vals = []
        for _idx, _z in enumerate(included_left_theta_down):
            r_val = included_left_dist_down[_idx]*pct
            r_val = r_val + offset
            theta_val = included_left_theta_down[_idx]
            loc_x = cx + math.cos(theta_val)*r_val
            loc_y = cy + math.sin(theta_val)*r_val

            if loc_x >= 1024:
                loc_x = 1023
            if loc_y >= 1024:
                loc_y = 1023
            down_vals.append(og_gray[int(loc_y), int(loc_x)])
        down_vals = np.array(down_vals)
        image_transect_down[i,:] = down_vals

    return image_transect_down


def get_major_axes(ncontour, cx, cy):
    max_right_angle = 0
    max_left_angle = 0
    max_right_value = 0
    max_left_value = 0

    reading_value = 0
    reading_angle = 0
    sulcus_value = 99999
    sulcus_angle = 0
    for item in ncontour:
        r_test = np.sqrt(    (item[0][0]-cx)  **2 + (item[0][1]-cy)**2)
        theta_test = np.arctan2( (item[0][1]-cy), (item[0][0]-cx) )

        if r_test > reading_value:
            reading_value = r_test
            reading_angle = theta_test
        if r_test < sulcus_value:
            sulcus_value = r_test
            sulcus_angle = theta_test

    for item in ncontour:
        r_test = np.sqrt(    (item[0][0]-cx)  **2 + (item[0][1]-cy)**2)
        theta_test = np.arctan2( (item[0][1]-cy), (item[0][0]-cx) )

        if r_test > max_right_value and item[0][0] > cx:
            max_right_value = r_test
            max_right_angle = theta_test
            max_right_x = item[0][0]
            max_right_y = item[0][1]
        if r_test > max_left_value and item[0][0] < cx:
            max_left_value = r_test
            max_left_angle = theta_test
            max_left_x = item[0][0]
            max_left_y = item[0][1]
    return max_left_angle, max_right_angle

        

def peakdet(v, delta, x = None):
    """
    https://gist.github.com/endolith/250860
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    """
    maxtab = []
    mintab = []
       
    if x is None:
        x = arange(len(v))
    
    v = asarray(v)
    
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    
    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN
    
    lookformax = True
    
    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return array(maxtab), array(mintab)


class OtolithConfig(config.Config):
    """
    Config Base Class
    Copyright (c) 2017 Matterport, Inc.
    """
    
    NAME = "otolith"
    IMAGES_PER_GPU = 1
    IMAGE_PADDING = True
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    

class OtolithDataset(utils.Dataset):
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
                
                if a["filename"] in exclude_train:
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

