
import re
import pickle
import math
import scipy as sp
import scipy.ndimage
from skimage import img_as_ubyte
from scipy.signal import find_peaks, detrend
from scipy import signal
from scipy.interpolate import interp1d
from skimage.util import img_as_ubyte
from skimage.morphology import disk
from skimage import exposure
from skimage.filters import rank
from scipy import interpolate, stats
import sys
from numpy import NaN, Inf, arange, isscalar, asarray, array
from .imgproc_aux import *


def evaluate(run_name, settings):
    config = OtolithConfig()
    all_val = OtolithDataset()
    
    if settings['dataset'] == 'datasets_north':
        all_val.load_otolith_data('{}/data_sep/'.format(settings['dataset']), 2, exclude="train_{}_{}".format(settings['split_name'], settings['idr']), settings=settings)
    else:
        all_val.load_otolith_data('{}/isolated/'.format(settings['dataset']), 1, exclude="train_{}_{}".format(settings['split_name'], settings['idr']), settings=settings)
    all_val.prepare()

    if settings['dataset'] == 'datasets_north':
        arc_width = 0.05
        read_limit = np.pi/2.0
    else:
        arc_width = 0.1
        read_limit = 0

    offset = 20

    exact = 0
    off_by_one = 0
    pred_lines = []
    for idx, image_id in enumerate(all_val.image_ids):
        info = all_val.image_info[image_id]
        image_path = info['path']
        fname = image_path.replace('\\','/').split('/')[-1]
        
        if settings['dataset'] == 'datasets_north':
            folder = int(fname.split('_age_')[1].split('.')[0])
            manual_age = int(folder)
        else:
            with open('datasets_baltic/all_data_map.json') as fson:
                data_map = json.load(fson)
            manual_age = int(data_map[fname])

        image = all_val.load_image(image_id)
        multi_masks, class_ids = all_val.load_mask(image_id)

        whole_mask = multi_masks[0]
        nucleus_mask = None
        if settings['dataset'] == 'datasets_north':
            nucleus_mask = multi_masks[1]

        original_image, window, scale, padding = all_val.resize_image(
            image, 
            min_dim=config.IMAGE_MIN_DIM,
            max_dim=config.IMAGE_MAX_DIM,
            padding=True
        )

        original_copy = original_image.copy()
        
        whole_mask = utils.resize_mask(whole_mask, scale, padding)
        if nucleus_mask is not None:
            nucleus_mask = utils.resize_mask(nucleus_mask, scale, padding)

        wh_cts, cx, cy = get_center_cts(whole_mask, nucleus_mask, settings)
        
        ncontour = wh_cts
        max_left_angle, max_right_angle = get_major_axes(ncontour, cx, cy)

        original_copy = original_image.copy()
        og_gray = cv2.cvtColor(img_as_ubyte(original_copy), cv2.COLOR_BGR2GRAY)

        all_peaks = []
        strip_id = 0
        max_left_angle = np.pi
        while max_left_angle - arc_width >  read_limit:
            image_transect_down = create_transect(og_gray, ncontour, max_left_angle, arc_width, offset, cx, cy)
            strip_id +=1

            if 'pxl_averaging' in settings and settings['pxl_averaging'] == 'median':
                raw_signal = np.median(image_transect_down, axis=1)
            else:
                raw_signal = np.mean( image_transect_down, axis=1)

            window = signal.general_gaussian(5, p=1,sig=5)
            filtered = signal.fftconvolve(window,raw_signal)
            filtered = (np.average(raw_signal)/np.average(filtered))*filtered
            mxtab, mntab = peakdet(filtered, 25)

            all_peaks.append(len(mxtab))
            max_left_angle = max_left_angle - arc_width
            strip_id += 1
        print(all_peaks)

        if 'peak_averaging' in settings and settings['peak_averaging'] == 'median':
            ai_age_val = np.round(np.median(all_peaks))
        else:
            ai_age_val = np.round(np.mean(all_peaks))

        if abs(ai_age_val-manual_age) < 1:
            exact += 1
        if abs(ai_age_val-manual_age) < 2:
            off_by_one += 1

        print("{}==={} vs {} ::: count {}".format(idx, ai_age_val, manual_age, exact))
        pred_lines.append("{},{},{}".format(fname, ai_age_val, manual_age))

    print(exact, off_by_one)
    
    output_file = "{}/{}/rdc_{}_{}_of_{}.txt".format(settings['dataset'], run_name, exact, off_by_one, len(all_val.image_ids) )
    with open(output_file, 'w') as fout:
        fout.write("\n".join(pred_lines))

