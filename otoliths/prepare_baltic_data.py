import os
import numpy as np
import cv2
import glob
import skimage
from skimage.color import rgb2gray
from scipy.ndimage.filters import gaussian_filter
from skimage.filters import sobel, gaussian, rank
from skimage.morphology import disk, dilation, square, erosion
from skimage.segmentation import watershed, random_walker
from scipy import ndimage
from skimage.measure import label
from skimage import morphology
from skimage.measure import regionprops
from skimage.measure import find_contours
from skimage import img_as_ubyte

import shutil
import json
import random

def segregate_train_fold(fold, age, folder_name="NEWFOLD"):
    destination = 'datasets_baltic/'
    with open("datasets_baltic/all_baltic_age_metadata.json") as fn:
        all_baltic_files = json.load(fn)
#     raise ValueError
    try:
        os.makedirs('datasets_baltic/{}_{}/'.format(folder_name, fold))
    except:
        pass
    with open("datasets_baltic/all_bounds_1155.json".format(destination)) as fin:
        all_bounds = json.load(fin)
    
    linux_files = all_baltic_files['age_{}'.format(age)]
    
    
    random.seed(int('{}0{}'.format(fold+1, fold+1) ) )
    sorted_imgs = sorted(linux_files)
    selected = random.sample(sorted_imgs, 30)
    selected = sorted(selected)

    for item in selected:
        filename = item.replace('\\', '/').split('/')[-1]

        print(filename)
        shutil.copyfile('datasets_baltic/images/{}'.format(filename), 'datasets_baltic/{}_{}/{}'.format(folder_name, fold, filename))



def create_train_val_set(fold, split_name, source="data_fold"):
    with open("datasets_baltic/all_data_map.json") as fson:
        data_map = json.load(fson)
    fold_num = fold
    mode = split_name

    train_dir = "datasets_baltic/train_{}_{}".format(mode, fold_num)
    valid_dir = "datasets_baltic/valid_{}_{}".format(mode, fold_num)
    mixed_train_dir = "datasets_baltic/mixed_{}_{}".format(mode,fold_num)

    fold_dir = "datasets_baltic/{}_{}".format(source,fold_num)

    try:
        os.makedirs(train_dir)
    except:
        pass
    try:
        os.makedirs(valid_dir)
    except:
        pass
    try:
        os.makedirs(mixed_train_dir)
    except:
        pass

    age_map = {}
    all_fold_files = glob.glob("{}/*.png".format(fold_dir) )
    all_lines = ["img,age,readable"]

    for ff in all_fold_files:
        strs = ff.replace("\\","/").split("/")
        img_name = strs[-1]
        
        manual_age = data_map[img_name] #img_name.split("_age_")[1].split(".png")[0]

        img = cv2.imread("{}/{}".format(fold_dir, img_name))
        cv2.imwrite("{}/{}".format(train_dir, img_name), img)
        cv2.imwrite("{}/{}_a.jpg".format(mixed_train_dir, img_name), img)

        flip_img = cv2.flip(img, 1)
        cv2.imwrite("{}/{}".format(valid_dir, img_name), flip_img)
        cv2.imwrite("{}/{}_b.jpg".format(mixed_train_dir, img_name), flip_img)

        all_lines.append("{}_a.jpg,{},2".format(img_name, manual_age))
        all_lines.append("{}_b.jpg,{},2".format(img_name, manual_age))

    with open("{}/all_baltic_{}_{}_train.csv".format(mixed_train_dir,mode,fold_num), "w") as fout:
        fout.write("\n".join(all_lines))

    all_json_files = glob.glob("{}/*.json".format(fold_dir) )
    for jsonf in all_json_files:
        strs = jsonf.replace("\\","/").split("/")
        shutil.copyfile(jsonf, "{}/{}".format(train_dir, strs[-1])  )
        shutil.copyfile(jsonf, "{}/{}".format(valid_dir, strs[-1])  )

        
        
def check_any_annotations(folder_name, destination="datasets_baltic"):
    count = 0

    all_files = glob.glob("{}/{}/*.png".format(destination, folder_name))
    sorted_files = sorted(all_files)

    all_label_files = glob.glob("{}/annotations/*.json".format(destination))
    label_annotations = {}
    all_list = {}
    for json_file in all_label_files:
        _annotations = json.load(open(json_file))
        _annotations = list(_annotations.values())
        _annotations = [a for a in _annotations if a['regions']]
        for a in _annotations:
            label_annotations[a['filename']] = (a, a["size"])

    new_count = 0
    count = 0
    
    with open("json_template.json") as fin:
        main_json = json.load(fin)
        main_json['_via_settings']['project']['name'] = 'age'
    
    for item in sorted_files:

        strs = item.replace("\\", "/").split("/")
        fname = strs[-1]
        if fname in label_annotations:
            a, fsize = label_annotations[fname]
            main_json["_via_image_id_list"].append("{}{}".format(fname, fsize) )
            new_item = {}
            new_item["{}{}".format(fname,fsize)] = a
            new_item["{}{}".format(fname,fsize)]["filename"] = fname
            new_item["{}{}".format(fname,fsize)]["size"] = int(fsize)
            main_json["_via_img_metadata"].update(new_item)
        else:
            count +=1
            print("Warning: The following file needs annotation! ==> {}".format(fname) )

    with open("{}/{}/final_{}_import.json".format( destination, folder_name, folder_name), "w") as fout:
        json.dump(main_json["_via_img_metadata"], fout, indent=4)


