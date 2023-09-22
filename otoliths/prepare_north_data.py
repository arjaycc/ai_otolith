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
import re
import random


def split_random(fold, prop, destination="datasets_north"):
    
    random.seed(int("{}0{}".format(fold+1, fold+1) ))

    all_files = glob.glob("{}/images/*.png".format(destination))
    sorted_files = sorted(all_files)
    train_map = {}
    train = random.sample(sorted_files, int(prop*len(sorted_files)) )
    for item in train:
        strs = item.replace("\\", "/").split("/")
        train_map[strs[-1]] = 1
        
    prev = {}
    for item in all_files:
        strs = item.replace("\\", "/").split("/")
        if strs[-1] in train_map:
            try:
                os.makedirs("{}/data_random_{}/".format(destination, fold))
            except:
                pass
                
            shutil.copyfile(item, "{}/data_random_{}/{}".format(destination, fold, strs[-1]))


def split_balance(fold, destination="datasets_north"):
    random.seed(int("{}0{}".format(fold+1, fold+1) ))
    limit = 12
    old_train_map = {}
    old_age_map = {}

    old_train_files = glob.glob("{}/data_random_{}/*.png".format(destination, fold))
    for item in old_train_files:
        strs = item.replace("\\", "/").split("/")
        age = strs[-1].split(".png")[0].split("_age_")[1]

        old_train_map[strs[-1]] = 1

        if age in old_age_map:
            old_age_map[age].append(item)
        else:
            old_age_map[age] = [item]

    all_files = glob.glob("{}/images/*.png".format(destination))

    sorted_files = sorted(all_files)
    age_map = {}
    for item in sorted_files:
        strs = item.replace("\\", "/").split("/")
        age = strs[-1].split(".png")[0].split("_age_")[1]

        if strs[-1] not in old_train_map:
            if age in age_map:
                age_map[age].append(item)
            else:
                age_map[age] = [item]

    try:
        os.makedirs("{}/data_balanced_{}/".format(destination, fold))
    except:
        pass

    count  = 0
    for k, v in age_map.items():
        train_v = old_age_map[k]
        print("{} with {}".format(k, len(train_v)))
        if len(train_v) < limit:
            train = random.sample(v, limit - len(train_v) )
            for item in train:
                count +=1
                strs = item.replace("\\", "/").split("/")
                shutil.copyfile(item, "{}/data_balanced_{}/{}".format(destination, fold, strs[-1]))

            train = train_v
            for item in train:
                strs = item.replace("\\", "/").split("/")
                shutil.copyfile(item, "{}/data_balanced_{}/{}".format(destination, fold, strs[-1]))
        else:
            train = random.sample(train_v, limit) #sorted(train_v)[:limit]
            for item in train:
                strs = item.replace("\\", "/").split("/")
                shutil.copyfile(item, "{}/data_balanced_{}/{}".format(destination, fold, strs[-1]))

    print(">>>>", count)
    

def create_annotations(folder_name, destination="datasets_north"):
    count = 0

    all_files = glob.glob("{}/{}/*.png".format(destination, folder_name))
    sorted_files = sorted(all_files)

    with open("json_template.json") as fin:
        main_json = json.load(fin)
        main_json['_via_settings']['project']['name'] = 'age'

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
    for item in sorted_files:
        strs = item.replace("\\", "/").split("/")
        # if strs[-1] not in anno:
        fname = strs[-1]

        a, fsize = label_annotations[fname]
        main_json["_via_image_id_list"].append("{}{}".format(fname, fsize) )
        new_item = {}
        new_item["{}{}".format(fname,fsize)] = a
        new_item["{}{}".format(fname,fsize)]["filename"] = fname
        new_item["{}{}".format(fname,fsize)]["size"] = int(fsize)
        main_json["_via_img_metadata"].update(new_item)

    with open("{}/{}/final_{}_import.json".format( destination, folder_name, folder_name), "w") as fout:
        json.dump(main_json["_via_img_metadata"], fout, indent=4)

        
def create_train_val_set(fold, split_name, source="data_balanced"):

    fold_num = fold
    mode = split_name

    train_dir = "datasets_north/train_{}_{}".format(mode, fold_num)
    valid_dir = "datasets_north/valid_{}_{}".format(mode, fold_num)
    mixed_train_dir = "datasets_north/mixed_{}_{}".format(mode,fold_num)

    fold_dir = "datasets_north/{}_{}".format(source,fold_num)

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
        manual_age = img_name.split("_age_")[1].split(".png")[0]

        img = cv2.imread("{}/{}".format(fold_dir, img_name))
        cv2.imwrite("{}/{}".format(train_dir, img_name), img)
        cv2.imwrite("{}/{}_a.jpg".format(mixed_train_dir, img_name), img)

        flip_img = cv2.flip(img, 1)
        cv2.imwrite("{}/{}".format(valid_dir, img_name), flip_img)
        cv2.imwrite("{}/{}_b.jpg".format(mixed_train_dir, img_name), flip_img)

        all_lines.append("{}_a.jpg,{},2".format(img_name, manual_age))
        all_lines.append("{}_b.jpg,{},2".format(img_name, manual_age))

    with open("{}/all_north_{}_{}_train.csv".format(mixed_train_dir,mode,fold_num), "w") as fout:
        fout.write("\n".join(all_lines))

    all_json_files = glob.glob("{}/*.json".format(fold_dir) )
    for jsonf in all_json_files:
        strs = jsonf.replace("\\","/").split("/")
        shutil.copyfile(jsonf, "{}/{}".format(train_dir, strs[-1])  )
        shutil.copyfile(jsonf, "{}/{}".format(valid_dir, strs[-1])  )

        
def create_species_set(species, fold, destination="datasets_north" ):

    with open("datasets_north/all_species_metadata.json") as fn:
        all_species = json.load(fn)
        
    random.seed(int("{}0{}".format(fold+1, fold+1) ))

    sp_folder = "{}/data_{}_{}".format(destination, species, fold)
    try:
        os.makedirs(sp_folder)
    except:
        pass
    for k,v in all_species.items():
        if k == species:
            sorted_vu = sorted(v)
            train_vu = random.sample(sorted_vu, 132 )
            for vu in train_vu:
                shutil.copyfile("{}/images/{}".format(destination, vu),"{}/{}".format(sp_folder, vu))
    for k,v in all_species.items():
        print(k, len(v))

        
def shuffle_train_data(fold, prop, destination="datasets_north/"):

    random.seed(int("{}0{}".format(fold+1, fold+1) ))

    all_files = glob.glob("{}/images/*.png".format(destination))
    sorted_files = sorted(all_files)
    train_map = {}

    random.shuffle(sorted_files)
    num_train = len(sorted_files)
    for idx,item in enumerate(train):
        if idx >= (fold*prop)*num_train and idx < (fold+1)*prop*num_train:
            strs = item.replace("\\", "/").split("/")
            train_map[strs[-1]] = 1

    prev = {}
    for item in all_files:
        strs = item.replace("\\", "/").split("/")

        if strs[-1] in train_map:
            try:
                os.makedirs("{}/data_shuffle_{}/".format(destination, fold))
            except:
                pass       
            shutil.copyfile(item, "{}/data_shuffle_{}/{}".format(destination, fold, strs[-1]))


