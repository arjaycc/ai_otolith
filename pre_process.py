import os
import numpy as np
import cv2
import glob
import shutil
import json


def create_train_val_from_json(fold_num, split_name, domain="datasets_north"):

    train_dir = "{}/train_{}_{}".format(domain, split_name, fold_num)
    valid_dir = "{}/valid_{}_{}".format(domain, split_name, fold_num)
    mixed_train_dir = "{}/mixed_{}_{}".format(domain, split_name, fold_num)
    
    source_dir = "{}/images".format(domain)
    
    labeled_images = {}
    all_json_files = glob.glob("{}/*.json".format(train_dir))
    for json_file in all_json_files:
        _annotations = json.load(open(json_file))
        _annotations = list(_annotations.values())
        _annotations = [a for a in _annotations if a['regions']]
        for a in _annotations:
            labeled_images[a['filename']] = a['filename']
            
    try:
        os.makedirs(valid_dir)
    except:
        pass
    try:
        os.makedirs(mixed_train_dir)
    except:
        pass

    
    all_lines = ["img,age,readable"]
    for k,v in labeled_images.items():
        img_name = k
        if domain == 'datasets_north':
            manual_age = img_name.split("_age_")[1].split(".png")[0]
        else:
            with open("datasets_baltic/all_data_map.json") as fson:
                data_map = json.load(fson)
            manual_age = data_map[img_name]

        img = cv2.imread("{}/{}".format(source_dir, img_name))
        cv2.imwrite("{}/{}".format(train_dir, img_name), img)
        cv2.imwrite("{}/{}_a.jpg".format(mixed_train_dir, img_name), img)

        flip_img = cv2.flip(img, 1)
        cv2.imwrite("{}/{}".format(valid_dir, img_name), flip_img)
        cv2.imwrite("{}/{}_b.jpg".format(mixed_train_dir, img_name), flip_img)

        all_lines.append("{}_a.jpg,{},2".format(img_name, manual_age))
        all_lines.append("{}_b.jpg,{},2".format(img_name, manual_age))

    with open("{}/{}_{}_{}_train.csv".format(mixed_train_dir, domain, split_name,fold_num), "w") as fout:
        fout.write("\n".join(all_lines))

    all_json_files = glob.glob("{}/*.json".format(train_dir) )
    for jsonf in all_json_files:
        strs = jsonf.replace("\\","/").split("/")
        shutil.copyfile(jsonf, "{}/{}".format(valid_dir, strs[-1])  )
        
        
domain = "datasets_north"
all_train_json = []
all_train_json.extend(glob.glob("{}/train_randsub*/".format(domain)))
all_train_json.extend(glob.glob("{}/train_cod*/".format(domain)))
all_train_json.extend(glob.glob("{}/train_saithe*/".format(domain)))
for item in all_train_json:
    strs = item.replace("\\", "/").split("/")
    folder_name = strs[1]
    folder_str = folder_name.split("_")
    split_name = folder_str[1]
    fold_num = int(folder_str[2])
    create_train_val_from_json(fold_num, split_name, domain=domain)
    

domain = "datasets_baltic"
all_train_json = []
all_train_json.extend(glob.glob("{}/train_fold*/".format(domain)))

for item in all_train_json:
    strs = item.replace("\\", "/").split("/")
    folder_name = strs[1]
    folder_str = folder_name.split("_")
    split_name = folder_str[1]
    fold_num = int(folder_str[2])
    create_train_val_from_json(fold_num, split_name, domain=domain)
    
    
