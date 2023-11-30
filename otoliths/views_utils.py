from django.shortcuts import render
import skimage.io
import time
import shutil
import json
import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import colorsys
import random

def predict_mrcnn(raw_image, sq_img, window, dataset, folder, image_name, mode="annulus"):
    from mrcnn import model as modellib
    from mrcnn.mrcnn_aux import InferenceConfig
    if mode == 'annulus':
        name = "mrcnn_arcring0run1_2"
    else:
        name = "mrcnncore_core_6"
    inference_config = InferenceConfig()
    inference_config.RPN_NMS_THRESHOLD =  0.8 #train_params["rpn_nms"]
    inference_config.DETECTION_MIN_CONFIDENCE = 0.6 #train_params["detection_confidence"]
    inference_config.DETECTION_NMS_THRESHOLD = 0.2 #train_params["detection_nms"]
    model = modellib.MaskRCNN(mode="inference", 
                              config=inference_config,
                              model_dir='datasets_baltic/{}/'.format(name))
    OTO_MODEL_PATH = 'datasets_baltic/{}/mrcnn_checkpoint.h5'.format(name)
    model.load_weights(OTO_MODEL_PATH, by_name=True)
    results = model.detect([sq_img], verbose=1)
    r = results[0]
    print(r)
    image_copy = sq_img.copy()
    result_name = "mrcnn_image_{}.png".format(image_name)
    result_path = "autolith/static/detail/{}".format(result_name)

    cx = 255
    cy = 255
    count_detected_masks(results, cx, cy, image_copy=image_copy, class_names=['bg','annulus'], fname=result_name)
    from keras import backend as K
    K.clear_session()


def predict_unet(raw_image, sq_img, window, dataset, folder, image_name):
    import skimage.io
    import time
    import tensorflow as tf

    from tensorflow.keras.models import load_model
    model = load_model('datasets_baltic/unet_arcring0run1_37/unet_arcring0run1_37.h5', compile=False)
    model._make_predict_function()


    print(image_name)
    print("#########")

    img_gray = skimage.color.rgb2gray(sq_img)
    Z_val =  np.zeros( ( 1, 512,512,1), dtype=np.float32)
    Z_val[0,:,:,0] = img_gray

    preds_test = model.predict(Z_val[0:1], verbose=1)
    print(preds_test.shape)
    cv2.imwrite("autolith/static/detail/unetmask_{}".format(image_name), preds_test[0].squeeze()*255)
    preds_test_t = (preds_test > 0.50).astype(np.uint8)
    #---------

    print("loading json")
    with open("autolith/static/json_template.json") as fin:
        main_json = json.load(fin)

    fname = image_name 
    with open("autolith/static/extra_json.json") as fin:
        extra_json = json.load(fin)
    image_path = 'autolith/static/data/{}/{}/{}'.format(dataset, folder, image_name)

    st = os.stat(image_path)
    fsize = int(st.st_size)

    main_json["_via_image_id_list"].append("{}{}".format(fname, fsize) )

    new_item = {}
    new_item["{}{}".format(fname,fsize)] = extra_json["namesize"].copy()
    new_item["{}{}".format(fname,fsize)]["filename"] = fname
    new_item["{}{}".format(fname,fsize)]["size"] = int(fsize)
    new_item["{}{}".format(fname,fsize)]["regions"] = []

    item =  preds_test_t[0].squeeze() #[window[0]:window[2], :]
    # item = cv2.resize(item, (raw_image.shape[1],raw_image.shape[0]) )
    _contours, hierarchy = cv2.findContours(item.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    _contours = sorted(_contours,key=cv2.contourArea, reverse=False)
    _contours = [c for c in _contours if cv2.contourArea(c) > 50 ]

    img_no_mask = sq_img.copy()
    cv2.drawContours(img_no_mask, _contours, -1, (0,255,0), 2)
    skimage.io.imsave('autolith/static/detail/unetcontour_{}'.format(image_name), img_no_mask)
    for c in _contours:
        new_sub_item = {
                'shape_attributes': {'name': 'polyline',
                'all_points_x': [],
                'all_points_y': []},
                'region_attributes': {}
        }
        interval = int(len(c)/20)
        for item_idx, item in enumerate(c):
            if interval>0 and item_idx%interval == 0:
                x,y = item[0]
                new_sub_item["shape_attributes"]["all_points_x"].append(int(x))
                new_sub_item["shape_attributes"]["all_points_y"].append(int(y))
        new_item["{}{}".format(fname,fsize)]["regions"].append(new_sub_item)
    main_json["_via_img_metadata"].update(new_item)
    main_json["_via_settings"]['project']['name'] = '{}'.format(image_name)

    with open("autolith/static/detail/{}.json".format(image_name), "w") as fout:
        json.dump(main_json, fout, indent=4)
    result_name = 'unetcontour_{}'.format(image_name)


def load_blank_marks(request, raw_image, sq_img, window, dataset, folder, image_name, markings="dots", mode='default'):

    print(image_name)
    print("#########")
    all_json_files = glob.glob("autolith/static/new_annotations/{}.json".format(image_name))
    json_annotations = {}
    all_list = {}
    for json_file in all_json_files:
        _annotations = json.load(open(json_file))
        
        _annotations = list(_annotations['_via_img_metadata'].values())
        print(_annotations)
        _annotations = [a for a in _annotations if a['regions']]

        for a in _annotations:
            json_annotations[a['filename']] = (a, a["size"])

    img_gray = skimage.color.rgb2gray(sq_img)
    print("loading json")
    with open("autolith/static/json_template.json") as fin:
        main_json = json.load(fin)

    fname = image_name 
    with open("autolith/static/extra_json.json") as fin:
        extra_json = json.load(fin)
    image_path = 'autolith/static/data/{}/{}/{}'.format(dataset, folder, image_name)

    shutil.copyfile(image_path, 'autolith/static/detail/{}'.format(image_name))

    st = os.stat(image_path)
    fsize = int(st.st_size)

    main_json["_via_image_id_list"].append("{}{}".format(fname, fsize) )
    
    new_item = {}
    if fname in json_annotations:
        a, _ = json_annotations[fname]
        new_item["{}{}".format(fname,fsize)] = a
        new_item["{}{}".format(fname,fsize)]["filename"] = fname
        new_item["{}{}".format(fname,fsize)]["size"] = int(fsize)
    else:
        new_item["{}{}".format(fname,fsize)] = extra_json["namesize"].copy()
        new_item["{}{}".format(fname,fsize)]["filename"] = fname
        new_item["{}{}".format(fname,fsize)]["size"] = int(fsize)
        new_item["{}{}".format(fname,fsize)]["regions"] = []

    main_json["_via_img_metadata"].update(new_item)
    
    with open("autolith/static/detail/main_var.json", "w") as fout:
        json.dump(main_json, fout, indent=4)

    with open("autolith/static/detail/main_var.json") as fin:
        json_var = json.load(fin)
        json_var = json.dumps(json_var)

    if mode == 'brush':
        return render(request, 'otoliths/researchers_interact.html', {'pass_var': json_var, 'fname': image_name, 'mode': 'default'})
    return render(request, 'otoliths/researchers.html', {'pass_var': json_var, 'fname': image_name, 'mode': 'brush'})


def convert_data_marks(request, raw_image, sq_img, window, dataset, folder, image_name, markings="dots", mode='default'):

    all_json_files = glob.glob("autolith/static/new_annotations/{}.json".format(image_name))
    json_annotations = {}
    all_list = {}
    for json_file in all_json_files:
        _annotations = json.load(open(json_file))
        
        _annotations = list(_annotations['_via_img_metadata'].values())
        print(_annotations)
        _annotations = [a for a in _annotations if a['regions']]

        for a in _annotations:
            json_annotations[a['filename']] = (a, a["size"])

    with open("autolith/static/json_template.json") as fin:
        main_json = json.load(fin)
        
    subset_count = 0
    
    fname = image_name
    image = raw_image

    gray_copy = raw_image.copy()
    gray_copy_copy = raw_image.copy()
    a, fsize = json_annotations[fname]
    poly = [r['shape_attributes'] for r in a['regions']]

    cy, cx = int(0.33*raw_image.shape[0]), int(raw_image.shape[1]/2.0)
    nr_cy, nr_cx = cy, cx
    csv_points_left = []
    csv_points_right = []
    for idx, pol in enumerate(poly):
        print(pol)
        if pol['name'] == 'point':
            rr_cx = pol['cx']
            rr_cy = pol['cy']
            dist = np.sqrt((rr_cx-cx)  **2 + (rr_cy-cy)**2)
            if rr_cx > nr_cx:
                csv_points_right.append( (int(rr_cx), int(rr_cy), dist) )
            else:
                csv_points_left.append( (int(rr_cx), int(rr_cy), dist) )
    print(csv_points_left)
    if len(csv_points_left) > 0:
        all_x = [nr_cx]
        all_y = [nr_cy]
        for item in csv_points_left:
            xx,yy, dd = item
            all_x.append(xx)
            all_y.append(yy)
        mleft,bleft = np.polyfit(all_x, all_y, 1)
        csv_points_left =  sorted(csv_points_left, key=lambda x: x[2], reverse=False)
        
    if len(csv_points_right) > 0:
        all_x = [nr_cx]
        all_y = [nr_cy]
        for item in csv_points_right:
            xx,yy, dd = item
            all_x.append(xx)
            all_y.append(yy)
        mright,bright = np.polyfit(all_x, all_y, 1)
        csv_points_right =  sorted(csv_points_right, key=lambda x: x[2], reverse=False)
    
    count = 0
    cand_pts = []
    best_diff = 9999
    intersect_x = 0
    intersect_y = 0
    subset_pts = []
    with_diff = {}
    prev_dd = 0
    
    gray_mask = np.zeros([gray_copy_copy.shape[0], gray_copy_copy.shape[1], 1], dtype=np.uint8)
    ncx = cx
    if len(csv_points_left) > 0:
        ncy =int(mleft*cx + bleft)
    
        for idx, item in enumerate(csv_points_left):
            xx,yy, dd = item

            rr = np.sqrt(    (xx-ncx)  **2 + (yy-ncy)**2)
            theta_incline = int(np.arctan2( (yy-ncy), (xx-ncx) ) * 180/np.pi) + 180.0

            arc_len = 2000*0.1
            theta = ((arc_len/rr)/2.0)*180/np.pi
            start_angle = int(180 - theta ) #+ 180
            end_angle = int(180 + theta) #+ 180

            if True: #dd - prev_dd < 45:
                cv2.ellipse(gray_mask, (ncx,ncy), (int(rr),int(rr/3)), theta_incline, start_angle, end_angle, (255,255,255), thickness=30) 

            prev_dd = dd
        
    prev_dd = 0
    ncx = cx
    if len(csv_points_right) > 0:
        ncy = int(mright*cx + bright)
        for idx, item in enumerate(csv_points_right):
            xx,yy,dd = item
    #         if xx > cx:
    #             continue
            rr = np.sqrt(    (xx-ncx)  **2 + (yy-ncy)**2)
            theta_incline = int(np.arctan2( (yy-ncy), (xx-ncx) ) * 180/np.pi) + 180.0

            arc_len = 2000*0.1
            theta = ((arc_len/rr)/2.0)*180/np.pi
            start_angle = int(180 - theta ) 
            if dd - prev_dd < 45:
                end_angle = int(180 + theta-int(theta/3)) 
            else:
                end_angle = int(180 + theta) 

            if True: #dd - prev_dd < 45:
                cv2.ellipse(gray_mask, (ncx,ncy), (int(rr),int(rr/3)), theta_incline, start_angle, end_angle, (255,255,255), thickness=30)
            else:
                cv2.ellipse(gray_mask, (ncx,ncy), (int(rr),int(rr/4)), theta_incline, start_angle, end_angle, (255,255,255), thickness=30)
            prev_dd = dd
        
    _contours, hierarchy = cv2.findContours(gray_mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    _contours = sorted(_contours,key=cv2.contourArea, reverse=False)
    _contours = [c for c in _contours if cv2.contourArea(c) > 10 ]
    
    print("loading json")
    with open("autolith/static/extra_json.json") as fin:
        extra_json = json.load(fin)
        
    main_json["_via_image_id_list"].append("{}{}".format(fname, fsize) )

    new_item = {}
    new_item["{}{}".format(fname,fsize)] = extra_json["namesize"].copy()
    new_item["{}{}".format(fname,fsize)]["filename"] = fname
    new_item["{}{}".format(fname,fsize)]["size"] = int(fsize)
    new_item["{}{}".format(fname,fsize)]["regions"] = []
    
    for c in _contours:
        new_sub_item = {
                'shape_attributes': {'name': 'polyline',
                'all_points_x': [],
                'all_points_y': []},
                'region_attributes': {}
        }

        interval = int(len(c)/20)
        if interval < 1:
            interval = 1
        for item_idx, item in enumerate(c):
            if interval>0 and item_idx%interval == 0:
                x,y = item[0]
                new_sub_item["shape_attributes"]["all_points_x"].append(int(x))
                new_sub_item["shape_attributes"]["all_points_y"].append(int(y))
        new_item["{}{}".format(fname,fsize)]["regions"].append(new_sub_item)

    main_json["_via_img_metadata"].update(new_item)
    subset_count += 1
   
    with open("autolith/static/detail/main_var.json", "w") as fout:
        json.dump(main_json, fout, indent=4)
    with open("autolith/static/detail/main_var.json") as fin:
        json_var = json.load(fin)
        json_var = json.dumps(json_var)

    mode = 'def'
    if mode == 'brush':
        return render(request, 'otoliths/researchers_interact.html', {'pass_var': json_var, 'fname': image_name, 'mode': 'default'})
    return render(request, 'otoliths/researchers.html', {'pass_var': json_var, 'fname': image_name, 'mode': 'brush'})



def load_data_marks(request, raw_image, sq_img, window, dataset, folder, image_name, markings="dots", mode='default'):
    import skimage.io
    import time
    import shutil
    
    all_json_files = glob.glob("autolith/static/new_annotations/{}.json".format(image_name))
    json_annotations = {}
    all_list = {}
    for json_file in all_json_files:
        _annotations = json.load(open(json_file))
        
        _annotations = list(_annotations['_via_img_metadata'].values())
        print(_annotations)
        _annotations = [a for a in _annotations if a['regions']]

        for a in _annotations:
            json_annotations[a['filename']] = (a, a["size"])

    print("loading json")
    with open("autolith/static/json_template.json") as fin:
        main_json = json.load(fin)

    fname = image_name 
    with open("autolith/static/extra_json.json") as fin:
        extra_json = json.load(fin)
    image_path = 'autolith/static/data/{}/{}/{}'.format(dataset, folder, image_name)

    shutil.copyfile(image_path, 'autolith/static/detail/{}'.format(image_name))
    a, fsize = json_annotations[fname]

    main_json["_via_image_id_list"].append("{}{}".format(fname, fsize) )

    new_item = {}
    new_item["{}{}".format(fname,fsize)] = a
    new_item["{}{}".format(fname,fsize)]["filename"] = fname
    new_item["{}{}".format(fname,fsize)]["size"] = int(fsize)

    main_json["_via_img_metadata"].update(new_item)
    
    with open("autolith/static/detail/main_var.json", "w") as fout:
        json.dump(main_json, fout, indent=4)
    with open("autolith/static/detail/main_var.json") as fin:
        json_var = json.load(fin)
        json_var = json.dumps(json_var)

    if mode == 'brush':
        return render(request, 'otoliths/researchers_interact.html', {'pass_var': json_var, 'fname': image_name, 'mode': 'default'})
    return render(request, 'otoliths/researchers.html', {'pass_var': json_var, 'fname': image_name, 'mode': 'brush'})



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

def count_detected_masks(results, cx, cy, image_copy=None, class_names=[], fname="test"):
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
        clist.append([contour,dist, idx, midx])

    sorted_c = sorted(clist, key=lambda kx: kx[1], reverse=True)

    labels = []
    labels_left = []
    labels_right = []
    for c in sorted_c:
        idx = c[2]
        dist = c[1]
        midx = c[-1]
        val = 1
        key = 0
        for label in labels:
            intersect, start_angle, end_angle = check_angle_intersection(label, c, alist)
            if intersect:
                val = label[0] + 1
                key = label[1]
        labels.append([val, idx, dist, c , midx])
        if midx > cx:
            labels_right.append([val, idx, dist, c ])
        else:
            labels_left.append([val, idx, dist, c  ])
    label_list = [l for l in labels]
    label_list_left = [l[0:2] for l in labels_left]
    label_list_right = [l[0:2] for l in labels_right]
    try:
        sorted_labels = sorted(label_list, key=lambda x: x[0])
        ai_reading = sorted_labels[-1][0]
    except:
        ai_reading = 0
    try:
        sorted_labels_left = sorted(label_list_left, key=lambda x: x[0])
        ai_reading_left = sorted_labels_left[-1][0]
    except:
        ai_reading_left = 0
    try:
        sorted_labels_right = sorted(label_list_right, key=lambda x: x[0])
        ai_reading_right = sorted_labels_right[-1][0]
    except:
        ai_reading_right = 0

    sorted_labels = sorted(label_list, key=lambda x: x[1])
    scores = []
    for s in sorted_labels:
        scores.append(s[0])
    scores = np.array(scores, dtype=np.float32)

    captions = []
    for sc in scores:
        captions.append("{}".format(int(sc)) )
    print_instances(image_copy, r['rois'], r['masks'], r['class_ids'], 
                    class_names, scores= r['scores'], ax="autolith/static/detail/{}".format(fname), captions=captions, 
                    sorted_labels=sorted_labels, ai_left=ai_reading_left, ai_right=ai_reading_right, cx=cx, cy=cy)
    return ai_reading


### FOR VISUALIZATION, we use the code by Matterport (Copyright (c) 2017) from Mask RCNN implementation

def apply_mask(image, mask, color, alpha=0.5):
    """
    >> Copyright (c) 2017 Matterport, Inc. <<
    Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def random_colors(N, bright=True):
    """
    >> Copyright (c) 2017 Matterport, Inc. <<
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    # colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    colors = []
    for hh in hsv:
        colors.append(colorsys.hsv_to_rgb(*hh))
    random.shuffle(colors)
    return colors


def print_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None, sorted_labels=None, ai_left=0, ai_right=0, cx=255, cy=255):
    """
    >> Copyright (c) 2017 Matterport, Inc. <<
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
#     if not ax:
    _, axp = plt.subplots(1, figsize=figsize)
#         auto_show = True
    axp.axis('off')
    plt.tight_layout()

    # Generate random colors
    colors = colors or random_colors(N)
    
    # Show area outside image boundaries.
    height, width = image.shape[:2]

    from matplotlib import patches, lines
    from matplotlib.patches import Polygon
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=4,
                            alpha=0.7, linestyle="dashed",
                            edgecolor=color, facecolor='none')
        axp.add_patch(p)
        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            x = random.randint(x1, (x1 + x2) // 2)
            caption = "{} {}".format(label, score) if score else label
        else:
            caption = captions[i]
        lab = sorted_labels[i]
        if lab[-1] > cx:
            caption = '{}'.format(ai_right - int(caption) + 1)
        else:
            caption = '{}'.format(ai_left - int(caption) + 1)
        score_top = "annulus {:.3f}".format(scores[i])
        axp.text(x1-27, y1+8, score_top, color='w', size=24, backgroundcolor="none")  
        axp.text(x2, y2, caption, color='w', size=24, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

    axp.imshow(masked_image.astype(np.uint8))
    plt.savefig(ax, transparent=False)
    return masked_image #plt.gcf()
