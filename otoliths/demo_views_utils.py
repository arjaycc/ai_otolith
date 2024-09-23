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
import warnings
import scipy

def predict_mrcnn(raw_image, sq_img, window, dataset, folder, image_name, mode="annulus", model_name='mrcnn_baltic_sample.h5', model_path='', run_label='mrcnn_annuli'):
    from mrcnn import model as modellib
    from mrcnn.mrcnn_aux import InferenceConfig
    if mode == 'annulus':
        name = "models"
    else:
        name = "core"
    inference_config = InferenceConfig()
    inference_config.RPN_NMS_THRESHOLD =  0.8 #train_params["rpn_nms"]
    inference_config.DETECTION_MIN_CONFIDENCE = 0.6 #train_params["detection_confidence"]
    inference_config.DETECTION_NMS_THRESHOLD = 0.2 #train_params["detection_nms"]
    model = modellib.MaskRCNN(mode="inference", 
                              config=inference_config,
                              model_dir='datasets_user/{}/'.format(name))

    try:
        model.load_weights(model_path, by_name=True)
        print("LOADING FROM PATH")
    except:
        OTO_MODEL_PATH = 'datasets_user/{}/{}'.format(name, model_name)
        model.load_weights(OTO_MODEL_PATH, by_name=True)
    results = model.detect([sq_img], verbose=1)
    r = results[0]
    print(r)
    image_copy = sq_img.copy()

    try:
        with open("{}/CENTER.json".format(dataset)) as fin:
            center_map = json.load(fin)
        cx, cy = center_map[image_name]
    except:
        # print(image_name)
        cx = 255
        cy = 255
    ai_reading = count_detected_masks(results, cx, cy, image_copy=image_copy, class_names=['bg','annulus'], fname=image_name, folder=folder, run_label=run_label)


    #==================================================
    # if dataset == 'datasets_user':
    print("loading json")
    with open("autolith/static/json_template.json") as fin:
        main_json = json.load(fin)

    fname = image_name 
    with open("autolith/static/extra_json.json") as fin:
        extra_json = json.load(fin)

    image_path = '{}/{}/{}'.format(dataset, folder, image_name)
    st = os.stat(image_path)
    fsize = int(st.st_size)

    main_json["_via_image_id_list"].append("{}{}".format(fname, fsize) )

    new_item = {}
    new_item["{}{}".format(fname,fsize)] = extra_json["namesize"].copy()
    new_item["{}{}".format(fname,fsize)]["filename"] = fname
    new_item["{}{}".format(fname,fsize)]["size"] = int(fsize)
    new_item["{}{}".format(fname,fsize)]["regions"] = []

    for current_idx in range(len(r['rois'])):
        main_idx = current_idx #get_main_detection(r, "nucleus")
        boxes = r['rois']
        scores = r['scores']
        masks = r['masks']
        print(scores)
        try:
            new_mask = masks[:,:,main_idx]
            print("*********")
            print(new_mask.shape)
        except:
            print("except---------------")
            continue

        new_mask = new_mask.astype(np.uint8)

        item =  new_mask.squeeze()[window[0]:window[2], :]
        new_mask = cv2.resize(item, (raw_image.shape[1],raw_image.shape[0]) )

        ypred = new_mask.copy()
        pthresh = np.zeros([ypred.shape[0], ypred.shape[1],1], dtype=np.uint8)
        pthresh[:,:,0] = ypred[:,:]
        pcontours, hierarchy = cv2.findContours(pthresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        pred_contour = max(pcontours, key=cv2.contourArea)

        c = pred_contour
        new_sub_item = {
                'shape_attributes': {'name': 'polyline',
                'all_points_x': [],
                'all_points_y': []},
                'region_attributes': {}
        }
        interval = int(len(c)/30)
        for item_idx, item in enumerate(c):
            if interval>0 and item_idx%interval == 0:
                x,y = item[0]
                new_sub_item["shape_attributes"]["all_points_x"].append(int(x))
                new_sub_item["shape_attributes"]["all_points_y"].append(int(y))
        new_item["{}{}".format(fname,fsize)]["regions"].append(new_sub_item)
    main_json["_via_img_metadata"].update(new_item)
    main_json["_via_settings"]['project']['name'] = '{}'.format(image_name)

    if dataset == 'datasets_user':
        with open('{}/{}/{}_json.json'.format(dataset, folder, image_name), "w") as fout:
            json.dump(main_json["_via_img_metadata"], fout, indent=4)
    else:
        # if dataset != 'datasets_user':
        print("????????????????????????????")
        print(dataset)
        with open("autolith/static/new_annotations/{}.json".format(image_name), "w") as fout:
            json.dump(main_json, fout, indent=4)

    #==========================================

    # fig = plt.figure()
    # # plt.plot(self.epoch_counts, self.losses, label="{}".format(self.measurement))
    # # plt.plot(self.epoch_counts, self.val_losses, label="val_{}".format(self.measurement))

    
    # plt.legend()
    # fig.savefig("autolith/static/plots/{}.png".format("test") )



    from keras import backend as K
    K.clear_session()

    return ai_reading


def predict_unet(raw_image, sq_img, window, dataset, folder, image_name, model_name='unet_baltic_sample.h5', model_path='', run_label='unet_annuli'):
    import skimage.io
    import time
    import tensorflow as tf

    from tensorflow.keras.models import load_model
    try:
        model = load_model(model_path, compile=False)
        print("LOADING FROM PATH")
    except:
        model = load_model('datasets_user/models/{}'.format(model_name), compile=False)
    model._make_predict_function()

    input_shape = model.layers[0].input_shape[0]
    CHANNELS = input_shape[-1]
    print(image_name)
    print("#########")

    img_gray = skimage.color.rgb2gray(sq_img)
    img = skimage.color.gray2rgb(img_gray)
    if CHANNELS == 1:
        Z_val =  np.zeros( ( 1, 512, 512,1), dtype=np.float32)
        Z_val[0,:,:,0] = img[:,:,0]
    else:
        Z_val =  np.zeros( ( 1, 512, 512,3), dtype=np.float32)
        Z_val[0,:,:,:] = img


    preds_test = model.predict(Z_val[0:1], verbose=1)
    print(preds_test.shape)
    preds_test_t = (preds_test > 0.50).astype(np.uint8)
    #---------

    try:
        with open("{}/CENTER.json".format(dataset)) as fin:
            center_map = json.load(fin)
        cx, cy = center_map[image_name]
    except:
        cx = 255
        cy = 255

    item =  preds_test_t[0].squeeze()
    ai_reading, ai_contours, ai_labels = count_prediction_output(dataset, item, cx, cy)
    print([cx, cy])
    print("AI READING:: {} :: {}", image_name, ai_reading)
    print("loading json")
    with open("autolith/static/json_template.json") as fin:
        main_json = json.load(fin)

    fname = image_name 
    with open("autolith/static/extra_json.json") as fin:
        extra_json = json.load(fin)

    image_path = '{}/{}/{}'.format(dataset, folder, image_name)
    st = os.stat(image_path)
    fsize = int(st.st_size)

    main_json["_via_image_id_list"].append("{}{}".format(fname, fsize) )

    new_item = {}
    new_item["{}{}".format(fname,fsize)] = extra_json["namesize"].copy()
    new_item["{}{}".format(fname,fsize)]["filename"] = fname
    new_item["{}{}".format(fname,fsize)]["size"] = int(fsize)
    new_item["{}{}".format(fname,fsize)]["regions"] = []


    sq_item = np.zeros([512, 512], dtype=np.uint8)
    cv2.drawContours(sq_item, ai_contours, -1, (255,255,255), -1)
    # sq_item =  preds_test_t[0].squeeze() #[window[0]:window[2], :]
    item =  sq_item.squeeze()[window[0]:window[2], :]
    item = cv2.resize(item, (raw_image.shape[1],raw_image.shape[0]) )
    _contours, hierarchy = cv2.findContours(item.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    _contours = sorted(_contours,key=cv2.contourArea, reverse=False)
    _contours = [c for c in _contours if cv2.contourArea(c) > 50 ]

    # sq_contours, hierarchy = cv2.findContours(sq_item.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # sq_contours = sorted(sq_contours,key=cv2.contourArea, reverse=False)
    # sq_contours = [c for c in sq_contours if cv2.contourArea(c) > 50 ]

    img_no_mask = sq_img.copy()

    for ai_label in ai_labels:
        (xpos, ypos, wd, ht) = cv2.boundingRect(ai_label[-1])
        cv2.putText(img_no_mask, str(ai_label[0]), (int(xpos+wd),int(ypos+ht) ),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0) )
        cv2.drawContours(img_no_mask, [ai_label[-1]], -1, (0,255,0), 2)
    # cv2.drawContours(img_no_mask, ai_contours, -1, (0,255,0), 2)
    skimage.io.imsave('autolith/static/detail/{}'.format(image_name), img_no_mask)
    try:
        os.makedirs('autolith/static/data/{}/{}'.format(run_label, folder))
    except:
        pass
    skimage.io.imsave('autolith/static/data/{}/{}/{}'.format(run_label, folder, image_name), img_no_mask)

    # if dataset == 'datasets_user':
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

    if dataset == 'datasets_user':
        with open('{}/{}/{}_json.json'.format(dataset, folder, image_name), "w") as fout:
            json.dump(main_json["_via_img_metadata"], fout, indent=4)
    else:
        with open("autolith/static/new_annotations/{}.json".format(image_name), "w") as fout:
            json.dump(main_json, fout, indent=4)


    from keras import backend as K
    K.clear_session()

    return ai_reading


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
    image_path = '{}/{}/{}'.format(dataset, folder, image_name)

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



# def load_data_marks(request, raw_image, sq_img, window, dataset, folder, image_name, markings="dots", mode='default'):
#     import skimage.io
#     import time
#     import shutil
    
#     print("started processing...")
#     # all_json_files = glob.glob("autolith/static/new_annotations/{}.json".format(image_name))
#     all_json_files = glob.glob("{}/{}/*.json".format(dataset, folder))
#     # if not all_json_files:
#     #     all_json_files = glob.glob("autolith/static/annotations/*.json")
#     json_annotations = {}
#     all_list = {}
#     for json_file in all_json_files:
#         _annotations = json.load(open(json_file))
#         try:
#             _annotations = list(_annotations['_via_img_metadata'].values())
#         except:
#             _annotations = list(_annotations.values())
#         _annotations = [a for a in _annotations if a['regions']]
#         for a in _annotations:
#             json_annotations[a['filename']] = (a, a["size"])

#     print("loading json")
#     with open("autolith/static/json_template.json") as fin:
#         main_json = json.load(fin)

#     fname = image_name 
#     with open("autolith/static/extra_json.json") as fin:
#         extra_json = json.load(fin)
#     # image_path = 'autolith/static/data/{}/{}/{}'.format(dataset, folder, image_name)
#     image_path = '{}/{}/{}'.format(dataset, folder, image_name)
#     shutil.copyfile(image_path, 'autolith/static/detail/{}'.format(image_name))
#     try:
#         a, fsize = json_annotations[fname]
#     except:
#         # with open("autolith/static/detail/main_var.json") as fin:
#         #     json_var = json.load(fin)
#         #     json_var = json.dumps(json_var)
#         print("empty")
#         json_var = {}
#         if mode == 'brush':
#             return render(request, 'demo/researchers_interact.html', {'pass_var': json_var, 'fname': image_name, 'mode': 'annuli_annotate', 'drawing_tool': 'default', 'dataformat': 'annuli'})
#         return render(request, 'demo/researchers.html', {'pass_var': json_var, 'fname': image_name, 'mode': 'annuli_annotate', 'drawing_tool': 'brush', 'dataformat': 'annuli',})


#     print("annotation found")
#     main_json["_via_image_id_list"].append("{}{}".format(fname, fsize) )

#     new_item = {}
#     new_item["{}{}".format(fname,fsize)] = a
#     new_item["{}{}".format(fname,fsize)]["filename"] = fname
#     new_item["{}{}".format(fname,fsize)]["size"] = int(fsize)

#     main_json["_via_img_metadata"].update(new_item)
    
#     with open("autolith/static/detail/main_var.json", "w") as fout:
#         json.dump(main_json, fout, indent=4)
#     with open("autolith/static/detail/main_var.json") as fin:
#         json_var = json.load(fin)
#         json_var = json.dumps(json_var)

#     if mode == 'brush':
#         return render(request, 'demo/researchers_interact.html', {'pass_var': json_var, 'fname': image_name, 'mode': 'annuli_annotate', 'drawing_tool': 'default', 'dataformat': 'annuli',})
#     return render(request, 'demo/researchers.html', {'pass_var': json_var, 'fname': image_name, 'mode': 'annuli_annotate', 'drawing_tool': 'brush', 'dataformat': 'annuli',})


# def load_data_marks_new(request, dataset, folder, annotype='both', drawing_tool='default'):
def load_data_marks(request, raw_image, sq_img, window, dataset, folder, image_name, markings="dots", mode='default'):
    all_image_files = glob.glob("{}/{}/*.png".format(dataset, folder))


    all_json_files = glob.glob("{}/{}/*.json".format(dataset, folder))

    print(all_json_files)
    # raise ValueError

    # with open("autolith/static/new_annotations/{}.json".format(image_name), "w") as fout:
    # train_json_files = glob.glob('REPO/all_sulcus/*.json')
    train_annotations = {}
    for json_file in all_json_files:
        # print(json_file)
        _annotations = json.load(open(json_file))
        # print(_annotations)
        train_annotations.update(_annotations)
    train_annotations = list(train_annotations.values())
    # print(train_annotations)
    train_annotations = [a for a in train_annotations if a['regions']]

    train_anno_map = {}
    for a in train_annotations:
        train_anno_map[a['filename']] = a
    for img_file in all_image_files:
        strs = img_file.replace("\\", "/").split("/") 
        fname = strs[-1] #a['filename']
        json_file = "autolith/static/new_annotations/{}.json".format(fname)
        if os.path.isfile(json_file):
            _annotations = json.load(open(json_file))
            _alist = list(_annotations["_via_img_metadata"].values())

            for _a in _alist:
                if _a['regions'] and fname == _a['filename']:
                    train_anno_map[fname] = _a
                    break


    with open("autolith/static/json_template.json") as fin:
        main_json = json.load(fin)

    for image_path in all_image_files:
        strs = image_path.replace("\\", "/").split("/") 
        sep_name = strs[-1]
        if sep_name != image_name:
            continue

        shutil.copyfile(image_path, 'autolith/static/detail/{}'.format(strs[-1]))

        # image_path = ff #"{}/ALL/{}".format(destination, key)
        st = os.stat("{}".format(image_path))
        fsize = int(st.st_size)
        main_json["_via_image_id_list"].append("{}{}".format(sep_name, fsize) )

        with open("autolith/static/extra_json.json") as fin:
            extra_json = json.load(fin)
        new_item = {}
        new_item["{}{}".format(sep_name,fsize)] = extra_json["namesize"].copy()
        new_item["{}{}".format(sep_name,fsize)]["filename"] = sep_name
        new_item["{}{}".format(sep_name,fsize)]["size"] = int(fsize)
        try:
            new_item["{}{}".format(sep_name,fsize)]["regions"] = train_anno_map[sep_name]['regions']
        except:
            new_item["{}{}".format(sep_name,fsize)]["regions"] = []

        new_sub_item = {
                'shape_attributes': {'name': 'polyline',
                'all_points_x': [],
                'all_points_y': []},
                'region_attributes': {}
        }
        new_item["{}{}".format(sep_name,fsize)]["regions"].append(new_sub_item)

        main_json["_via_img_metadata"].update(new_item)
    main_json['_via_settings']['project']['name'] = folder


    with open("autolith/static/detail/main_var.json", "w") as fout:
        json.dump(main_json, fout, indent=4)

    with open("autolith/static/detail/main_var.json") as fin:
        json_var = json.load(fin)
        json_var = json.dumps(json_var)

    if True:# drawing_tool == 'default':
        return render(request, 'demo/researchers.html', {'pass_var': json_var, 'fname': 'test', 'mode': 'annuli_annotate', 'drawing_tool': 'brush', 'dataformat': 'annuli', 'dataset': dataset, 'folder': folder})
    return render(request, 'demo/researchers_interact.html', {'pass_var': json_var, 'fname': 'test', 'mode': 'annuli_annotate', 'drawing_tool': 'default', 'dataformat': 'annuli', 'dataset': dataset, 'folder': folder})


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

def count_detected_masks(results, cx, cy, image_copy=None, class_names=[], fname="test", folder="test", run_label='mrcnn_annuli'):
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

    result_name = "{}".format(fname)
    print_instances(image_copy, r['rois'], r['masks'], r['class_ids'], 
                    class_names, scores= r['scores'], ax="autolith/static/detail/{}".format(result_name), captions=captions, 
                    sorted_labels=sorted_labels, ai_left=ai_reading_left, ai_right=ai_reading_right, cx=cx, cy=cy)
    try:
        os.makedirs("autolith/static/data/{}/{}".format(run_label, folder))
    except:
        pass
    print_instances(image_copy, r['rois'], r['masks'], r['class_ids'], 
                    class_names, scores= r['scores'], ax="autolith/static/data/{}/{}/{}".format(run_label, folder, fname), captions=captions, 
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

def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    """Resizes an image keeping the aspect ratio unchanged.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    min_scale: if provided, ensure that the image is scaled up by at least
        this percent even if min_dim doesn't require it.
    mode: Resizing mode.
        none: No resizing. Return the image unchanged.
        square: Resize and pad with zeros to get a square image
            of size [max_dim, max_dim].
        pad64: Pads width and height with zeros to make them multiples of 64.
               If min_dim or min_scale are provided, it scales the image up
               before padding. max_dim is ignored in this mode.
               The multiple of 64 is needed to ensure smooth scaling of feature
               maps up and down the 6 levels of the FPN pyramid (2**6=64).
        crop: Picks random crops from the image. First, scales the image based
              on min_dim and min_scale, then picks a random crop of
              size min_dim x min_dim. Can be used in training only.
              max_dim is not used in this mode.

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None
    
    if mode == "none":
        return image, window, scale, padding, crop

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    if min_scale and scale < min_scale:
        scale = min_scale

    # Does it exceed max dim?
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # Resize image using bilinear interpolation
    if scale != 1:
        image = skimage.transform.resize(
            image, (round(h * scale), round(w * scale)),
            order=0, mode="constant", preserve_range=True)

    # Need padding or cropping?
    if mode == "square":
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y:y + min_dim, x:x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))
    return image.astype(image_dtype), window, scale, padding, crop


def resize_mask(mask, scale, padding, crop=None):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    # Suppress warning from scipy 0.13.0, the output shape of zoom() is
    # calculated with round() instead of int()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    if crop is not None:
        y, x, h, w = crop
        mask = mask[y:y + h, x:x + w]
    else:
        mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask
#===================================================================================================================================================================================================================================================


def sort_contours(domain, mask_item, cx, cy):
    import math
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


def get_unused_axis_label(sorted_c, alist, cx, cy, ai_reading):
    label_list = []
    for c in sorted_c:
        idx = c[2]
        xpos = c[3]
        ypos = c[4]
        # unused axis (right axis)
        if xpos <= cx:
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
    return sorted_labels[:ai_reading]


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

    all_labels = []
    for label in sorted_labels:
        all_labels.append([ai_reading-label[0]+1, label[-1]])
    if domain == 'datasets_north':
        try:
            extra_labels = get_unused_axis_label(sorted_c, alist, cx, cy, ai_reading)
            max_label = extra_labels[-1][0]
            for label in extra_labels:
                all_labels.append([max_label-label[0]+1, label[-1]])
        except:
            pass

    return ai_reading, contours, all_labels