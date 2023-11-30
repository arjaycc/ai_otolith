from django.shortcuts import render
import skimage.io
import time
import shutil
import json
import os
import glob
import numpy as np
import cv2


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
    #     for i in range(N):
#         y1, x1, y2, x2 = boxes[i]
#         cv2.putText(image, str(captions[i]), (int(x1),int(y1) ),  cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0), 3 )

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
    # result_path = "autolith/static/test_again/{}".format(image_name)
    # print_instances_unet(image_copy, r['rois'],  r['masks'], r['class_ids'], ['bg', 'annulus'], 
    #     captions=[str(t) for t in range(len(r['masks']))], scores=range(len(r['masks'])), axl=result_path)
    

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

