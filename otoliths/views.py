from django.shortcuts import render
from django.http import HttpResponse
from django.core.paginator import Paginator
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from .forms import UploadForm, DataFilterForm, RunFormPhase1, RunFormPhase2, EnsembleFilterForm, UnetForm
from .models import OtolithImage, Image

from scipy.signal import find_peaks, detrend
from scipy import signal
from skimage.color import rgb2gray
from skimage.filters import sobel, gaussian, rank
from skimage.morphology import disk, dilation, square, erosion
from skimage.segmentation import watershed, random_walker
from scipy import ndimage
from skimage.measure import label
from skimage import morphology
from skimage.measure import regionprops
from skimage.measure import find_contours
from skimage import img_as_ubyte
from scipy import ndimage as ndi
from scipy.interpolate import interp1d
from skimage.util import img_as_ubyte
from skimage.morphology import disk
from skimage import feature
from skimage.color import rgb2gray
from skimage import feature, exposure
from skimage import img_as_float
from scipy.ndimage import gaussian_filter
from skimage.morphology import reconstruction
from skimage.filters import rank
from scipy import interpolate
import skimage

import pickle
import json
import cv2
import numpy as np
import glob
import re
import math
import random
import sklearn
import matplotlib.pyplot as plt
import os

from .experiments_imgproc import *
from .experiments_mrcnn import *
from .experiments_unet import *
from .views_utils import *

@login_required
def index(request):
    return render(request, 'otoliths/index.html')

@login_required
def images(request):
    qs = OtolithImage.objects.all()
    try:
        group = request.GET['group']
    except:
        group = 'Northsea'
    
    if group == 'Northsea':
        source_dir = 'data/northsea/train'
    elif group == 'Baltic':
        source_dir = 'data/baltic/train'
    else:
        source_dir = 'data/northsea/train'
        
    try:
        year = int(request.GET['year'])
    except:
        pass

    img_files = glob.glob('autolith/static/{}/*.png'.format(source_dir))
    all_images = []
    count = 0
    for img_file in img_files:
        strs = img_file.replace("\\", "/").split("/")
        all_images.append(strs[-1])
        count += 1
        if count > 25:
            break
    return render(request, 'otoliths/images.html', {'images': all_images, 'source_dir': source_dir} )


@csrf_exempt
def map(request):
    if request.method=='POST':
        for k, v in request.POST.items():
            img_data = json.loads(k)
            print(img_data)
    return render(request, 'otoliths/map.html')

@login_required
def researchers(request):
    northsea_files = glob.glob("core/static/*detail_no_mask.png")
    if len(northsea_files) > 0:
        pngfiles = glob.glob("core/static/*_no_mask.png")
        item = pngfiles[0]
        fname = item.replace("\\", "/").split("/")[-1]
        num = fname.split("_")[0]
        with open("core/static/datasets/{}/{}.json".format(num, num)) as fin:
            json_var = json.load(fin)
            json_var = json.dumps(json_var)
    else:
        pngfiles = glob.glob("autolith/static/*_no_mask.png")
        item = pngfiles[0]
        fname = item.replace("\\", "/").split("/")[-1]
        with open("autolith/static/source/temp/{}/{}.json".format(fname.replace("_no_mask.png",""), fname.replace("_no_mask.png",""))) as fin:
            json_var = json.load(fin)
            json_var = json.dumps(json_var)

    return render(request, 'otoliths/researchers.html', {'pass_var': json_var})

@login_required
def ai(request):
    images = []
    return render(request, 'otoliths/aimethod.html', {'images': images } )

# @login_required
# def upload(request):
#     form = UploadForm()
#     return render(request, 'otoliths/upload.html', {'form': form})

@login_required
def analysis(request):
    image1 = 'result1.png'
    image2 = 'result2.png'
    return render(request, 'otoliths/analysis.html', {'image1': image1, 'image2': image2})

@login_required
def detail(request, image_id):
    img_obj = OtolithImage.objects.get(pk=image_id)
    with open('core/static/{}'.format(img_obj.data_path), 'rb') as fin:
        img = pickle.load(fin)
    try:
        process = request.GET['process']
    except:
        process = "Show"
    if process == "Show":
        pngfiles = glob.glob('core/static/*.png')
        for ff in pngfiles:
            os.remove(ff)
        img_new = img.copy()
    elif process == "Contour":
        img_new = img.copy()
    elif process == "SelectRegions":
        img_new = img.copy()
    elif process == "Segment":
        img_resized = img.copy()
        black = np.zeros_like(img_resized)
        black_copy = black.copy()
        mask = black.copy()
        
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 70, 155, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        n = len(contours)-1
        contours = sorted(contours,key=cv2.contourArea, reverse=False)
        c = contours[-2]
        hull = cv2.convexHull(c)
        rect = cv2.boundingRect(c)
        x, y, w, h = rect
        mask = np.ones(img_resized.shape[:2], dtype="uint8") * 255
        cv2.drawContours(mask, [hull], -1,0,-1 )
        new_mask = cv2.bitwise_not(mask)
        img_new = cv2.bitwise_and(img_resized, img_resized, mask=new_mask)

        ofs = 50
        img_resized = img_resized[y-ofs:y+h+ofs, x-ofs:x+w+ofs]
        height = h+100
        width = w+100

        if height >= width:
            new_height = 800
            new_width = int((new_height*width)/height)
            offset_height = 0
            offset_width = 400 - int(new_width/2)
        else:
            new_width = 800
            new_height = int((new_width*height)/width)
            offset_height = 400 - int(new_height/2)
            offset_width = 0

        img_part = cv2.resize(img_resized, (new_width,new_height))
        print(img_part.shape)
        img_new = np.zeros( (800,800, 3), dtype="uint8")
        img_new[offset_height:offset_height+img_part.shape[0], offset_width:offset_width+img_part.shape[1]] = img_part
        print(img_new.shape)

    elif process == "EdgeDetection":
        img = cv2.imread("core/static/{}_detail.png".format(image_id))
        img_resized = img.copy()
        im = img_resized[:,:, ::-1]
        img_g = rgb2gray(im)
        
        img_n = feature.canny(img_g, sigma=2)
        img_new = img_as_ubyte(img_n)

    elif process == 'LogAdjustment':
        img = cv2.imread("core/static/{}_detail.png".format(image_id))
        img_resized = img.copy()
        im = img_resized[:,:, ::-1]
        img_n = exposure.adjust_log(im, 1.5, True)
        img_new = img_n[:,:, ::-1]
        img_new = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)
    else:
        img_new = img.copy()

    cv2.imwrite('core/static/{}_detail.png'.format(image_id), img_new)
    return render(request, 'otoliths/detail.html', {'image_id': image_id, 'group': img_obj.fish_group})

@login_required
def visualize(request):
    img_name = "test.jpg"
    convert_to_gray("core/static/visualize/{}".format(img_name), 'core/static/visualize/result.png')
    return render(request, 'otoliths/visualization.html', {'image_name': img_name})

@login_required
def northsea(request):
    img_files = glob.glob('staticfiles/data/northsea/train/*.png')
    print(img_files)
    all_images = []
    count = 0
    for img_file in img_files:
        strs = img_file.replace("\\", "/").split("/")
        all_images.append(strs[-1])
        count += 1
        if count > 25:
            break
    return render(request, 'otoliths/northsea.html', {'images': all_images})

@login_required
def balticsea(request):
    img_files = glob.glob('staticfiles/data/baltic/train/*.png')
    print(img_files)
    all_images = []
    count = 0
    for img_file in img_files:
        strs = img_file.replace("\\", "/").split("/")
        all_images.append(strs[-1])
        count += 1
        if count > 25:
            break
    return render(request, 'otoliths/baltic.html', {'images': all_images})

@login_required
def data_detail(request, image_name):
    print(image_name)
    image_id = image_name
    return render(request, 'otoliths/data_detail.html', {'image_name': image_name})


@login_required
def experiments(request):
    return render(request, 'otoliths/experiments.html')


@login_required
def experiments_unet(request):

    dataset = 'datasets_baltic'
    if 'dataset' in request.GET:
        dataset = request.GET['dataset']

    run_type = 'test'
    if 'run_type' in request.GET:
        run_type = request.GET['run_type']

    run_label = 'randfold'
    if 'run_label' in request.GET:
        run_label = request.GET['run_label']

    idr = 0
    if 'idr' in request.GET:
        idr = int(request.GET['idr'])

    split_name = 'fold'
    if 'split_name' in request.GET:
        split_name = request.GET['split_name']

    selected = 47
    if 'selected' in request.GET:
        selected = request.GET['selected']

    settings = {
        'dataset' : dataset,
        'run_type': run_type,
        'run_label': run_label,
        'search_mode': False,
        'idr': idr,
        'selected': [selected],
        'split_type': 'rs',
        'split_name': split_name,
        'checkpoint': '_checkpoint',
    }

    target_species = 0
    if 'target_species' in request.GET:
        target_species = request.GET['target_species']
        settings.update({'target_species': target_species})

    age_limit = 0
    if 'age_limit' in request.GET:
        age_limit = int(request.GET['age_limit'])
        settings.update({'age_limit': age_limit})

    brighten = True
    if 'brighten' in request.GET:
        brighten = bool(request.GET['brighten'])
        settings.update({'brighten': brighten})

    source_dataset = 'datasets_north'
    if 'source_dataset' in request.GET:
        source_dataset = request.GET['source_dataset']
        settings.update({'source_dataset': source_dataset})

    run_unet(settings)
    return render(request, 'otoliths/experiments.html')


@login_required
def experiments_mrcnn(request):

    dataset = 'datasets_baltic'
    if 'dataset' in request.GET:
        dataset = request.GET['dataset']

    run_type = 'test'
    if 'run_type' in request.GET:
        run_type = request.GET['run_type']

    run_label = 'randfold'
    if 'run_label' in request.GET:
        run_label = request.GET['run_label']

    idr = 0
    if 'idr' in request.GET:
        idr = int(request.GET['idr'])

    split_name = 'fold'
    if 'split_name' in request.GET:
        split_name = request.GET['split_name']

    selected = 47
    if 'selected' in request.GET:
        selected = request.GET['selected']

    settings = {
        'dataset' : dataset,
        'run_type': run_type,
        'run_label': run_label,
        'search_mode': False,
        'idr': idr,
        'selected': [selected],
        'split_type': 'rs',
        'split_name': split_name,
        'checkpoint': '_checkpoint',
    }

    target_species = 0
    if 'target_species' in request.GET:
        target_species = request.GET['target_species']
        settings.update({'target_species': target_species})

    age_limit = 0
    if 'age_limit' in request.GET:
        age_limit = int(request.GET['age_limit'])
        settings.update({'age_limit': age_limit})

    brighten = True
    if 'brighten' in request.GET:
        brighten = bool(request.GET['brighten'])
        settings.update({'brighten': brighten})

    source_dataset = 'datasets_north'
    if 'source_dataset' in request.GET:
        source_dataset = request.GET['source_dataset']
        settings.update({'source_dataset': source_dataset})

    run_mrcnn(settings)
    return render(request, 'otoliths/experiments.html')


@login_required
def interact(request):
    request = json.loads(request.body.decode('utf-8'))
    print(request['_via_image_id_list'])
    img_name = '{}.png'.format(request['_via_image_id_list'][0].split('.png')[0])
    with open("autolith/static/new_annotations/{}.json".format(img_name), "w") as fout:
        json.dump(request, fout, indent=4)
        
    return HttpResponse("success")

# @login_required
# def dataview_sets(request, dataset):
#     img_files = glob.glob('autolith/static/data/{}/*/*.png'.format(dataset))
#     img_files.extend(glob.glob('autolith/static/data/{}/*/*.jpg'.format(dataset)))
#     print(img_files)
#     all_folders = []
#     for img_file in img_files:
#         strs = img_file.replace("\\", "/").split("/")
#         all_folders.append(strs[-2])
#     all_folders = list(set(all_folders))
#     return render(request, 'otoliths/dataview_sets.html', {'folders': all_folders})

# @login_required
# def dataview_images(request, dataset, folder):
#     img_files = glob.glob('autolith/static/data/{}/{}/*.png'.format(dataset, folder) )
#     img_files.extend(glob.glob('autolith/static/data/{}/{}/*.jpg'.format(dataset, folder)))
#     print(img_files)
#     all_images = []
#     count = 0
#     for img_file in img_files:
#         strs = img_file.replace("\\", "/").split("/")
#         all_images.append(strs[-1])
#         count += 1
#         if count > 25:
#             break
#     return render(request, 'otoliths/dataview_images.html', {'dataset': dataset, 'folder': folder, 'images': all_images})

@login_required
def data_detail(request, dataset, folder, image_name):
    import skimage.io
    from mrcnn.utils import resize_image

    og_img = skimage.io.imread('autolith/static/data/{}/{}/{}'.format(dataset, folder, image_name))
    sq_img, window, scale, padding, _ = resize_image(
    og_img, 
    min_dim=512,
    max_dim=512,
    #padding=True,
    )

    print(request.POST)
    print(request.GET)
    if 'process' in request.GET:
        if request.GET['process'] == 'SelectRegions':
            predict_mrcnn(og_img, sq_img, window, dataset, folder, image_name)
            result_name ="mrcnn_image_{}.png".format(image_name)
        elif request.GET['process'] == 'CoreDetection':
            predict_mrcnn(og_img, sq_img, window, dataset, folder, image_name, mode="core")
            result_name ="mrcnn_image_{}.png".format(image_name) 
        elif request.GET['process'] == 'Combined':
            predict_mrcnn_combined(og_img, sq_img, window, dataset, folder, image_name, mode="combined")
            result_name ="mrcnn_image_{}.png".format(image_name) 
        elif request.GET['process'] == 'LogAdjustment':
            return load_data_marks(request, og_img, sq_img, window, dataset, folder, image_name)
        elif request.GET['process'] == 'Blank':
            return load_blank_marks(request, og_img, sq_img, window, dataset, folder, image_name)
        elif request.GET['process'] == 'Segment':
            if 'convert' in request.GET:
                return convert_data_marks(request, og_img, sq_img, window, dataset, folder, image_name, markings='poly')
            elif 'mode' in request.GET and request.GET['mode'] == 'brush':
                return load_data_marks(request, og_img, sq_img, window, dataset, folder, image_name, markings='poly', mode='brush')
            else:
                return load_data_marks(request, og_img, sq_img, window, dataset, folder, image_name, markings='poly')
        else:
            predict_unet(og_img, sq_img, window, dataset, folder, image_name)
            result_name ='unetcontour_{}'.format(image_name) 
        image_name = result_name
    else:
        skimage.io.imsave("autolith/static/detail/{}".format(image_name), sq_img)

    return render(request, 'otoliths/image_detail.html', {'dataset': dataset, 'folder': folder, 'image_name': image_name})


#-=---------------------------------------------------------

from django.shortcuts import redirect

def load_images_from_json(domain, split_name):

    train_dir = "{}/{}".format(domain, split_name)
    valid_dir = "{}/{}".format(domain, split_name.replace('train_', 'valid_'))

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

    all_lines = ["img,age,readable"]
    for k,v in labeled_images.items():
        img_name = k
        img = cv2.imread("{}/{}".format(source_dir, img_name))
        cv2.imwrite("{}/{}".format(train_dir, img_name), img)

        flip_img = cv2.flip(img, 1)
        cv2.imwrite("{}/{}".format(valid_dir, img_name), flip_img)

    all_json_files = glob.glob("{}/*.json".format(train_dir) )
    for jsonf in all_json_files:
        strs = jsonf.replace("\\","/").split("/")
        shutil.copyfile(jsonf, "{}/{}".format(valid_dir, strs[-1])  )
        

def load_images(dataset, split_name):
    load_images_from_json(dataset, split_name)

def load_annuli_annotations(request, dataset, folder, annotype='both', drawing_tool='default'):
    all_image_files = glob.glob("{}/{}/*.png".format(dataset, folder))
    all_json_files = glob.glob("{}/{}/*.json".format(dataset, folder))

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
        shutil.copyfile(image_path, 'autolith/static/detail/{}'.format(strs[-1]))
        sep_name = strs[-1]
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

    if drawing_tool == 'default':
        return render(request, 'otoliths/researchers.html', {'pass_var': json_var, 'fname': 'test', 'mode': 'annuli_annotate', 'drawing_tool': 'brush', 'dataformat': 'annuli', 'dataset': dataset, 'folder': folder})
    return render(request, 'otoliths/researchers_interact.html', {'pass_var': json_var, 'fname': 'test', 'mode': 'annuli_annotate', 'drawing_tool': 'default', 'dataformat': 'annuli', 'dataset': dataset, 'folder': folder})


def load_annotations(request, dataset, folder, drawing_tool='default'):
    all_image_files = glob.glob("{}/{}/*.png".format(dataset, folder))
    json_file = '{}/{}/0/whole_proj.json'.format(dataset, folder)
    try:
        main_json = json.load(open(json_file))
        for img_file in all_image_files:
            strs = img_file.replace("\\", "/").split("/") 
            shutil.copyfile(img_file, 'autolith/static/detail/{}'.format(strs[-1]))
    except:
        with open("autolith/static/json_template.json") as fin:
            main_json = json.load(fin)

        for image_path in all_image_files:
            strs = image_path.replace("\\", "/").split("/") 
            sep_name = strs[-1]
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
            new_item["{}{}".format(sep_name,fsize)]["regions"] = []

            new_sub_item = {
                    'shape_attributes': {'name': 'polyline',
                    'all_points_x': [],
                    'all_points_y': []},
                    'region_attributes': {}
            }
            new_item["{}{}".format(sep_name,fsize)]["regions"].append(new_sub_item)

            main_json["_via_img_metadata"].update(new_item)
        main_json['_via_settings']['project']['name'] = 'whole'
    with open("autolith/static/detail/main_var.json", "w") as fout:
        json.dump(main_json, fout, indent=4)
    with open("autolith/static/detail/main_var.json") as fin:
        json_var = json.load(fin)
        json_var = json.dumps(json_var)

    if drawing_tool == 'default':
        return render(request, 'otoliths/researchers.html', {'pass_var': json_var, 'fname': 'outer', 'mode': 'outer', 'drawing_tool': 'brush', 'dataformat': 'outer', 'dataset': dataset, 'folder': folder})
    return render(request, 'otoliths/researchers.html', {'pass_var': json_var, 'fname': 'outer', 'mode': 'outer', 'drawing_tool': 'default', 'dataformat': 'outer', 'dataset': dataset, 'folder': folder})




def isolate_contours(ff, new_name="", new_dir=""):
    image = skimage.io.imread(ff)
    # If grayscale. Convert to RGB for consistency.
    if image.ndim != 3:
        image = skimage.color.gray2rgb(image)
    # If has an alpha channel, remove it for consistency
    if image.shape[-1] == 4:
        image = image[..., :3]

    img = image.copy()
    original_image = img #_as_ubyte(img)

    original_copy = original_image.copy()
    hsv_img = skimage.color.rgb2hsv(original_image)
    gray_copy = cv2.cvtColor(original_copy, cv2.COLOR_BGR2GRAY)


    test = np.arange(0,1,0.05)
    max_diff = 0
    current_ch = -1
    for ch in [0,1]:
        all_val = []
        for item in test:

            lower_mask = (hsv_img[:,:,ch] < item)  & (hsv_img[:,:,2] > 0.015)

            wh_thresh = np.zeros([lower_mask.shape[0],lower_mask.shape[1],1], dtype=np.uint8)
            wh_thresh[:,:,0] = lower_mask[:,:]

            tab_up = wh_thresh[:int(0.20*wh_thresh.shape[0]),int(0.33*wh_thresh.shape[1]):int(0.66*wh_thresh.shape[1])]
            bg_up = np.mean(tab_up )
            tab_down = wh_thresh[int(wh_thresh.shape[0]-0.20*wh_thresh.shape[0]):,int(0.33*wh_thresh.shape[1]):int(0.66*wh_thresh.shape[1])]
            bg_down = np.mean(tab_down )
            ce = np.mean(wh_thresh[int(0.33*wh_thresh.shape[0]):int(0.66*wh_thresh.shape[0]),int(0.33*wh_thresh.shape[1]): int(0.66*lower_mask.shape[1])])
            bg = (bg_up+bg_down)/2.0
            
            ce = np.mean(wh_thresh[int(0.33*wh_thresh.shape[0]):int(0.66*wh_thresh.shape[0]),int(0.33*wh_thresh.shape[1]): int(0.66*lower_mask.shape[1])])
            val = abs(ce - bg)
            all_val.append(val)
        all_val = np.array(all_val)
        diff = np.max(all_val)
        print(diff)
        if diff > (max_diff+0.05):
            max_diff = diff
            current_ch = ch
            current_thresh = test[np.argmax(all_val)]

    test = np.arange(0,1,0.05)
    max_diff = 0
    current_ch = -1
    for ch in [0,1]:
        all_val = []
        for item in test:

            lower_mask = (hsv_img[:,:,ch] < item)  & (hsv_img[:,:,2] > 0.015)# (hsv_img[:,:,2] < 0.95)

            wh_thresh = np.zeros([lower_mask.shape[0],lower_mask.shape[1],1], dtype=np.uint8)
            wh_thresh[:,:,0] = lower_mask[:,:]

            tab_up = wh_thresh[:int(0.20*wh_thresh.shape[0]),int(0.33*wh_thresh.shape[1]):int(0.66*wh_thresh.shape[1])]
            bg_up = np.mean(tab_up )
            tab_down = wh_thresh[int(wh_thresh.shape[0]-0.20*wh_thresh.shape[0]):,int(0.33*wh_thresh.shape[1]):int(0.66*wh_thresh.shape[1])]
            bg_down = np.mean(tab_down )
            ce = np.mean(wh_thresh[int(0.33*wh_thresh.shape[0]):int(0.66*wh_thresh.shape[0]),int(0.33*wh_thresh.shape[1]): int(0.66*lower_mask.shape[1])])
            bg = (bg_up+bg_down)/2.0
            val = abs(ce - bg)
            all_val.append(val)
        all_val = np.array(all_val)
        diff = np.max(all_val)
        print(diff)
        if diff > (max_diff+0.05):
            max_diff = diff
            current_ch = ch
            current_thresh = test[np.argmax(all_val)]
            

    #---------------
    lower_mask = (hsv_img[:,:,current_ch] < current_thresh)  & (hsv_img[:,:,2] > 0.015)
    lh_thresh = np.zeros([lower_mask.shape[0],lower_mask.shape[1],1], dtype=np.uint8)
    lh_thresh[:,:,0] = lower_mask[:,:]
    lh_thresh = cv2.medianBlur(lh_thresh, 15)

    lh_thresh =  cv2.erode(lh_thresh.squeeze(), np.ones( (10,10), np.uint8 ), iterations=2 )


    sh_thresh = np.zeros([lower_mask.shape[0],lower_mask.shape[1],1], dtype=np.uint8)
    sh_thresh[:,:,0] = lh_thresh[:,:]


    sh_contours, hierarchy = cv2.findContours(sh_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contours = sorted(sh_contours,key=cv2.contourArea, reverse=True)

    all_contours = [c for c in contours if cv2.contourArea(c) > 10000]
    min_idx = -1
    if len(all_contours)>1:
        print("test")
        min_dist = 99999
        min_idx = -1
        for ix, ct in enumerate(all_contours):
            M = cv2.moments(ct)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            print(cx, cy, cv2.contourArea(ct))
            current_dist = abs(lower_mask.shape[1]/2.0-cx)
            if current_dist < min_dist :
                min_dist = current_dist
                min_idx = ix
        sh_cts = all_contours[min_idx]
    else:
        sh_cts = all_contours[0]
        
    test_copy = img.copy()

    gray_copy = cv2.cvtColor(test_copy, cv2.COLOR_BGR2GRAY)

    lower_mask = (hsv_img[:,:,current_ch] < current_thresh)  & (hsv_img[:,:,2] > 0.015)

    wh_thresh = np.zeros([lower_mask.shape[0],lower_mask.shape[1],1], dtype=np.uint8)
    wh_thresh[:,:,0] = lower_mask[:,:]


    wh_contours, hierarchy = cv2.findContours(wh_thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # wh_cts = max(wh_contours, key=cv2.contourArea)
    wh_contours = sorted(wh_contours,key=cv2.contourArea, reverse=True)[:3]
    whole_contours = [c for c in wh_contours if cv2.contourArea(c) > 10000]
    # min_idx = -1
    if len(whole_contours)>1:
        print("test")
        wh_dist = 99999
        wh_idx = -1
        for ix, ct in enumerate(whole_contours):
            M = cv2.moments(ct)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            print(cx, cy, cv2.contourArea(ct))
            current_dist = abs(lower_mask.shape[1]/2.0-cx)
            if current_dist < wh_dist :
                wh_dist = current_dist
                wh_idx = ix
        wh_cts = whole_contours[wh_idx]
    else:
        wh_cts = whole_contours[0]

    mask = np.ones(test_copy.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [wh_cts], -1,0,-1 )

    mapFg =  cv2.dilate(mask, np.ones( (3,3), np.uint8 ), iterations=10 )
    mapFg =  cv2.erode(mapFg, np.ones( (10,10), np.uint8 ), iterations=10 )


    mapFg =  cv2.dilate(mapFg, np.ones( (5,5), np.uint8 ), iterations=10 )
    mapFg =  cv2.erode(mapFg, np.ones( (10,10), np.uint8 ), iterations=10 )
    new_mask = mapFg < 1 #cv2.bitwise_not(mapFg)

    new_thresh = np.zeros([new_mask.shape[0],new_mask.shape[1],1], dtype=np.uint8)
    new_thresh[:,:,0] = new_mask[:,:]

    wh_contours, hierarchy = cv2.findContours(new_thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    wh_cts = max(wh_contours, key=cv2.contourArea)


    #---------
    if min_idx != -1:
        ex_mask = np.ones(test_copy.shape[:2], dtype="uint8")
        for ix, ct in enumerate(all_contours):
            if ix != min_idx:
                cv2.drawContours(ex_mask, [ct], -1,0,-1 )
    #----------

    mask = np.ones(test_copy.shape[:2], dtype="uint8")
    # hull = cv2.convexHull(wh_cts)
    cv2.drawContours(mask, [sh_cts], -1,0,-1 )

    scaled_mask = mask < 1 #cv2.bitwise_not(mapFg)

    markers = np.zeros_like(gray_copy)

    markers[scaled_mask>0] = 2
    markers[new_mask<1] = 1
    if min_idx != -1:
        markers[ex_mask<1] = 1

    elevation_map = sobel(gray_copy)
    #         break
    seg = watershed(elevation_map, markers)
    seg_bin = ndimage.binary_fill_holes(seg-1)


    seg_bin = erosion(seg_bin.squeeze(), selem=disk(10))
    seg_bin = dilation(seg_bin, selem=disk(10))



    labeled_seg = label(seg_bin)


    regions = regionprops(labeled_seg)

    curr_seg = None
    curr_max = 0
    for ridx,item in enumerate(regionprops(labeled_seg)):
        if item.area > curr_max:
            curr_max = item.area
            curr_seg = ridx

    try:
        contour = find_contours(labeled_seg==regions[curr_seg].label, 0.5)[0]
    except:
        return None
    ncontour = np.expand_dims(np.flip(contour, axis=1).astype(int), axis=1)
    return ncontour


def create_outer_contours(dataset, folder):

    try:
        os.makedirs('{}/{}/0'.format(dataset, folder))
    except:
        pass
    test_files = glob.glob('{}/{}/*.png'.format(dataset, folder))
    len(test_files)

    with open("autolith/static/json_template.json") as fin:
        main_json = json.load(fin)
    try:
        with open("{}/{}/0/whole_proj.json".format(dataset, folder) ) as fin:
            existing = json.load(fin)
    except:
        existing = {"_via_img_metadata": {}}
    use_nucleus = True
    all_polar_images = []
    all_original_images = []
    all_contours = []
    all_centers = []
    all_max_angles = []
    all_reading_coords = []
    for idx, ff in enumerate(test_files):
        print(idx)

        key = ff.replace("\\", "/").split("/")[-1]
        sep_name = key
        fname = key
        # try:
        #     os.makedirs("{}/ALL/".format(destination) )
        # except:
        #     pass
        
        # shutil.copy(ff, "{}/ALL/".format(destination))
        image_path = ff #"{}/ALL/{}".format(destination, key)

        st = os.stat("{}".format(image_path))

        fsize = int(st.st_size)

        with open("autolith/static/extra_json.json") as fin:
            extra_json = json.load(fin)

        if "{}{}".format(sep_name,fsize) in existing["_via_img_metadata"]:
            main_json["_via_image_id_list"].append("{}{}".format(sep_name, fsize) )

            new_item = {}
            new_item["{}{}".format(sep_name,fsize)] = extra_json["namesize"].copy()
            new_item["{}{}".format(sep_name,fsize)]["filename"] = sep_name
            new_item["{}{}".format(sep_name,fsize)]["size"] = int(fsize)
            new_item["{}{}".format(sep_name,fsize)]["regions"] = existing["_via_img_metadata"]["{}{}".format(sep_name,fsize)]["regions"]
            main_json["_via_img_metadata"].update(new_item)

            # main_json["_via_img_metadata"].update()
            continue

        try:
            wh_cts = isolate_contours(ff)
        except:
            wh_cts = None
        #     continue
        if wh_cts is None:
            # wh_cts = prev_cts
            wh_cts = []


        main_json["_via_image_id_list"].append("{}{}".format(sep_name, fsize) )

        new_item = {}
        new_item["{}{}".format(sep_name,fsize)] = extra_json["namesize"].copy()
        new_item["{}{}".format(sep_name,fsize)]["filename"] = sep_name
        new_item["{}{}".format(sep_name,fsize)]["size"] = int(fsize)
        new_item["{}{}".format(sep_name,fsize)]["regions"] = []

        c = wh_cts #_contours[-1]
        new_sub_item = {
                'shape_attributes': {'name': 'polyline',
                'all_points_x': [],
                'all_points_y': []},
                'region_attributes': {}
        }
        interval = int(len(c)/300)
        for item_idx, item in enumerate(c):
            if interval>0 and item_idx%interval == 0:
                x,y = item[0]
                new_sub_item["shape_attributes"]["all_points_x"].append(int(x))
                new_sub_item["shape_attributes"]["all_points_y"].append(int(y))
        new_item["{}{}".format(sep_name,fsize)]["regions"].append(new_sub_item)

        main_json["_via_img_metadata"].update(new_item)
        


        # prev_cts = wh_cts
        image_raw = skimage.io.imread(ff)
        img_no_mask = image_raw.copy()

        # for ai_label in ai_labels:
        #     (xpos, ypos, wd, ht) = cv2.boundingRect(ai_label[-1])
        #     cv2.putText(img_no_mask, str(ai_label[0]), (int(xpos+wd),int(ypos+ht) ),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0) )
        #     cv2.drawContours(img_no_mask, [ai_label[-1]], -1, (0,255,0), 2)
        cv2.drawContours(img_no_mask, [wh_cts], -1, (0,255,0), 5)
        # skimage.io.imsave('autolith/static/detail/{}'.format(image_name), img_no_mask)
        try:
            os.makedirs('autolith/static/data/watershed/{}'.format(folder))
        except:
            pass
        skimage.io.imsave('autolith/static/data/watershed/{}/{}'.format(folder, fname), img_no_mask)
        
    main_json['_via_settings']['project']['name'] = 'contour'
    with open("{}/{}/0/whole_proj.json".format(dataset, folder), "w") as fout:
        json.dump(main_json, fout, indent=4)

def scale_data_from_contours(request, dataset, folder):
    train_path = folder.replace("raw_", "train_")
    try:
        os.makedirs("{}/{}/0".format(dataset, train_path))
    except:
        pass
    try:
        os.makedirs("{}/images".format(dataset))
    except:
        pass
    try:
        os.makedirs("{}/images/0".format(dataset))
    except:
        pass

    wh_files = glob.glob("{}/{}/0/*.json".format(dataset, folder))
    json_annotations = {}
    for json_file in wh_files:
        _annotations = json.load(open(json_file))
        try:
            _annotations = list(_annotations['_via_img_metadata'].values())
        except:
            _annotations = list(_annotations.values())

        _annotations = [a for a in _annotations if a['regions']]
        for a in _annotations:
            json_annotations[a['filename']] = a #(a, a["size"])

    with open("autolith/static/json_template.json") as fin:
        main_json = json.load(fin)
        main_json['_via_settings']['project']['name'] = 'whole_scaled'
    
    all_bounds = {}
    ring_map = {}
    for k, a in json_annotations.items():
        fname = a['filename']
        sep_name = fname

        img_path = '{}/{}/{}'.format(dataset, folder, fname)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        polygons = [r['shape_attributes'] for r in a['regions']]
        whole_mask = np.zeros([img.shape[0], img.shape[1], 1], dtype=np.uint8)
        for poly_idx, poly in enumerate(polygons):
            rr, cc = skimage.draw.polygon(poly['all_points_y'], poly['all_points_x'])
            whole_mask[rr, cc, 0] = 1 #255.0

        raw_whole_mask = whole_mask.copy()
        whole_mask = cv2.dilate(whole_mask, np.ones( (4,4), np.uint8 ), iterations=10 )

        _contours, hierarchy = cv2.findContours(whole_mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        _contours = sorted(_contours,key=cv2.contourArea, reverse=False)
        rect = cv2.boundingRect(_contours[-1])
        x, y, w, h = rect

        img_resized = img.copy()
        wh_resized = raw_whole_mask.copy()

        ofs = 50
        img_new = img_resized[max([y-ofs,0]):min([y+h+ofs, img_resized.shape[0]]), max([x-ofs,0]):min([x+w+ofs,img_resized.shape[1]])]
        wh_new = wh_resized[max([y-ofs,0]):min([y+h+ofs, img_resized.shape[0]]), max([x-ofs,0]):min([x+w+ofs,img_resized.shape[1]])]
        
        image_path = "{}/{}/{}".format(dataset, train_path, sep_name)
        cv2.imwrite(image_path, img_new)
        cv2.imwrite("{}/images/{}".format(dataset, sep_name), img_new)
        
        with open("autolith/static/extra_json.json") as fin:
            extra_json = json.load(fin)

        st = os.stat(image_path)
        fsize = int(st.st_size)
        main_json["_via_image_id_list"].append("{}{}".format(fname, fsize) )
        
        new_item = {}
        new_item["{}{}".format(fname,fsize)] = extra_json["namesize"].copy()
        new_item["{}{}".format(fname,fsize)]["filename"] = fname
        new_item["{}{}".format(fname,fsize)]["size"] = int(fsize)
        new_item["{}{}".format(fname,fsize)]["regions"] = []
        
        item = wh_new

        _contours, hierarchy = cv2.findContours(item.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        _contours = sorted(_contours,key=cv2.contourArea, reverse=False)
        _contours = [_contours[-1]]
        for c in _contours:
            new_sub_item = {
                    'shape_attributes': {'name': 'polyline',
                    'all_points_x': [],
                    'all_points_y': []},
                    'region_attributes': {}
            }
            for item_idx, item in enumerate(c):
                x,y = item[0]
                new_sub_item["shape_attributes"]["all_points_x"].append(int(x))
                new_sub_item["shape_attributes"]["all_points_y"].append(int(y))
            new_item["{}{}".format(fname,fsize)]["regions"].append(new_sub_item)
        main_json["_via_img_metadata"].update(new_item)
        main_json['_via_settings']['project']['name'] = "whole_scaled"

    with open("{}/{}/0/whole_scaled.json".format(dataset, train_path), "w") as fout:
        json.dump(main_json, fout, indent=4)
    with open("{}/images/0/whole_{}_import.json".format(dataset, folder.replace("data_", "", 1)), "w") as fout:
        json.dump(main_json["_via_img_metadata"], fout, indent=4)


def get_url_params(request_dict, mode='GET'):

    method = 'U-Net'
    if 'method' in request_dict:
        method = request_dict['method']

    dataset = 'datasets_baltic'
    if 'dataset' in request_dict:
        dataset = request_dict['dataset']

    run_type = 'test'
    if 'run_type' in request_dict:
        run_type = request_dict['run_type']

    run_label = 'randfold'
    if 'run_label' in request_dict:
        run_label = request_dict['run_label']

    idr = 0
    if 'idr' in request_dict:
        idr = int(request_dict['idr'])

    split_name = 'fold'
    if 'split_name' in request_dict:
        split_name = request_dict['split_name']

    selected = 47
    if 'selected' in request_dict:
        selected = request_dict['selected']
        if mode == 'POST':
            selected = [int(selected)]

    settings = {
        'method': method,
        'dataset' : dataset,
        'run_type': run_type,
        'run_label': run_label,
        'search_mode': False,
        'idr': idr,
        'selected': selected,
        'split_type': 'rs',
        'split_name': split_name,
        'checkpoint': '_checkpoint',
    }

    target_species = 0
    if 'target_species' in request_dict and request_dict['target_species'] != '':
        target_species = request_dict['target_species']
        settings.update({'target_species': target_species})

    age_limit = 0
    if 'age_limit' in request_dict and request_dict['age_limit'] != '':
        age_limit = int(request_dict['age_limit'])
        settings.update({'age_limit': age_limit})

    brighten = True
    if 'brighten' in request_dict and request_dict['brighten'] != '':
        brighten = int(request_dict['brighten'])
        if mode == 'POST':
            brighten = bool(brighten)
        settings.update({'brighten': brighten})

    base = 'none'
    if 'base' in request_dict and request_dict['base'] != '':
        base = request_dict['base']
        settings.update({'base': base})

    base_id = 0
    if 'base_id' in request_dict and request_dict['base_id'] != '':
        base_id = request_dict['base_id']
        if mode == 'POST':
            base_id = int(base_id)
        settings.update({'base_id': base_id})

    continual = 0
    if 'continual' in request_dict and request_dict['continual'] != '':
        continual = int(request_dict['continual'])
        if mode == 'POST':
            continual = bool(continual)
        settings.update({'continual': continual})

    eval_balanced_set = 0
    if 'eval_balanced_set' in request_dict and request_dict['eval_balanced_set'] != '':
        eval_balanced_set = int(request_dict['eval_balanced_set'])
        if mode == 'POST':
            eval_balanced_set = bool(eval_balanced_set)
        settings.update({'eval_balanced_set': eval_balanced_set})

    source_dataset = 'datasets_north'
    if 'source_dataset' in request_dict and request_dict['source_dataset'] != '':
        source_dataset = request_dict['source_dataset']
        settings.update({'source_dataset': source_dataset})

    return settings


def create_validation(dataset, split_name):
    train_dir = "{}/{}".format(dataset, split_name)
    valid_dir = "{}/{}".format(dataset, split_name.replace('train_', 'valid_'))
    try:
        os.makedirs(valid_dir)
    except:
        pass

    all_img_files = glob.glob("{}/*.png".format(train_dir))
    for imagef in all_img_files:
        strs = imagef.replace("\\","/").split("/")
        img_name = strs[-1]
        img = cv2.imread(imagef)
        flip_img = cv2.flip(img, 1)
        cv2.imwrite("{}/{}".format(valid_dir, img_name), flip_img)
    all_json_files = glob.glob("{}/*.json".format(train_dir) )
    for jsonf in all_json_files:
        strs = jsonf.replace("\\","/").split("/")
        shutil.copyfile(jsonf, "{}/{}".format(valid_dir, strs[-1]) )


def get_constituents(request):

    experimental = [
        ("mrcnn", "basednone"),
        ("mrcnn", "basedcoco"),
        ("mrcnn", "basedbaltic"),
        ("unet", "basednone"),
        ("unet", "basedvgg"),
        ("unet", "basedbaltic"),
    ]
    all_items = [x for x in request.GET.getlist('weights')]

    all_constituents = []
    for exp in experimental:
        for idx, item in enumerate(all_items):
            strs = item.split(" // ")
            if exp[0] == strs[1] and exp[1] in strs[2]:
                all_constituents.append(item)
                all_items.pop(idx)
                break
    for idx, item in enumerate(all_items):
        all_constituents.append(item)

    return all_constituents


def get_regex_keys(constituents):

    regex_keys = []
    for item in constituents:
        strs = item.split(" // ")
        substrs = strs[1].strip().split("_")
        if 'basedbaltic' in strs[1]:
            key = "{}_*{}*".format(substrs[0], 'basedbaltic' )
        elif 'basednorth' in strs[1]:
            key = "{}_*{}*".format(substrs[0], 'basednorth' )
        elif 'basednone' in strs[1]:
            key = "{}_*{}*".format(substrs[0], 'basednone' )
        elif 'basedvgg' in strs[1]:
            key = "{}_*{}*".format(substrs[0], 'basedvgg' )
        elif 'basedcoco' in strs[1]:
            key = "{}_*{}*".format(substrs[0], 'basedcoco' )
        else:
            # substrs = strs[1].strip().split("_")
            print(strs)
            print(substrs)
            key = "{}_{}*".format(substrs[0], substrs[1].split()[0])
        regex_keys.append(key)


    return regex_keys


def start_ensemble_training(dataset, folder, ensemble_method, constituents):

    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVR
    import pandas as pd
    import pickle

    base_rep = 0

    regex_keys = get_regex_keys(constituents)
    print(regex_keys)

    manual_readings_on_training_data = []

    key_pred_map = {}
    for kidx, key in enumerate(regex_keys):
        key_pred_map[kidx] = []

    for kidx, key in enumerate(regex_keys):
        all_files = glob.glob("{}/{}/{}.txt".format(dataset, folder, key))
        all_files = sorted(all_files)
        for ff in all_files:
            # print(ff)
            with open(ff) as fin:
                lines = fin.readlines()
            for line in lines:
                strs = line.strip().split(",")

                ai_reading = round(float(strs[-2]))
                key_pred_map[kidx] .append(ai_reading)
                if kidx == 0:
                    manual_reading = round(float(strs[-1]))
                    manual_readings_on_training_data.append(manual_reading)

    manual_readings_on_training_data = np.array(manual_readings_on_training_data)
    ai_training_data = pd.DataFrame()

    for kidx, key in enumerate(regex_keys):
        ai_training_data[key] = np.array(key_pred_map[kidx])

    if ensemble_method == 'LinearRegression':
        linear_model = LinearRegression()
        linear_model.fit(ai_training_data, manual_readings_on_training_data)
        with open('{}/{}/LinearRegression.pkl'.format(dataset, folder), 'wb') as fout:
            pickle.dump({'model': linear_model, 'keys': regex_keys }, fout)
    elif ensemble_method == 'RandomForest':
        rf_model = RandomForestClassifier(max_depth=5, random_state=101)
        rf_model.fit(ai_training_data, manual_readings_on_training_data)
        with open('{}/{}/RandomForest.pkl'.format(dataset, folder), 'wb') as fout:
            pickle.dump({'model': rf_model, 'keys': regex_keys }, fout)
    else:
        with open('{}/{}/Averaging.pkl'.format(dataset, folder), 'wb') as fout:
            pickle.dump({'model': [], 'keys': regex_keys }, fout)


def start_ensemble_testing(dataset, folder, ensemble_method, ensemble_path):
    import pandas as pd
    import pickle

    with open('{}/{}.pkl'.format(ensemble_path, ensemble_method), 'rb') as fin:
        ensemble = pickle.load(fin)

    key_pred_map = {}
    for kidx, key in enumerate(ensemble['keys']):
        key_pred_map[kidx] = []

    names_test_data= []
    manual_readings_on_test_data = []

    keys = ensemble['keys']
    print(keys)
    for kidx, key in enumerate(keys):
        all_files = glob.glob("{}/{}/{}.txt".format(dataset, folder, key))
        for ff in all_files:
            print(ff)
            with open(ff) as fin:
                lines = fin.readlines()
            for line in lines:
                strs = line.strip().split(",")

                ai_reading = round(float(strs[-2]))
                key_pred_map[kidx].append(ai_reading)

                if kidx == 0:
                    manual_reading = round(float(strs[-1]))
                    manual_readings_on_test_data.append(manual_reading)
                    names_test_data.append(strs[0])

    manual_readings_on_test_data = np.array(manual_readings_on_test_data)
    ai_test_data = pd.DataFrame()

    for kidx, key in enumerate(keys):
        print((kidx, key))
        ai_test_data[key] = np.array(key_pred_map[kidx])

    if ensemble_method == 'LinearRegression':
        linear_model = ensemble['model']
        ensemble_pred = linear_model.predict(ai_test_data)
    elif ensemble_method == 'RandomForest':
        rf_model = ensemble['model']
        ensemble_pred = rf_model.predict(ai_test_data)
    else:
        ensemble_pred = ai_test_data.mean(axis=1)

    all_north_acc = []
    all_results = []
    
    ensemble_correct = 0
    for idx, actual in enumerate(manual_readings_on_test_data):
        fname = names_test_data[idx]
        diff = actual - round(ensemble_pred[idx])
        if diff == 0:
            ensemble_correct += 1
        all_results.append("{},{},{}".format(fname, round(ensemble_pred[idx]), actual))

    print(ensemble_correct)
    all_north_acc.append(100*ensemble_correct/len(manual_readings_on_test_data))
    results_path = "datasets_ensemble/{}".format(folder)
    try:
        os.makedirs(results_path)
    except:
        pass

    with open("{}/{}.txt".format(results_path, ensemble_method), "w") as fout:
        fout.write("\n".join(all_results))


def show_results_table_from_file(request):

    dataset = None
    if 'dataset' in request.GET:
        dataset = request.GET['dataset']

    source = dataset
    if 'source' in request.GET:
        source = request.GET['source']


    folder = None
    if 'folder' in request.GET:
        folder = request.GET['folder']

    
    if 'fname' in request.GET:
        fname = request.GET['fname']
    else:
        raise ValueError('Cannot be empty')

    if dataset is None:
        path = fname
    elif folder is None:
        path = dataset + fname
    else:
        path = dataset + '/' + folder + '/' + fname

    
    with open(path) as fin:
        lines = fin.readlines()
        if len(lines[0].split(",")) == 3:
            headers = ["Image Name", "AI Reading", "Manual Reading"]
            correct_tally = [0]
        else:
            headers = ["Image Name", "RandomForest", "LinearRegression", "MeanPred", "Manual Reading"]
            correct_tally = [0,0,0]

    results_list = []
    results_list.append(headers)
    correct = 0

    total = 0
    for line in lines:
        strs = line.strip().split(",")
        image_name = strs[0]
        manual_reading = round(float(strs[-1]))
        if len(strs) == 3:
            ai_reading = round(float(strs[1]))
            diff = manual_reading - ai_reading
            if diff == 0:
                correct_tally[0] += 1
            total += 1
            results_list.append([image_name, ai_reading, manual_reading])
        else:  
            for ix, item in enumerate(strs[1:-1]):
                ai_reading = round(float(item))
                diff = manual_reading - ai_reading
                if diff == 0:
                    correct_tally[ix] += 1

            results_list.append([image_name]+ strs[1:-1] + [manual_reading])

    for ix, item in enumerate(headers[1:-1]):
        accuracy = round(100.0*correct_tally[ix]/len(lines))
        headers[1+ix] = headers[1+ix] + ' (' + str(accuracy) + '%)'
    
    return render(request, 'otoliths/results_table.html', {'source': source, 'results_list': results_list, 'correct': correct, 'total': total, 'accuracy': 100*correct/total if total else 0})



# start login_required >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

@login_required
def upload(request):
    print("++++++++++++++++++++++++++++++")
    username = request.user.username
    print("++++++++++++++++++++++++++++++")
    from .forms import UploadForm
    if request.method == 'POST':
        print(request.POST)
        form  = UploadForm(request.POST, request.FILES)
        data_id = request.POST['data_id']

        
        for ff in request.FILES.getlist('files'):
            print(ff.name)
            if ff.name.endswith('.csv'):
                try:
                    os.makedirs("datasets_{}/raw_{}_{}/0/".format(username, request.POST['folder'], data_id))
                except:
                    pass
                with open("datasets_{}/raw_{}_{}/{}".format(username, request.POST['folder'], data_id, ff.name), "wb+") as dest:
                    for chunk in ff.chunks():
                        dest.write(chunk)
                with open("datasets_{}/raw_{}_{}/{}".format(username, request.POST['folder'], data_id, ff.name)) as fin:
                    lines = fin.readlines()

                all_data_map = {}
                try:
                    with open("datasets_{}/all_data_map.json".format(username)) as jsonfin:
                        all_data_map = json.load(jsonfin)
                except:
                    pass
                for line in lines:
                    strs = line.split(",")
                    try:
                        all_data_map[strs[0].strip()] = str(round(float(strs[1].strip())))
                    except:
                        pass
                with open("datasets_{}/all_data_map.json".format(username), "w") as jsonfout:
                    json.dump(all_data_map, jsonfout, indent=4)
            elif ff.name.endswith('.h5'):
                try:
                    os.makedirs("datasets_{}/models/".format(username))
                except:
                    pass

                with open("datasets_{}/models/{}".format(username, ff.name), "wb+") as dest:
                    for chunk in ff.chunks():
                        dest.write(chunk) 
            else:
                try:
                    os.makedirs("datasets_{}/raw_{}_{}/0/".format(username, request.POST['folder'], data_id))
                except:
                    pass
                with open("datasets_{}/raw_{}_{}/{}".format(username, request.POST['folder'], data_id, ff.name), "wb+") as dest:
                    for chunk in ff.chunks():
                        dest.write(chunk) 
        for ff in request.FILES.getlist('ring_groundtruth'):
            print(ff.name)
            try:
                with open("datasets_{}/train_{}_{}/{}".format(username, request.POST['folder'], data_id, ff.name), "wb+") as dest:
                    for chunk in ff.chunks():
                        dest.write(chunk) 
            except:
                pass
        for ff in request.FILES.getlist('outer_contour'):
            print(ff.name)
            with open("datasets_{}/raw_{}_{}/0/{}".format(username, request.POST['folder'], data_id, ff.name), "wb+") as dest:
                for chunk in ff.chunks():
                    dest.write(chunk) 

        all_json_files = glob.glob("datasets_{}/train_{}_{}/*.json".format(username, request.POST['folder'], data_id) )
        for jsonf in all_json_files:
            strs = jsonf.replace("\\","/").split("/")
            shutil.copyfile(jsonf, "datasets_{}/annotations/{}".format(username, strs[-1]) )

        return redirect('/otoliths/dataview/datasets_{}/?folder_type=raw'.format(username))
    else:

        upload_type = 'images'
        try:
            upload_type = request.GET['upload_type']
        except:
            pass
        form = UploadForm()



    raw_data = True
    all_dir_paths = glob.glob('datasets_{}/raw_*'.format(username))
    print(all_dir_paths)
    all_folders = []
    count = 0
    for dir_path in all_dir_paths:
        strs = dir_path.replace("\\", "/").split("/")
        path = strs[-1]
        all_folders.append(path)
        count += 1

    return render(request, 'otoliths/upload.html', {'form': form, 'upload_type': upload_type, 'folders': all_folders })


@login_required
def dataview(request):
    return render(request, 'otoliths/dataview.html')


@login_required
def dataview_sets(request, dataset):

    try:
        os.makedirs('autolith/static/detail/')
    except:
        pass
    try:
        os.makedirs('autolith/static/new_annotations/')
    except:
        pass

    try:
        os.makedirs('autolith/static/data/{}/images/'.format(dataset))
    except:
        pass

    if request.method == 'POST':
        
        # split_name = request.POST['action'].split(":")[-1].strip()
        process = None if 'process' not in request.POST else request.POST['process']
        split_name = request.POST['split_name']
        if process is not None:
            if 'image_scale' in request.POST:
                print(request.POST)
                load_images(dataset, split_name)
            elif 'proportion' in request.POST:
                create_validation(dataset, split_name)
            elif 'create_annotation' in request.POST:
                # create_validation(dataset, split_name)
                # return load_annuli_annotations(request, dataset, split_name)
                raise ValueError("Todo create  anno")
            else:
                # load_annotations(request, dataset, split_name)
                raise ValueError
        else:
            run_type = 'train' if request.POST['action'].startswith('Train') else 'test'
            aimethod = request.POST['aimethod']
            return redirect("/demo/{}_setup/".format(aimethod)+
                            "?split_name={}".format(split_name) +
                            "&run_type={}".format(run_type))

    else:
        if 'mode' in request.GET:
            drawing_tool = 'default'
            try:
                if request.GET['drawing_tool'] == 'brush':
                    drawing_tool = 'brush' 
            except:
                pass
            try:
                annotype = request.GET['annotype']
            except:
                annotype = 'both'
            split_name = request.GET['split_name']
            print("#######")
            print(split_name)
            return load_annuli_annotations(request, dataset, split_name, annotype=annotype, drawing_tool=drawing_tool)
        elif 'aiprocess' in request.GET:
            # from mrcnn.utils import resize_image, resize_mask
            # dataset = 'datasets_user'

            folder = request.GET['split_name']
            method = request.GET['method_name']
            if dataset == 'datasets_north':
                sample_model_unet = 'unet_north_sample.h5'
                sample_model_mrcnn = 'mrcnn_north_sample.h5'
            else:
                sample_model_unet = 'unet_baltic_sample.h5'
                sample_model_mrcnn = 'mrcnn_baltic_sample.h5'

            img_files = glob.glob("{}/{}/*.png".format(dataset, folder))
            print("********")
            print(len(img_files))
            print("********")
            for img_file in img_files:
                og_img = skimage.io.imread(img_file)
                strs = img_file.replace("\\", "/").split("/")
                image_name = strs[-1]

                # if os.path.isfile('autolith/static/data/{}_annuli/{}/{}'.format(method, folder, image_name)):
                #     print("pred exists")
                #     continue
                print("starting prediction")
                # og_img = skimage.io.imread('{}/{}/{}'.format(dataset, folder, image_name))
                sq_img, window, scale, padding, _ = resize_image(
                og_img, 
                min_dim=512,
                max_dim=512,
                #padding=True,
                )
                try:
                    print("predicting...")
                    # folder = 'train_sample_0'
                    # predict_mrcnn(og_img, sq_img, window, dataset, folder, image_name)
                    if method == 'unet':
                        predict_unet(og_img, sq_img, window, dataset, folder, image_name, model_name=sample_model_unet)
                    else:
                        predict_mrcnn(og_img, sq_img, window, dataset, folder, image_name, model_name=sample_model_mrcnn)
                except:
                    raise
                    pass

    all_split_name = []
    all_dir_paths = glob.glob('{}/train_*'.format(dataset))
    for dir_path in all_dir_paths:
        strs = dir_path.replace("\\", "/").split("/")
        path = strs[-1]
        all_split_name.append(path.split("_")[1])


    subset = ''
    if 'subset' in request.GET:
        subset = request.GET['subset']
        if subset == 'all':
            subset = ''

    folder_type = 'train'
    if 'folder_type' in request.GET:
        folder_type = request.GET['folder_type']

    form = DataFilterForm(initial={'subset': subset}, request=request, choices=['all'] + sorted(list(set(all_split_name))) )   


    if folder_type == 'raw':
        raw_data = True
        all_dir_paths = glob.glob('{}/{}_{}*'.format(dataset, folder_type, subset))
        # all_dir_paths.extend(glob.glob('{}/raw_{}*'.format(dataset, subset)))
        print(all_dir_paths)
        all_folders = []
        count = 0
        for dir_path in all_dir_paths:
            strs = dir_path.replace("\\", "/").split("/")
            path = strs[-1]

            og_name = path.replace('train_', '', 1).replace('raw_', '', 1)

            img_files = glob.glob('{}/{}/*.png'.format(dataset, path))
            with_images = bool(img_files)

            train_files = glob.glob('{}/train_{}/*.png'.format(dataset, og_name))
            valid_files = glob.glob('{}/valid_{}/*.png'.format(dataset, og_name))
            with_valid = bool(valid_files) and bool(train_files)


            json_files = glob.glob('{}/train_{}/*.json'.format(dataset, og_name))
            with_json = bool(json_files)

            
            for_valid = with_images and with_json and not with_valid
            print(for_valid)
            ready = with_images and with_valid and with_json


            all_folders.append([path, raw_data, with_valid, with_json, ready, for_valid])
            count += 1
    else:
        raw_data =  False
        all_dir_paths = glob.glob('{}/train_{}*'.format(dataset, subset))
        # all_dir_paths.extend(glob.glob('{}/raw_{}*'.format(dataset, subset)))
        print(all_dir_paths)
        all_folders = []
        count = 0
        for dir_path in all_dir_paths:
            strs = dir_path.replace("\\", "/").split("/")
            path = strs[-1]

            og_name = path.replace('train_', '', 1).replace('raw_', '', 1)

            img_files = glob.glob('{}/{}/*.png'.format(dataset, path))
            with_images = bool(img_files)

            train_files = glob.glob('{}/train_{}/*.png'.format(dataset, og_name))
            valid_files = glob.glob('{}/valid_{}/*.png'.format(dataset, og_name))
            with_valid = bool(valid_files) and bool(train_files)
            if with_valid:
                with_valid = "valid_{}".format(og_name)


            json_files = glob.glob('{}/train_{}/*.json'.format(dataset, og_name))
            with_json = bool(json_files)

            
            for_valid = with_images and with_json and not with_valid
            print(for_valid)
            ready = with_images and with_valid and with_json


            all_folders.append([path, raw_data, with_valid, with_json, ready, for_valid])
            count += 1

    return render(request, 'otoliths/dataview_sets.html', {'dataset': dataset, 'folders': all_folders, 'form': form})




@login_required
def dataview_images(request, dataset, folder):
    username = request.user.username
    from django.core.paginator import Paginator

    try:
        current_page = request.GET['page']
    except:
        current_page = 1

    #------------------------------
    user_uploaded = False
    # if dataset == 'datasets_user' and folder.startswith('raw_'):
    if dataset != 'datasets_north' and dataset != 'datasets_baltic' and folder.startswith('raw_'):
        user_uploaded = True

    print(user_uploaded)
    print("=============")
    try:
        os.makedirs('autolith/static/detail/')
    except:
        pass
    try:
        os.makedirs('autolith/static/new_annotations/')
    except:
        pass

    img_files = glob.glob('{}/{}/*.png'.format(dataset, folder) )
    json_files = glob.glob('{}/{}/*.json'.format(dataset, folder) )
    # img_files.extend(glob.glob('autolith/static/data/{}/{}/*.jpg'.format(dataset_val, folder)))
    img_files = sorted(img_files)
    print(img_files)
    all_images = []
    all_thumbnails = []
    count = 0
    for img_file in img_files:
        strs = img_file.replace("\\", "/").split("/")
        all_images.append(strs[-1])
        # all_images.append('autolith/static/data/{}/{}/{}'.format(dataset, folder, strs[-1]))
        count += 1


    p = Paginator(all_images,30) # 30
    print(">>>>>>>>>>")
    print(p.num_pages)
    print(p.count)

    page_obj = p.page(current_page)
    print(page_obj.object_list)
    for image_name in page_obj.object_list:
        # if not os.path.isfile('autolith/static/data/{}/{}/{}'.format(dataset, folder, img_item )):
        if os.path.isfile("autolith/static/data/{}/{}/{}".format(dataset, folder, image_name)):
            # sq_img = skimage.io.imread("autolith/static/data/{}/{}/{}".format(dataset, folder, image_name))
            pass
        else:
            og_img = skimage.io.imread('{}/{}/{}'.format(dataset, folder, image_name))
            sq_img, window, scale, padding, _ = resize_image(
            og_img, 
            min_dim=512,
            max_dim=512,
            #padding=True,
            )
            try:
                os.makedirs("autolith/static/data/{}/{}/".format(dataset, folder))
            except:
                pass
            skimage.io.imsave("autolith/static/data/{}/{}/{}".format(dataset, folder, image_name), sq_img)

    print(page_obj.number)
    print(page_obj.has_next())
    print(page_obj.has_previous())

    #------------------------------

    with_preds = False
    current_ai_method = 'mrcnn'
    # dataset_val = dataset
    if request.method == 'POST':
        print(request.POST)
        if 'alpha' in request.POST:
            create_outer_contours(dataset, folder)
            with_preds = 'watershed'
            user_uploaded = False
            if dataset != 'datasets_north' and dataset != 'datasets_baltic' and folder.startswith('raw'):
                user_uploaded = True
            return render(request, 'otoliths/dataview_images_list.html', 
                {'dataset': dataset, 'folder': folder, 'images': page_obj.object_list, 'user_uploaded': user_uploaded, 'with_preds': with_preds, 'page_obj': page_obj}   
            )


        elif 'scale' in request.POST:
            scale_data_from_contours(request, dataset, folder)
            return redirect('/otoliths/dataview/{}/'.format(dataset))
        elif 'split_name' in request.POST:
            split_name = request.POST['split_name']
            load_images(dataset, split_name)
            return redirect('/otoliths/dataview/{}/{}'.format(dataset,folder))
    else:

        if 'mode' in request.GET:
            drawing_tool = 'default'
            try:
                if request.GET['drawing_tool'] == 'brush':
                    drawing_tool = 'brush' 
            except:
                pass
            if request.GET['mode'] == 'annuli_annotate':
                try:
                    annotype = request.GET['annotype']
                except:
                    annotype = 'both'


                return load_annuli_annotations(request, dataset, folder, annotype=annotype, drawing_tool=drawing_tool)
            else:
                return load_annotations(request, dataset, folder, drawing_tool=drawing_tool)


        elif 'aitrain' in request.GET:
            current_ai_method = 'mrcnn'
            if 'aimethod' in request.GET and request.GET['aimethod'] == 'U-Net':
                current_ai_method = 'unet'
            print(request.GET)
            settings = get_url_params(request.GET, mode='POST')

            strs = folder.split("_")
            idr = int(strs[-1])
            split_name = strs[-2]
            settings['split_name'] = split_name
            settings['idr'] = idr
            settings['run_type'] = 'train'

            print(settings)


            weights = request.GET['weights']
            if weights == 'None':
                pass
            else:
                strs = weights.split(" // ")
                source_dataset = strs[0]
                source_folder = strs[1].replace(" model", "")
                model_path = glob.glob("{}/{}/*.h5".format(source_dataset, source_folder))[0]
                print(model_path)
                settings['base'] = model_path

                if current_ai_method == 'unet':
                    from tensorflow.keras.models import load_model
                    try:
                        model = load_model(model_path, compile=False)
                        print("LOADING FROM PATH")
                    except:
                        raise ValueError('no model found') #model = load_model('datasets_user/models/{}'.format(model_name), compile=False)
                    model._make_predict_function()
                    input_shape = model.layers[0].input_shape[0]
                    CHANNELS = input_shape[-1]
                    settings['channels'] = CHANNELS
                    settings['transfer'] = True


            if current_ai_method == 'unet':
                if dataset == 'datasets_north':
                    settings['selected'] = [47]
                else:
                    settings['selected'] = [37]
                run_unet(settings)
            else:
                if dataset == 'datasets_north':
                    settings['selected'] = [6]
                else:
                    settings['selected'] = [2]
                run_mrcnn(settings)
                print("=======================================")

            all_h5 = glob.glob('datasets_*/{}*/*.h5'.format(current_ai_method) )
            print(len(all_h5))
            all_models = []
            all_models.append('None')
            for item in all_h5:
                strs = item.replace("\\", "/").split("/")
                all_models.append("{} // {} model".format(strs[-3], strs[-2] ))

            return render(request, 'otoliths/dataview_images_train.html', 
                {'dataset': dataset, 'folder': folder, 'split_name': folder, 'images': page_obj.object_list, 'all_models': all_models, 
                'current_task': "train", 'current_ai_method': current_ai_method,
                'page_obj': page_obj, 'current_label': 'samplerun'}   
            )

        elif 'aiprocess' in request.GET or 'aiprocessrow' in request.GET:
            current_ai_method = 'mrcnn'
            if 'aimethod' in request.GET and request.GET['aimethod'] == 'U-Net':
                current_ai_method = 'unet'

            print(request.GET)
            weights = request.GET['weights']
            strs = weights.split(" // ")
            source_dataset = strs[0]
            source_folder = strs[1].replace(" model", "")
            model_path = glob.glob("{}/{}/*.h5".format(source_dataset, source_folder))[0]
            print(model_path)


            # raise ValueError
            if dataset == 'datasets_north':
                sample_model_unet = 'unet_north_sample.h5'
                sample_model_mrcnn = 'mrcnn_north_sample.h5'
            else:
                sample_model_unet = 'unet_baltic_sample.h5'
                sample_model_mrcnn = 'mrcnn_baltic_sample.h5'

            try:
                with open('{}/all_data_map.json'.format(dataset)) as fson:
                    age_data_map = json.load(fson)
            except:
                pass

            pred_lines = []
            count = 0
            with_preds = []
            for img_file in img_files:
                

                og_img = skimage.io.imread(img_file)
                strs = img_file.replace("\\", "/").split("/")
                image_name = strs[-1]

                # if os.path.isfile('autolith/static/data/{}/{}/{}'.format(source_folder, folder, image_name)):
                #     print("pred exists")
                #     with_preds.append(image_name)
                #     continue

                count += 1
                if 'aiprocessrow' in request.GET and count > 6:
                    break


                try:
                    manual_age = int(age_data_map[image_name])
                except:
                    manual_age = int(image_name.split('_age_')[1].split('.')[0])
                # og_img = skimage.io.imread('{}/{}/{}'.format(dataset, folder, image_name))
                sq_img, window, scale, padding, _ = resize_image(
                og_img, 
                min_dim=512,
                max_dim=512,
                #padding=True,
                )
                try:
                    print("predicting..")
                    if current_ai_method == 'unet':
                        ai_reading = predict_unet(og_img, sq_img, window, dataset, folder, image_name, model_name=sample_model_unet, model_path=model_path, run_label=source_folder)
                    else:
                        ai_reading = predict_mrcnn(og_img, sq_img, window, dataset, folder, image_name, model_name=sample_model_mrcnn, model_path=model_path, run_label=source_folder)
                    pred_lines.append("{},{},{}".format(image_name, ai_reading, manual_age))
                    with_preds.append(image_name)
                except:
                    raise
                    pass

            output_file = "{}/{}/{}.txt".format(dataset, folder, source_folder)
            with open(output_file, 'w') as fout:
                fout.write("\n".join(pred_lines))

            
            try:
                current_task = request.GET['current_task']
            except:
                current_task = 'test'

            all_h5 = glob.glob('datasets_*/{}*/*.h5'.format(current_ai_method) )
            print(len(all_h5))
            all_models = []
            for item in all_h5:
                strs = item.replace("\\", "/").split("/")
                all_models.append("{} // {} model".format(strs[-3], strs[-2] ))


            if current_task == 'test':
                template_to_load = 'otoliths/dataview_images_test.html'
            else:
                template_to_load = 'otoliths/dataview_images_annotate.html'

            return render(request, template_to_load, 
                {'dataset': dataset, 'folder': folder, 'images': page_obj.object_list, 'all_models': all_models, 
                'current_task': current_task, 'current_ai_method': current_ai_method, 'with_preds': with_preds, 'run_label': source_folder, 'current_model': weights,
                'page_obj': page_obj}   
            )


        elif 'ensemble_creation' in request.GET:

            constituents = get_constituents(request)
            ensemble_method = request.GET['aimethod']
            error_found = False
            if len(constituents) < 2:
                error_found = True
            else:
                try:
                    start_ensemble_training(dataset, folder, ensemble_method, constituents)
                except:
                    error_found = True

            if error_found:
                all_models = []
                all_txt = glob.glob('{}/{}/*.txt'.format(dataset, folder) )
                for item in all_txt:
                    strs = item.replace("\\", "/").split("/")
                    substrs = strs[-1].split('_')
                    midpart = "_".join(substrs[1:-1])
                    model_name = midpart[:-5]
                    # model_name = substrs[1][1:-5]

                    all_models.append("{} // {}_{} predictions".format(strs[0], substrs[0], model_name ))

                all_models = list(set(all_models))

                ensemble_list = []
                all_pkl = glob.glob('{}/train_*/*.pkl'.format(dataset) )
                for item in all_pkl:
                    strs = item.replace("\\", "/").split("/")
                    ensemble_list.append("{} // {} // {} ".format(strs[0], strs[1], strs[2]) )


                return render(request, 'otoliths/dataview_images_ensemble_train.html', 
                    {'dataset': dataset, 'folder': folder, 'images': page_obj.object_list, 'all_models': all_models, 
                    'current_task': 'ensemble_train', 'current_ai_method': current_ai_method, 'ensemble_list': ensemble_list, 'split_name': folder,
                    'page_obj': page_obj, 'error_found': error_found }   
                )
            else:
                return redirect('/otoliths/dataview/{}/'.format(dataset))

        elif 'ensemble_prediction' in request.GET:
            print(request.GET)
            weights = request.GET['weights']
            strs = weights.split(" // ")
            ensemble_path = "{}/{}".format(strs[0], strs[1])
            ensemble_method = strs[-1].replace(".pkl", "").strip()
            print(ensemble_method)
            error_found = False
            try:
                start_ensemble_testing(dataset, folder, ensemble_method, ensemble_path)
            except:
                error_found = True
            if error_found:
                all_models = []
                all_txt = glob.glob('{}/{}/*.txt'.format(dataset, folder) )
                for item in all_txt:
                    strs = item.replace("\\", "/").split("/")
                    substrs = strs[-1].split('_')
                    midpart = "_".join(substrs[1:-1])
                    model_name = midpart[:-5]

                    all_models.append("{} // {}_{} predictions".format(strs[0], substrs[0], model_name ))

                all_models = list(set(all_models))

                ensemble_list = []
                all_pkl = glob.glob('{}/train_*/*.pkl'.format(dataset) )
                for item in all_pkl:
                    strs = item.replace("\\", "/").split("/")
                    ensemble_list.append("{} // {} // {} ".format(strs[0], strs[1], strs[2]) )


                return render(request, 'otoliths/dataview_images_ensemble_test.html', 
                    {'dataset': dataset, 'folder': folder, 'images': page_obj.object_list, 'all_models': all_models, 
                    'current_task': 'ensemble_test', 'current_ai_method': current_ai_method, 'ensemble_list': ensemble_list, 'split_name': folder,
                    'page_obj': page_obj, 'error_found': error_found }   
                )
            else:
                fname = "{}.txt".format(ensemble_method)
                return redirect("/otoliths/results/"+
                                "?dataset={}".format("datasets_ensemble") +
                                "&source={}".format(dataset) +
                                "&folder={}".format(folder) +
                                "&fname={}".format(fname))


        else:

            try:
                current_task = request.GET['current_task']
            except:
                current_task = 'list'


            if current_task == 'list':
                user_uploaded = False
                if dataset != 'datasets_north' and dataset != 'datasets_baltic' and folder.startswith('raw'):
                    user_uploaded = True


                return render(request, 'otoliths/dataview_images_list.html', 
                    {'dataset': dataset, 'folder': folder, 'images': page_obj.object_list, 'user_uploaded': user_uploaded, 
                    'json_files': json_files, 'page_obj': page_obj}   
                )

            elif current_task == 'annotate':
                current_ai_method = 'mrcnn'
                if 'aimethod' in request.GET and request.GET['aimethod'] == 'U-Net':
                    current_ai_method = 'unet'

                all_h5 = glob.glob('datasets_*/{}*/*.h5'.format(current_ai_method) )
                print(len(all_h5))
                all_models = []
                for item in all_h5:
                    strs = item.replace("\\", "/").split("/")
                    all_models.append("{} // {} model".format(strs[-3], strs[-2] ))

                return render(request, 'otoliths/dataview_images_annotate.html', 
                    {'dataset': dataset, 'folder': folder, 'images': page_obj.object_list, 
                     'current_ai_method': current_ai_method, 'all_models': all_models, 'current_task': current_task,
                     'page_obj': page_obj}   
                )

            elif current_task == 'ensemble_test':


                try:
                    current_ensemble = request.GET['current_ensemble']
                    disabled_edit = True
                except:
                    current_ensemble = ''
                    disabled_edit = False

                try:
                    current_model = request.GET['current_model']
                except:
                    current_model = ''

                all_models = []
                all_txt = glob.glob('{}/{}/*.txt'.format(dataset, folder) )
                for item in all_txt:
                    strs = item.replace("\\", "/").split("/")
                    substrs = strs[-1].split('_')
                    midpart = "_".join(substrs[1:-1])
                    model_name = midpart[:-5]

                    all_models.append("{} // {}_{} predictions".format(strs[0], substrs[0], model_name ))

                all_models = list(set(all_models))

                ensemble_list = []
                all_pkl = glob.glob('{}/train_{}*/{}*.pkl'.format(dataset, current_model, current_ensemble) )
                for item in all_pkl:
                    strs = item.replace("\\", "/").split("/")
                    ensemble_list.append("{} // {} // {} ".format(strs[0], strs[1], strs[2]) )


                return render(request, 'otoliths/dataview_images_ensemble_test.html', 
                    {'dataset': dataset, 'folder': folder, 'images': page_obj.object_list, 'all_models': all_models, 
                    'current_task': current_task, 'current_ai_method': current_ai_method, 'ensemble_list': ensemble_list, 'split_name': folder,
                    'page_obj': page_obj }   
                )

            elif current_task == 'ensemble_train':
                try:
                    current_ensemble = request.GET['current_ensemble']
                    disabled_edit = True
                except:
                    current_ensemble = ''
                    disabled_edit = False


                all_models = []
                all_txt = glob.glob('{}/{}/*.txt'.format(dataset, folder) )
                for item in all_txt:
                    strs = item.replace("\\", "/").split("/")
                    substrs = strs[-1].split('_')
                    midpart = "_".join(substrs[1:-1])
                    model_name = midpart[:-5]

                    all_models.append("{} // {}_{} predictions".format(strs[0], substrs[0], model_name ))

                all_models = list(set(all_models))

                ensemble_list = []
                all_pkl = glob.glob('{}/train_*/*.pkl'.format(dataset) )
                for item in all_pkl:
                    strs = item.replace("\\", "/").split("/")
                    ensemble_list.append("{} // {} // {} ".format(strs[0], strs[1], strs[2]) )


                return render(request, 'otoliths/dataview_images_ensemble_train.html', 
                    {'dataset': dataset, 'folder': folder, 'images': page_obj.object_list, 'all_models': all_models, 
                    'current_task': current_task, 'current_ai_method': current_ai_method, 'ensemble_list': ensemble_list, 'split_name': folder,
                    'page_obj': page_obj, 'disabled_edit': disabled_edit }   
                )


            elif current_task == 'train':
                current_ai_method = 'mrcnn'
                if 'aimethod' in request.GET and request.GET['aimethod'] == 'U-Net':
                    current_ai_method = 'unet'

                try:
                    current_model = request.GET['current_model']
                    disabled_edit = True
                except:
                    current_model = ''
                    disabled_edit = False

                try:
                    current_label = request.GET['current_label']
                except:
                    current_label = 'test'

                print(current_model)
                # raise ValueError
                all_h5 = glob.glob('datasets_*/{}_{}*/*.h5'.format(current_ai_method, current_model) )
                print(len(all_h5))
                all_models = []
                if not disabled_edit:
                    all_models.append('None')

                for item in all_h5:
                    strs = item.replace("\\", "/").split("/")
                    all_models.append("{} // {} model".format(strs[-3], strs[-2] ))

                return render(request, 'otoliths/dataview_images_train.html', 
                    {'dataset': dataset, 'folder': folder, 'split_name': folder, 'images': page_obj.object_list, 'all_models': all_models, 
                    'current_task': current_task, 'current_ai_method': current_ai_method, 'current_label': current_label,
                    'page_obj': page_obj, 'disabled_edit': disabled_edit}   
                )
            elif current_task == 'test':
                current_ai_method = 'mrcnn'
                if 'aimethod' in request.GET and request.GET['aimethod'] == 'U-Net':
                    current_ai_method = 'unet'

                try:
                    current_model = request.GET['current_model']
                    disabled_edit = True
                except:
                    current_model = ''
                    disabled_edit = False

                if disabled_edit:
                    all_h5 = glob.glob('datasets_{}/{}_{}*/*.h5'.format(username, current_ai_method, current_model) )
                else:
                    all_h5 = glob.glob('datasets_*/{}_{}*/*.h5'.format(current_ai_method, current_model) )
                print(len(all_h5))
                all_models = []
                for item in all_h5:
                    strs = item.replace("\\", "/").split("/")
                    all_models.append("{} // {} model".format(strs[-3], strs[-2] ))


                return render(request, 'otoliths/dataview_images_test.html', 
                    {'dataset': dataset, 'folder': folder, 'images': page_obj.object_list, 'all_models': all_models, 
                    'current_task': current_task, 'current_ai_method': current_ai_method,
                    'page_obj': page_obj, 'disabled_edit': disabled_edit}   
                )


@login_required
def data_detail(request, dataset, folder, image_name):
    import skimage.io
    # from mrcnn.utils import resize_image, resize_mask

    if dataset == 'datasets_north':
        sample_model_unet = 'unet_north_sample.h5'
        sample_model_mrcnn = 'mrcnn_north_sample.h5'
    else:
        sample_model_unet = 'unet_baltic_sample.h5'
        sample_model_mrcnn = 'mrcnn_baltic_sample.h5'


    print(request.POST)
    print(request.GET)
    current_process = 'view'
    try:
        current_process = request.GET['process']
        if current_process.startswith('U-Net'):
            current_process = 'unet'
            print("-------")
        elif current_process.startswith('Mask'):
            current_process = 'mrcnn'
    except:
        pass

    try:
        os.makedirs("autolith/static/detail/")
    except:
        pass

    if 'process' in request.GET:
        print("started loading image")
        og_img = skimage.io.imread('{}/{}/{}'.format(dataset, folder, image_name))
        sq_img, window, scale, padding, _ = resize_image(
        og_img, 
        min_dim=512,
        max_dim=512,
        #padding=True,
        )
        print("started loading whcontour")

        if request.GET['process'].startswith('Brighten') or request.GET['process'].startswith('Background'):
            # if os.path.isfile("{}/{}/0/whole_scaled.json")
            wh_files = glob.glob("{}/{}/0/*.json".format(dataset, folder))

            if dataset == 'datasets_north':
                wh_files.extend(glob.glob("datasets_north/images/0/*.json") )
            elif dataset == 'datasets_baltic':
                wh_files.extend(glob.glob("datasets_baltic/images/0/*.json") )
            else:
                wh_files.extend(glob.glob("datasets_users/images/0/*.json") )
            
            json_annotations = {}
            for json_file in wh_files:
                _annotations = json.load(open(json_file))
                try:
                    _annotations = list(_annotations['_via_img_metadata'].values())
                except:
                    _annotations = list(_annotations.values())

                _annotations = [a for a in _annotations if a['regions']]
                found = False
                for a in _annotations:
                    if a['filename'] == image_name:
                        json_annotations[a['filename']] = a
                        found = True
                        break
                if found:
                    break
            wh_mask = None
            if image_name in json_annotations:
                a = json_annotations[image_name]
                h, w = og_img.shape[:2]
                polygons = [r['shape_attributes'] for r in a['regions']]
                wh_mask = np.zeros([h, w, 1], dtype=np.uint8)
                for idx, poly in enumerate(polygons):
                    rr, cc = skimage.draw.polygon(poly['all_points_y'], poly['all_points_x'])
                    wh_mask[rr, cc, 0] = 1 
                whole_mask = resize_mask(wh_mask, scale, padding)
            print("started loading linking")


            current_process = 'preproc'
            if request.GET['process'] == 'Brighten':
                
                img_mod = modify_image_brighten(sq_img, whole_mask, dataset)
                skimage.io.imsave("autolith/static/detail/{}".format(image_name), img_mod)
                skimage.io.imsave("autolith/static/data/{}/{}/{}".format(dataset, folder, image_name), img_mod)
                print("Brighten")
            elif request.GET['process'] == 'Background Removal':
                
                img_mod = modify_image_bgremove(sq_img, whole_mask, dataset)
                # skimage.io.imsave("autolith/static/detail/{}".format(image_name), img_mod)
                skimage.io.imsave("autolith/static/detail/{}".format(image_name), img_mod)
                skimage.io.imsave("autolith/static/data/{}/{}/{}".format(dataset, folder, image_name), img_mod)
                print("BG Removal")
            elif request.GET['process'] == 'Brighten with Background Removal':
                
                img_mod = modify_image_both(sq_img, whole_mask, dataset)
                skimage.io.imsave("autolith/static/detail/{}".format(image_name), img_mod)
                # skimage.io.imsave("autolith/static/detail/{}".format(image_name), img_mod)
                skimage.io.imsave("autolith/static/data/{}/{}/{}".format(dataset, folder, image_name), img_mod)
                print("Brighten with BG Removal")
        else:
            # predict_mrcnn(og_img, sq_img, window, dataset, folder, image_name, mode="core")
            # raise ValueError
            if request.GET['process'] == 'Mask R-CNN Detection':
                predict_mrcnn(og_img, sq_img, window, dataset, folder, image_name)
                # raise ValueError
                # result_name ="mrcnn_image_{}.png".format(image_name)
            elif request.GET['process'] == 'CoreDetection':
                predict_mrcnn(og_img, sq_img, window, dataset, folder, image_name, mode="core")
                # result_name ="mrcnn_image_{}.png".format(image_name) 
            elif request.GET['process'] == 'Combined':
                predict_mrcnn_combined(og_img, sq_img, window, dataset, folder, image_name, mode="combined")
                # result_name ="mrcnn_image_{}.png".format(image_name) 
            elif request.GET['process'] == 'User Annotation':
                print("user mode")

                return load_blank_marks(request, og_img, sq_img, window, dataset, folder, image_name)
            elif request.GET['process'] == 'AI-Assisted Annotation' or request.GET['process'] == "Segment":
                if 'convert' in request.GET:
                    return convert_data_marks(request, og_img, sq_img, window, dataset, folder, image_name, markings='poly')
                elif 'mode' in request.GET and request.GET['mode'] == 'brush':
                    print("brush mode")
                    return load_data_marks(request, og_img, sq_img, window, dataset, folder, image_name, markings='poly', mode='brush')
                else:
                    print("default mode")
                    return load_data_marks(request, og_img, sq_img, window, dataset, folder, image_name, markings='poly')
            else:
                predict_unet(og_img, sq_img, window, dataset, folder, image_name, model_name=sample_model_unet)
        result_name = image_name
    else:
        if os.path.isfile("autolith/static/data/{}/{}/{}".format(dataset, folder, image_name)):
            sq_img = skimage.io.imread("autolith/static/data/{}/{}/{}".format(dataset, folder, image_name))
        else:
            og_img = skimage.io.imread('{}/{}/{}'.format(dataset, folder, image_name))
            sq_img, window, scale, padding, _ = resize_image(
            og_img, 
            min_dim=512,
            max_dim=512,
            #padding=True,
            )
            skimage.io.imsave("autolith/static/data/{}/{}/{}".format(dataset, folder, image_name), sq_img)

    return render(request, 'otoliths/image_detail.html', {'dataset': dataset, 'folder': folder, 'image_name': image_name, 
        'current_process': current_process, 'sample_model_unet': sample_model_unet, 'sample_model_mrcnn': sample_model_mrcnn})


@csrf_exempt
def interact(request):
    import copy
    request = json.loads(request.body.decode('utf-8'))
    # print(request['_via_image_id_list'])
    print(request['dataset'])
    print(request['folder'])
    print(request['dataformat'])
    # print(request)   
    print("+++++++++++++@@@@@@@@@@@@@@@@@@@@@@@@@___{}".format(request['_via_data_format_version']) )

    try:
        os.makedirs('autolith/static/new_annotations/')
    except:
        pass

    dataset = request['dataset']
    folder = request['folder']
    dataformat = request['dataformat']

    if dataset == 'datasets_user':
        if dataformat == 'outer':
            try:
                os.makedirs("{}/{}/0/".format(dataset, folder))
            except:
                pass
            with open("{}/{}/0/whole_proj.json".format(dataset, folder), "w") as fout:
                json.dump(request, fout, indent=4)

        else:
            for image_item in request['_via_image_id_list']:
                main_json = copy.deepcopy(request)
                img_name = '{}.png'.format(image_item.split('.png')[0])

                main_json['_via_settings']['project']['name']  = img_name
                main_json['_via_image_id_list'] = [image_item]
                main_json['_via_img_metadata'] = {image_item: request['_via_img_metadata'][image_item] }

                if len(main_json['_via_img_metadata'][image_item]['regions'])> 0:
                    with open("{}/{}/{}_json.json".format(dataset, folder, img_name), "w") as fout:
                        json.dump(main_json['_via_img_metadata'], fout, indent=4)
                # else:
                # if len(main_json['_via_img_metadata'][image_item]['regions'])> 0:
                #     with open("autolith/static/new_annotations/{}.json".format(img_name), "w") as fout:
                #         json.dump(main_json, fout, indent=4)
    else:
        for image_item in request['_via_image_id_list']:
            main_json = copy.deepcopy(request)
            img_name = '{}.png'.format(image_item.split('.png')[0])

            main_json['_via_settings']['project']['name']  = img_name
            main_json['_via_image_id_list'] = [image_item]
            main_json['_via_img_metadata'] = {image_item: request['_via_img_metadata'][image_item] }

            # if len(main_json['_via_img_metadata'][image_item]['regions'])> 0:
            #     with open("{}/{}/{}_json.json".format(dataset, folder, img_name), "w") as fout:
            #         json.dump(main_json['_via_img_metadata'], fout, indent=4)
            # else:
            if len(main_json['_via_img_metadata'][image_item]['regions'])> 0:
                with open("autolith/static/new_annotations/{}.json".format(img_name), "w") as fout:
                    json.dump(main_json, fout, indent=4)
                     
    return HttpResponse("success")