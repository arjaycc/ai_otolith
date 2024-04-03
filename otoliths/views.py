from django.shortcuts import render
from django.http import HttpResponse
from django.core.paginator import Paginator
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from .forms import UploadForm
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

@login_required
def upload(request):
    form = UploadForm()
    return render(request, 'otoliths/upload.html', {'form': form})

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
    settings = {
        'dataset' : 'datasets_north',
        'run_type': 'test',
        'run_label': 'randsub',
        'search_mode': False,
        'idr': 0,
        'selected': [47],
        'split_type': 'rs', # 5-fold, 5-random subsampling
        'split_name': 'randsub',
        'checkpoint': '_checkpoint',
        #'target_species': 'cod',
        #'age_limit': 4,
        #'brighten': True,
        #'source_dataset': 'datasets_north'
    }
    run_unet(settings)
    return render(request, 'otoliths/experiments.html')


@login_required
def interact(request):
    request = json.loads(request.body.decode('utf-8'))
    print(request['_via_image_id_list'])
    img_name = '{}.png'.format(request['_via_image_id_list'][0].split('.png')[0])
    with open("autolith/static/new_annotations/{}.json".format(img_name), "w") as fout:
        json.dump(request, fout, indent=4)
        
    return HttpResponse("success")

@login_required
def dataview_sets(request, dataset):
    img_files = glob.glob('autolith/static/data/{}/*/*.png'.format(dataset))
    img_files.extend(glob.glob('autolith/static/data/{}/*/*.jpg'.format(dataset)))
    print(img_files)
    all_folders = []
    for img_file in img_files:
        strs = img_file.replace("\\", "/").split("/")
        all_folders.append(strs[-2])
    all_folders = list(set(all_folders))
    return render(request, 'otoliths/dataview_sets.html', {'folders': all_folders})

@login_required
def dataview_images(request, dataset, folder):
    img_files = glob.glob('autolith/static/data/{}/{}/*.png'.format(dataset, folder) )
    img_files.extend(glob.glob('autolith/static/data/{}/{}/*.jpg'.format(dataset, folder)))
    print(img_files)
    all_images = []
    count = 0
    for img_file in img_files:
        strs = img_file.replace("\\", "/").split("/")
        all_images.append(strs[-1])
        count += 1
        if count > 25:
            break
    return render(request, 'otoliths/dataview_images.html', {'dataset': dataset, 'folder': folder, 'images': all_images})

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


