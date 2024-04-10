from django.shortcuts import render, redirect
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
import shutil

from .experiments_imgproc import *
from .experiments_mrcnn import *
from .experiments_unet import *
from .views_utils import *
# from .views_aux import *


def index(request):
    return render(request, 'demo/index.html')


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


def generate_run_folder_from_ensemble_data(self, domain, method):
    all_files = glob.glob("datasets_ensemble/{}_{}_*/*.txt".format(method, domain))
    for ff in all_files:
        fname = ff.replace("\\", "/").split("/")[-1]
        print(fname)
        strs = fname.split('_{}_'.format(method))
        print(strs[0])

        try:
            os.makedirs('datasets_{}/{}/'.format(domain, strs[0]))
        except:
            pass

        shutil.copyfile(ff, 'datasets_{}/{}/{}_{}'.format(domain, strs[0], method, strs[1]))


def ai(request):
    images = []
    img_files = glob.glob('autolith/static/data/datasets_*/*/*.png')
    print(img_files)
    all_images = []
    count = 0
    for img_file in img_files:
        strs = img_file.replace("\\", "/").split("/")

        image_name = strs[-1]
        folder_name = strs[-2] 
        dataset = strs[-3]
        all_images.append(image_name)
        images.append([image_name, dataset, folder_name])
        count += 1
        if count > 25:
            break
    return render(request, 'demo/aimethod.html', {'images': images } )


# def upload(request):
#     form = UploadForm()
#     return render(request, 'demo/upload.html', {'form': form})
def upload(request):
    from .forms import UploadForm
    if request.method == 'POST':
        print(request.POST)
        form  = UploadForm(request.POST, request.FILES)
        data_id = request.POST['data_id']

        
        for ff in request.FILES.getlist('files'):
            print(ff.name)
            if ff.name.endswith('.h5'):
                try:
                    os.makedirs("datasets_user/models/")
                except:
                    pass

                with open("datasets_user/models/{}".format(ff.name), "wb+") as dest:
                    for chunk in ff.chunks():
                        dest.write(chunk) 
            else:
                try:
                    os.makedirs("datasets_user/data_{}_{}/0/".format(request.POST['folder'], data_id))
                except:
                    pass
                with open("datasets_user/data_{}_{}/{}".format(request.POST['folder'], data_id, ff.name), "wb+") as dest:
                    for chunk in ff.chunks():
                        dest.write(chunk) 
        for ff in request.FILES.getlist('ring_groundtruth'):
            print(ff.name)
            try:
                with open("datasets_user/train_{}_{}/{}".format(request.POST['folder'], data_id, ff.name), "wb+") as dest:
                    for chunk in ff.chunks():
                        dest.write(chunk) 
            except:
                pass
        for ff in request.FILES.getlist('outer_contour'):
            print(ff.name)
            with open("datasets_user/data_{}_{}/0/{}".format(request.POST['folder'], data_id, ff.name), "wb+") as dest:
                for chunk in ff.chunks():
                    dest.write(chunk) 

        all_json_files = glob.glob("datasets_user/train_{}_{}/*.json".format(request.POST['folder'], data_id) )
        for jsonf in all_json_files:
            strs = jsonf.replace("\\","/").split("/")
            shutil.copyfile(jsonf, "datasets_user/annotations/{}".format(strs[-1]) )
        return redirect('/demo/index/')
    else:
        form = UploadForm()
    return render(request, 'demo/upload.html', {'form': form})

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


def run(request):
    if request.method == 'POST':
        print(request.POST)

    initial = get_url_params(request.GET)
    print(initial)
    if 'base' in initial:
        form = RunFormPhase2(initial=initial)
        return render(request, 'demo/ai_run_v2.html', {'form': form})
    else:
        form = RunFormPhase1(initial=initial)
        return render(request, 'demo/ai_run.html', {'form': form})


def unet_setup(request):
    if request.method == 'POST':
        print(request.POST)
        raise ValueError

    initial = get_url_params(request.GET)
    initial.update({'method': 'U-Net'})
    dataset = initial['dataset']
    folder = "unet_{}{}run1_{}".format(initial['run_label'], initial['idr'], initial['selected'])
    outfiles = glob.glob("{}/{}/*_of_*.txt".format(dataset, folder))

    existing_results = False
    if len(outfiles) > 0:
        outfiles.sort(key=os.path.getctime)
        existing_results = outfiles[-1].replace("\\", "/").split("/")[-1]
    print(initial)
    if 'base' in initial:
        form = RunFormPhase2(initial=initial)
        return render(request, 'demo/ai_run_v2.html', {'form': form, 'existing_results': existing_results, 'dataset': dataset, 'folder': folder})
    else:
        form = RunFormPhase1(initial=initial)
        return render(request, 'demo/ai_run.html', {'form': form})

def mrcnn_setup(request):
    if request.method == 'POST':
        print(request.POST)
        raise ValueError

    initial = get_url_params(request.GET)
    initial.update({'method': 'Mask R-CNN'})

    dataset = initial['dataset']
    folder = "mrcnn_{}{}run1_{}".format(initial['run_label'], initial['idr'], initial['selected'])
    outfiles = glob.glob("{}/{}/*_of_*.txt".format(dataset, folder))

    existing_results = False
    if len(outfiles) > 0:
        outfiles.sort(key=os.path.getctime)
        existing_results = outfiles[-1].replace("\\", "/").split("/")[-1]

    print(initial)
    if 'base' in initial:
        form = RunFormPhase2(initial=initial)
        return render(request, 'demo/ai_run_v2.html', {'form': form, 'existing_results': existing_results, 'dataset': dataset, 'folder': folder})
    else:
        form = RunFormPhase1(initial=initial)
        return render(request, 'demo/ai_run.html', {'form': form})


def setup_run(request):
    initial = get_url_params(request.GET)
    form = RunFormPhase2(initial=initial)
    return render(request, 'demo/ai_run_v2.html', {'form': form})

def start_run(request):
    if request.method == 'POST':
        print(request.POST)
        settings = get_url_params(request.POST, mode='POST')
        print(settings)
        if settings['method'] == 'U-Net':
            run_unet(settings)
        elif settings['method'] == 'Mask R-CNN':
            run_mrcnn(settings)

        dataset = settings['dataset']
        # name = 'unet_{}{}run1_{}'.format(run_label, idr, param_count)
        if settings['method'] == 'U-Net':
            folder = "unet_{}{}run1_{}".format(settings['run_label'], settings['idr'], settings['selected'][0])
        elif settings['method'] == 'Mask R-CNN':
            folder = "mrcnn_{}{}run1_{}".format(settings['run_label'], settings['idr'], settings['selected'][0])
        
        outfiles = glob.glob("{}/{}/*_of_*.txt".format(dataset, folder))
        outfiles.sort(key=os.path.getctime)
        fname = outfiles[-1].replace("\\", "/").split("/")[-1]
        return redirect("/demo/results/"+
                        "?dataset={}".format(dataset) +
                        "&folder={}".format(folder) +
                        "&fname={}".format(fname))


def analysis(request):
    image1 = 'result1.png'
    image2 = 'result2.png'
    pairs = [(image1, image2), (image1, image2)]
    return render(request, 'demo/analysis.html', {'pairs': pairs})


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


def visualize(request):
    img_name = "test.jpg"
    convert_to_gray("core/static/visualize/{}".format(img_name), 'core/static/visualize/result.png')
    return render(request, 'otoliths/visualization.html', {'image_name': img_name})


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


def balticsea_sets(request):
    all_dir_paths = glob.glob('datasets_baltic/train_*')
    print(all_dir_paths)
    all_folders = []
    count = 0
    for dir_path in all_dir_paths:
        strs = dir_path.replace("\\", "/").split("/")
        all_folders.append(strs[-1])
        count += 1
        # if count > 25:
        #     break
    # return render(request, 'otoliths/baltic_sets.html', {'images': all_images})
    return render(request, 'demo/dataview_sets.html', {'dataset': 'datasets_baltic', 'folders': all_folders})

def balticsea_images(request, folder):
    img_files = glob.glob('datasets_baltic/{}/*.png'.format(folder))
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


def data_detail(request, image_name):
    print(image_name)
    image_id = image_name
    return render(request, 'otoliths/data_detail.html', {'image_name': image_name})


def experiments(request):
    return render(request, 'demo/experiments.html')


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

@csrf_exempt
def interact(request):
    request = json.loads(request.body.decode('utf-8'))
    print(request['_via_image_id_list'])
    img_name = '{}.png'.format(request['_via_image_id_list'][0].split('.png')[0])
    with open("autolith/static/annotations/{}.json".format(img_name), "w") as fout:
        json.dump(request, fout, indent=4)
        
    return HttpResponse("success")



@csrf_exempt
def dataview_sets(request, dataset):

    if request.method == 'POST':
        
        # split_name = request.POST['action'].split(":")[-1].strip()
        process = None if 'process' not in request.POST else request.POST['process']
        split_name = request.POST['split_name']
        if process is not None:
            if 'image_scale' in request.POST:
                print(request.POST)
                load_images(dataset, split_name)
            elif 'proportion':
                create_validation(dataset, split_name)
            else:
                # load_annotations(request, dataset, split_name)
                raise ValueError
        else:
            run_type = 'train' if request.POST['action'].startswith('Train') else 'test'
            return redirect("/demo/run/"+
                            "?split_name={}".format(split_name) +
                            "&run_type={}".format(run_type))

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

    form = DataFilterForm(initial={'subset': subset}, request=request, choices=['all'] + sorted(list(set(all_split_name))) )   

    all_dir_paths = glob.glob('{}/train_{}*'.format(dataset, subset))
    all_dir_paths.extend(glob.glob('{}/data_{}*'.format(dataset, subset)))
    print(all_dir_paths)
    all_folders = []
    count = 0
    for dir_path in all_dir_paths:
        strs = dir_path.replace("\\", "/").split("/")
        path = strs[-1]

        og_name = path.replace('train_', '', 1).replace('data_', '', 1)

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


        all_folders.append([path, with_images, with_valid, with_json, ready, for_valid])
        count += 1

    return render(request, 'demo/dataview_sets.html', {'dataset': dataset, 'folders': all_folders, 'form': form})

@csrf_exempt
def dataview_images(request, dataset, folder):

    if 'mode' in request.GET:
        return load_annotations(request, dataset, folder)
    # dataset_val = dataset
    if request.method == 'POST':
        print(request.POST)
        if 'alpha' in request.POST:
            create_outer_contours(dataset, folder)
        elif 'scale' in request.POST:
            scale_data_from_contours(request, dataset, folder)
            return redirect('/demo/dataview/{}/'.format(dataset))
        elif 'split_name' in request.POST:
            split_name = request.POST['split_name']
            load_images(dataset, split_name)
            return redirect('/demo/dataview/{}/{}'.format(dataset,folder))

    user_uploaded = False
    if folder.startswith('data_'):
        user_uploaded = True

    img_files = glob.glob('{}/{}/*.png'.format(dataset, folder) )
    # img_files.extend(glob.glob('autolith/static/data/{}/{}/*.jpg'.format(dataset_val, folder)))
    print(img_files)
    all_images = []
    count = 0
    for img_file in img_files:
        strs = img_file.replace("\\", "/").split("/")
        all_images.append(strs[-1])
        count += 1
        if count > 25:
            break
    return render(request, 'demo/dataview_images.html', {'dataset': dataset, 'folder': folder, 'images': all_images, 'user_uploaded': user_uploaded, 'with_images': bool(img_files)})


def data_detail(request, dataset, folder, image_name):
    import skimage.io
    from mrcnn.utils import resize_image, resize_mask

    og_img = skimage.io.imread('{}/{}/{}'.format(dataset, folder, image_name))
    sq_img, window, scale, padding, _ = resize_image(
    og_img, 
    min_dim=512,
    max_dim=512,
    #padding=True,
    )

    # if os.path.isfile("{}/{}/0/whole_scaled.json")
    wh_files = glob.glob("{}/{}/0/*.json".format(dataset, folder))
    if dataset == 'datasets_north':
        wh_files.extend(glob.glob("datasets_north/images/0/*.json") )
    else:
        wh_files.extend(glob.glob("datasets_baltic/images/0/*.json") )
    
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

    if dataset == 'datasets_north':
        sample_model_unet = 'unet_north_sample.h5'
        sample_model_mrcnn = 'mrcnn_north_sample.h5'
    else:
        sample_model_unet = 'unet_baltic_sample.h5'
        sample_model_mrcnn = 'mrcnn_baltic_sample.h5'
    print(request.POST)
    print(request.GET)
    if 'process' in request.GET:
        if request.GET['process'] == 'Brighten':
            img_mod = modify_image_brighten(sq_img, whole_mask, dataset)
            skimage.io.imsave("autolith/static/detail/{}".format(image_name), img_mod)
        elif request.GET['process'] == 'Background Removal':
            img_mod = modify_image_bgremove(sq_img, whole_mask, dataset)
            skimage.io.imsave("autolith/static/detail/{}".format(image_name), img_mod)
        elif request.GET['process'] == 'Brighten with Background Removal':
            img_mod = modify_image_both(sq_img, whole_mask, dataset)
            skimage.io.imsave("autolith/static/detail/{}".format(image_name), img_mod)
        elif request.GET['process'] == 'Mask R-CNN Detection':
            predict_mrcnn(og_img, sq_img, window, dataset, folder, image_name)
            # result_name ="mrcnn_image_{}.png".format(image_name)
        elif request.GET['process'] == 'CoreDetection':
            predict_mrcnn(og_img, sq_img, window, dataset, folder, image_name, mode="core")
            # result_name ="mrcnn_image_{}.png".format(image_name) 
        elif request.GET['process'] == 'Combined':
            predict_mrcnn_combined(og_img, sq_img, window, dataset, folder, image_name, mode="combined")
            # result_name ="mrcnn_image_{}.png".format(image_name) 
        elif request.GET['process'] == 'User Annotation':
            return load_blank_marks(request, og_img, sq_img, window, dataset, folder, image_name)
        elif request.GET['process'] == 'AI-Assisted Annotation' or request.GET['process'] == "Segment":
            if 'convert' in request.GET:
                return convert_data_marks(request, og_img, sq_img, window, dataset, folder, image_name, markings='poly')
            elif 'mode' in request.GET and request.GET['mode'] == 'brush':
                return load_data_marks(request, og_img, sq_img, window, dataset, folder, image_name, markings='poly', mode='brush')
            else:
                return load_data_marks(request, og_img, sq_img, window, dataset, folder, image_name, markings='poly')
        else:
            predict_unet(og_img, sq_img, window, dataset, folder, image_name, model_name=sample_model_unet)
        # result_name ='{}'.format(image_name) 
        #     # result_name ='{}'.format(image_name)
        #     # dataset = "unet_annuli"
        # image_name = result_name
        result_name = image_name
    else:
        skimage.io.imsave("autolith/static/detail/{}".format(image_name), sq_img)
        try:
            os.makedirs("autolith/static/data/{}/{}".format(dataset, folder))
        except:
            pass
        skimage.io.imsave("autolith/static/data/{}/{}/{}".format(dataset, folder, image_name), sq_img)

    return render(request, 'demo/image_detail.html', {'dataset': dataset, 'folder': folder, 'image_name': image_name, 'sample_model_unet': sample_model_unet, 'sample_model_mrcnn': sample_model_mrcnn})

#------------------


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

#---------------------------------------------------------------- [AUX] ------------------------------------------


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


    use_nucleus = True
    all_polar_images = []
    all_original_images = []
    all_contours = []
    all_centers = []
    all_max_angles = []
    all_reading_coords = []
    for idx, ff in enumerate(test_files):
        print(idx)

        try:
            wh_cts = isolate_contours(ff)
        except:
            wh_cts = None
        #     continue
        if wh_cts is None:
            # wh_cts = prev_cts
            wh_cts = []

        with open("autolith/static/extra_json.json") as fin:
            extra_json = json.load(fin)

            
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
        main_json['_via_settings']['project']['name'] = sep_name


        # prev_cts = wh_cts
        

    with open("{}/{}/0/whole_proj.json".format(dataset, folder), "w") as fout:
        json.dump(main_json, fout, indent=4)


def load_annotations(request, dataset, folder):
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

    return render(request, 'otoliths/researchers.html', {'pass_var': json_var, 'fname': 'test', 'mode': 'brush'})



def scale_data_from_contours(request, dataset, folder):
    train_path = folder.replace("data_", "train_")
    try:
        os.makedirs("{}/{}/0".format(dataset, train_path))
    except:
        pass
    try:
        os.makedirs("{}/images".format(dataset))
    except:
        pass
    try:
        os.makedirs("{}/annotations/0".format(dataset))
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
    with open("{}/annotations/0/whole_{}_import.json".format(dataset, folder.replace("data_", "", 1)), "w") as fout:
        json.dump(main_json["_via_img_metadata"], fout, indent=4)

@csrf_exempt
def ensemble_sets(request, dataset, subset):

    if request.method == 'POST':
        print(request.POST)
        split_name = request.POST['split_name']
        replicate_id = int(split_name.split('replicate')[-1])
        print(replicate_id)
        if subset == 'north':
            ensemble_train_and_predict_north(subset, replicate_id)
        else:
            ensemble_train_and_predict_baltic(subset, replicate_id)
        folder = "results"
        fname = "{}_combined.txt".format(subset)
        return redirect("/demo/results/"+
                        "?dataset={}".format(dataset) +
                        "&source=datasets_{}".format(subset) +
                        "&folder={}".format(folder) +
                        "&fname={}".format(fname))

    all_split_name = []
    all_folders = []
    all_split_sets = {}
    all_dir_paths = glob.glob('datasets_ensemble/unet_{}_*/*.txt'.format(subset))
    for dir_path in all_dir_paths:
        strs = dir_path.replace("\\", "/").split("/")
        path = strs[-1]
        raw_split_name = path.split("_")[1].split("based")[0]
        print(raw_split_name)
        split_name = raw_split_name
        if 'normal' in split_name or 'retain' in split_name:
            continue
        if 'comb' in raw_split_name:
            split_id = raw_split_name.split('comb')[0].replace('ex', '')
            split_name = 'all-folds-combined replicate{}'.format(split_id)
        if 'kfold' in raw_split_name:
            continue
            # split_id = raw_split_name.split('kfold')[0].replace('ex', '')
            # split_name = '3-fold replicate{}'.format(split_id)
        if split_name in all_split_sets:
            continue
        all_split_name.append(split_name)
        all_split_sets[split_name] = 1
        all_folders.append([split_name, True, True, False, False])

    form = EnsembleFilterForm(initial={'subset': subset}, request=request, choices=['all'] + sorted(list(set(all_split_name))) ) 

    return render(request, 'demo/ensemble_sets.html', {'dataset': dataset, 'folders': all_folders, 'form': form})


def ensemble_train_and_predict_baltic(domain, base_rep):
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVR
    import glob
    import numpy as np
    import pandas as pd

    mrcnn_none_data = []
    mrcnn_coco_data = []

    if domain == 'north':
        pass
    else:
        mrcnn_north_data = []

    unet_none_data = []
    unet_vgg_data = []

    if domain == 'north':
        pass
    else:
        unet_north_data = []

    manual_readings_on_training_data = []

    all_files = glob.glob("datasets_ensemble/mrcnn_{}_exfold/mrcnn_ex{}*coco*.txt".format(domain, base_rep))
    all_files = sorted(all_files)
    for ff in all_files:
        with open(ff) as fin:
            lines = fin.readlines()
        for line in lines:
            strs = line.strip().split(",")
            
            ai_reading = round(float(strs[-2]))
            mrcnn_coco_data.append(ai_reading)

            manual_reading = round(float(strs[-1]))
            manual_readings_on_training_data.append(manual_reading)
    


    all_files = glob.glob("datasets_ensemble/mrcnn_{}_exfold/mrcnn_ex{}*none*.txt".format(domain, base_rep))
    all_files = sorted(all_files)
    for ff in all_files:
        with open(ff) as fin:
            lines = fin.readlines()
        for line in lines:
            strs = line.strip().split(",")
            ai_reading = round(float(strs[-2]))
            mrcnn_none_data.append(ai_reading)


    if domain == 'north':
        all_files = glob.glob("datasets_ensemble/mrcnn_{}_exfold/mrcnn_{}ex*baltic*.txt".format(domain, base_rep))
    else:
        all_files = glob.glob("datasets_ensemble/mrcnn_{}_exfold/mrcnn_ex{}*north*.txt".format(domain, base_rep))
    all_files = sorted(all_files)
    for ff in all_files:
        with open(ff) as fin:
            lines = fin.readlines()
        for line in lines:
            strs = line.strip().split(",")
            ai_reading = round(float(strs[-2]))
            mrcnn_north_data.append(ai_reading)

    all_files = glob.glob("datasets_ensemble/unet_{}_exfold/unet_{}ex*none*.txt".format(domain, base_rep))
    all_files = sorted(all_files)
    for ff in all_files:
        with open(ff) as fin:
            lines = fin.readlines()
        for line in lines:
            strs = line.strip().split(",")
            ai_reading = round(float(strs[-2]))
            unet_none_data.append(ai_reading)

    all_files = glob.glob("datasets_ensemble/unet_{}_exfold/unet_{}ex*vgg*.txt".format(domain, base_rep))
    all_files = sorted(all_files)
    for ff in all_files:
        with open(ff) as fin:
            lines = fin.readlines()
        for line in lines:
            strs = line.strip().split(",")
            ai_reading = round(float(strs[-2]))
            unet_vgg_data.append(ai_reading)

    if domain == 'north':
        pass
    else:
        all_files = glob.glob("datasets_ensemble/unet_{}_exfold/unet_{}ex*north*.txt".format(domain, base_rep))
    all_files = sorted(all_files)
    for ff in all_files:
        with open(ff) as fin:
            lines = fin.readlines()
        for line in lines:
            strs = line.strip().split(",")
            ai_reading = round(float(strs[-2]))
            unet_north_data.append(ai_reading) 
            

    manual_readings_on_training_data = np.array(manual_readings_on_training_data)

    ai_training_data = pd.DataFrame()
    ai_training_data['mrcnn_none'] = np.array(mrcnn_none_data)
    ai_training_data['mrcnn_coco'] = np.array(mrcnn_coco_data)

    if domain == 'north':
        pass
    else:
        ai_training_data['mrcnn_north'] = np.array(mrcnn_north_data)

    ai_training_data['unet_none'] = np.array(unet_none_data)
    ai_training_data['unet_vgg'] = np.array(unet_vgg_data)

    if domain == 'north':
        pass
    else:
        ai_training_data['unet_north'] = np.array(unet_north_data)


    linear_model = LinearRegression()
    linear_model.fit(ai_training_data, manual_readings_on_training_data)

    rf_model = RandomForestClassifier(max_depth=5, random_state=101)
    rf_model.fit(ai_training_data, manual_readings_on_training_data)

    #-----------------------------------------------------------------------------------------------------------------

    mrcnn_none_data = []
    mrcnn_coco_data = []
    mrcnn_north_data = []

    unet_none_data = []
    unet_vgg_data = []
    unet_north_data = []

    names_test_data = []
    manual_readings_on_test_data = []

    all_files = glob.glob("datasets_ensemble/mrcnn_{}_combined/mrcnn_ex{}*coco*.txt".format(domain, base_rep))
    
    for ff in all_files:
        with open(ff) as fin:
            lines = fin.readlines()
        for line in lines:
            strs = line.strip().split(",")

            ai_reading = round(float(strs[-2]))
            mrcnn_coco_data.append(ai_reading)

            manual_reading = round(float(strs[-1]))
            manual_readings_on_test_data.append(manual_reading)

            names_test_data.append(strs[0])
    

    all_files = glob.glob("datasets_ensemble/mrcnn_{}_combined/mrcnn_ex{}*none*.txt".format(domain, base_rep))
    for ff in all_files:
        with open(ff) as fin:
            lines = fin.readlines()
        for line in lines:
            strs = line.strip().split(",")
            ai_reading = round(float(strs[-2]))
            mrcnn_none_data.append(ai_reading)

    all_files = glob.glob("datasets_ensemble/mrcnn_{}_combined/mrcnn_ex{}*north*.txt".format(domain, base_rep))
    for ff in all_files:
        with open(ff) as fin:
            lines = fin.readlines()
        for line in lines:
            strs = line.strip().split(",")
            ai_reading = round(float(strs[-2]))
            mrcnn_north_data.append(ai_reading)

    all_files = glob.glob("datasets_ensemble/unet_{}_combined/unet_ex{}*none*.txt".format(domain, base_rep))
    for ff in all_files:
        with open(ff) as fin:
            lines = fin.readlines()
        for line in lines:
            strs = line.strip().split(",")
            ai_reading = round(float(strs[-2]))
            unet_none_data.append(ai_reading)

    all_files = glob.glob("datasets_ensemble/unet_{}_combined/unet_ex{}*vgg*.txt".format(domain, base_rep))
    for ff in all_files:
        with open(ff) as fin:
            lines = fin.readlines()
        for line in lines:
            strs = line.strip().split(",")
            ai_reading = round(float(strs[-2]))
            unet_vgg_data.append(ai_reading)

    all_files = glob.glob("datasets_ensemble/unet_{}_combined/unet_ex{}*north*.txt".format(domain, base_rep))
    for ff in all_files:
        with open(ff) as fin:
            lines = fin.readlines()
        for line in lines:
            strs = line.strip().split(",")
            ai_reading = round(float(strs[-2]))
            unet_north_data.append(ai_reading)
            

    manual_readings_on_test_data = np.array(manual_readings_on_test_data)

    ai_test_data = pd.DataFrame()
    ai_test_data['mrcnn_none'] = np.array(mrcnn_none_data)
    ai_test_data['mrcnn_coco'] = np.array(mrcnn_coco_data)
    ai_test_data['mrcnn_north'] = np.array(mrcnn_north_data)
    ai_test_data['unet_none'] = np.array(unet_none_data)
    ai_test_data['unet_vgg'] = np.array(unet_vgg_data)
    ai_test_data['unet_north'] = np.array(unet_north_data)



    all_baltic_acc = []
    all_results = []
    rf_pred = rf_model.predict(ai_test_data)
    linear_pred = linear_model.predict(ai_test_data)
    mean_pred = ai_test_data.mean(axis=1)

    rf_correct = 0
    lin_correct = 0
    mean_correct = 0
    for idx, actual in enumerate(manual_readings_on_test_data):
        fname = names_test_data[idx]
        diff = actual - round(rf_pred[idx])
        if diff == 0:
            rf_correct += 1
        diff = actual - round(linear_pred[idx])
        if diff == 0:
            lin_correct += 1
        diff = actual - round(mean_pred[idx])
        if diff == 0:
            mean_correct += 1

        all_results.append("{},{},{},{},{}".format(fname, round(rf_pred[idx]), round(linear_pred[idx]), round(mean_pred[idx]), actual))

    print(rf_correct)
    print(lin_correct)
    print(mean_correct)
    all_baltic_acc.append(100*rf_correct/len(manual_readings_on_test_data))
    all_baltic_acc.append(100*lin_correct/len(manual_readings_on_test_data))
    all_baltic_acc.append(100*mean_correct/len(manual_readings_on_test_data))

    results_path = "datasets_ensemble/results/"
    try:
        os.makedirs(results_path)
    except:
        pass

    with open("{}/{}_combined.txt".format(results_path, domain), "w") as fout:
        fout.write("\n".join(all_results))

    return rf_correct, lin_correct, mean_correct

def ensemble_train_and_predict_north(domain, base_rep):
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVR
    import glob
    import numpy as np
    import pandas as pd
    # import random

    mrcnn_none_data = []
    mrcnn_coco_data = []
    mrcnn_baltic_data = []

    unet_none_data = []
    unet_vgg_data = []
    unet_baltic_data = []

    manual_readings_on_training_data = []
    all_files = glob.glob("datasets_ensemble/mrcnn_north_exfold/mrcnn_{}*baltic*.txt".format(base_rep))
    all_files = sorted(all_files)
    for ff in all_files:
        with open(ff) as fin:
            lines = fin.readlines()
        for line in lines:
            strs = line.strip().split(",")

            ai_reading = round(float(strs[-2]))
            mrcnn_baltic_data.append(ai_reading)

            manual_reading = round(float(strs[-1]))
            manual_readings_on_training_data.append(manual_reading)
    
    all_files = glob.glob("datasets_ensemble/mrcnn_north_exfold/mrcnn_{}*none*.txt".format(base_rep))
    all_files = sorted(all_files)
    for ff in all_files:
        with open(ff) as fin:
            lines = fin.readlines()
        for line in lines:
            strs = line.strip().split(",")
            ai_reading = round(float(strs[-2]))
            mrcnn_none_data.append(ai_reading)

    all_files = glob.glob("datasets_ensemble/mrcnn_north_exfold/mrcnn_{}*coco*.txt".format(base_rep))
    all_files = sorted(all_files)
    for ff in all_files:
        with open(ff) as fin:
            lines = fin.readlines()
        for line in lines:
            strs = line.strip().split(",")
            ai_reading = round(float(strs[-2]))
            mrcnn_coco_data.append(ai_reading)
            
            
    all_files = glob.glob("datasets_ensemble/unet_north_exfold/unet_{}*none*.txt".format(base_rep))
    all_files = sorted(all_files)
    for ff in all_files:
        with open(ff) as fin:
            lines = fin.readlines()
        for line in lines:
            strs = line.strip().split(",")
            ai_reading = round(float(strs[-2]))
            unet_none_data.append(ai_reading)

    all_files = glob.glob("datasets_ensemble/unet_north_exfold/unet_{}*vgg*.txt".format(base_rep))
    all_files = sorted(all_files)
    for ff in all_files:
        with open(ff) as fin:
            lines = fin.readlines()
        for line in lines:
            strs = line.strip().split(",")
            ai_reading = round(float(strs[-2]))
            unet_vgg_data.append(ai_reading)

    all_files = glob.glob("datasets_ensemble/unet_north_exfold/unet_{}*baltic*.txt".format(base_rep))
    all_files = sorted(all_files)
    for ff in all_files:
        with open(ff) as fin:
            lines = fin.readlines()
        for line in lines:
            strs = line.strip().split(",")
            ai_reading = round(float(strs[-2]))
            unet_baltic_data.append(ai_reading) 
            

    manual_readings_on_training_data = np.array(manual_readings_on_training_data)
    ai_training_data = pd.DataFrame()
    ai_training_data['mrcnn_none'] = np.array(mrcnn_none_data)
    ai_training_data['mrcnn_coco'] = np.array(mrcnn_coco_data)
    ai_training_data['mrcnn_baltic'] = np.array(mrcnn_baltic_data)
    ai_training_data['unet_none'] = np.array(unet_none_data)
    ai_training_data['unet_vgg'] = np.array(unet_vgg_data)
    ai_training_data['unet_baltic'] = np.array(unet_baltic_data)

    linear_model = LinearRegression()
    linear_model.fit(ai_training_data, manual_readings_on_training_data)

    rf_model = RandomForestClassifier(max_depth=5, random_state=101)
    rf_model.fit(ai_training_data, manual_readings_on_training_data)

    names_test_data= []
    manual_readings_on_test_data = []

    mrcnn_none_data = []
    mrcnn_coco_data = []
    mrcnn_baltic_data = []

    unet_none_data = []
    unet_vgg_data = []
    unet_baltic_data = []


    all_files = glob.glob("datasets_ensemble/mrcnn_north_combined/mrcnn_{}*baltic*.txt".format(base_rep))
    for ff in all_files:
        with open(ff) as fin:
            lines = fin.readlines()
        for line in lines:
            strs = line.strip().split(",")

            ai_reading = round(float(strs[-2]))
            mrcnn_baltic_data.append(ai_reading)

            manual_reading = round(float(strs[-1]))
            manual_readings_on_test_data.append(manual_reading)
            names_test_data.append(strs[0])

    all_files = glob.glob("datasets_ensemble/mrcnn_north_combined/mrcnn_{}*none*.txt".format(base_rep))
    for ff in all_files:
        with open(ff) as fin:
            lines = fin.readlines()
        for line in lines:
            strs = line.strip().split(",")
            ai_reading = round(float(strs[-2]))
            mrcnn_none_data.append(ai_reading)

    all_files = glob.glob("datasets_ensemble/mrcnn_north_combined/mrcnn_{}*coco*.txt".format(base_rep))
    for ff in all_files:
        with open(ff) as fin:
            lines = fin.readlines()
        for line in lines:
            strs = line.strip().split(",")
            ai_reading = round(float(strs[-2]))
            mrcnn_coco_data.append(ai_reading)
              
            
    all_files = glob.glob("datasets_ensemble/unet_north_combined/unet_{}*none*.txt".format(base_rep))
    for ff in all_files:
        with open(ff) as fin:
            lines = fin.readlines()
        for line in lines:
            strs = line.strip().split(",")
            ai_reading = round(float(strs[-2]))
            unet_none_data.append(ai_reading)

    all_files = glob.glob("datasets_ensemble/unet_north_combined/unet_{}*vgg*.txt".format(base_rep))
    for ff in all_files:
        with open(ff) as fin:
            lines = fin.readlines()
        for line in lines:
            strs = line.strip().split(",")
            ai_reading = round(float(strs[-2]))
            unet_vgg_data.append(ai_reading)

    all_files = glob.glob("datasets_ensemble/unet_north_combined/unet_{}*baltic*.txt".format(base_rep))
    for ff in all_files:
        with open(ff) as fin:
            lines = fin.readlines()
        for line in lines:
            strs = line.strip().split(",")
            ai_reading = round(float(strs[-2]))
            unet_baltic_data.append(ai_reading)
            

    manual_readings_on_test_data = np.array(manual_readings_on_test_data)
    ai_test_data = pd.DataFrame()
    ai_test_data['mrcnn_none'] = np.array(mrcnn_none_data)
    ai_test_data['mrcnn_coco'] = np.array(mrcnn_coco_data)
    ai_test_data['mrcnn_baltic'] = np.array(mrcnn_baltic_data)
    ai_test_data['unet_none'] = np.array(unet_none_data)
    ai_test_data['unet_vgg'] = np.array(unet_vgg_data)
    ai_test_data['unet_baltic'] = np.array(unet_baltic_data)



    all_north_acc = []
    all_results = []
    rf_pred = rf_model.predict(ai_test_data)
    linear_pred = linear_model.predict(ai_test_data)
    mean_pred = ai_test_data.mean(axis=1)

    rf_correct = 0
    lin_correct = 0
    mean_correct = 0
    for idx, actual in enumerate(manual_readings_on_test_data):
        fname = names_test_data[idx]
        diff = actual - round(rf_pred[idx])
        if diff == 0:
            rf_correct += 1
        diff = actual - round(linear_pred[idx])
        if diff == 0:
            lin_correct += 1
        diff = actual - round(mean_pred[idx])
        if diff == 0:
            mean_correct += 1

        all_results.append("{},{},{},{},{}".format(fname, round(rf_pred[idx]), round(linear_pred[idx]), round(mean_pred[idx]), actual))

    print(rf_correct)
    print(lin_correct)
    print(mean_correct)
    all_north_acc.append(100*rf_correct/len(manual_readings_on_test_data))
    all_north_acc.append(100*lin_correct/len(manual_readings_on_test_data))
    all_north_acc.append(100*mean_correct/len(manual_readings_on_test_data))

    results_path = "datasets_ensemble/results/"
    try:
        os.makedirs(results_path)
    except:
        pass

    with open("{}/{}_combined.txt".format(results_path, domain), "w") as fout:
        fout.write("\n".join(all_results))

    return rf_correct, lin_correct, mean_correct


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
    
    return render(request, 'demo/results_table.html', {'source': source, 'results_list': results_list, 'correct': correct, 'total': total, 'accuracy': 100*correct/total if total else 0})



def modify_image_brighten(img_new, mask_new, dataset):
    image_res = img_as_ubyte(img_new.copy())
    if dataset == 'datasets_north':
        img_new = cv2.convertScaleAbs(img_as_ubyte(image_res), alpha=1.5, beta=10)
    else:
        img_new = cv2.convertScaleAbs(img_as_ubyte(image_res), alpha=2, beta=20)
    return img_new

def modify_image_bgremove(img_new, mask_new, dataset):
    nwh_thresh = np.zeros([mask_new.shape[0], mask_new.shape[1],1], dtype=np.uint8)
    nwh_thresh = mask_new[:,:,0]
    nwh_contours, h = cv2.findContours(nwh_thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    nwh_cts = max(nwh_contours, key=cv2.contourArea)

    test_mask = np.ones(mask_new.shape[:2], dtype="uint8") * 255
    cv2.drawContours(test_mask, [nwh_cts], -1, 0, -1)

    mapFg = cv2.erode(test_mask, np.ones( (5,5), np.uint8), iterations=10)
    test_img_mask = cv2.bitwise_not(mapFg)

    imagec = img_as_ubyte(img_new.copy())
    image_res = cv2.bitwise_and(imagec, imagec, mask=test_img_mask)

    return image_res

def modify_image_both(img_new, mask_new, dataset):
    nwh_thresh = np.zeros([mask_new.shape[0], mask_new.shape[1],1], dtype=np.uint8)
    nwh_thresh = mask_new[:,:,0]
    nwh_contours, h = cv2.findContours(nwh_thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    nwh_cts = max(nwh_contours, key=cv2.contourArea)

    test_mask = np.ones(mask_new.shape[:2], dtype="uint8") * 255
    cv2.drawContours(test_mask, [nwh_cts], -1, 0, -1)

    mapFg = cv2.erode(test_mask, np.ones( (5,5), np.uint8), iterations=10)
    test_img_mask = cv2.bitwise_not(mapFg)

    imagec = img_as_ubyte(img_new.copy())
    image_res = cv2.bitwise_and(imagec, imagec, mask=test_img_mask)

    if dataset == 'datasets_north':
        img_new = cv2.convertScaleAbs(img_as_ubyte(image_res), alpha=1.5, beta=10)
    else:
        img_new = cv2.convertScaleAbs(img_as_ubyte(image_res), alpha=2, beta=20)
    return img_new
