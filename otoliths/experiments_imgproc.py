from django.shortcuts import render
import os
import json


def start_run_imgproc(settings):
    import imgproc.imgproc_main as imgproc_setup
    print(settings) 
    idr = settings['idr']
    run_label = "{}".format(settings['run_label'])
    name = 'imgproc_{}{}run1'.format(run_label, idr)
    try:
        os.makedirs('{}/{}'.format(settings['dataset'], name))
    except:
        pass

#     if settings['run_type'] == 'train' or settings['run_type'] == 'both':
#         imgproc_setup.train(settings=settings)

    if settings['run_type'] == 'test' or settings['run_type'] == 'both':
        imgproc_setup.evaluate(name, settings=settings)


def run_imgproc(request):

    if 'dataset' in request:
        settings = request
    else:
        settings = request.GET
        
    start_run_imgproc(settings)
    
    images = [] 
    if 'dataset' in request:
        return
    return render(request, 'otoliths/experiments.html', {'images': images })

