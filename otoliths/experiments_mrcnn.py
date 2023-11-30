from django.shortcuts import render
import os
import json
import numpy as np



def start_run_mrcnn(settings):
    import mrcnn.mrcnn_main as mrcnn_setup
    search_mode = settings['search_mode']
    idr = settings['idr']
    selected = settings['selected']
    
    all_edge_params = [(True, 1.0, ["sobel-y"]), (True, 1.0, ["sobel-x"])] # smoothing, weight, edge
    all_data_params = [(1024, True), (1024, False)]  # size ,min
    all_train_params = [( 0.75, 0.2, 0.85), (0.8, 0.2, 0.7),  ( 0.6, 0.2, 0.8)] # , detection_thresh, detect_nms, rpn_nms
    
    param_count = 0
    for u_params in all_edge_params:
        for d_params in all_data_params:
            for t_params in all_train_params:
                smooth, wfactor, edge_loss = u_params
                
                sz, mini = d_params
                
                det_conf, det_nms, rpn_nms = t_params

                edge_params = {
                    'smoothing': smooth,
                    'weight_factor': wfactor,
                    'edge_loss': edge_loss
                }
                data_params = {
                    'img_size': sz,
                    'mini_mask': mini
                }
                train_params = {
                    'rpn_nms': rpn_nms,
                    'detection_confidence': det_conf,
                    'detection_nms': det_nms
                }
                
                full_params = {}
                full_params.update(edge_params)
                full_params.update(data_params)
                full_params.update(train_params)
                
                param_count += 1
                if not search_mode:
                    if param_count not in selected:
                        continue
                
                if search_mode:
                    mark = "param"
                else:
                    mark = "{}".format(settings['run_label'])
                name = 'mrcnn_{}{}run1_{}'.format(mark, idr, param_count)
                print(">>>>>>>>>>  PROCESSING {} <<<<<<<<".format(name))
                try:
                    os.makedirs('{}/{}'.format(settings['dataset'], name))
                except:
                    pass
                
                with open('{}/{}.json'.format(settings['dataset'], name), 'w') as f:
                    json.dump(full_params,f,sort_keys=True, indent=4)
                try:
                    os.makedirs('{}/{}/pred/'.format(settings['dataset'], name))
                except:
                    pass

                try:
                    os.makedirs('{}/{}/output/'.format(settings['dataset'], name))
                except:
                    pass

                mrcnn_setup.train(
                    name=name, 
                    data_params=data_params, 
                    edge_params=edge_params, 
                    train_params=train_params,
                    settings=settings)

def run_mrcnn(request):

    if 'dataset' in request:
        settings = request
    else:
        settings = request.GET
        
    start_run_mrcnn(settings)
    
    images = [] 
    if 'dataset' in request:
        return
    return render(request, 'otoliths/experiments.html', {'images': images })

