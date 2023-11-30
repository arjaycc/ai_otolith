from django.shortcuts import render
import os
import json



def start_run_unet(settings):
    import unet.unet_main as unet_setup
    print(settings) 
    set_num = settings['idr']
    if set_num < 0:
        idr = 999
    else:
        idr = set_num
    search_mode = settings['search_mode']
    selected = settings['selected']

    # TEMP
    all_unet_params = [(0,0,True), (10,5, True), (40,20, True), (10,5, False), (40,20, False)]
    all_data_params = [(1,False, False), (3,False, False), (3, False, True), (1,True, False), (3,True, False), (3, True, True)]
    all_train_params = [False, True]
    param_count = 0
    for u_params in all_unet_params:
        for d_params in all_data_params:
            for t_params in all_train_params:
                w0, sigma, edge = u_params
                ch, aug, tr = d_params
                fr = t_params
                if edge:
                    if w0 == 0:
                        func = 'edge'
                        w0 = 10
                        sigma = 5
                    else:
                        func = 'both'
                else:
                    func = 'weighted'
                unet_params = {
                    'w0': w0,
                    'sigma': sigma,
                    'loss_function':func
                }
                data_params = {
                    'channels': ch,
                    'normalize': True,
                    'augment': aug,
                    'transfer': tr
                }
                train_params = {
                    'full_ring': fr
                }
                
                full_params = {}
                full_params.update(unet_params)
                full_params.update(data_params)
                full_params.update(train_params)
                
                param_count += 1
                if len(selected) > 0:
                    if param_count not in selected:
                        continue

                if search_mode:
                    run_label = "param"
                else:
                    run_label = "{}".format(settings['run_label'])

                name = 'unet_{}{}run1_{}'.format(run_label, idr, param_count)

                print(">>>>>>>>>>> PROCESSING {} <<<<<<<<".format(name))
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

                if settings['run_type'] == 'train' or settings['run_type'] == 'both':
                    unet_setup.train(name=name, data_params=data_params, unet_params=unet_params, train_params=train_params, settings=settings)

                if settings['run_type'] == 'test' or settings['run_type'] == 'both':
                    unet_setup.evaluate(name=name, full_ring_type=fr, data_params=data_params, settings=settings)


def run_unet(request):

    if 'dataset' in request:
        settings = request
    else:
        settings = request.GET

    start_run_unet(settings)
    
    images = [] 
    if 'dataset' in request:
        return
    return render(request, 'otoliths/experiments.html', {'images': images })

