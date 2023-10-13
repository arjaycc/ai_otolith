import os
import numpy as np
import cv2
import glob
import re
import shutil


def get_results_list(dataset_filter='datasets_north', algo_filter='None', run_filter='None', mode_filter='None'):
    if dataset_filter is None:
        dataset_filter = '*'
    if algo_filter is None:
        algo_filter = '*'
    if run_filter is None:
        run_filter = '*'
    if mode_filter is None:
        mode_filter = '*'
    files = glob.glob("{}/{}_{}/{}".format(dataset_filter, algo_filter, run_filter, mode_filter))
    run_files = {}
    for ff in sorted(files):
        strs = ff.replace("\\", "/").split("/")
        key = '{}:{}'.format(strs[-2], mode_filter)
        if key in run_files:
            run_files[key].append(ff)
        else:
            run_files[key] = [ff]
    
    run_results = []
    for k,v in run_files.items():
        v.sort(key=os.path.getctime)
        run_results.append([k, v[-1]])
    return run_results
        

def write_results(run_results, label, total_filter=None, idx_max=9999):
    import os
    import shutil
    all_rr = []
    exact = []
    oneoff = []
    try:
        os.makedirs('all_results/{}/'.format(label))
    except:
        pass
    for idx,item in enumerate(run_results):
        strs = item[1].split('_of_')
        left_strs = strs[0].split('_')
        exact_val = left_strs[-2]
        oneoff_val = left_strs[-1]
        total_val = strs[1].split('_')[0].split('.')[0]
        
        if total_filter is not None:
            if total_val != total_filter:
                continue
        print(item[0], item[1])
        shutil.copyfile(item[1], 'all_results/{}/{}_{}'.format(label, item[0].split(":")[0],item[1].replace("\\", "/").split("/")[-1] ))
        exact.append(exact_val)
        oneoff.append(oneoff_val)

    lines = []
    lines.append("{}_exact = c({})".format(label, ','.join(exact)))
    lines.append("{}_oneoff = c({})".format(label, ','.join(oneoff)))
    return lines


all_lines = []
algo_list = ['cnnreg', 'mrcnn', 'unet']
norm_modes = ['cnn', 'mrcnn_rd', 'unet_rd']
br_modes = ['brighten', 'mrcnn_br', 'unet_br']
north_run_flags = ['run1', 'run1_6', 'run1_47']
baltic_run_flags = ['run1', 'run1_2', 'run1_37']


all_lines.append("##### --------- NORTH RS fold")
for i in range(3):
    algo_name = algo_list[i]
    mode = norm_modes[i]
    run_name = 'randsub*{}'.format(north_run_flags[i])
    algo_rr = get_results_list(dataset_filter='datasets_north', algo_filter=algo_name, run_filter='{}'.format(run_name), mode_filter='{}*_of_{}*.txt'.format(mode, 528))
    all_lines.extend(write_results(algo_rr, '{}_rsfold'.format(algo_name)))    


all_lines.append("##### --------- BALTIC RS Fold ")
for i in range(3):
    algo_name = algo_list[i]
    mode = norm_modes[i]
    run_name = 'randfold*{}'.format(baltic_run_flags[i])
    algo_rr = get_results_list(dataset_filter='datasets_baltic', algo_filter=algo_name, run_filter='{}'.format(run_name), mode_filter='{}*_of_{}*.txt'.format(mode, 1005))
    all_lines.extend(write_results(algo_rr, '{}_baltic_rsfold'.format(algo_name)))   
    

all_lines.append("##### --------- SAITHE train, N-COD test ")
for i in range(3):
    algo_name = algo_list[i]
    mode = norm_modes[i]
    run_name = 'saithe*{}'.format(north_run_flags[i])
    algo_rr = get_results_list(dataset_filter='datasets_north', algo_filter=algo_name, run_filter='{}'.format(run_name), mode_filter='{}*_of_{}*.txt'.format(mode, 194))
    all_lines.extend(write_results(algo_rr, '{}_saithe_on_ncod_test'.format(algo_name)) )   
    
all_lines.append("##### --------- SAITHE train, B-COD test ")
for i in range(3):
    algo_name = algo_list[i]
    mode = norm_modes[i]
    run_name = 'saithe*{}'.format(north_run_flags[i])
    algo_rr = get_results_list(dataset_filter='datasets_baltic', algo_filter=algo_name, run_filter='{}'.format(run_name), mode_filter='{}*_of_{}*.txt'.format(mode, 1155))
    all_lines.extend(write_results(algo_rr, '{}_saithe_on_bcod_test'.format(algo_name)) )   
    
all_lines.append("##### --------- N-COD on Saithe Test ")
for i in range(3):
    algo_name = algo_list[i]
    mode = norm_modes[i]
    run_name = 'cod*{}'.format(north_run_flags[i])
    algo_rr = get_results_list(dataset_filter='datasets_north', algo_filter=algo_name, run_filter='{}'.format(run_name), mode_filter='{}*_of_{}*.txt'.format(mode, 351))
    all_lines.extend(write_results(algo_rr, '{}_ncod_on_saithe_test'.format(algo_name)) )    
    
all_lines.append("##### --------- N-COD on B-COD test ")
for i in range(3):
    algo_name = algo_list[i]
    mode = norm_modes[i]
    run_name = 'cod*{}'.format(north_run_flags[i])
    algo_rr = get_results_list(dataset_filter='datasets_baltic', algo_filter=algo_name, run_filter='{}'.format(run_name), mode_filter='{}*_of_{}*.txt'.format(mode, 1155))
    all_lines.extend(write_results(algo_rr, '{}_ncod_on_bcod_test'.format(algo_name)) )    

all_lines.append("##### --------- NORTH age 7 lim")
for i in range(3):
    algo_name = algo_list[i]
    mode = norm_modes[i]
    run_name = 'age7lim*{}'.format(north_run_flags[i])
    algo_rr = get_results_list(dataset_filter='datasets_north', algo_filter=algo_name, run_filter='{}'.format(run_name), mode_filter='{}*_of_{}*.txt'.format(mode, 188))
    all_lines.extend(write_results(algo_rr, '{}_age7lim'.format(algo_name)))


all_lines.append("##### --------- NORTH RS FOLD brighten")
for i in range(3):
    algo_name = algo_list[i]
    mode = br_modes[i] #'brighten'
    run_name = 'randsub*{}'.format(north_run_flags[i])
    algo_rr = get_results_list(dataset_filter='datasets_north', algo_filter=algo_name, run_filter='{}'.format(run_name), mode_filter='{}*_of_{}*.txt'.format(mode, 528))
    all_lines.extend(write_results(algo_rr, '{}_rs_brighten'.format(algo_name)))    

    
all_lines.append("##### --------- BALTIC RS FOLD brighten ")
for i in range(3):
    algo_name = algo_list[i]
    mode = br_modes[i] #'brighten'
    run_name = 'randfold*{}'.format(baltic_run_flags[i])
    algo_rr = get_results_list(dataset_filter='datasets_baltic', algo_filter=algo_name, run_filter='{}'.format(run_name), mode_filter='{}*_of_{}*.txt'.format(mode, 1005))
    all_lines.extend(write_results(algo_rr, '{}_baltic_rs_brighten'.format(algo_name)))   
    
    
all_lines.append("##### --------- NORTH on Baltic Test")
for i in range(3):
    algo_name = algo_list[i]
    mode = norm_modes[i]
    run_name = 'randsub*{}'.format(north_run_flags[i])
    algo_rr = get_results_list(dataset_filter='datasets_baltic', algo_filter=algo_name, run_filter='{}'.format(run_name), mode_filter='{}*_of_{}*.txt'.format(mode, 1155))
    all_lines.extend(write_results(algo_rr, '{}_north_on_baltic_test'.format(algo_name)))   
    
    
all_lines.append("##### --------- BALTIC age 4 lim")
for i in range(3):
    algo_name = algo_list[i]
    mode = norm_modes[i]
    run_name = 'age4lim*{}'.format(baltic_run_flags[i])
    algo_rr = get_results_list(dataset_filter='datasets_baltic', algo_filter=algo_name, run_filter='{}'.format(run_name), mode_filter='{}*_of_{}*.txt'.format(mode, 42))
    all_lines.extend(write_results(algo_rr, '{}_age4lim'.format(algo_name)))

    
all_lines.append("##### --------- BALTIC on NORTH Test (INTERCHANGE)")
for i in range(3):
    algo_name = algo_list[i]
    mode = norm_modes[i]
    run_name = 'randfold*{}'.format(baltic_run_flags[i])
    algo_rr = get_results_list(dataset_filter='datasets_north', algo_filter=algo_name, run_filter='{}'.format(run_name), mode_filter='{}*_of_{}*.txt'.format(mode, 660))
    all_lines.extend(write_results(algo_rr, '{}_baltic_on_north_test'.format(algo_name)))   


all_lines.append("##### --------- ImgProc NORTH RS Fold")
algo_name = 'imgproc'
run_name = 'randsub*run1'
mode = 'reading'
algo_rr = get_results_list(dataset_filter='datasets_north', algo_filter=algo_name, run_filter='{}'.format(run_name), mode_filter='{}*_{}*.txt'.format(mode, 528))
all_lines.extend(write_results(algo_rr, '{}_rsfold'.format(algo_name)))
    
    
all_lines.append("##### --------- ImgProc BALTIC RS Fold")
algo_name = 'imgproc'
run_name = 'randfold*run1'
mode = 'reading'
algo_rr = get_results_list(dataset_filter='datasets_baltic', algo_filter=algo_name, run_filter='{}'.format(run_name), mode_filter='{}*_{}*.txt'.format(mode, 1005))
all_lines.extend(write_results(algo_rr, '{}_baltic_rsfold'.format(algo_name)))

all_lines.append("##### --------- RANDOM")
for i in range(3):
    algo_name = algo_list[i]
    mode = norm_modes[i]
    run_name = 'testrand*{}'.format(north_run_flags[i])
    algo_rr = get_results_list(dataset_filter='datasets_north', algo_filter=algo_name, run_filter='{}'.format(run_name), mode_filter='{}*_of_{}*.txt'.format(mode, 528))
    all_lines.extend(write_results(algo_rr, '{}_rsfold'.format(algo_name)))    


with open("summary_results.R", "w") as fout:
    fout.write('\n'.join(all_lines))
    
    
    




