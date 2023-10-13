from otoliths.experiments_imgproc import *
from otoliths.experiments_mrcnn import *
from otoliths.experiments_unet import *

# IMPORTANT!
# Make sure to first run: python pre_process.py 
# before running this code to prepare the training+validation data

fold_list = [0,1,2,3,4]
for fold in fold_list:

    settings = {
        'dataset' : 'datasets_north',
        'run_type': 'both',                  # train, test or both
        'run_label': 'randsub',              # name of the run
        'search_mode': False,                # whether to search for optimal hyper-parameters
        'idr': fold,                         # fold id 
        'selected': [9],                    # hyper-parameter id list (i.d. of optimal parameter set)
        'split_type': 'rs',                  # (cv)5-fold cv, (rs)5-random subsampling
        'split_name': 'randsub',             # label of prepared folders (e.g. train_randsub_0, valid_randsub_0)
        'checkpoint': '_checkpoint',         # whether to use model saved during checkpoint or at the end (use empty '')
        #'target_species': 'saithe',         # filter test images by species
        #'age_limit': 7,                     # whether to create age limit for training and testing (7 for North, 4 for Baltic) 
        #'brighten': True,                   # whether to brighten the test images
        #'source_dataset': 'datasets_north'  # if the model is trained in one source dataset but to be tested in another set
    }
    
    # For the traditional algorithm involving signal processing (comment out the next line to skip)
    run_imgproc(settings) 
    
    # For running M-RCNN algorithm (comment out the next line to skip)
    run_mrcnn(settings) 
    
    # For running U-Net algorithm (comment out the next line to skip)
    run_unet(settings) 
    
# IMPORTANT!
# The 'selected' parameter id (e.g. the val 9 specified in example) used in this study are as follows:
#          NORTH 
# U-Net param_id selected == 47
# M-RCNN param_id selected == 6
#         BALTIC 
# U-Net param_id selected == 37
# M-RCNN param_id selected == 2
    
# IMPORTANT!
# the prediction results are inside the txt files generated inside run folders per algorithm 
# for example, a UNet run involving training set with split_name='randsub' on fold(idr)=0 and param_id(selected)=47) will
# have a folder named unet_randsub0run1_47

# To efficiently collect the results without going to the run folders,
# Run: python post_process.py
# and check the folder 'all_results'