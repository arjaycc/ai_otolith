from otoliths.experiments_imgproc import *
from otoliths.experiments_mrcnn import *
from otoliths.experiments_unet import *



fold_list = [0,1,2,3,4]
for fold in fold_list:

    settings = {
        'dataset' : 'datasets_north',
        'run_type': 'both',                  # train, test or both
        'run_label': 'randsub',              # name of the run
        'search_mode': False,                # whether to search for optimal hyper-parameters
        'idr': fold,                         # fold id 
        'selected': [47],                    # hyper-parameter id list 
        'split_type': 'rs',                  # (cv)5-fold cv, (rs)5-random subsampling
        'split_name': 'randsub',             # label of prepared folders (e.g. train_randsub_0, valid_randsub_0)
        'checkpoint': '_checkpoint',         # whether to use model saved during checkpoint or at the end (use empty '')
        #'target_species': 'saithe',         # filter test images by species
        #'age_limit': 4,                     # whether to create age limit for training and testing  
        #'brighten': True,                   # whether to brighten the test images
        #'source_dataset': 'datasets_north'
    }
    
    # For the traditional algorithm involving signal processing
    run_imgproc(settings) 
    
    # For running M-RCNN algorithm
    run_mrcnn(settings) 
    
    # For running U-Net algorithm
    run_unet(settings) 

    
