AI_otolith -> Fish Age Reading using Object Detection and Segmentation


To run the project, one can use either Command-Line or Web-based method (requires installation of the necessary packages and starting the development server).

The following are the summarized steps for running the methods:

1. First, the two datasets must be downloaded from the following repositories:
- North Sea dataset: DOI https://doi.org/10.5281/zenodo.8341092
- Baltic Sea dataset: DOI https://doi.org/10.5281/zenodo.8341149

2. Unzip the the downloaded image zip files into corresponding dataset folder (i.e. datasets_north/ or datasets_baltic/)

3. Run: python pre_process.py

4. Input the desired run configuration by editing the settings dictionary inside main.py (use default values for initial testing)

5. Run: python main.py

6. Inspect the results into the newly created folders in both either datasets_north/ or datasets_baltic/

7. Or skip manual checking of results by running: python post_process.py which will then create a folder named "all_results" containing the txt files summarizing the ai predictions for each image file as well as the expected age (from manual age readers).

In case one does not want to train a new model and just want to test existing models from the study, the following are the needed steps:

1. A set of previously trained models can be downloaded from the following location:
- models.zip: https://doi.org/10.5281/zenodo.10000645

2. Unzip into the corresponding dataset folder (i.e. datasets_north/ or datasets_baltic/)

    2.1 This zip file contains the trained models involving a single train-val-test split (n=1 subsampling replicate numbered accordingly) for both U-Net and Mask R-CNN. Due to the large sizes of the models, the models for other train-val-test splits (n=20 for North Sea, n=4 for Baltic) are not uploaded but are available upon request.

3. To test these models against the images not use during training, edit the main.py and use the following settings:

U-Net with the North Sea Dataset:

    settings = {
        'dataset' : 'datasets_north',
        'run_type': 'test',                  
        'run_label': 'randsub',   
        'search_mode': False,                
        'idr': 0,                  
        'selected': [47],                
        'split_type': 'rs',                
        'split_name': 'randsub',          
        'checkpoint': '_checkpoint'
        }

        
U-Net with the Baltic Sea Dataset:

    settings = {
        'dataset' : 'datasets_baltic',
        'run_type': 'test',                  
        'run_label': 'randfold',   
        'search_mode': False,                
        'idr': 0,                  
        'selected': [47],                
        'split_type': 'rs',                
        'split_name': 'fold',          
        'checkpoint': '_checkpoint'
        }

Mask R-CNN with the North Sea Dataset:

    settings = {
        'dataset' : 'datasets_north',
        'run_type': 'test',                  
        'run_label': 'randsub',   
        'search_mode': False,                
        'idr': 0,                  
        'selected': [6],                
        'split_type': 'rs',                
        'split_name': 'randsub',          
        'checkpoint': '_checkpoint'
        }
        
Mask R-CNN with the Baltic Sea Dataset:

    settings = {
        'dataset' : 'datasets_baltic',
        'run_type': 'test',                  
        'run_label': 'randfold',   
        'search_mode': False,                
        'idr': 0,                  
        'selected': [2]                
        'split_type': 'rs',                
        'split_name': 'fold',          
        'checkpoint': '_checkpoint'
        }

4. One can also specify the target species by adding the 'target_species' in the settings dictionary (requires species_map.json file -> earlier commits may not contain this file)

5. Also, the test sets from each dataset can be interchanged by specifying the destination set on 'dataset' keyword while indicating the original 'source_dataset' keyword in the settings.

   U-Net trained on North Sea Dataset to be tested on Baltic Sea Dataset:

   ````
    settings = {
        'dataset' : 'datasets_baltic',
        'run_type': 'test',                  
        'run_label': 'randsub',   
        'search_mode': False,                
        'idr': 0,                  
        'selected': [47],                
        'split_type': 'rs',                
        'split_name': 'randsub',          
        'checkpoint': '_checkpoint',
        'source_dataset': 'datasets_north',
        }
    ````
7. For web-based usage (localhost), run "python manage.py runserver". Then, use the following URLs to perform experiments and customize the settings with GET parameters (e.g. ?dataset=datasets_baltic&run_type=test&run_label=randsub ...)

   * U-Net: localhost:8000/otoliths/experiments/run_unet/

   * Mask R-CNN: localhost:8000/otoliths/experiments/run_mrcnn/
  
   Alternatively, one can visit the experiments page (http://localhost:8000/demo/experiments/) for examples/demo of experimental setups used.

