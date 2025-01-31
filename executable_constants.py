"""
Original work began May 2022
@ author: Alicia Dagle
@ contributors: Gabriel Trigo, Madeline Skeel
"""

# Import packages:
import numpy as np

######################################################## Constants to follow: ##########################\#####################################
# Save result dataframe to csv file

"""static final constants"""
RAD_TO_DEG = 57.29 # conversion from radians to degrees, this could also be called from a numpy package
ROTATION_LIMIT = 30 # degree limit, for allowable rotations, to align histological internal os line vertically
COLORS = {"green": np.array([0, 1, 0], dtype="float32"), 
          "yellow": np.array([1, 1, 0], dtype="float32"),
          "pink": np.array([1, 0, 1], dtype="float32"),
          "blue": np.array([0, 1, 1], dtype="float32"),
          "black": np.array([0, 0, 0], dtype="float32"),
          "white": np.array([1, 1, 1], dtype="float32"),
          "gray": np.array([0.5, 0.5, 0.5], dtype="float32"),
          "violet": np.array([0.5, 0, 1], dtype="float32"),
          "red": np.array([1, 0, 0], dtype="float32"),
          "orange": np.array([1, 0.65, 0], dtype="float32")} 


COLORS_TUPLES = {"green": (0,1,0,),
                 "yellow": (1, 1, 0),
                 "pink": (1, 0, 1),
                 "blue": (0, 1, 1),
                 "black": (0, 0, 0)}
               
"""model constants"""
NUM_FEATURES = 2 # the number of cervical features returned by get_results.py, saved in the csv file --> in this case CL in pixels and CL in metric units
#Constants for determining internal os location, from plateau of width along cervical canal:
CERV_CANAL_CM_THRESHOLD = 0.3 # cm thickness thresholdhold, used to determine anatomical internal os --> can be thought of as maximum allowable size for mucus plug
MOVING_AVG_WIDTH = 5 # the number of pixels to be considered in the moving average
TRIGGER_THRESHOLD = 0.9 # must be bigger than 0.5 because this will be the average of the closed cervical canal
TRIGGER_SIZE = 20 # how many points to consider to the right of the internal os location (where 2nd derivative plateaus)
START_IDX = 10 # The index at which the model begins to look at the change in derivative (note: needs to be least as big as the MOVING_AVG_WIDTH to avoid edge artifacts)

#Constants used to determine location of internal os, external os, and cervical line points
CL_PADDING = 5 # introduced to avoid weird behavior at edges, cause by angles of wall


def define_folder_paths(dataset_type):
    """
        description:
            Define where ultrasound images and/or masks are saved.
            NOTE: dataset type defined as input to get_results.py, for example 'ATOPS' fed to command line when running python3 get_results.py ATOPS

        parameters:
            @ dataset_type (str): dataset type variable which will be referenced later as a switch to trigger different paths and processing
            
        model constants:

        returns:
            @ dataset_type (str): dataset type variable which will be referenced later as a switch to trigger different paths and processing
            @ US_IMGS_PATH (str): location of the raw ultrasound images for each data type
            @ FOLDER_PATH (str): location of either predicted or ground truth segmented mask images
            @ BOUNCE (bool): boolean because bounce dataset is processed differnetly than the rest
        """



    if dataset_type == 'BOUNCE':
        US_IMGS_PATH = '/home/obsegment/code/ResearchDataset/Bounce/data/23weeks_MH' # BOUNCE images
        FOLDER_PATH = '/home/obsegment/code/ResearchDataset/Bounce/Results/combined/Combined_Predictions_OrigSize/PostProcessed_25' # BOUNCE prediction images
        BOUNCE = True
        # Use a specific pixel conversion for each image:, specific to dataset type:
        SCALE_FILEPATH = None

    elif dataset_type == 'ATOPS':
        US_IMGS_PATH = '/home/obsegment/code/ResearchDataset/ATOPS/data' # ATOPS jpg images
        FOLDER_PATH = '/home/obsegment/code/ResearchDataset/ATOPS/Results/combined/Combined_Predictions_OrigSize/PostProcessed_25' # location of ATOPS prediction masks
        BOUNCE = False
        # Use a specific pixel conversion for each image:, specific to dataset type:
        SCALE_FILEPATH = '/home/obsegment/code/ResearchDataset/ATOPS/Feature_Extraction/ATOPS_Pixel2mm_Scale.csv'
        
    elif dataset_type == 'ATOPS_GT':
        US_IMGS_PATH = '/home/obsegment/code/ResearchDataset/ATOPS/data' # ATOPS jpg images
        FOLDER_PATH = '/home/obsegment/code/ResearchDataset/ATOPS/data/20241002' # location of ATOPS GT masks
        BOUNCE = False
        # Define location of csv file which contains the scale information for each image:
        SCALE_FILEPATH = '/home/obsegment/code/ResearchDataset/ATOPS/Feature_Extraction/ATOPS_Pixel2mm_Scale.csv'

    elif dataset_type == 'HIRI':
        US_IMGS_PATH = '/home/obsegment/code/ResearchDataset/HIRI/data' # HIRI jpg images
        FOLDER_PATH = '/home/obsegment/code/ResearchDataset/HIRI/Results/combined/Combined_Predictions_OrigSize/PostProcessed_25' #HIRI prediction masks
        BOUNCE = False
        # Define location of csv file which contains the scale information for each image:
        SCALE_FILEPATH = '/home/obsegment/code/ResearchDataset/HIRI/Feature_Extraction/Hiri_Pixel2mm_Scales_UnlabeledNames.csv'

        
    elif dataset_type == 'HIRI_GT':
        US_IMGS_PATH = '/home/obsegment/code/ResearchDataset/HIRI/data' # HIRI jpg images
        FOLDER_PATH = '/home/obsegment/code/ResearchDataset/HIRI/data/20240926' # location of HIRI GT masks
        BOUNCE = False
        # Define location of csv file which contains the scale information for each image:
        SCALE_FILEPATH = '/home/obsegment/code/ResearchDataset/HIRI/Feature_Extraction/Hiri_Pixel2mm_Scales_UnlabeledNames.csv'

        
    elif dataset_type == 'LORI':
        US_IMGS_PATH = '/home/obsegment/code/ResearchDataset/LORI/data' # LORI jpg images
        FOLDER_PATH = '/home/obsegment/code/ResearchDataset/LORI/Results/combined/Combined_Predictions_OrigSize/PostProcessed_25' # LORI prediction images
        BOUNCE = False
        SCALE_FILEPATH = None

    elif dataset_type == 'CLEAR':
        US_IMGS_PATH = '/home/obsegment/code/ResearchDataset/CLEAR/data/20220625' # CLEAR jpg images
        FOLDER_PATH = '/home/obsegment/code/ResearchDataset/CLEAR/Results/combined/Combined_Predictions_OrigSize/PostProcessed_25' # CLEAR prediction images
        BOUNCE = False
        SCALE_FILEPATH = None
    
    print(f'folder_path = {FOLDER_PATH}')
    
    return dataset_type, US_IMGS_PATH, FOLDER_PATH, BOUNCE, SCALE_FILEPATH


def define_result_path(dataset_type, FOLDER_PATH):
    """
        description:
            Define the results path to save the cervical feature images and csv with quantitative measurements

        parameters:
            @ dataset_type (str): dataset type (BOUNCE, ATOPS, ATOPS_GT, HIRI, HIRI_GT, LORI, LORI_GT, CLEAR) used to trigger switch that loads the dataset from the proper folder and in the proper format
            @ FOLDER_PATH (str): folder path where TVUS images are stored, defined in define_result_path()
            
        model constants:

        returns:
            @ RESULT_PATH_PARENT_DIR (str): parent directory for the results, derived by splitting FOLDER_PATH variable, and adding a new subdirectory: 'Feature_Extraction/predicted'
            @ RESULT_PATH (str): path to store cervical feature results, derived by combining parent directory and MODEL_NAME variable
            @ MODEL_NAME (str): model name deduced from the FOLDER_PATH variable if cervical feature extraction is xxecuted on predicted segmentation masks (subfolder after Results taken from FOLDER_PATH) 
                                'GT' if the cervical feature extraction is executed on ground truth segmentation masks
            @ POSTPROCESS_NAME (str): None or 'PostProcessed25' indicated whether postprocesing has been applied to the segmentations before cervical feature extraction
        """
    
    if dataset_type == 'ATOPS_GT' or dataset_type == 'HIRI_GT':
        # For the ground truth segmentation msak, the structure is as follows:
        RESULT_PATH_PARENT_DIR = FOLDER_PATH.split('data/')[0] + 'Feature_Extraction/predicted'
        MODEL_NAME = 'GT'
        RESULT_PATH = RESULT_PATH_PARENT_DIR + '/' +  MODEL_NAME
        POSTPROCESS_NAME = None
        
    elif dataset_type != 'BOUNCE':
        # For all datasets except bounce, the structure for the predicted segmentation mask results is as follows:
        MODEL_NAME = FOLDER_PATH.split('Results/')[-1].split('/')[0]
        POSTPROCESS_NAME = 'PostProcessed25'
        RESULT_PATH_PARENT_DIR = FOLDER_PATH.split('Results/')[0] + 'Feature_Extraction/predicted'
        RESULT_PATH = RESULT_PATH_PARENT_DIR + '/' +  MODEL_NAME + '/' + POSTPROCESS_NAME
        
    elif dataset_type == 'BOUNCE':
        # The bounce data is structured slightly differently and therefore is defined as follows:
        MODEL_NAME = FOLDER_PATH.split('Results/')[-1].split('/')[0]
        POSTPROCESS_NAME = 'PostProcessed25'
        RESULT_PATH_PARENT_DIR = '/home/obsegment/code/ResearchDataset/Bounce/Feature_Extraction/23weeks/predicted'
        RESULT_PATH = RESULT_PATH_PARENT_DIR + '/' +  MODEL_NAME + '/' + POSTPROCESS_NAME
    
    return RESULT_PATH_PARENT_DIR, RESULT_PATH, MODEL_NAME, POSTPROCESS_NAME
