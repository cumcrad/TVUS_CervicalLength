"""
Original work began May 2022
@ author: Alicia Dagle
@ contributors: Gabriel Trigo, Madeline Skeel
"""

# Import packages:
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from feature_extraction_TVUS.NPJ_CL_only_Jan2025.extractor import Extractor_Obj
import executable_constants
import sys


def get_imagefeatures_save_df(dataset_type, US_IMGS_PATH, FOLDER_PATH, result_path_parent_directory, result_path, SCALE_FILEPATH, debug = False): 
    """
    description:
        Primary function in get_results.py, this function manipulates the segmentation masks to derive cervical features (i.e. cervical length).
        The features are saved overlayed on top of the original TVUS image and/or segmentation masks and also exported to a csv file in result_path_parent_directory.

    parameters:
        @ dataset_type (str): dataset type variable which will be referenced later as a switch to trigger different paths and processing
        @ US_IMGS_PATH (str): location of the raw ultrasound images for each data type (named US_IMGS_PATH)
        @ FOLDER_PATH (str): location of either predicted or ground truth segmented mask images
        @ dataset_type (str): dataset type variable which will be referenced later as a switch to trigger different paths and processing
        
        
    model constants:

    returns:
        @ cervical_feature_df (pd.DataFrame): dataframe containing all cervical features, which is also saved as a csv within this function
    """
    # Initialize empty lists for the US image paths, mask paths and image scales:
    US_files = []
    mask_files = []
    pixel_cm_conversion_list_orig = []

    
    # List all subdirectories within FOLDER_PATH, pointing to masks (png image folders only):
    # Iterate through each image path (f) in FOLDER_PATH:
    for f in os.listdir(FOLDER_PATH):
        filename, extension = os.path.splitext(f)
        if extension == '.png':
            # Generate list of all mask file paths:
            mask_files.append(f)
            
            if dataset_type == 'HIRI' or dataset_type == 'ATOPS':
                # Get patient name and pixel to mm conversion for each image path (f) in FOLDER_PATH:
                patient_name = filename.split('/')[-1].split('.')[0]
            elif dataset_type == 'ATOPS_GT' or dataset_type == 'HIRI_GT':
                # Get patient name and pixel to mm conversion for each image path (f) in FOLDER_PATH:
                patient_name = filename.split('/')[-1].split('_lab')[0]
            if SCALE_FILEPATH != None:
                # Define scale from location of csv file which contains the scale information for each image:
                scale_df = pd.read_csv(SCALE_FILEPATH) 
                patient_scale_pixel2mm = scale_df[scale_df.Patient_ID == patient_name].pixel_to_mm_conversion.item()
                # Convert scale to cm/pixel, so that we can multiply the number of pixels by this value to find the number of cm, etc:
                patient_scale = float(patient_scale_pixel2mm)/10
                # Generate list of all image scales:
                pixel_cm_conversion_list_orig.append(patient_scale)
            
    if debug == True:
        print(f"pixel_cm_conversion_list = {pixel_cm_conversion_list_orig}")
    
    # Create list of all raw US files (excluding inpainting images):
    for f in os.listdir(US_IMGS_PATH):
        if 'im.png' in f:
            if not "Inpaint" in f:
                US_files.append(f)
        elif ".jpg" in f: #catch ATOPS US images...add a key/switch for this later
            if not "Inpaint" in f:
                US_files.append(f)
    if not len(US_files) == len(mask_files):
        print("ERROR: length of US_files and mask_files are not equal!!!")
    
    
    # Sort mask  and ultrasound files by name, rearrange pixel scale list accordingly 
    index_order = np.argsort(mask_files)
    pixel_cm_conversion_list = [pixel_cm_conversion_list_orig[index] for index in index_order]
    mask_files = sorted(mask_files)
    US_files = sorted(US_files)
    
    if debug == True:
        print(f"index order = {index_order}")
        print(f"length of mask order = {len(mask_files)}")
        print(f"length of index order = {len(index_order)}")
        print(f"1st elem index order = {index_order[0]}")
        print(f"pixel_cm_conversion_list_orig = {pixel_cm_conversion_list_orig}")    
        print(f"pixel_cm_conversion_list_orig[0] = {pixel_cm_conversion_list_orig[0]}")
    
   
    # Initialize empty table to store cervix features:
    table = np.zeros((len(mask_files), executable_constants.NUM_FEATURES))
    home = os.getcwd()
    
    # For each subdiretory, create a list of features:
    for i, mask_file in enumerate(mask_files):
        features = np.zeros(executable_constants.NUM_FEATURES)
        extractor_input_path = "{}/{}".format(FOLDER_PATH, mask_file) 
        US_file = US_files[i]
        US_img_input_path = "{}/{}".format(US_IMGS_PATH, US_file)
        if debug == True:
            print(f"mask_file = {mask_file}")
            print('extractor_input_path = ' + extractor_input_path)
            print(f"US_file = {US_file}")
        
        # Get the scale conversion for the image of interest:
        if SCALE_FILEPATH!= None:
            pixel_cm_conversion = pixel_cm_conversion_list[i]
        elif dataset_type== "BOUNCE":
            pixel_cm_conversion = 0.0102
        else: #No conversion defined yet:
            pixel_cm_conversion = np.nan
        
        # Call extractor object
        obj = Extractor_Obj(extractor_input_path, US_img_input_path, pixel_cm_conversion)
       
        # remove ".png" extension from mask path:
        mask_filename = mask_file.split('.')[0] #[:-4]
        
        # Create result directory substructure - if it does not yet exist:
        if not os.path.exists(result_path_parent_directory):
            os.makedirs(result_path_parent_directory)
            #os.chdir(result_path_parent_directory)
            
        # Create model directory substructure - if it does not yet exist:
        if not os.path.exists(os.path.join(result_path_parent_directory, MODEL_NAME)):
            os.makedirs(os.path.join(result_path_parent_directory, MODEL_NAME))
        #os.chdir(os.path.join(result_path_parent_directory, MODEL_NAME))
        
         # If postprocessing is used, make a post-processing sub-directory:
        if POSTPROCESS_NAME != None:
            if not os.path.exists(os.path.join(os.path.join(result_path_parent_directory,MODEL_NAME), POSTPROCESS_NAME)):
                os.makedirs(os.path.join(os.path.join(result_path_parent_directory,MODEL_NAME), POSTPROCESS_NAME))
                #os.chdir(os.path.join(os.path.join(result_path_parent_directory,MODEL_NAME), POSTPROCESS_NAME))
        
        #create a math extension to save the resulting images created in get_df()
        result_images_folder = "{}/{}".format(result_path, mask_filename)
        if not os.path.exists(result_images_folder):
            os.makedirs(result_images_folder)
            
        #os.chdir(result_path)
        #os.chdir(home)
    
        
        if debug == True:
            print(f"test= {result_path}")
            print(f"result_image_folder here = {result_images_folder}")
            print(f"parent dir name = {result_path_parent_directory}")
            print(f"model name = {MODEL_NAME}")
            print(f"was {MODEL_NAME} file created? ")
            print(os.path.exists(MODEL_NAME))
            
       
        # Measure the circle length (in pixels):
        try:
            # Fill cervical length feature
            features[0] = obj.get_cervical_length()
        except:
            # Raise an error if obj.get_cervical_length() fails, and fill feature with NaN
            features[0] = np.nan
            print("Error: not able to measure cervical length")
        
        # Convert cervical length value from pixels to cm:
        try:
            # If a pixel scale is defined for the given dataset, convert the pixel value of cervical length to a metric value (cm):
            if dataset_type == "BOUNCE" or dataset_type == "HIRI" or dataset_type == 'HIRI_GT' or dataset_type == 'ATOPS' or dataset_type == 'ATOPS_GT':
                features[1] = features[0]*pixel_cm_conversion
        except:
            # If no pixel scale is defined, fill feature value with NaN and raise error:
            features[1] = np.nan
            print("Error: not able to convert pixel to cm scale")

        # Save an image with the internal and external os indicated with a white dot overlay on image:
        try: 
            # Internal and external os overlay on mask, in original orientation:
            fig1 = plt.figure(figsize=(10,5))
            obj.plot_os(underlying = 'mask', coordinates = 'original')
            fig1.savefig("{}/os_mask".format(result_images_folder), dpi = 500, bbox_inches = "tight")
            
            # Internal and external os overlay on TVUS, in original orientation:
            fig2 = plt.figure(figsize=(10,5))
            obj.plot_os(underlying = 'image', coordinates = 'original')
            fig2.savefig("{}/os_image".format(result_images_folder), dpi = 500, bbox_inches = "tight")
            
            # Internal and external os overlay on mask, in rotated orientation:
            fig3 = plt.figure(figsize=(10,5))
            obj.plot_os(underlying = 'mask', coordinates = 'dynamic')
            fig3.savefig("{}/os_mask_dynamic".format(result_images_folder), dpi = 500, bbox_inches = "tight")
        except:
            # Raise an error if plot_os() function fails, and image cannot be saved:
            print("Error: Not able to visualize internal & external os")
            pass

        # Save an image with the cervical line indicated in white overlay:
        try: 
            # Create a figure, display white cervical line overlay on segmentation mask:
            fig1 = plt.figure(figsize=(10,5))
            obj.plot_cervical_line(underlying = 'mask', coordinates = 'original')
            fig1.savefig("{}/cervical_line".format(result_images_folder),bbox_inches = "tight")
            # Create a figure, display white cervical line overlay on TVUS image:
            fig2 = plt.figure(figsize=(10,5))
            obj.plot_cervical_line(underlying = 'image', coordinates = 'original')
            fig2.savefig("{}/cervical_line_overlay".format(result_images_folder),bbox_inches = "tight")
        except:
            # Raise an error if plot_cervical_line() function fails, and image cannot be saved:
            print("Error: Not able to visualize cervical line")
            pass
        
        # Fill list with features
        table[i] = features # this may be redundant?

        # Convert the list to a pandas dataframe type:
        cervical_feature_df = pd.DataFrame(table, index = mask_files, columns = ["cervical_length_pixels", "cervical_length_cm"])
    
    # Return pandas dataframe of cervical features:
    return  cervical_feature_df
                    



if __name__ == '__main__':
    '''example of command line to execute: python3 get_results.py ATOPS'''
    debug = True
    
    # Map command line arguments to function arguments:
    dataset_type, US_IMGS_PATH, FOLDER_PATH, BOUNCE, SCALE_FILEPATH = executable_constants.define_folder_paths(*sys.argv[1:])
    # Define directories to save cervical feature results:
    RESULT_PATH_PARENT_DIR, RESULT_PATH, MODEL_NAME, POSTPROCESS_NAME = executable_constants.define_result_path(dataset_type, FOLDER_PATH)
    # Create a dataframe of cervical features:
    result = get_imagefeatures_save_df(dataset_type, US_IMGS_PATH, FOLDER_PATH, RESULT_PATH_PARENT_DIR, RESULT_PATH, SCALE_FILEPATH)
    # Save cervical features dataframe as a csv file:
    result.to_csv(os.path.join(RESULT_PATH, 'CervixFeature_result.csv'))
    
    if debug == True:
        print(f"dataset_type = {dataset_type}")
        print(f"FOLDER_PATH = {FOLDER_PATH}")
        print(result)
    