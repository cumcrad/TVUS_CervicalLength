# How to execute the algorithm

The `get_results.py` is the primary script to generate automated cervical length extraction, from predicted segmentation mask outputs. This function takes 1 input, which is the dataset name. In order to run this script, you will need to first update executable_constants.py to provide the appropriate dataset type and associated file location for the saved model outputs. For example, from command line, one can execute:
`{python3 main_model_run.py CLEAR}` to run the automatic cervical length extraction algorithm on the CLEAR dataset.


# How to update values in executable_constants.py

If you have not previously set the information for a dataset (ex: CLEAR, BOUNCE, ATOPS, LORI, HIRI, MTS_PREEMIE), follow these instructions before running get_results.py


* Change `US_IMGS_PATH` to where the underlying ultrasound images are read from, these are used to create the overlay cervical feature pictures
* Change `FOLDER_PATH` to where the predicted images (postprocessed or otherwise) are read from
* Set `BOUNCE` boolean to indicate whether we are evaluating the Bounce dataset (which has a preset image scale). If you have a known image scale for another dataset, you can set this in the `get_results.py` file by modifying the if statement containing the variable `pixel_cm_conversion`
* Change `scale_filepath` to the location where a csv contains the scale information for each image within a dataset. This csv should have one column titled `Patient_ID` and another titled 'pixel_to_mm_conversion'. If the image scale is not available, leave this variable as None. In this case, the `get_results.py` script will return pixel values for cervical length, but not metric values.
* There is no need to change `RESULT_PATH`, images will be saved here and the path will automatically be created using: `FOLDER_PATH` and `MODEL_NAME`

The static final constants and model constants do not need to be updated, but these values may be modified if the image masks you are working with are a different color, or if you would like to explore other constant threshold values within the cervical length algorithm deployment. 

# Summary of the algorithm

`get_results.py` contains the primary functions to load the image/mask files, locate the internal/external os and subsequent cervical length features, and save these features to a csv file sepecified by `RESULT_PATH`. This script calls executable_constant methods `define_folder_paths()` and `define_result_path()` to locate system files. Then it executes the Primary function `get_imagefeatures_save_df()` in `get_results.py`:  this function manipulates the segmentation masks to derive cervical features (i.e. cervical length) by calling methods within the `Extractor_Obj` and `Img_Obj classes`, defined in `extractor.py` and `image_utils.py` respectively. The features (cervical length) are then saved as an overlay on top of the original TVUS image and/or segmentation masks, to the `RESULT_PATH`. Finally, the quantitative values (cervical length in pixels and metric values if available) also exported to a csv file in `result_path_parent_directory`.
