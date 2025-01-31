"""
Original work began May 2022
@ author: Alicia Dagle
@ contributors: Gabriel Trigo, Madeline Skeel
"""

#Import pre-built python packages:
import numpy as np
import matplotlib.pyplot as plt
from image_utils import Img_Obj
from scipy.optimize import curve_fit
import cv2 as cv
from PIL import Image
from matplotlib import patches

#Import custom scripts
from static_methods import moving_average, linear_fnc, find_start_index, get_perpendicular_line
import executable_constants

#Imports to remove artifacts:
from skimage.measure import regionprops, regionprops_table, label
import copy
import pandas as pd
import cv2

import skimage
import torchvision
import torch
import torchvision.transforms.functional as F
from skimage import io, color, segmentation #used for displaying transparent segmentation mask


from matplotlib.patches import Arc
from matplotlib.transforms import Bbox, IdentityTransform, TransformedBbox
import math
from scipy import ndimage


class Extractor_Obj:
    
    def __init__(self, mask_path, raw_US_path, pixel_cm_conversion, debug = True):
        print("Initializing Exctractor_Obj...")
        self.img_obj = Img_Obj(mask_path, raw_US_path)
        self.pixel_cm_conversion = pixel_cm_conversion
        # Initialize cervical line features:
        self.CL_line = None
        self.CL_line_coordinates = None, None
        
        
        """
        description:
            Start by identifying (x,y) location of internal and external os
            on all images, as these landmarks will be necessary to extract
            other features. Then, identify the cervical canal, whether it is
            closed or a mucus plug is present. Measure the midpoint along the
            cervical canal to return the cervical line trace. Take the number
            of pixels in the cervical line (only 1 point per column) as the
            pixel-wise cervical lenth. Convert the pixel value to a metric
            value, if the scale is available.
            
        """
        
   
        
        # Try to locate internal os of the cervix, in original and rotated coordinate system
        try:
            self.hist_internal_os_dynamic, self.hist_internal_os_orig, self.anat_internal_os_dynamic, self.anat_internal_os_orig = self.get_internal_os(plateau_method = True)
            if debug == True:
                print(f"self.hist_internal_os_dynamic = {self.hist_internal_os_dynamic}")
                print(f"self.hist_internal_os_orig = {self.hist_internal_os_orig}")
                print(f"self.anat_internal_os_dynamic = {self.anat_internal_os_dynamic}")
                print(f"self.anat_internal_os_orig = {self.anat_internal_os_orig}")
        except:
            print(" internal os not found, look at self.get_internal_os()")
            self.hist_internal_os_dynamic = np.array([np.nan, np.nan])
            self.hist_internal_os_orig = np.array([np.nan, np.nan])
        
        #Try to locate the external os of the cervix, in the original and rotated coordinate system:
        try: 
            self.external_os = self.get_external_os()
            if debug == True:
                print(f"self.external_os_dynamic = {self.external_os_dynamic}")
                print(f"self.external_os_orig = {self.external_os_orig}")
        except:
            self.external_os_dynamic = np.array([np.nan, np.nan])
            self.external_os_orig = np.array([np.nan, np.nan])
        
        # Try to plot internal and external os locations, by overlaying on the original TVUS image or segmented mask
        try:
            self.plot_os()
        except:
            print("Error in plot_os()")
            pass
        
        # Try to detect cervical line --> this will define self.cervical_line_dynamic and self.closed_cervical_line_dynamic   
        try: 
            print("getting cervical line")
            self.get_cervical_line()
            print("cervical line points retrieved successfully")
            # If cervical line is detected, plot it as an overlay on the original TVUS image and segmented mask
            try:
                self.plot_cervical_line(underlying = 'mask', coordinates = 'original')
                self.plot_cervical_line(underlying = 'image', coordinates = 'original')
            except:
                print("Error in plot_cervical_line()")
                pass
        except:
            print("WARNING: cervical line extraction unsuccessful, not able to generate plot")
            pass
    
    
    # Helper function used to find the internal os location of the cervix        
    def find_plateau(self, debug = False):
        """
        description:
            Identifies the region where the number of cervical canal points in a column plateaus
            by scanning from left to right and looking for successive low values of the derivative
            
        parameters:
            
        model constants:
            @ executable_constants.MOVING_AVG_WIDTH (int): window size to be considered
            @ executable_constants.CERV_CANAL_CM_THRESHOLD (int): cm thickness thresholdhold, used to determine anatomical internal os --> can be thought of as maximum allowable size for mucus plug
            @ executable_constants.START_IDX (int): The index at which the model begins to look at the change in derivative (note: needs to be least as big as the MOVING_AVG_WIDTH to avoid edge artifacts)
            @ executable_constants.TRIGGER_THRESHOLD (int): the derivative needs to be smaller than this value to activate the trigger
            @ executable_constants.TRIGGER_SIZE:  number of sucessive points that must match the condition for the trigger to activate
            
        returns:
            @ int_os_index (int): index of the column in which the start of the plateau was detected,
                this is then identified as the histological internal os location
        """
        
        # Take the moving average of the difference in the number of pixels per column, across an specified with for the moving average: this can be thought of as the 2nd derivative of cervical canal diameter
        diff_avg = moving_average(np.diff(self.img_obj.cerv_canal_pixel_count), executable_constants.MOVING_AVG_WIDTH)
        
        # Initialize trigger and padding values: 
        trigger = 0
        padding = (executable_constants.MOVING_AVG_WIDTH - 1)/2
        cerv_canal_threshold_pixel = executable_constants.CERV_CANAL_CM_THRESHOLD*(1/self.pixel_cm_conversion) 
        hist_int_os_index = None
        anat_int_os_index = None
        
        # Start looking at the change in derivative after a certain threshold (START_IDX) that is at least as big as the MOVING_AVG_WIDTH to avoid edge artifacts:
        min_col_index = np.nonzero(self.img_obj.cerv_canal_pixel_count)[0][executable_constants.START_IDX]
        
        if debug == True:
            print(f"self.img_obj.cerv_canal_pixel_count = {self.img_obj.cerv_canal_pixel_count}")
            print(f"check length of diff_avg:{len(diff_avg)} and np.diff(self.img_obj.cerv_canal_pixel_count): {len(np.diff(self.img_obj.cerv_canal_pixel_count))}")
        
        # For each column, iterate through the number of cervical canal pixels in a given column:
        for plateau_index, value in enumerate(diff_avg):
            # If the column is some distance (defined by min column index) from the leftmost column of this class (to avoid edge artifacts),
            if plateau_index > min_col_index:
                #If there is a sufficiently small (think: 2nd derivative) change in the rate of change of cervical width (value smaller than trigger threshold - default value defined in executeable_constants.py)
                if abs(value) <= executable_constants.TRIGGER_THRESHOLD:
                    #first_deriv = moving_average(self.img_obj.cerv_canal_pixel_countexecutable_constants.MOVING_AVG_WIDTH)
                    first_deriv = np.diff(self.img_obj.cerv_canal_pixel_count)[int(plateau_index+padding)] #should this be a moving avg?
                    if debug == True:
                        print(f"entered loop to check plateau values for each index, value = {value}")
                        #If the cervical width is decreasing (negative 1st derivate):
                        print(f"1st deriv index: {int(plateau_index+padding)}")
                        print(f"1st derivative: {first_deriv}")
                    if first_deriv <= 0: # Allow for small positive fluctuations less than or equal to 1
                        trigger += 1
                else:
                    # Reset trigger count to 0
                    trigger = 0
                if debug == True:
                    print(f"trigger: {trigger}")
                    
                # Check if the column width is below the cervical canal threshold, or not:
                if self.img_obj.cerv_canal_pixel_count[plateau_index]<= cerv_canal_threshold_pixel and anat_int_os_index == None:
                    anat_int_os_index = plateau_index
                else:
                    pass
        
            # If the plateau leveled off for an iteration larger than the trigger size, calculate the internal os location
            # by taking the point where the plateau behavior started, and return this value
            if trigger > executable_constants.TRIGGER_SIZE and hist_int_os_index == None:
                hist_int_os_index = plateau_index - executable_constants.TRIGGER_SIZE # subtraction indicates moving left by TRIGGER_SIZE
            
        return hist_int_os_index, anat_int_os_index
        
    # Fill variables with internal os coordinates: self.hist_internal_os_dynamic, self.hist_internal_os_orig   
    def get_internal_os(self, plateau_method = True, debug = False):
        """
        description:
            Gets the row and column indices, i.e. (x,y) position of the internal os
            
        parameters:
            @ pixel_cm_conversion: the pixel to cm conversions for given image
            
        model constants:
        
        returns:
            @ self.hist_internal_os_dynamic (int, int): indices of the row and column of the internal os
                in the dynamic/rotated coordinate systems
            @ self.hist_internal_os_orig (int, int): indices of the row and column of the internal os
                in the original coordinate systems
        """
        bool_int_os_found = False
        plateau_success = False
        # Initialize dynamic internal os variable as np array of empty values (both histological and anatomical)
        self.hist_internal_os_dynamic = np.array([np.nan, np.nan])
        self.anat_internal_os_dynamic = np.array([np.nan, np.nan])
        
        # Use the plateau method to find the internal os:
        if plateau_method == True:
            # find index of internal os:
            try:
                column_hist_int_os, column_anat_int_os = self.find_plateau()
                print(f"column_hist_int_os = {column_hist_int_os}, column_anat_int_os = {column_anat_int_os}")
                if column_anat_int_os < column_hist_int_os:
                    # redefine anatomical internal os to ensure it is not to the left of the histological internal os (which better captures closed cervix location in case of v-or y-shaped funnel)
                    column_anat_int_os = column_hist_int_os
                print(f"cerv_canal_class all columns = {self.img_obj.cerv_canal_class}")
                # take all pixels in the column of the internal os:
                column_hist_int_os_all_pixels = self.img_obj.cerv_canal_class[:, column_hist_int_os]
                column_anat_int_os_all_pixels = self.img_obj.cerv_canal_class[:, column_anat_int_os]
                    
                # Identify top and bottom-most point in the column of interest, take the average between them to find row location of histological int. os:
                hist_int_os_top = np.nonzero(column_hist_int_os_all_pixels)[0][0]
                hist_int_os_bottom = np.nonzero(column_hist_int_os_all_pixels)[0][-1]
                row_hist_int_os = (hist_int_os_top + hist_int_os_bottom) // 2
                
                # Identify top and bottom-most point in the column ov interest, take the average between them to find row location of anatomical int. os:
                anat_int_os_top = np.nonzero(column_anat_int_os_all_pixels)[0][0]
                anat_int_os_bottom = np.nonzero(column_anat_int_os_all_pixels)[0][-1]
                row_anat_int_os = (anat_int_os_top + anat_int_os_bottom) // 2
                
                # Save the internal os row, col coordinates:
                self.hist_internal_os_dynamic = row_hist_int_os, column_hist_int_os
                self.anat_internal_os_dynamic = row_anat_int_os, column_anat_int_os
                
                # Also return internal os, translated/rotated back to original coordinate system
                if self.img_obj.dynamic_rotation_angle == None:
                    # the image was not rotated, therefore a coordinate shift is not required  
                    self.hist_internal_os_orig = self.hist_internal_os_dynamic
                    self.anat_internal_os_orig = self.anat_internal_os_dynamic
                else:
                    # the image was rotated, shift coordinates accordingly
                    self.hist_internal_os_orig = self.rotate_features_to_default_coord(row_hist_int_os, column_hist_int_os, -self.img_obj.dynamic_rotation_angle, image_coord = True)
                    print(f'row_anat_int_os = {row_anat_int_os}, column_anat_int_os = {column_anat_int_os}, self.img_obj.dynamic_rotation_angle = {self.img_obj.dynamic_rotation_angle} ')
                    self.anat_internal_os_orig = self.rotate_features_to_default_coord(row_anat_int_os, column_anat_int_os, -self.img_obj.dynamic_rotation_angle, image_coord = True)
                
                # optional, print debugging statements:
                if debug == True:
                    print(f"column_all_pixels = {column_hist_int_os_all_pixels}")
                    print(f"np.nonzero(column_hist_int_os_all_pixels)[0] = {np.nonzero(column_hist_int_os_all_pixels)[0]}")
                    print(f"internal os location = {row_hist_int_os},{column_hist_int_os}")
                    print(f"rotation angle = {self.img_obj.dynamic_rotation_angle}")
                    print(f"Plateau method used to find internal os at {self.hist_internal_os_dynamic}")
                    print(f"check variable: {self.img_obj.cerv_canal_pixel_count}")
                plateau_success = True
            
            except:
                print("find_plateau method did not work, try alternate method")
                plateau_success = False
                pass
        
        # NOTE - To account for rare artifacts, it may be nice to add a catch statement to look if there are 2 separate instances of the cervical class
        # if this is true, we should take only the leftmost instance to find the plateau,
        # this will avoid artifacts from the mucus plug in the middle region of the cervix 
        
        # If the plateau method is not being used or was unsuccessful (can occur if no cervical canal class pixels present in image), consider adjacent cervix tissues:
        if (plateau_method == False) or (plateau_success == False):
            # If there are no green cervical canal pixels or plateau method is not used ------------------------------------------
            # try looking for adjacent cervix tissue
            if debug == True:
                print("Looking for internal os as the intserection of cervical canal, anterior cervix and posterior cervix")
            try:
                # Now try triple point method, regardless of whether plateau_method worked: 
                list_triple_points = []
                # Iterate through image row, columns:
                for i in range(0,self.img_obj.img_dynamic.shape[0]):
                    for j in range(0,self.img_obj.img_dynamic.shape[1]):
                        bool_cerv_canal = np.array_equal(self.img_obj.img_dynamic[i, j, :], executable_constants.COLORS["green"])
                        if bool_cerv_canal == True:
                            if debug == True:
                                print(f"point i,j = {i,j} is a cervical canal class")
                            # Reset booleans before checkign neighboring pixels
                            bool_pos_cerv = False
                            bool_ant_cerv = False    
                            # Check all perpendicular and diagnoally touching points:
                            for row_dim in range(i-1,i+1):
                                for col_dim in range(j-1,j+1):
                                    adjacent_anterior_cervix = np.array_equal(self.img_obj.img_dynamic[row_dim, col_dim, :], executable_constants.COLORS["pink"])
                                    adjacent_posterior_cervix = np.array_equal(self.img_obj.img_dynamic[row_dim, col_dim, :], executable_constants.COLORS["blue"])
                                    if debug == True:
                                        print(f"neighboring point to explore = {row_dim, col_dim}")
                                        print(f"adjacent_anterior_cervix = {adjacent_anterior_cervix}")
                                        print(f"adjacent_posterior_cervix = {adjacent_posterior_cervix}")
                                    if adjacent_anterior_cervix == True:
                                        bool_ant_cerv = True
                                    if adjacent_posterior_cervix == True:
                                        bool_pos_cerv = True
                            # If cervical canal has anterior and posterior cervix neighbors:
                            if bool_pos_cerv and bool_ant_cerv:
                                #identify left-most intersection only:
                                if np.isnan(self.hist_internal_os_dynamic).any():
                                    list_triple_points.append([i,j])
                                    #assign internal os
                                    self.hist_internal_os_dynamic = i,j
                                else:
                                    #reassign internal os if column value is smaller than previously assigned value
                                    if j < self.hist_internal_os_dynamic[1]:
                                        # incase there are multiple points that intersect all 3 classes, look for left-most intersection
                                        self.hist_internal_os_dynamic = i,j
                                self.anat_internal_os_dynamic = self.hist_internal_os_dynamic
                                self.hist_internal_os_orig = self.rotate_features_to_default_coord(self.hist_internal_os_dynamic[0], self.hist_internal_os_dynamic[1], -self.img_obj.dynamic_rotation_angle, image_coord = True)
                                self.anat_internal_os_orig = self.hist_internal_os_orig  
                                     
                                if not np.isnan(self.hist_internal_os_dynamic).any():
                                    self.hist_internal_os_orig = self.rotate_features_to_default_coord(self.hist_internal_os_dynamic[0], self.hist_internal_os_dynamic[1], -self.img_obj.dynamic_rotation_angle, image_coord = True)
                                    # Histological os and anatomical internal os are the same if we are using the triple point method (cervix is closed):
                                    self.anat_internal_os_orig =  self.hist_internal_os_orig
                # If there is exactly one intersection between pink/green/blue:           
                if len(list_triple_points) == 1:
                    bool_int_os_found = True
                    print(f"Found internal os using triple point intersection at: {self.hist_internal_os_dynamic}")
                elif len(list_triple_points)>1:
                    print("Multiple triple point intersections found")
                elif len(list_triple_points)==0:
                    print("There was no intersection found between pink/green/blue classes")
                if debug == True:
                    print("looked for int os , results to follow:")         
            except:
                print("Triple point method unsucessful...")  
                pass
                
            # If we did not find the internal os yet, look at the adjacent cervical tissue and take the left-most adjacement point:                        
            if bool_int_os_found == False:
                print("Internal os NOT FOUND using triple point intersection")
            elif bool_int_os_found == True:
                print("Internal os was FOUND using triple point intersection, but we will still check adjacent tissue:")
            
            # Regardless of whether triple point was found, also look at adjacent cervix tissue:
            try:
                #take left most point of the adjacent cervix as the internal os
                adjacent_cervix = self.img_obj.closed_cervix_points_orig
                
                # Take the minimum column and reassign the internal os based on the minimum column for adjacent cervical tissue:    
                for idx in range(0,len(adjacent_cervix)):
                    if idx == 0:
                        int_os = adjacent_cervix[idx]
                        col_min = int_os[1]
                    elif adjacent_cervix[idx][1] < col_min:
                        col_min = adjacent_cervix[idx][1]
                        int_os = adjacent_cervix[idx]
                    else:
                        pass
                    
                # Define the original coordinate system of the internal os:
                if self.img_obj.dynamic_rotation_angle == None:
                    #the image was not rotated, therefore a coordinate shift is not required
                    dynamic_os = int_os
                    if not np.isnan(self.hist_internal_os_dynamic).any():
                        if dynamic_os[1] < self.hist_internal_os_dynamic[1]:
                            self.hist_internal_os_dynamic = dynamic_os
                            self.hist_internal_os_orig = int_os
                            self.anat_internal_os_orig = self.hist_internal_os_orig
                else:
                    dynamic_os = self.rotate_features_to_dynamic_coord(int_os[0], int_os[1], +self.img_obj.dynamic_rotation_angle, image_coord = True)
                    if not np.isnan(self.hist_internal_os_dynamic).any():
                        if dynamic_os[1] < self.hist_internal_os_dynamic[1]:
                            self.hist_internal_os_dynamic = dynamic_os
                            self.hist_internal_os_orig = int_os
                            self.anat_internal_os_orig = self.hist_internal_os_orig
                            
                # Define the dynamic/rotated coordinate system of the internal os:
                self.hist_internal_os_dynamic = int(self.hist_internal_os_dynamic[0]), int(self.hist_internal_os_dynamic[1])
                # Keep same definition for anatomical internal os:
                self.anat_internal_os_dynamic = self.hist_internal_os_dynamic
                
                
                if debug == True:
                    print("Looking for internal os by analzing the adjacent cervical tissue, taking leftmost point (between adjacent tissue method and triple point method)")
                    print(f"adjacent_cervix = {adjacent_cervix}")    
                    print(f"Internal os FOUND using adjacent cervix at {int_os}")
                    print(f"self.hist_internal_os_orig = {self.hist_internal_os_orig}")
                    print(f"self.hist_internal_os_orig[0] {self.hist_internal_os_orig[0]}, self.hist_internal_os_orig[1] = {self.hist_internal_os_orig[1]}, self.img_obj.dynamic_rotation_angle = {self.img_obj.dynamic_rotation_angle}")  
                    print(f"self.hist_internal_os_dynamic = {self.hist_internal_os_dynamic}")
                    print(f"Internal os FOUND after checking adjacent cervix at {self.hist_internal_os_dynamic}")
            except:
                print("Adjacent cervix tissue method did not work to find internal os")
                pass
            
        # Return internal os in the rotated/dynamic coordinate system and in the original coordinate system        
        return self.hist_internal_os_dynamic, self.hist_internal_os_orig, self.anat_internal_os_dynamic, self.anat_internal_os_orig
        
    # Fill variables with external os coordinates: self.external_os_dynamic, self.external_os_orig       
    def get_external_os(self, debug = False):
        """
        description:
            Gets the row and column indexes of the external os.
            
        parameters:
            
        model constants:
            @ executable_constants.CL_PADDING (int): number of columns to ignore at the right end to avoid
                unpredictable behavior due to angled wall
                
        returns:
            @ self.external_os_dynamic (int, int): indices of the row and column of the external os, in external/rotated coordinate system
            @ self.external_os_orig (int, int): indices of the row and column of the external os, in original coordinate system
        """
        
        column = np.nonzero(self.img_obj.cerv_canal_pixel_count)[0][-executable_constants.CL_PADDING]
        top = np.nonzero(self.img_obj.cerv_canal_class[:, column])[0][0]
        bottom = np.nonzero(self.img_obj.cerv_canal_class[:, column])[0][-1] #this same method could be applied in get_internal_os()
        row_midpoint = (top + bottom) // 2
        row_ext_os = row_midpoint
        col_ext_os = column
        
        if debug == True:
            print(f"columns for external os calculation are: {np.nonzero(self.img_obj.cerv_canal_pixel_count)[0][:]}")
            print(f"top for external os calculation is: {top}")
            print(f"bottom for external os calculation is: {bottom}")
            print(f"external os location = {row_ext_os},{col_ext_os}")
        
        # Save external os in the rotated/dynamic coordinate system and in the original coordinate system
        self.external_os_dynamic = row_ext_os, col_ext_os
        if self.img_obj.dynamic_rotation_angle == None:
            self.external_os_orig = self.external_os_dynamic
        else:
            self.external_os_orig = self.rotate_features_to_default_coord(row_ext_os, col_ext_os, -self.img_obj.dynamic_rotation_angle, image_coord = True)
        
        print(f"External os FOUND using adjacent cervix at {self.external_os_dynamic}")
        
        #Return external os in dynamic/rotated and original coordinate system:
        return self.external_os_dynamic, self.external_os_orig
    
    # Return row, col of points along the cervical line (in the original and rotated/dynamic coordinate system)  
    def get_cervical_line(self, debug = False):
        """
        description:
            obtain the line that goes along the cervix (inner canal), this will be used to denote cervical length
        parameters:
            
        model constants:
            
        assigned variabled:
            @ self.cervical_line_dynamic: a tupple of arrays for the row and column indices of the cervical line in its rotated orientation (dynamic)
            @ row_line, @ col_line (np.array, np.array): vertical and
                horizontal coordinates of the points that make up the line
        
        returns:
            @ (cl_row, cl_col), (cl_row_orig, cl_col_orig)
        """
        
        # Get row and column coordinate for points along the cervical canal, in rotated/dynamic image orientation:
        row_line = []
        col_line = []
        var_store = int(self.hist_internal_os_dynamic[1])
        if debug == True:
            print(f"sanity check 1: internal os starting indx = {var_store}")
        for col_idx in range(int(self.hist_internal_os_dynamic[1]), int(self.external_os_dynamic[1])+1): #added +1 here 4/25/2024
            element = self.img_obj.cerv_canal_pixel_count[col_idx]
            if debug == True:
                print(f"col_idx = {col_idx}, element = {element}")
                print(f"count of cervical canal pixels = {element}")
            # account for zero values in some columns (there is a gap in the labeling of this class, ie more than one instance)
            if element == 0:
                # Pass to the next column, there may be a gap in the class labels:
                pass
            
            #assuming nonzero values in every column between internal and external os:    
            elif element > 0:
                top = np.nonzero(self.img_obj.cerv_canal_class[:, col_idx])[0][0]
                bot = np.nonzero(self.img_obj.cerv_canal_class[:, col_idx])[0][-1]
                #take midpoint between top and bottom pixel in each column
                mid = (top + bot) // 2
                
                #take avg instead of midpoint:
                avg_green = np.average(np.nonzero(self.img_obj.cerv_canal_class[:, col_idx])[0][:])
                row_idx = mid
                if debug == True:
                    print(f"top = {top}, bottom = {bot}")
                    print(np.nonzero(self.img_obj.cerv_canal_class[:, col_idx]))
                    print(f"avg_green = {int(avg_green)}")
                
                # if there is a missing point, not labeled --> reference the previous row index
                # preserve image coordinates (row, column)
                row_line.append(row_idx)
                col_line.append(col_idx)
                if debug == True:
                    print(f"row_idx, col_idx = {row_idx, col_idx}")
                
        cl_row, cl_col = np.array(row_line), np.array(col_line)
        self.cervical_line_dynamic =  cl_row, cl_col
        
        # Find the portion of the cervical line over which the cervix is closed (to measure cervical length):
        closed_cl_row_dynamic, closed_cl_col_dynamic = [], []
        for idx, col in enumerate(cl_col):
            if  col >= self.anat_internal_os_dynamic[1]:
                closed_cl_col_dynamic.append(col)
                closed_cl_row_dynamic.append(cl_row[idx])
        self.closed_cervical_line_dynamic = closed_cl_row_dynamic, closed_cl_col_dynamic
        
        if debug == True:
            print(f"cl_row, cl_col = {cl_row, cl_col}")
        
        
        # CONVERT ROW/COLUMN COORDINATES TO ORIGINAL IMAGE ORIENTATION:
        # start by abstracting row, column list of coordinates in rotated image space:
        cl_row_orig = []
        cl_col_orig = []
        
        # For each row coordinte:
        for idx in range(0,len(cl_row)):
            # Transform rotated image coordinates to original row, column image space:
            if self.img_obj.dynamic_rotation_angle == None:
                # no need to rotate coordinates if there is no angle
                row_orig, col_orig = cl_row[idx], cl_col[idx]
            else:
                # otherwise, rotate coordiantes based on the assigned angle
                row_orig, col_orig = self.rotate_features_to_default_coord(cl_row[idx], cl_col[idx], -self.img_obj.dynamic_rotation_angle, image_coord=True)
            # append original coordinates to separate row/col points
            cl_row_orig.append(row_orig)
            cl_col_orig.append(col_orig)
       
        # Save cervical line row and column coordinates in two separate lists, for original coordinate system
        self.cervical_line_orig =  cl_row_orig, cl_col_orig
        
        # Find the portion of the cervical line over which the cervix is closed (to measure cervical length):
        closed_cl_row, closed_cl_col = [], []
        for idx, col in enumerate(cl_col_orig):
            if  col >= self.anat_internal_os_orig[1]:
                closed_cl_col.append(col)
                closed_cl_row.append(cl_row_orig[idx])
        self.closed_cervical_line_orig = closed_cl_row, closed_cl_col
                
        
        if debug == True:
            print(f"x distance between internal & external os = {self.external_os_orig[1] - self.hist_internal_os_orig[1]}")
        
        # Return row/col coordinate lists for the dynamic and original coordinate systems      
        return self.cervical_line_dynamic, self.cervical_line_orig, self.closed_cervical_line_dynamic, self.closed_cervical_line_orig
        
    # Return the cervical length (number of pixel of this class between internal/external os)
    def get_cervical_length(self, debug = False):
        """
        description:
            Integrate the cervical length by taking the distance between each successive set of (row,col) points, along the row dimension
        
        parameters:
            
        model constants:
        
        returns:
            @ length (float): cervical length counted in pixels
        """
        #list of row, col coordinates comprising the cervical canal line
        cl_rows, cl_cols = self.closed_cervical_line_orig
        
        # initialize empty list of x, y coordinates comprising the cervical canal line
        cl_x, cl_y = np.zeros_like(cl_rows), np.zeros_like(cl_cols)
        
        # fill list of x, y coordinates comprising the cervical canal line
        for idx in range(0,len(cl_rows)):
            cl_x[idx], cl_y[idx] = self.image_to_xy_coordinates(cl_rows[idx], cl_cols[idx], coordinates = 'original')
    
        #calculate distance between two points, using sum of squares:
        delta_x = (cl_x[1:] - cl_x[:-1]) ** 2
        delta_y = (cl_y[1:] - cl_y[:-1]) ** 2
        delta = delta_x + delta_y
        length = np.sum((delta) ** 0.5)
        
        if debug == True:
            print(f"cervical length = {length}")
            
        # Return cervical length as the sum of distance, between consecutive points
        return length
 
    # Given dynamic/rotated coordinate system, rotate the x/y or row/col coordinates to the static/original coordinate system:
    def rotate_features_to_default_coord(self, row_or_x, column_or_y, angle_rad, image_coord = True):
        """
        description:
            By default, this function assumes it is given image coordinates in (row, column) format and is used to rotate an image back to the original/static coordinate system.
            If image_coord = False, then the function then performs the rotation using cartesian coordinates in (x,y) format.
            This function rotates the given coordiantes (image or cartesian) by a specified angle (angle_rad) about the center of the image.
            It accounts for the expansion/translation of image dimensions during this process so that the same point can be tracked in both coordinate systems.
            
        parameters:
            @ row_or_x (int): if image_coord = True, a row coordinate to be transformed (if image_coord = False, an x-coordinate to be transformed)
            @ column_or_y (int): if image_coord = True, a column coordinate to be transformed (if image_coord = False, an y-coordinate to be transformed)
            @ angle_rad (float): the angle by which the coordinate system needs to be rotated, about the center of the image
            @ image_coord (boolean): default value True, this assumes that the coordinates are in row/column image coordiantes (False, assumes coordiantes are in x/y cartesian coordinates)
            
        model constants:
        
        returns:
            row_prime: returned if image_coord = True, the row coordinate of the point in the transformed coordinate system
            column_prime: returned if image_coord = True, the column coordinate of the point in the transformed coordinate system
            x_prime: returned if image_coord = False, the x-coordinate of the point in the transformed coordinate system
            y_prime: returned if image_coord = False, the y-coordinate of the point in the transformed coordinate system
        """
        
        
        if image_coord == True:
            #note: row index for y, column index for x
            x = column_or_y
            y = self.img_obj.img_dynamic.shape[0] - row_or_x
        elif image_coord == False:
            x = row_or_x
            y = column_or_y
            
        # Need to translate the points by how much the image shrinks/expands, before applying a rotation to restore original image coordinates
        # define center of image as the pivot point for rotation:
        x_center_orig = self.img_obj.img.shape[1]/2 - 1
        y_center_orig = self.img_obj.img.shape[0]/2 - 1
        x_center_rot = self.img_obj.img_dynamic.shape[1]/2 - 1
        y_center_rot = self.img_obj.img_dynamic.shape[0]/2 - 1
        # we now have the original and rotated center points
    
        # translate to (x,y) locations in original coordinate system 
        column_expansion = x_center_rot - x_center_orig
        row_expansion = y_center_rot - y_center_orig
        x_trans = x - column_expansion
        y_trans = y - row_expansion
        
        #get offset between point to be rotated(x,y) and center point(x_center, y_center) --> translate so that new center of rotation is around origin (0,0)
        #this is done in original coordinate system
        x_diff = x_trans - x_center_orig
        y_diff = y_trans - y_center_orig
        
        #calculate new graph coordinates x_prime,y_prime
        if angle_rad != None:
            x_prime = x_center_orig + np.cos(angle_rad)*x_diff - np.sin(angle_rad)*y_diff
            y_prime = y_center_orig + np.sin(angle_rad)*x_diff + np.cos(angle_rad)*y_diff
        else:
            x_prime = row_or_x
            y_prime = column_or_y
    
        #convert back to image coordinates if necessary
        if image_coord == True:
            if angle_rad != None:
                row_prime = self.img_obj.img.shape[0] - y_prime
                column_prime = x_prime
            else:
                column_prime = column_or_y
                row_prime = row_or_x
            return row_prime, column_prime
        
        elif image_coord == False:
            return x_prime, y_prime
    
    # Given static/original coordinate system, rotate the x/y or row/col coordinates to the dynamic/rotated coordinate system:
    def rotate_features_to_dynamic_coord(self, row_or_x, column_or_y, angle_rad, image_coord = True):
        """
        description:
            By default, this function assumes it is given image coordinates in (row, column) format, and is used to rotate an image back to the new dynamic/rotated coordinate system.
            It can also be fed x/y coordinates which skips the first step and the final conversion back to row/column.
            If image_coord = False, then the function then performs the rotation using cartesian coordinates in (x,y) format.
            This function rotates the given coordiantes (image or cartesian) by a specified angle (angle_rad) about the center of the image.
            It accounts for the expansion/translation of image dimensions during this process so that the same point can be tracked in both coordinate systems.
            
        parameters:
            @ row_or_x (int): if image_coord = True, a row coordinate to be transformed (if image_coord = False, an x-coordinate to be transformed)
            @ column_or_y (int): if image_coord = True, a column coordinate to be transformed (if image_coord = False, an y-coordinate to be transformed)
            @ angle_rad (float): the angle by which the coordinate system needs to be rotated, about the center of the image
            @ image_coord (boolean): default value True, this assumes that the coordinates are in row/column image coordiantes (False, assumes coordiantes are in x/y cartesian coordinates)
            
        model constants:
        
        returns:
            row_prime: returned if image_coord = True, the row coordinate of the point in the transformed coordinate system
            column_prime: returned if image_coord = True, the column coordinate of the point in the transformed coordinate system
            x_prime: returned if image_coord = False, the x-coordinate of the point in the transformed coordinate system
            y_prime: returned if image_coord = False, the y-coordinate of the point in the transformed coordinate system
        """
        
    
        if image_coord == True:
            #note: row index for y, column index for x
            x = column_or_y
            y = self.img_obj.img.shape[0] - row_or_x
        elif image_coord == False:
            x = row_or_x
            y = column_or_y
            
        # Need to translate the points by how much the image shrinks/expands, before applying a rotation to restore original image coordinates
        # define center of image as the pivot point for rotation:
        x_center_orig = self.img_obj.img.shape[1]/2 - 1
        y_center_orig = self.img_obj.img.shape[0]/2 - 1
        x_center_rot = self.img_obj.img_dynamic.shape[1]/2 - 1
        y_center_rot = self.img_obj.img_dynamic.shape[0]/2 - 1
    
        # translate to (x,y) locations in original coordinate system
        # note: this is the inverse of the rotate_default function:
        column_expansion = x_center_rot - x_center_orig
        row_expansion = y_center_rot - y_center_orig
        
        # translated corodinate system: add because we are removing
        x_trans = x + column_expansion
        y_trans = y + row_expansion
        
        # get offset between point to be rotated(x,y) and center point(x_center, y_center) --> translate so that new center of rotation is around origin (0,0)
        # this is done in original coordinate system
        x_diff = x_trans - x_center_rot #should be negative
        y_diff = y_trans - y_center_rot
        
        #calculate new graph coordinates x_prime,y_prime
        x_prime = x_center_rot + np.cos(angle_rad)*x_diff - np.sin(angle_rad)*y_diff
        y_prime = y_center_rot + np.sin(angle_rad)*x_diff + np.cos(angle_rad)*y_diff
  
        #convert back to image coordinates if necessary
        if image_coord == True:
            row_prime = self.img_obj.img_dynamic.shape[0] - y_prime
            column_prime = x_prime
            return row_prime, column_prime
        elif image_coord == False:
            return x_prime, y_prime
       
    # Convert from image to cartesian coordinate systems:
    def image_to_xy_coordinates(self, row, column, coordinates = 'original'):
        """
        description:
            Utility function used to convert from image (row,col) to cartesian (x,y) coordinate systems.
            This can be used either in the original/static or rotated/dynamic coordinate system.
            
        parameters:
            @ row (int): a row coordinate to be transformed to cartesian coordiantes
            @ column (int): a column coordinate to be transformed to cartesian coordiantes
            @ coordiantes (string): by default, this is set to 'original', but this transformation can also be done in the 'dynamic' coordinate system.
            
        model constants:
        
        returns:
            @ x (int): the x-coordinate of the point in the transformed coordinate system
            @ y (int): the y-coordinate of the point in the transformed coordinate system
        """
        
        x = column
        #assumes original image row/column coordinates:
        if coordinates == 'original':
            y = self.img_obj.img.shape[0] - row
        elif coordinates == 'dynamic':
            y = self.img_obj.img_dynamic.shape[0] - row
        return x,y

    # Convert from cartesian to image coordinate system
    def xy_to_image_coordinates(self,x,y,coordinates = 'original'):
        """
        description:
            Utility function used to convert from image (row,col) to cartesian (x,y) coordinate systems.
            This can be used either in the original/static or rotated/dynamic coordinate system.
            
        parameters:
            @ x (int): a x-coordinate to be transformed to cartesian coordiantes
            @ y (int): a y-coordinate to be transformed to cartesian coordiantes
            @ coordiantes (string): by default, this is set to 'original', but this transformation can also be done in the 'dynamic' coordinate system.
            
        model constants:
        
        returns:
            @ row (int): the row-coordinate of the point in the transformed coordinate system
            @ col (int): the col-coordinate of the point in the transformed coordinate system
        """
        
        column = x
        #assumes original image x/y coordinates:
        if coordinates == 'original':
            row = self.img_obj.img.shape[0] - y
        elif coordinates == 'dynamic':
            row = self.img_obj.img_dynamic.shape[0] - y
        return row, column
    
    # Plot the internal os points overlaid on the underlying ultrasound image or mask
    def plot_os(self, underlying = 'mask', coordinates = 'original'):
        """
        description:
            Plot the internal and external os location overlayed on the segmentation image.
            By default, this is done in the original coordinate system, but this can also be done in the rotated/dynamic coordinate system.
            This function will display the image with plotted points, but will not explicitly save the image or return any variables.
            
        parameters:
            @ underlying (string): 'image' or 'mask' indicates what underlying image will be displayed/saved underneath the circle overlay
            @ coordinates (string): 'original' or 'dynamic' defines which coordine system will be used to plot the int/ext os points
           
        model constants:
    
        returns:
        """
        # Plot the internal and external os in the original coordinate system
        if coordinates == 'original':
            ax = plt.gca()
            ax.set_axis_off()
            if underlying == 'image':
                plt.imshow(self.img_obj.USimg)
            elif underlying == 'mask':
                plt.imshow(self.img_obj.img)
            plt.scatter(self.hist_internal_os_orig[1],self.hist_internal_os_orig[0], s = 10, color = "gray")
            plt.scatter(self.anat_internal_os_orig[1],self.anat_internal_os_orig[0], s = 10, color = "white")
            plt.scatter(*self.external_os_orig[::-1], s = 10, color = "white")
            plt.show()
        
        # Plot the internal and external os in the dynamic coordinate system
        elif coordinates == 'dynamic':
            ax = plt.gca()
            ax.set_axis_off()
            if underlying == 'image':
                print("Error: check inputs to this function, it does not make sense to display a rotated ultrasound image")
            elif underlying == 'mask':
                plt.imshow(self.img_obj.img_dynamic)
            plt.scatter(self.hist_internal_os_dynamic[1],self.hist_internal_os_dynamic[0], s = 10, color = "gray")
            plt.scatter(self.anat_internal_os_dynamic[1],self.anat_internal_os_dynamic[0], s = 10, color = "white")
            plt.scatter(*self.external_os_dynamic[::-1], s = 10, color = "white")
            plt.show()
            
    # Plot the cervical line overlaid on the underlying ultrasound image or mask:
    def plot_cervical_line(self, underlying = 'mask', coordinates = 'original'):
        """
        description:
            Plot the cervical line overlayed on the original transvaginal ultrasound image or masked/segmentation image.
            This can be done in the original coordinate system (image or mask) or in the rotated/dynamic coordinate sytem (mask only).
        
        parameters:
            
        model constants:
        
        returns:
        """
        
        ax = plt.gca()
        ax.set_axis_off()
        # Plot the internal and external os in the original coordinate system
        if coordinates == 'original':
            cl_row_orig, cl_col_orig = self.closed_cervical_line_orig
            if underlying == 'image':
                plt.imshow(self.img_obj.USimg)
            elif underlying == 'mask':
                plt.imshow(self.img_obj.img)
            plt.plot(cl_col_orig, cl_row_orig, linewidth = 3, color = "white")
            plt.show()
        
        elif coordinates == 'dynamic':
            cl_row_dynamic, cl_col_dynamic = self.closed_cervical_line_dynamic
            if underlying == 'image':
                print("Error: check inputs to this function, it does not make sense to display a rotated ultrasound image")
            elif underlying == 'mask':
                plt.imshow(self.img_obj.img)
            plt.plot(cl_col_dynamic, cl_row_dynamic, linewidth = 3, color = "white")
            plt.show()
        
