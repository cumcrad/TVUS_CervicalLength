"""
Original work began May 2022
@ author: Alicia Dagle
@ contributors: Gabriel Trigo, Madeline Skeel
"""

# Import pre-built python packages:
import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit
import matplotlib.image as mpimg

# Import custom scripts
from static_methods import linear_fnc
import executable_constants

class Img_Obj:
    
    def __init__(self, mask_path, raw_US_path, debug = False):
        print("Initializing Img_Obj...")
        self.mask_path = mask_path
        self.raw_US_path = raw_US_path
        self.USimg = mpimg.imread(self.raw_US_path)
        self.img = mpimg.imread(self.mask_path)
        self.img_dynamic = mpimg.imread(self.mask_path)
        self.cerv_canal_class = None
        self.cerv_canal_pixel_count = None
        self.cerv_canal_top_edge = None
        self.cerv_canal_bottom_edge = None
        self.hist_int_os_angle = None
        self.cervcanal_angle = None
        self.dynamic_rotation_angle = None # angle to rotate original image to dynamic coordinate system (determined by hist_int_os_angle or cervcanal_angle)
        self.display_img = None
        self.closed_cervix_points_orig = None
        self.closed_cervix_points_dyn = None

        
        # Label the closed cervical canal, using the original coordinate system: list of [row,col] points
        self.closed_cervix_points_orig = self.relabel_closed_cervical_canal(image_type = self.img) # Note: we may need to turn this off if using triple point intersection for internal os location
        
        # Attempt to rotate image, to align the superior-most boundary of the cervical canal + funnel feature (which we call the internal os line) vertically":
        try:
            print("trying to rotate image")
            x_leftbound, y_leftbound, popt = self.rotate_vertical_anatomical_internal_os_line()
            if debug == True:
                print(f"self.x_leftbound = {self.x_leftbound}")
                print(f"self.y_leftbound = {self.y_leftbound}")
        except:
            # If the rotation fails, return the original image as the dynamic or "rotated" image:
            print('rotate_vertical_anatomical_internal_os_line failed! Original image orientation will be returned as dynamic image')
            self.img_dynamic = self.img
        
        # Repopulate image attributes (after rotation) and add new attributes:
        self.repopulate_addnew_attributes()
        
    
    def identify_adjacent_cervix_tissue(self, image_type, row, column):
        """
        description:
            Identifiy points where the anterior and posterior cervix are in direct contact,
            any such adjacent points (from the anterior cervix class)
            will be relabeled as points along the cervical canal,
            forming the basis for the cervical lenth measurement.

        parameters:
            @ row (int): row index of the point under consideration
            @ column (int): column index of the point under consideration
            
        model constants:

        returns:
            @ bool (boolean): True if the point belongs
                to the artificial green class
        """
        
        #check if pixel 1 is anterior cervix tissue:
        if (image_type[row, column,:] == executable_constants.COLORS["pink"]).all():
            #check if below pixel is posterior cervix:
            if (row - 1 >= 0) and (image_type[row - 1, column,:] == executable_constants.COLORS["blue"]).all():
                #print("left case: closed cervix found")
                return True
        
            #otherwise, check if above pixel is posterior cervix: 
            elif (row + 1 < image_type.shape[0]) and (image_type[row + 1, column,:] == executable_constants.COLORS["blue"]).all():
                #print("right case: closed cervix found")
                return True
        
            #otherwise, check if left pixel is posterior cervix:    
            elif (column - 1 >= 0) and (image_type[row, column - 1,:] == executable_constants.COLORS["blue"]).all():
                #print("bottom case: closed cervix found")
                return True
        
            #otherwise, check if right pixel is posterior cervix   
            elif (column + 1 < image_type.shape[1]) and (image_type[row, column + 1,:] == executable_constants.COLORS["blue"]).all():
                #print("top case: closed cervix found")
                return True
            
        else:
            return False
    
    
    def relabel_closed_cervical_canal(self, image_type):
        """
        description:
            Create a single pixel thickness label (green) for adjacent anterior/posterior cervix tissue, this will make it easier to extract cervical length later:
            This is done be relabeling the anterior cervix class pixels (that are immediately adjacent to posterior cervix class pixels) to cervical canal class. 
        
        parameters:
            @ image_type (np.array): self.img or self.img_dynamic, array-like representation of image, in origina or rotated coordiante system
            
        model constants:

        returns:
            @ closed_cervix_points (list): all (row, col) points idnetified in the closed cervical line
        """
        
        print("Entering relabel_closed_cervical_canal() method")
        
        # Initialize emtpy list of all points (row,col) in the closed cervical line:
        closed_cervix_points = []
        for row in range(image_type.shape[0]):
            for column in range(image_type.shape[1]): 
                # Reassign pixel as (closed) cervical canal class type, and create list of all (row,col) points in the closed cervical line:
                if self.identify_adjacent_cervix_tissue(image_type, row, column):
                    self.img_dynamic[row, column,:] = executable_constants.COLORS["green"]
                    closed_cervix_points.append([row, column])
                    
        return closed_cervix_points


    def rotate_vertical_anatomical_internal_os_line(self, debug = False):
            """
            description:
                Fit the anatomical internal os line using a linear curve fit
                rotate the image to align the superior-most (left-most)
                edge with the vertical direction. Limit the maximum allowable rotation
                
            parameters: None
                
            model constants:
                @ executable_constants.RAD_TO_DEG : defined in executable_constants.py
                @ executable_constants.ROTATION_LIMIT : defined in executable_constants.py
                
            returns:
                @ X (np.array), @ Y (np.array): x and y coordinates of the points that make up the anatomical internal os line 
                @ popt (np.array): [intercept, slope] = optimal values of parameters from curve fitting superior-most boundary of cervical canal/funnel feature 
                
            """
            
            print("going to extract hist int os line...")
            
            # Generate array of X and Y points that comprise the superior most points of the cervical canal + funnel class:
            X, Y = self.extract_hist_internal_os_line()
            
            # Fit the superior most boundary of the cervical canal and funnel class to a line, return optimal values of parameters from curve fitting:
            popt, pcov = curve_fit(linear_fnc, X[:len(X) // 2], Y[:len(Y) // 2])
            perr = np.sqrt(np.diag(pcov))
            slope = popt[1]
            
            # Use inverse slope to find angle rotation necessary to align this with the vertical, rotate image accordingly:
            hist_int_os_angle = -np.arctan(1 / slope)
            # If the necessary rotation angle is below the predefined limit, rotate the image and save variables for dynamic (rotated) image, original displacement angle, and rotated angle:
            if np.absolute(executable_constants.RAD_TO_DEG*hist_int_os_angle) < executable_constants.ROTATION_LIMIT:
                self.img_dynamic = ndimage.rotate(self.img_dynamic, hist_int_os_angle * executable_constants.RAD_TO_DEG, order = 0)
                self.hist_int_os_angle = hist_int_os_angle
                self.dynamic_rotation_angle = self.hist_int_os_angle
                print(f"rotated image successfully to align hist int os, with angle = {executable_constants.RAD_TO_DEG*hist_int_os_angle}")
            # If angle is larger than allowable rotation, raise an error and do not define variables (self.img_dynamic, self.hist_int_os_angle, self.dynamic_rotation_angle)
            else:
                print(f"angle larger than {executable_constants.ROTATION_LIMIT} degrees, image was not rotated to align hist int os, calculated angle = {executable_constants.RAD_TO_DEG*hist_int_os_angle}")
                pass
                
            # Optional debugging step, to understand what is happening:
            if debug == True:
                print(f"X = {X}")
                print(f"Y = {Y}")
                print(f" popt = {popt}, pcov = {pcov}")
                print(f"perr = {perr}")
            
            # Return list of x and y coordinates along the superior most boundary of the cervical canal class (X,Y) and the linear fit slope and intercept (popt)
            return X, Y, popt
    
    
    def repopulate_addnew_attributes(self):
        """
        description:
            Repopulate the image atributes after image rotation (if applicable). Also fill new attributes:
                self.cerv_canal_pixel_count
                self.cerv_canal_top_edge
                self.cerv_canal_bottom_edge

        parameters:
            
        model constants:

        returns:

        """
        
        # Identify closed cervical points in the dynamic coordinate system, call the same method used before rotation but feed in dynamic image
        try:
            # NOTE: if image not rotated, then self.img = self.img_dynamic and values will be the same
            self.closed_cervix_points_dyn = self.relabel_closed_cervical_canal(image_type = self.img_dynamic)
        except:
            print("ERORR: identifying cervical canal class variable closed_cervix_points_dyn")
            pass
        
        # Identify all cervical canal class points in the dynamic image:
        try:
            array_cerv_canal_class_xy = np.empty((self.img_dynamic.shape[0], self.img_dynamic.shape[1]))
            for i in range(0,self.img_dynamic.shape[0]):
                for j in range(0,self.img_dynamic.shape[1]):
                    bool_cerv_canal = np.array_equal(self.img_dynamic[i, j, :], executable_constants.COLORS["green"])
                    array_cerv_canal_class_xy[i,j] = bool_cerv_canal

            self.cerv_canal_class = array_cerv_canal_class_xy
        except:
            print("ERORR: identifying cervical canal class variable self.cerv_canal_class")
            pass
        
        # Count the number of cervical canal class points in each column, along the image:    
        try:
            self.cerv_canal_pixel_count = self.count_green_points_per_column()
        except:
            print("ERORR: identifying cervical canal pixel count value")
            pass
        
        # Isolate the top edge of the cervical canal class
        try:
            self.cerv_canal_top_edge = self.extract_cervicalcanal_edge("top")
        except:
            pass
        
        # Isolate the bottom edge of the cervical canal class:
        try:
            self.cerv_canal_bottom_edge = self.extract_cervicalcanal_edge("bottom")
        except:
            pass
    
    
    
    def belongs_to_hist_int_os_line(self, row, column):
        """
        description:
            For all pixels in the class corresponding to the potential space between the histological
            internal os and the external os, this function checks if a point is part of the superior
            most portion, which forms a line connecting the anterior and posterior cervix tissue at
            the histological internal os.

        parameters:
            @ row (int): row index of the point being checked
            @ column (int): column index of the point being checked
        
        model constants:

        returns:
            @ hist_os_line_bool (bool): True if the point belongs to line connecting anterior/posterior
              cervix at internal os, False otherwise.
        """

        # Only consider pixels belonging to the class "potential space between the histological internal os and the external os" sometimes referred to by shorthand name "cervical canal class", denoted by green pixels
        if (self.img_dynamic[row,column,:] == executable_constants.COLORS["green"]).all():
            # If leftmost pixel (to pixel of interest) is background, return true
            if (self.img_dynamic[row, column - 1,:] == executable_constants.COLORS["black"]).all():
                # Note: do not include green pixel in border of image, if there is no left adjacent black pixel
                hist_os_line_bool = True
        
            else:
                hist_os_line_bool = False
    
        return hist_os_line_bool
    
    def extract_hist_internal_os_line(self, debug = False):
        """
        description:
            Extracts a line connecting the anterior to posterior cervix tissue at the histological internal os

        parameters:
            
        model constants:

        returns:
            @ x_edge_revised (np.array) @ y_edge_revised (np.array): x and y coordinates
            of the points that make up the histological internal os line
        """
        
        print("entered extract_hist_internal_os_line")
        
        # Intialize emtpy list for x and y coordinates on the edge of the "potential space between the histological internal os and the external os" class
        x_edge = []
        y_edge = []
        x_edge_revised = []
        y_edge_revised = []
        
        
        # Create a list of (x,y) points constituting the histological internal os line between anterior/posterior cervix:
        for row in range(self.img_dynamic.shape[0]):
            for column in range(self.img_dynamic.shape[1]):
                if self.belongs_to_hist_int_os_line(row, column):
                    x_edge.append(column)
                    y_edge.append(row)
    
        
        # Convert list to numpy array & and order with respect to x coordinate:
        x_edge = np.array(x_edge)
        y_edge = np.array(y_edge)
        unique_x = np.unique(x_edge)
        
        # Check if there are multiple pixels in a single column: If there are, check top and bottom pixel. If they are both black, then do not include this pixel
        
        # Iterate through all unique x values
        for x_elem in (unique_x):
            # Find the corresponding index in x_edge
            idx = np.where(x_edge == x_elem)[0]
            if debug == True:
                print(f"idx = {idx}")
                print(f"idx = {idx} for elem = {x_elem}")
                
            
            # Find instances of y corresponding to idx values:
            y_vals = []
            for idx_num in idx:
                y_vals.append(y_edge[idx_num])
                    
            # If there are multiple y elements for this single x value, take the min and max y values:    
            max_y = np.max(y_vals)
            min_y = np.min(y_vals)
            if debug == True:
                print(f"y_vals = {y_vals}")
                print(f"max_y = {max_y}, min_y = {min_y}")
            
            # Iterate through each y value at a given value x_elem:
            for y_val in y_vals:
                # Check if min value has black pixel below, or if max value has black pixel above (this would indicate it is not intersecting the anterior/posterior cervix):
                if not (self.img_dynamic[min_y - 1, x_elem,:] == executable_constants.COLORS["black"]).all() or not (self.img_dynamic[max_y + 1, x_elem,:] == executable_constants.COLORS["black"]).all():
                    x_edge_revised.append(x_elem)
                    y_edge_revised.append(y_val)
                    if debug == True:
                        print(f"There is not black below, it is: {self.img_dynamic[min_y - 1, x_elem,:]}")

        return x_edge_revised, y_edge_revised
          
               
    #def rotate_horizontal_adjacent_cervix_line(self, debug = False):
        """
        description:
            Fit the points of adjacent anterior and posterior cervix tissue (closed cervical canal) to a line using a linear curve fit,
            rotate the image to align the line in the horixontal direction.
            
            NOTE: This function was explored but ultimately not used in the cervical length feature extraction
            
        parameters: None
            
        model constants: None
            
        returns:
            @ X (np.array), @ Y (np.array): x and y coordinates of the points that make up the closed portion of the cervix
            @ popt (np.array): [intercept, slope] = optimal values of parameters from curve fitting the closed portion of the cervix
            
        """
        
        # Identify ajacent anterior/posterior cervix points in the dynamic coordinate system:
        print("going to extract adjacent cervical tissue line...")
        
        # Initialize empty list of X and Y coordinates:
        X = []
        Y = []
        
        # Iterate through all coordinated in the closed portion of the cervix:
        for coordinate in self.closed_cervix_points_orig:
            #row,col --> translate to x,y coordinates or fit inverse
            
            # Convert (row,col) to (x,y) coordinates --> translation modified from extractor_updated.Extractor_Obj.image_to_xy_coordinates() function:
            row, column = coordinate[0], coordinate[1]
            x = column
            #assumes original image row/column coordinates:
            y = self.img.shape[0] - row
            X.append(x)
            Y.append(y)
        
        # Fit the X and Y coordinates to a line, return optimal values of parameters from curve fitting:
        popt, pcov = curve_fit(linear_fnc, X[:len(X) // 2], Y[:len(Y) // 2])
        perr = np.sqrt(np.diag(pcov))
        slope = popt[1]
        
        if debug == True:
            print(f"X = {X}")
            print(f"Y = {Y}")
            print(f" popt = {popt}, pcov = {pcov}")
            print(f"perr = {perr}")
            print(f"slop = {slope}")
        
        
        
        # Use inverse slope to find angle rotation necessary from vertical line, rotate image accordingly:
        closed_cervical_line_angle = -np.arctan(slope)
        self.img_dynamic = ndimage.rotate(self.img_dynamic, closed_cervical_line_angle * executable_constants.RAD_TO_DEG, order = 0)
        self.cervcanal_angle = closed_cervical_line_angle
        self.dynamic_rotation_angle = self.cervcanal_angle
        print("rotated image successfully to align cervical canal (adjacent cervical tissue)")
        
        return X, Y, popt
    

    
    def belongs_to_cervicalcanal_edge(self, row, column):
        """
        description:
            Identifies points that belong to the edge of the cervical canal class

        parameters:
            @ row (int): row index of the point of interest
            @ column (int): column index of the point of interest
            
        model constants:

        returns:
            @ cerv_canal_edge_bool (boolean): True if the point has a green neighboor, False if not
        """
        
        # If row,column contains neither anterior/posterior cervix class, return False
        if (self.img_dynamic[row, column,:] == executable_constants.COLORS["pink"]).all():
            # Given anterior cervix label, check if next row is cervical canal class
            if (row - 1 >= 0) and (self.img_dynamic[row + 1, column,:] == executable_constants.COLORS["green"]).all():
                cerv_canal_edge_bool = True
            
            # Given anterior cervix label, check if previous column is cervical canal class
            elif (column - 1 >= 0) and (self.img_dynamic[row, column - 1] == executable_constants.COLORS["green"]).all():
                cerv_canal_edge_bool = True
            
            # Given anterior cervix label, check if next column is cervical canal class 
            elif (column + 1 < self.img_dynamic.shape[1]) and (self.img_dynamic[row, column + 1] == executable_constants.COLORS["green"]).all():
                cerv_canal_edge_bool = True
            
            # If no neighboring pixel is cervical canal class:
            else:
                cerv_canal_edge_bool = False
            
        if (self.img_dynamic[row, column,:] == executable_constants.COLORS["blue"]).all():
            # Given posterior cervix label, check if next row is cervical canal class
            if (row + 1 < self.img_dynamic.shape[0]) and (self.img_dynamic[row + 1, column,:] == executable_constants.COLORS["blue"]).all():
                cerv_canal_edge_bool = True
                            
            # Given posterior cervix label, check if previous column is cervical canal class
            elif (column - 1 >= 0) and (self.img_dynamic[row, column - 1] == executable_constants.COLORS["green"]).all():
                cerv_canal_edge_bool = True
            
            # Given posterior cervix label, check if next column is cervical canal class 
            elif (column + 1 < self.img_dynamic.shape[1]) and (self.img_dynamic[row, column + 1] == executable_constants.COLORS["green"]).all():
                cerv_canal_edge_bool = True
            
            # If no neighboring pixel is cervical canal class:
            else:
                cerv_canal_edge_bool = False
        
        # If no pixels are pink/blue representing anterior or posterior cervica canal class, return false:
        else:
            cerv_canal_edge_bool = False
    
        return cerv_canal_edge_bool
    
    
    def extract_cervicalcanal_edge(self, edge_type):
        """
        description:
            Extract the upper or bottom edge of the cervical canal class

        parameters:
            @ edge_type (str): "top" to extract the top edge,
                               "bottom" to extract the bottom edge
            
        model constants:

        returns:
            @ x_edge (np.array), @ y_edge(np.array): sorted arrays of the x and y coordinates of the edge points
        """
        
        # Define color_assignment variable to check based on provided edge_type:
        if edge_type == "top":
            color_assignment = executable_constants.COLORS["pink"]
        elif edge_type == "bottom":
            color_assignment = executable_constants.COLORS["blue"]       
        
        # Initalize empty list for x and y coordinates of cervical canal edge: 
        x_edge = []
        y_edge = []
        # Fill list of x and y coordinates for cervical canal edge, by checking 
        for row in range(self.img_dynamic.shape[0]):
            for column in range(self.img_dynamic.shape[1]):
                # Check if a given coordiante (row,col) belongs to cervical canal class and matches the color assignment:
                if self.belongs_to_cervicalcanal_edge(row, column) and (self.img_dynamic[row, column,:] == color_assignment).all():
                    # Fill list for x and y coordinates of cervical canal edge:
                    x_edge.append(column)
                    y_edge.append(row)
        
        # Convert list of x and y coordinates of cervical canal edge to numpy array:
        x_edge = np.array(x_edge)
        y_edge = np.array(y_edge)
        # Reorder array of x and y coordiantes of cervical canal edge, according to order of x coordinates:
        order = np.argsort(x_edge)
        x_edge = x_edge[order]
        y_edge = y_edge[order]
                    
        return x_edge, y_edge
    
    
    def count_green_points_per_column(self):
        """
        description:
            scans the image from left to right and counts the number of green 
            points per column

        parameters:
            
        model constants:

        returns:
            @ column_count (np.array): array with the count of green points 
                for each column
        """
        
        # Initialize empty list to count the number of potential space between histologinal internal os and external os (green) class points per column:
        column_count = []
        
        # For each column value in the cervical canal class, count the number of rows containing green pixels, and append count to column_count list:
        for column_index in range(self.cerv_canal_class.shape[1]):
            num_points_cerv_canal_percolumn = np.count_nonzero(self.cerv_canal_class[:, column_index])
            column_count.append(num_points_cerv_canal_percolumn)
            
        # Convert column_count list to numpy array:    
        column_count = np.array(column_count)
                               
        return column_count
    
    
    
    def has_cervcanal_neighbor(self, row, column):
        """
        description:
            Given (row, column) indices for a point of interest,
            check to see if the point has any (green) neighboors
            beloning to the "potential space between histological
            internal os and external os" class

        parameters:
            @ row (int): row index of the point of interest
            @ column (int): column index of the point of interest
            
        model constants:

        returns:
            @ cerv_canal_neighbor_bool (boolean): True if the point has a green neighboor, false if not
        """
        
        # Ignoring first row (because of boundary conditions), if previous green, set to True:
        if row - 1 >= 0:
            if (self.img_dynamic[row - 1, column,:] == executable_constants.COLORS["green"]).all():
                cerv_canal_neighbor_bool = True
          
        # Ignoring last row (because of boundary conditions), if next row is green, set to True:  
        if row + 1 < self.img_dynamic.shape[0]:
            if (self.img_dynamic[row + 1, column,:] == executable_constants.COLORS["green"]).all():
                cerv_canal_neighbor_bool = True
        
        # Ignoring first column (because of boundary conditions), if column to the left is green, set to True:
        if column - 1 >= 0:
            if (self.img_dynamic[row, column - 1] == executable_constants.COLORS["green"]).all():
                cerv_canal_neighbor_bool = True
        
        # Ignoring first column (because of boundary conditions), if column to the right is green, set to True:    
        if column + 1 < self.img_dynamic.shape[1]:
            if (self.img_dynamic[row, column + 1] == executable_constants.COLORS["green"]).all():
                cerv_canal_neighbor_bool = True
        
        # Otherwise, if the (row,column) point is a boundary coordinate or the orthogonal points are not green, set to False:
        else:
            cerv_canal_neighbor_bool = False
    
        return cerv_canal_neighbor_bool