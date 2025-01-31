"""
Original work began May 2022
@ author: Alicia Dagle
@ contributors: Gabriel Trigo, Madeline Skeel

This file contains standard functions, easily found online or recreated from mathematical equations. These are static methods which are called in image_utils.py and extractor.py
"""

import numpy as np

def linear_fnc(x, a, b):
    """
    description:
        standard linear function used for fitting
    parameters:
        @ x (float): dependent variable
        @ a (float): offset
        @ b (float): slope
    
    model constants:
    returns:
        @ y (float): y coordinate of the line
            y = a + b * x
    """
    return a + b * x

def moving_average(X, moving_avg_width):
    """
    description:
        calculates the moving average of an array with specified
        window size
    parameters:
        @ X (numpy array): array whose moving average ought to
            be calculated
        @ moving_avg_width (int): window size to be considered
    returns:
        @ moving_average (np.array): moving average array 
            for each column
    """
    moving_average = np.convolve(X, np.ones(moving_avg_width), "valid") / moving_avg_width
    return moving_average

def find_start_index(X, min_x):
    """
    description:
        given a sorted array X and a minimum value
        min_x, finds the first index of the array
        for which the array value is greater than
        threshold of min_x
    parameters:
        @ X (np.array): sorted array to be scanned
        @ min_x (float): we are looking for values greater
            than this
    
    model constants:
    returns:
        @ i (int): index at which the array values
            start being greater than min_x
    """
    bool_array = X > min_x
    i = np.nonzero(bool_array)[0][0]
    return i

def get_perpendicular_line(popt, x_intercept):
    """
    description:
        given parameters of a line and the desired
        x coordinate of interception, obtain the
        parameters of a perpendicular line that
        intercepts the line passed in at such point
    parameters:
        @ popt (tuple(float)): parameters of the original
            line (intercept 0 and slope 1)
        @ x_intercept (float): desired x coordinate of interception
        
    model constants:
    returns:
        @ a (float), @ b (float): intercept and angular
            coefficients of the perpendicular line
    """
    #if y-intercept is provided explictly, force line through this point
    m = -1 / popt[1]
    b = (popt[1] - m)*x_intercept + popt[0]

    return b, m