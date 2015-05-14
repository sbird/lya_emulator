"""Module which defines a variant of where which works on floating point"""

import numpy as np

def wheref(array, value):
    """ A where function to find where a floating point value is equal to another"""
    #Floating point inaccuracy.
    eps=1e-7
    return np.where((array > value-eps)*(array < value+eps))

