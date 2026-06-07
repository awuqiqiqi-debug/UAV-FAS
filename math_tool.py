import math
import cmath
import builtins
import numpy as np
import pandas as pd
from numpy import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# NumPy 2.x compat: numpy.max shadows builtins.max but has different semantics;
# restore the builtins so callers using `from math_tool import *` get correct max/min.
max = builtins.max
min = builtins.min

def cartesian_coordinate_to_spherical_coordinate(cartesian_coordinate):
    """
    transmit cartesian_coordinate_to_spherical_coordinate
    input 1 X 3 np.array,   [x, y, z]
    output 1 X 3 np.array,  [r, theta, fai]
    """
    r = np.linalg.norm(cartesian_coordinate)
    if abs(cartesian_coordinate[2]) < 1e-8:
        theta = math.atan(np.linalg.norm(cartesian_coordinate[0:2])/1e-8)
    else:
        theta = math.atan(np.linalg.norm(cartesian_coordinate[0:2])/cartesian_coordinate[2])
    
    if abs(cartesian_coordinate[0]) < 1e-8:
        x = 1e-8
    else:
        x = cartesian_coordinate[0]

    y = cartesian_coordinate[1]    
    if abs(y) < 1e-8:
        y = 1e-8

    if y > 0 and x > 0:
        fai = math.atan(y/x)
    elif x < 0 and y > 0:
        fai = math.atan(y/x) + math.pi
    elif x < 0 and y < 0:
        fai = math.atan(y/x) - math.pi
    else:
        fai = math.atan(y/x)
    return r, theta, fai

def vecter_normalization(cartesian_coordinate):
    return cartesian_coordinate/np.linalg.norm(cartesian_coordinate)

def get_coor_ref(coor_sys, coor):
    """
    input:  coor_sys: normalized 1,3 np.array list (1,3)
            coor: coordinate under earth system
    output: referenced coordinate for x,y, normalized 1,3 np.array
    """
    x_ref = np.dot(coor_sys[0],coor)
    y_ref = np.dot(coor_sys[1],coor)
    z_ref = np.dot(coor_sys[2],coor)
    return np.array([x_ref, y_ref, z_ref])

def dB_to_normal(dB):
    """
    input: dB
    output: normal vaule
    """
    return math.pow(10, (dB/10))

def normal_to_dB(normal):
    """
    input: normal
    output: dB value
    """
    return -10 * math.log10(normal)

def diag_to_vector(diag):
    """
    transfer a diagnal matrix into a vector
    """
    vec_size = np.shape(diag)[0]
    vector = np.asmatrix(np.zeros((vec_size, 1), dtype=complex), dtype=complex)
    for i in range(vec_size):
        vector[i, 0] = diag[i, i]
    return vector
    
def vector_to_diag(vector):
    """
    transfer a vector into a diagnal matrix
    """
    vec_size = np.shape(vector)[0]
    diag = np.asmatrix(np.zeros((vec_size, vec_size), dtype=complex), dtype=complex)
    for i in range(vec_size):
        diag[i, i] = vector[i, 0]
    return diag

def bigger_than_zero(value):
    """
    max(0,value)
    """
    return max(0, value)

def dataframe_to_dictionary(df):
    """
    docstring
    """
    return {col_name : df[col_name].values for col_name in df.columns.values}

def convert_list_to_complex_matrix(list_real, shape):
    """
    list_real is a 2* N*K dim list, convert it to N X K complex matrix
    shape is a tuple (N, K)
    """
    N = shape[0]
    K = shape[1]
    matrix_complex =np.asmatrix(np.zeros((N, K), dtype=complex), dtype=complex) 
    for i in range(N):
        for j in range(K):
            matrix_complex[i, j] = list_real[2*(i*K + j)] + 1j * list_real[2*(i*K + j) + 1]
            
    return matrix_complex

def convert_list_to_complex_diag(list_real, diag_row_num):
    """
    list_real is a M dim list, convert it to M X M complex diag matrix
    diag_row_num is the M
    ARIS: 幅度(1~10) + 相位(0~2π)
    """
    M = diag_row_num
    diag_matrix_complex = np.asmatrix(np.zeros((M, M), dtype=complex), dtype=complex)
    for i in range(M):
        # ARIS: 将输入值映射到幅度[1,10]和相位[0,2π]
        # 前半部分控制幅度，后半部分控制相位
        amplitude = 1.0 + 9.0 * (0.5 + 0.5 * math.sin(list_real[i]))  # 映射到[1,10]
        phase = list_real[i] * math.pi  # 相位
        diag_matrix_complex[i, i] = amplitude * cmath.exp(1j * phase)
    return diag_matrix_complex

def map_to(x, x_range:tuple, y_range:tuple):
    x_min = x_range[0]
    x_max = x_range[1]
    y_min = y_range[0]
    y_max = y_range[1]
    y = y_min+(y_max - y_min) / (x_max - x_min) * (x - x_min)
    return y

