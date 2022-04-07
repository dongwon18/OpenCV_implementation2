'''
# Copyright 2021. Dongwon Kim All rights reserved.
#
# File name : 2d_transformations.py
#
# Written by Dongwon Kim
#
# CV assignment2
#   2D transformation: move, rotate, flip, scale
#
# Modificatoin history
#   written by Dongwon Kim on Nov 25, 2021
#
# OpenCV version: 4.5.3
# using virtual environment with Anaconda
'''

# TODO: change keyboards for uppercase
import cv2
import numpy as np
import math

SMILE_PATH = "../images/smile.png"
PLANE_WIDTH = 801
PLANE_HEIGHT = 801
ORIX = 400
ORIY = 400


def get_transformed_image(img, M):
    new_img = np.full([PLANE_WIDTH, PLANE_HEIGHT], 255)
    
    height = img.shape[0]
    width = img.shape[1]
    cen_x = width // 2
    cen_y = height // 2
    dx = cen_x * M[0, 0] + cen_y * M[0, 1]
    dy = cen_x * M[1, 0] + cen_y * M[1, 1]
    
    dx = 400 - dx
    dy = 400 - dy
    
    for i in range(height):
        for j in range(width):
            x = j * M[0, 0] + i * M[0, 1] + M[0, 2]
            x = round(x + dx)
            y = j * M[1, 0] + i * M[1, 1] + M[1, 2]
            y = round(y + dy)
            if(0 < x < PLANE_WIDTH) and (0 < y < PLANE_HEIGHT):
                new_img[y][x] = img[i][j]
 
    return new_img
    
    
def draw_arraw(img):
    new_img = img.copy()
    new_img = cv2.arrowedLine(new_img, (0, ORIY), (PLANE_WIDTH, ORIY), (0, 0, 0), tipLength=0.05)
    new_img = cv2.arrowedLine(new_img, (ORIX, PLANE_HEIGHT), (ORIX, 0), (0, 0, 0), tipLength=0.05)
    return new_img

def get_mat(mode, x, y):
    mat = np.zeros([3, 3])
    if(mode == "trans"):
        mat = np.array([[1, 0, x], [0, 1, y], [0, 0, 1]])  
     
    # x: degree, y: not used
    elif(mode == "rotate"):
        angle = math.radians(x)
        cos = math.cos(angle)
        sin = math.sin(angle)
        mat = np.array([[cos, -1*sin, 0], [sin, cos, 0], [0, 0, 1]])
  
    # x: fip dir(1: flip across x axis, 0: y axis), y: not used
    elif(mode == "flip"):
        if(x == 1):
            mat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        else:
            mat = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])    
    # scal x times, y times    
    elif(mode == "scale"):
        mat = np.array([[x, 0, 0], [0, y, 0], [0, 0, 1]])   
    else:
        print("Improper mode: {}".format(mode))    
        
    return mat    

def translate_input(key):
    mat = np.zeros([3,3])
    flag = 0
    if key == ord('a'):
        mat = get_mat("trans", -5, 0)
    elif key == ord('d'):
        mat = get_mat("trans", 5, 0)
    elif key == ord('w'):
        mat = get_mat("trans", 0, -5)
    elif key == ord('s'):
        mat = get_mat("trans", 0, 5)
    elif key == ord('r'):
        mat = get_mat("rotate", -5, 0)
    elif key == ord('R'):
    #elif key == ord('u'):
        mat = get_mat("rotate", 5, 0)
    elif key == ord('f'):
        mat = get_mat("flip", 0, 0)
    elif key == ord('F'):
    #elif key == ord('i'):
        mat = get_mat("flip", 1, 0)
    elif key == ord('x'):
        mat = get_mat("scale", 0.95, 1)
    elif key == ord('X'):
    #elif key == ord('o'):
        mat = get_mat("scale", 1.05, 1)
    elif key == ord('y'):
        mat = get_mat("scale", 1, 0.95)
    elif key == ord('Y'):
    #elif key == ord('p'):
        mat = get_mat("scale", 1, 1.05)
    else:
        print("The key is not specified")
        mat = None
        flag = 1
    return mat, flag


img = cv2.imread(SMILE_PATH, cv2.IMREAD_GRAYSCALE)
cv2.imshow('original', img)

M = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
result = get_transformed_image(img, M)
final = np.array(draw_arraw(result), dtype = np.uint8)
cv2.imshow('original', final)

while 1:    
    key = cv2.waitKey(0)
    print(key)
    if key == ord('Q'):
    #if key == ord('l'):
        break
    elif key == ord('H'):
    #elif key == ord('k'):
        result = get_transformed_image(img, M)
    else:
        mat, flag = translate_input(key)
        if(flag != 1):
            result = get_transformed_image(result, mat)
    final = np.array(draw_arraw(result), dtype = np.uint8)
    cv2.imshow(chr(key), final)

cv2.destroyAllWindows()