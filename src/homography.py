'''
# Copyright 2021. Dongwon Kim All rights reserved.
#
# File name : homography.py
#
# Written by Dongwon Kim
#
# CV assignment2
#   normalized homography, RANSAC homography, panorama
#
# Modificatoin history
#   written by Dongwon Kim on Nov 25, 2021
#
# OpenCV version: 4.5.3
# using virtual environment with Anaconda
#
# for normalized homography, cv2.BFMatcher also gives strange result
'''
import cv2
import numpy as np
import time

CV_COVER = "../images/cv_cover.jpg"
CV_DESK = "../images/cv_desk.png"
HP_COVER = '../images/hp_cover.jpg'
DIAMOND1 = "../images/diamondhead-10.png"
DIAMOND2 = "../images/diamondhead-11.png"

cover = cv2.imread(CV_COVER, cv2.IMREAD_GRAYSCALE)
desk = cv2.imread(CV_DESK, cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create()
cover_kp = orb.detect(cover, None)
cover_kp, cover_desc = orb.compute(cover, cover_kp)
desk_kp = orb.detect(desk, None)
desk_kp, desk_desc = orb.compute(desk, desk_kp)

'''
get hamming distance for each keypoints
for the cover, desk, keypoint: (500, 32)
compare 500 keypoints
result = (500, 500) distance for each keypoint
'''
def hamming_distance(a, b):
    distance = np.zeros([a.shape[0], a.shape[0]])
    if(a.shape != b.shape):
        print("<hamming_distance>shape of the inputs should be the same")
        return None
    
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            for k in range(b.shape[1]):
                distance[i][j] += bin(a[i][k] ^ b[j][k]).count('1')
    return distance

distance = hamming_distance(cover_desc, desk_desc)

def sorting(array):
    array_index = np.argmin(array, axis = 1) # get min dis index for each index of a
    top_match=[]
    for i in range(len(array)):
        dmatch = cv2.DMatch(_imgIdx = 0, _queryIdx = i, _trainIdx = array_index[i], _distance=array[i][array_index[i]])
        top_match.append(dmatch)
    
    top_match = sorted(top_match, key = lambda x:x.distance)
    return top_match

matches = sorting(distance)
res = cv2.drawMatches(cover, cover_kp, desk, desk_kp, matches[:10], None, flags=2)
cv2.imshow('matching point', res)


def compute_homography(srcP, destP):
    n = len(srcP)
    avg_src = srcP
    avg_dest = destP
    mat = []
    for i in range(n):        
        x1 = avg_src[i][0][0]
        y1 = avg_src[i][0][1]
        x2 = avg_dest[i][0][0]
        y2 = avg_dest[i][0][1]
        temp = [-x1, -y1, -1, 0, 0, 0, x1*x2, y1*x2, x2]
        mat.append(temp)
        temp = [0, 0, 0, -x1, -y1, -1, x1*y2, y1*y2, y2]
        mat.append(temp)
    A = np.matrix(mat)
    u, s, v = np.linalg.svd(A)
    h = np.reshape(v[8], (3, 3)) # smallest singular value locates at last(8) in s
    h = (1/h.item(8)) * h
    return h


def normalize(array):
    avg = np.average(array, axis = 0)
    avg_norm = array - avg  # mean -> origin
    maxi = 0
    for location in avg_norm:
        x = location[0][0]
        y = location[0][1]
        sum = (x**2 + y**2)**0.5 / 2**0.5
        if(maxi < sum):
            maxi = sum
    
    if maxi != 0:
        avg_norm /= maxi
    
    return avg_norm

src_point = np.array([cover_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2).astype(np.float32)
dst_point = np.array([desk_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2).astype(np.float32)

normalize_h = compute_homography(src_point, dst_point)
result = cv2.warpPerspective(cover, normalize_h, (desk.shape[1], desk.shape[0])) # (width, height)
cv2.imshow('<normalize> cover & desk warp', result)
composed_result = desk.copy()
for i in range(result.shape[0]):
    for j in range(result.shape[1]):
        if(result[i][j] != 0 ):
            composed_result[i][j] = result[i][j]

cv2.imshow('<normalize> cover & desk composed', composed_result)

def compute_homography_ransac(srcP, destP, th):
    np.random.seed(37)
    best_h = np.zeros([3,3])
    max_inlier = 0
    total_iter = 7000
    for i in range(total_iter):
        inlier = 0
        corr = np.random.choice(srcP.shape[0], 4)
        src = srcP[corr]
        dst = destP[corr]
        h = compute_homography(src, dst)
        for j in range(4):
            error = abs(dst[j][0][0] - (src[j][0][0] * h[0, 0] + src[j][0][1] * h[0, 1] + h[0, 2]))
            if error < th:
                inlier += 1
            error = abs(dst[j][0][1] - (src[j][0][0] * h[1, 0] + src[j][0][1] * h[1, 1] + h[1, 2]))
 
            if error < th:
                inlier += 1
        
        if max_inlier < inlier:
            max_inlier = inlier
            best_h = h
    return best_h

startTime = time.perf_counter()
ransac_h = compute_homography_ransac(src_point, dst_point, 20.0)
endTime = time.perf_counter()
print("<RANSAC homography computing time:desk>: {0:.4f}sec".format(endTime - startTime))

result2 = cv2.warpPerspective(cover, ransac_h, (desk.shape[1], desk.shape[0]))
cv2.imshow('<RANSAC> cover & desk warp', result2)
composed_result2 = desk.copy()
for i in range(result2.shape[0]):
    for j in range(result2.shape[1]):
        if(result2[i][j] != 0):
            composed_result2[i][j] = result2[i][j]

cv2.imshow('<RANSAC> cover & desk composed', composed_result2)

hp = cv2.imread(HP_COVER, cv2.IMREAD_GRAYSCALE)
result_hp = cv2.warpPerspective(hp, ransac_h, (desk.shape[1], desk.shape[0]))
cv2.imshow('hp & desk warp', result_hp)
composed_hp = desk.copy()
for i in range(result_hp.shape[0]):
    for j in range(result_hp.shape[1]):
        if(result2[i][j] != 0):
            composed_hp[i][j] = result_hp[i][j]
cv2.imshow('<RANSAC> hp & desk composed', composed_hp)


diamond1 = cv2.imread(DIAMOND1, cv2.IMREAD_GRAYSCALE)
diamond2 = cv2.imread(DIAMOND2, cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create()
dia1_kp = orb.detect(diamond1, None)
dia1_kp, dia1_desc = orb.compute(diamond1, dia1_kp)
dia2_kp = orb.detect(diamond2, None)
dia2_kp, dia2_desc = orb.compute(diamond2, dia2_kp)
dia_distane = hamming_distance(dia1_desc, dia2_desc)
dia_matches = sorting(dia_distane)


dia1_point = np.array([dia1_kp[m.queryIdx].pt for m in dia_matches]).reshape(-1, 1, 2).astype(np.float32)
dia2_point = np.array([dia2_kp[m.trainIdx].pt for m in dia_matches]).reshape(-1, 1, 2).astype(np.float32)
startTime = time.perf_counter()
dia_h = compute_homography_ransac(dia2_point, dia1_point, 20.0)
endTime = time.perf_counter()
print("<RANSAC homography computing time: diamond>: {0:.4f}sec".format(endTime - startTime))
result_dia = cv2.warpPerspective(diamond2, dia_h, (diamond1.shape[1] + diamond2.shape[1], diamond2.shape[0]))

result_dia[:diamond1.shape[0], :diamond1.shape[1]] = diamond1

cv2.imshow('diamond panorama', result_dia)

h = diamond1.shape[0]
w = diamond1.shape[1]
result_dia2 = np.zeros(result_dia.shape)
result_dia2 = result_dia.copy()

blend_range = 30
blend = 5

for i in range(h):
    for j in range(blend, blend_range, blend):
        lower = result_dia[i, w-j//2]
        upper = result_dia[i, w+j//2]
        for k in range(blend):
            result_dia2[i, w-j//2 + k] = lower * (1- k / blend) + upper * k / blend

for i in range(h):
    for j in range(blend, blend_range, blend):
        lower = result_dia2[i, w-j//2]
        upper = result_dia2[i, w+j//2]
        for k in range(blend):
            result_dia2[i, w-j//2 + k] = lower * (1- k / blend) + upper * k / blend

cv2.imshow('diamond blending(press any key to close all windows)', result_dia2)
cv2.waitKey(0)
cv2.destroyAllWindows()

