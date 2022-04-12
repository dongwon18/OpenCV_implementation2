---
finished_date: 2021-11-25
tags:
    - OpenCV
    - Numpy
    - Python
    - 2D_transformation
    - hamming_distance
    - homography
    - RANSAC
    - iamge_stitching
    - panorama
---
# OpenCV library implementation2
- implement OpenCV functions with Numpy
- get proper tranformation matrix with implemented *get_mat*
- apply transformation with *get_transformed_image*  

- get ORB features from built-in OpenCV function
- get haming distance for each keypoint with *hamming_distance*
- get normalized homography, RANSAC homography and apply homography with built-in OpenCV function

## 2D transformation
- transform *./images/smile.png* using 3 x3 afine transformation matrix M
    - get the matrix by *get_mat*
    - apply the matrix to the image by *get_transformed_image()*
- get user's keyboard input 
    - a: move left by 5 pixel
    - b: move right by 5 pixel
    - w: move up by 5 pixel
    - s: move down by 5 pixel
    - r: rotate counter-clockwise by 5 degree
    - R: rotate clickwise by 5 degree
    - f: flip across y axis
    - F: flip across x axis
    - x: shrink the size by 5% along x direction
    - X: enlarge the size by 5% along x direction
    - y: shrink the size by 5% along y direction
    - Y: enlarge the size by 5% along y direction
    - H: return to original image
    - Q: quit
- show the result in 801 x 801 2D plane

## homography
- read *./images/cv_desk.png, ./images/cv_cover.png* 
- get ORB feature using OpenCV function(*ORB_create(), detect*)
- calculate hamming distance and ifnd top 10 matched pair
- display the pair using OpenCV fuunction *cv2.drawMatches*

### Normalized Homography
- using matching feature points, get 3 x 3 homography transformation matrix
    - normalize points of src images and dst images
    - compute homography using built in function *numpy.linalg.svd*
### RANSAC Homography
- get RANSAC homography transformation matrix

### warping
- wrap *cv_cover.jpg* to *cv_desk.png* using *cv2.wrapPerspective()*
- wrap *hp_cover.jpg* to *cv_desk.png*

### image stitching
- get RANSAC homography of *diamondhead-10.png* and *diamondhead-11.png*
- stitch those images so that make a panorma image.
- set gradation effect for blending
    - set blend range and blend gap
    - set pixel value as weighted mean(weighted with distance between start and end of range)

## File structure
```
|-- src
    |-- 2d_transformations.py
    |-- homography.py
|-- images
    |-- cv_cover.jpg
    |-- cv_desk.png
    |-- diamondhead-10.png
    |-- diamondhead-11.png
    |-- hp_cover.jpg
    |-- smile.png
|-- result
    |-- desk & cover match.png
    |-- diamond blending.png
    |-- diamond matching.png
    |-- ransac cover.png
    |-- ransac hp.png
    |-- normalization homography.png
```

## Result
<p align=center>
    <img src="./result/desk & cover match.png" alt="desk & cover matching points"><br/>
    desk & cover matching points
</p>
<p align=center>
    <img src="./result/diamond matching.png" alt="diamond matching"><br/>
    diamondhead matching points
</p>
<p align=center>
    <img src="./result/diamond blending.png" alt="diamond blending"><br/>
    diamondhead panorama blending
</p>

## 배운 점
- transformation matrix를 구하여 원하는 대로 이미지를 연속적으로 변환할 수 있다.
- matching point를 구하여 homography를 구할 수 있다. 특히 ransac을 이용한 image wraping은 built in function과 거의 유사한 결과를 냈다.
- Numpy 사용이 능숙해졌다.

## 한계점
- normalization을 이용하여 matching을 할 때 matching point 중 하나가 alias point 여서 이상한 결과가 나온다.(이는 OpenCV built in 함수를 이용하였을 때도 동일하였다.) 이런 alias를 없애고 진행해보려고 하였으나 alias point를 설정하는 기준을 명확히 찾지 못했다.
<p align=center>
    <img src="./result/normalization homography.png" alt="desk & cover matching points"><br/>
    desk & cover matching points
</p>  

- image stitching시 붙여진 그림의 오른쪽 빈부분을 없앨 수 없었다. 사이즈를 특정하여 잘라 보려고 하였지만 구현에 실패하였다.
