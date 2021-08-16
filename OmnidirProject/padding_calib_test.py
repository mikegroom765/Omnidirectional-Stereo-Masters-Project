import glob
from detectChessboardCorners import detectChessboardConrner
from stereocam import *
from OCamCalib_model import *
from perspective_undistortion_LUT_OCamCalib import *
import time
import cv2 as cv
import numpy as np

sf = 5

tf_OCam = OCamCalib_model()
tf_OCam.readOCamFile("C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\data\\calibration_parameters_OCamCalib\\calib_results_tf.txt")

#  img = cv.imread("C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\data\\calibration_images\\top_front\\tf_0.bmp")
img = cv.imread("C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\Experiment\\tf\\tf_0.bmp")

cv.imshow("original", img)

pad = 1920

h, w, c = np.shape(img)

result = np.full((pad, pad, c), (0, 0, 0), dtype=np.uint8)
result1 = np.full((1080, 1080, c), (0, 0, 0), dtype=np.uint8)

pad_h = (pad - h) // 2
pad_w = (pad - w) // 2

result[pad_h:pad_h+h, pad_w:pad_w+w] = img

pad_h = pad_h // 2
pad_w = pad_w // 2

result1[pad_h:pad_h+h, pad_w:pad_w+w] = img

cv.imshow("pad", result)

tf_mapx, tf_mapy = perspective_undistortion_LUT(tf_OCam, sf, 640, 640, 0, 0)
pad_mapx, pad_mapy = perspective_undistortion_LUT(tf_OCam, sf, 640, 640, 1280, 1280)
pad1_mapx, pad1_mapy = perspective_undistortion_LUT(tf_OCam, sf, 640, 640, 640, 640)

a = time.time()
undistorted_image = cv.remap(img, tf_mapx, tf_mapy, cv.INTER_LINEAR)
b = time.time()
undistorted_image_pad = cv.remap(result, pad_mapx, pad_mapy, cv.INTER_LINEAR)
c = time.time()
undistorted_image_pad1 = cv.remap(result1, pad1_mapx, pad1_mapy, cv.INTER_LINEAR)
d = time.time()

print("remap no pad:", str(b-a))
print("remap pad 1280 ", str(c-b))
print("remap pad 640: ", str(d-c))

cv.imshow("undistorted_image", undistorted_image)
cv.imshow("undistorted_image_pad", undistorted_image_pad)
cv.imshow("undistorted_image_pad1", undistorted_image_pad1)



cv.waitKey(100000)

