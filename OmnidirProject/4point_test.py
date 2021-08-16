import numpy as np
import cv2
import csv
import glob
from depth_from_disparity_spherical import *
from depth_from_disparity_OCam import *
import pickle

pts = []

def draw_mask(event,x,y,flags,param):
    global pts
    if event == cv2.EVENT_LBUTTONUP:
        pts.append([x, y])

frontInFile = open(
    "C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\data\\calibration_parameters_OCV\\front_camera_params_OCV.xml",
    'rb')
frontCameras = pickle.load(frontInFile)
frontInFile.close()

# Load previous back camera caibration
backInFile = open(
    "C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\data\\calibration_parameters_OCV\\back_camera_params_OCV.xml",
    'rb')
backCameras = pickle.load(backInFile)
backInFile.close()

frontList = []
backList = []

for i in range(34):
    frontList.append('C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\Experiment\\disparity_maps\\OCV\\PSMNet\\front\\front_' + str(i) + '.bmp')

for i in range(33):
    backList.append('C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\Experiment\\disparity_maps\\OCV\\PSMNet\\back\\back_' + str(i) + '.bmp')
#backList = sorted(glob.glob("C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\Experiment\\disparity_maps\\OCV\\SGBM\\back\\back_*.bmp"))

tfImgList = sorted(glob.glob("C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\Experiment\\tf\\tf_*.bmp"))
tbImgList = sorted(glob.glob("C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\Experiment\\tb\\tb_*.bmp"))

cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_mask)

color = (0)
fields = ['Absolute Error', 'Standard Deviation', 'Distance to center']
rows = []

idx = 0
prev_idx = 0

while True:
    k = cv2.waitKey(20) & 0xFF
    if idx == prev_idx:
        try:
            original = cv2.imread(tbImgList[idx])
            img = cv2.imread(backList[idx])
            img = cv2.transpose(img)
            img = cv2.flip(img, 1)
            #img_depth = depth_from_disp_OCam(img, frontCameras.tvec, frontCameras.K1[0,0])
            img_depth = depth_from_disp_spherical(img, backCameras.tvec)
            idx += 1
        except IndexError:
            with open('PSMNet_OCV_back.csv', 'w', newline='') as f:
                write = csv.writer(f)
                write.writerow(fields)
                write.writerows(rows)
            break
    cv2.imshow('image', 2 * img)
    cv2.imshow("original", original)

    if np.shape(pts)[0] == 4:
        mask = np.zeros((640, 640), np.uint8) #400, 1280
        pts = np.asarray(pts)

        av_x = (pts[0, 0] + pts[1, 0] + pts[2, 0] + pts[3, 0]) / 4
        av_y = (pts[0, 1] + pts[1, 1] + pts[2, 1] + pts[3, 1]) / 4

        distance = np.sqrt((320 - av_x) ** 2) / 320 # + (320 - av_y) ** 2)

        pts = pts.reshape((1, 4, 2))
        mask = cv2.fillPoly(mask, pts, (255), 8)
        mean, std = cv2.meanStdDev(img_depth, mask=mask)
        error = np.absolute(3 - mean[0, 0])
        print("Absolute Error: ", error)
        print("Standard deviation", std[0, 0])
        print("Distance to center: ", distance)
        rows.append([error, std[0, 0], distance])
        pts = []
        prev_idx += 1

    if k == ord('s'):
        with open('test1.csv', 'w', newline='') as f:
            write = csv.writer(f)
            write.writerow(fields)
            write.writerows(rows)



