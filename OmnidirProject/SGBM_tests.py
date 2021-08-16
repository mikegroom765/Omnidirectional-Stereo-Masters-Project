import cv2 as cv
import math
import numpy as np
from PSMNet.models import *
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

def on_trackbar_set_disparities(value):
    sgbm.setNumDisparities(max(16, value * 16))

def on_trackbar_set_blocksize(value):
    if not(value % 2):
        value = value + 1
    sgbm.setBlockSize(max(3, value))

def on_trackbar_set_speckle_range(value):
    sgbm.setSpeckleRange(value)

def on_trackbar_set_speckle_window(value):
    sgbm.setSpeckleWindowSize(value)

def on_trackbar_set_setDisp12MaxDiff(value):
    sgbm.setDisp12MaxDiff(value)

def on_trackbar_set_setP1(value):
    sgbm.setP1(value)

def on_trackbar_set_setP2(value):
    sgbm.setP2(value)

def on_trackbar_set_setPreFilterCap(value):
    sgbm.setPreFilterCap(value)

def on_trackbar_set_setUniquenessRatio(value):
    sgbm.setUniquenessRatio(value)

def on_trackbar_set_wlsLmbda(value):
    wls_filter.setLambda(value)

def on_trackbar_set_wlsSigmaColor(value):
    wls_filter.setSigmaColor(value * 0.1)

def on_trackbar_null(value):
    return

def test(imgL, imgR):

    imgL = imgL.cuda()
    imgR = imgR.cuda()

    with torch.no_grad():
        disp = model(imgL, imgR)

    disp = torch.squeeze(disp)
    pred_disp = disp.data.cpu().numpy()

    return pred_disp

model_path = "C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\PSMNet\\pretrained_model_KITTI2015.tar"
model = stackhourglass(192)
model = nn.DataParallel(model, device_ids=[0])
model.cuda()
state_dict = torch.load(model_path)
model.load_state_dict(state_dict['state_dict'])
model.eval()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

infer_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

sgbm = cv.StereoSGBM_create(minDisparity=0,
                                    numDisparities=16 * 8,
                                    blockSize=3,
                                    P1=4 * 3 * 3,
                                    P2=32 * 3 * 3,
                                    disp12MaxDiff=1,
                                    preFilterCap=63,
                                    uniquenessRatio=10,
                                    speckleWindowSize=100,
                                    speckleRange=32)

left_matcher = sgbm
right_matcher = cv.ximgproc.createRightMatcher(left_matcher)

wls_lmbda = 800
wls_sigma = 1.2

wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(wls_lmbda)
wls_filter.setSigmaColor(wls_sigma)

#tf_img = cv.imread("C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\Experiment\\RectifiedImages\\ImprovedOCamCalib\\Scale_factor_5.0\\tf\\tf_0.bmp")
#bf_img = cv.imread("C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\Experiment\\RectifiedImages\\ImprovedOCamCalib\\Scale_factor_5.0\\bf\\bf_0.bmp")

#tf_img = cv.imread("C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\Experiment\\RectifiedImages\\Omnidir\\tb\\tb_0.bmp")
#bf_img = cv.imread("C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\Experiment\\RectifiedImages\\Omnidir\\bb\\bb_0.bmp")

tf_img = cv.imread("C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\PSMNet\\left.png")
bf_img = cv.imread("C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\PSMNet\\right.png")


height, width, channels = bf_img.shape

tf_img = tf_img[0:height, 0:int(width/2)]
bf_img = bf_img[0:height, 0:int(width/2)]

cv.imshow("tf", tf_img)
cv.imshow("bf", bf_img)

cv.namedWindow('controls', cv.WINDOW_NORMAL)
cv.createTrackbar("Max Disparity(x 16): ", 'controls', int(128 / 16), 16,
                 on_trackbar_set_disparities)
cv.createTrackbar("Window Size: ", 'controls', 21, 50, on_trackbar_set_blocksize)
cv.createTrackbar("Speckle Window: ", 'controls', 0, 200, on_trackbar_set_speckle_window)
cv.createTrackbar("LR Disparity Check Diff:", 'controls', 0, 25, on_trackbar_set_setDisp12MaxDiff)
cv.createTrackbar("Disparity Smoothness P1: ", 'controls', 0, 4000, on_trackbar_set_setP1)
cv.createTrackbar("Disparity Smoothness P2: ", 'controls', 0, 16000, on_trackbar_set_setP2)
cv.createTrackbar("Pre-filter Sobel-x- cap: ", 'controls', 0, 5, on_trackbar_set_setPreFilterCap)
cv.createTrackbar("Winning Match Cost Margin %: ", 'controls', 0, 20, on_trackbar_set_setUniquenessRatio)
cv.createTrackbar("Speckle Size: ", 'controls', math.floor((width * height) * 0.0005), 10000,
                 on_trackbar_null)
cv.createTrackbar("Max Speckle Diff: ", 'controls', 16, 2048, on_trackbar_null)
cv.createTrackbar("WLS Filter Lambda: ", 'controls', wls_lmbda, 10000, on_trackbar_set_wlsLmbda)
cv.createTrackbar("WLS Filter Sigma Color (x 0.1): ", 'controls', math.ceil(wls_sigma / 0.1), 50, on_trackbar_set_wlsSigmaColor)

img1 = cv.cvtColor(tf_img, cv.COLOR_BGR2RGB)
img2 = cv.cvtColor(bf_img, cv.COLOR_BGR2RGB)
img1 = infer_transform(img1)
img2 = infer_transform(img2)

if img1.shape[1] % 16 != 0:
    times = img1.shape[1] // 16
    top_pad = (times + 1) * 16 - img1.shape[1]
else:
    top_pad = 0

if img1.shape[2] % 16 != 0:
    times = img1.shape[2] // 16
    right_pad = (times + 1) * 16 - img1.shape[2]

else:
    right_pad = 0

img1 = F.pad(img1, (0, right_pad, top_pad, 0)).unsqueeze(0)
img2 = F.pad(img2, (0, right_pad, top_pad, 0)).unsqueeze(0)

pred_disp = test(img1, img2)
img = (pred_disp * (256 / 192)).astype('uint8')

grayL = cv.cvtColor(tf_img, cv.COLOR_BGR2GRAY)
grayR = cv.cvtColor(bf_img, cv.COLOR_BGR2GRAY)

#grayL = np.power(grayL, 0.75).astype('uint8')
#grayR = np.power(grayR, 0.75).astype('uint8')

while True:

    disparity_front_sgbm = sgbm.compute(tf_img, bf_img)
    _, disparity_front_sgbm = cv.threshold(disparity_front_sgbm, 0, 128, cv.THRESH_TOZERO)
    disparity_scaled_front_sgbm = (disparity_front_sgbm / 16.).astype(np.uint8)
    disparity_to_display_sgbm = (disparity_scaled_front_sgbm * (256. / 128)).astype(np.uint8)

    displ = left_matcher.compute(cv.UMat(grayL), cv.UMat(grayR))# .astype(np.int16) # tf_img, bf_img
    dispr = right_matcher.compute(cv.UMat(grayR), cv.UMat(grayL))# .astype(np.int16)
    displ = np.int16(cv.UMat.get(displ))
    dispr = np.int16(cv.UMat.get(dispr))
    disparity_front_wls = wls_filter.filter(displ, grayL, None, dispr)

    _, disparity_front_wls = cv.threshold(disparity_front_wls, 0, 128, cv.THRESH_TOZERO)
    disparity_scaled = (disparity_front_wls / 16.).astype(np.uint8)
    disparity_to_display_wls = (disparity_scaled * (256. / 128)).astype(np.uint8)

    cv.imshow("sgbm", disparity_to_display_sgbm)
    cv.imshow("sgbm wls", disparity_to_display_wls)
    cv.imshow("PSM for display", img)
    cv.imshow("PSM", pred_disp.astype(np.uint8))


    cv.waitKey(10000)