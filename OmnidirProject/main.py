import cv2 as cv
from detectChessboardCorners import detectChessboardConrner
from calib_img_gen import calib_img_gen
from stereocam import *
from Size import *
from OCamCalib_model import *
from perspective_undistortion_LUT_OCamCalib import *
from omnidirCalib import *
from improvedOCamCalib import *
import numpy as np
import glob
import pickle
import sys
import time
import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from PSMNet.models import *

# Calibration Parameters
calib_flags = 0
# Creates two objects of class StereoCam for omnidir calibration
frontCameras = StereoCam()
backCameras = StereoCam()

# Creates a object called boardSize of class Size (width, height)
boardSize = Size(8, 6)# small bboard is 5, 4  big board is 12, 8, new board is 9 x 7, kaiwen is 9 x 6

# Defaults board square sizes (mm)
square_width = 80.8 # small board is 37, big board is 28.6, new board is 51.5, kaiwen is 30
square_height = 80.8

# This is used for both omnidir and ImprovedOCamCalib calibrations since we do the calibrations on the same images!
calibImgDir = "C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\data\\calibration_images"

# Directories for OCV omnidir calibration
OCVcalibDir = "C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\data\\calibration_parameters_OCV"
OCVcalibFrontOut = "\\front_camera_params_OCV.xml"
OCVcalibBackOut = "\\back_camera_params_OCV.xml"

# Directories for ImprovedOCamCalib calibration
OCamCalibDir = "C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\data\\calibration_parameters_OCamCalib"
OCamCalibFrontOut = "\\front_camera_params_OCam.xml"
OCamCalibBackOut = "\\back_camera_params_OCam.xml"
OCamCalibTopFront = "\\calib_results_tf.txt"
OCamCalibTopBack = "\\calib_results_tb.txt"
OCamCalibBotFront = "\\calib_results_bf.txt"
OCamCalibBotBack = "\\calib_results_bb.txt"

# Image format for calibration image generation
imgFormat = ".bmp"

# Calibration image prefixes
imgTF = "\\top_front\\tf_"
imgTB = "\\top_back\\tb_"
imgBF = "\\bottom_front\\bf_"
imgBB = "\\bottom_back\\bb_"

# Run-Time Flags
CALIBRATION = bool(0)
IMG_GEN = bool(0)
SHOW_CALIB_IMG = bool(1)
CAMERA_STREAM = bool(0)
USE_SGBM = bool(1)
POINT_CLOUD = bool(0)
USE_GPS = bool(0)
USE_PSEUDO = bool(0)
MEDIA_OUT = bool(0)
LOG_FPS = bool(0)
IMPROVEDOCAMCALIB = bool(0)
OCV_OMNIDIR_CALIB = bool(0)
OCV_SGBM = bool(0)
OCV_CUDA_SGM = bool(0)
PSM = bool(0)
HSM = bool(0)
TRACKBAR = bool(0)
LEFTRIGHTLEFT = bool(0)

# Scale_factor for ImprovedOCamCalib calibration and stream

scale_factor = 5.0

# Trackbar functions from Prof. Toby Breckon's zed stereo GitHub

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

# Method for computing disparity maps using PSMNet
def PSM_disp(imgL, imgR):

    imgL = imgL.cuda()
    imgR = imgR.cuda()

    with torch.no_grad():
        disp = model(imgL, imgR)

    disp = torch.squeeze(disp)
    pred_disp = disp.data.cpu().numpy()

    return pred_disp

# Menu

Parameters = input("Please select an option(s): \n -g - Generate Calibration Images \n -c - Calibrate cameras (OpenCV omnidir module or ImprovedOCamCalib) \n -s - Stream Real Time Disparity Map \nPlease select an option: ")
if "-g" in Parameters:
    IMG_GEN = 1
elif "-c" in Parameters:

    calib_menu_flag = 0
    while calib_menu_flag == 0:
        calib_method = input("For OpenCV omnidir please select: o\nFor ImprovedOCamCalib please select: i \nWhich calibration technique would you like to use? ")
        if calib_method == "i":
            IMPROVEDOCAMCALIB = 1
            OCV_OMNIDIR_CALIB = 0
            calib_menu_flag = 1
        if calib_method == "o":
            OCV_OMNIDIR_CALIB = 1
            IMPROVEDOCAMCALIB = 0
            calib_menu_flag = 1

    key = input("Would you like to change calibration parameters? (y/n): ")
    while not CALIBRATION:
        if key == "y":
            width_int = False
            height_int = False
            while not width_int:
                width = input("Please enter the board width (# of inner corners, must be a integer!): ")
                try:
                    boardSize.width = int(width)
                    if boardSize.width > 0:
                        width_int = True
                    else:
                        print("This value must be a positive integer!")
                except ValueError:
                    print("This value must be a positive integer!")
            while not height_int:
                height = input("Please enter the board height (# of inner corners, must be a integer!")
                try:
                    boardSize.height = int(height)
                    if boardSize.height > 0:
                        height_int = True
                    else:
                        print("This value must be a positive integer!")
                except ValueError:
                    print("This value must be a positive integer!")
            CALIBRATION = 1
        elif key == "n":
            CALIBRATION = 1
        else:
            key = input("Please choose y or n!")
            #  print("\n Example: -w <board_width> -h <board_height> -sw <square_width> -sh <square_height> [-fs] [-fp] [-m]")
            #  calib_parameters = input("\nConfigurations: \n-w: width of board (number of inner corners) \n-h: height of board (number of inner corners) \n-sw: square width size \n-sh: square height size \n-fp: fix principal point (i.e. use image center) \n-fs: fix skew (i.e. no skew) \n-m: manual check of automatic detected corners \nPlease select an option: ")

elif "-s" in Parameters:
    CAMERA_STREAM = 1
    stream_flag = 0
    while stream_flag == 0:
        calib_method = input("For OpenCV omnidir please select: o\nFor ImprovedOCamCalib please select: i \nWhich calibration technique would you like to use? ")
        if calib_method == "i":
            IMPROVEDOCAMCALIB = 1
            OCV_OMNIDIR_CALIB = 0
            stream_flag = 1
        if calib_method == "o":
            OCV_OMNIDIR_CALIB = 1
            IMPROVEDOCAMCALIB = 0
            stream_flag = 1

    stream_flag = 0
    while stream_flag == 0:
        stereo_method = input("Please choose a stereo correspondence algorithm:\n - OpenCV SGBM: sgbm\n - OpenCV CUDA SGM: cuda\n - Heirarchical Stereo Matching (HSM): hsm\n - Pyramid Stereo Matching Network (PSMNet): psm\nPlease select an option: ")
        if stereo_method == "sgbm":
            OCV_SGBM = 1
            trackbar_flag = 1
            while trackbar_flag:
                trackbar = input("Show SGBM controls? (y/n): ")
                if trackbar == "y":
                    TRACKBAR = 1
                    trackbar_flag = 0
                if trackbar == "n":
                    trackbar_flag = 0
            while stream_flag == 0:
                left_right_left_flag = input("Left-Right Right-Left with WLS filtering? (y/n): ")
                if left_right_left_flag == 'y':
                    LEFTRIGHTLEFT = 1
                    stream_flag = 1
                if left_right_left_flag == 'n':
                    stream_flag = 1
        if stereo_method == "cuda":
            OCV_CUDA_SGM = 1
            stream_flag = 1
        if stereo_method == "hsm":
            HSM = 1
            stream_flag = 1
        if stereo_method == "psm":
            PSM = 1
            stream_flag = 1

if IMG_GEN:
    calib_img_gen(calibImgDir, imgTF, imgTB, imgBF, imgBB, imgFormat, OCamCalibDir, OCamCalibTopFront, OCamCalibTopBack,
                  OCamCalibBotFront, OCamCalibBotBack, scale_factor)

if CALIBRATION and OCV_OMNIDIR_CALIB:

    omnidirCalib(calibImgDir, imgFormat, boardSize, imgTF, imgTB, imgBF, imgBB, square_width, square_height, SHOW_CALIB_IMG)

    #  frontCameras, backCameras = omnidirCalib(...)

    # Save front camera object as a .xml file
    #frontOutFile = open(OCVcalibDir + OCVcalibFrontOut, 'wb')
    #pickle.dump(frontCameras, frontOutFile)
    #frontOutFile.close()
    # Save back camera object as a .xml file
   #backOutFile = open(OCVcalibDir + OCVcalibBackOut, 'wb')
    #pickle.dump(backCameras, backOutFile)
   # backOutFile.close()

if CALIBRATION and IMPROVEDOCAMCALIB:

    improvedOCamCalib(calibImgDir, imgFormat, boardSize, imgTF, imgTB, imgBF, imgBB, \
                                                  square_width, square_height, SHOW_CALIB_IMG, OCamCalibDir, \
                                                  OCamCalibTopFront, OCamCalibTopBack, OCamCalibBotFront, \
                                                  OCamCalibBotBack, scale_factor)

    #  frontCameras, backCameras = improvedOCamCalib(...)

    # Save front camera object as a .xml file
    #frontOutFile = open(OCamCalibDir + OCamCalibFrontOut, 'wb')
    #pickle.dump(frontCameras, frontOutFile)
    #frontOutFile.close()
    # Save back camera object as a .xml file
    #backOutFile = open(OCamCalibDir + OCamCalibBackOut, 'wb')
    #pickle.dump(backCameras, backOutFile)
    #backOutFile.close()

if not CALIBRATION:
    if OCV_OMNIDIR_CALIB:

        # Load previous front camera calibration
        frontInFile = open(OCVcalibDir + OCVcalibFrontOut, 'rb')
        frontCameras = pickle.load(frontInFile)
        frontInFile.close()
        # Load previous back camera caibration
        backInFile = open(OCVcalibDir + OCVcalibBackOut, 'rb')
        backCameras = pickle.load(backInFile)
        backInFile.close()


    if IMPROVEDOCAMCALIB:

        # Load previous front camera calibration
        frontInFile = open(OCamCalibDir + OCamCalibFrontOut, 'rb')
        frontCameras = pickle.load(frontInFile)
        frontInFile.close()
        # Load previous back camera calibration
        backInFile = open(OCamCalibDir + OCamCalibBackOut, 'rb')
        backCameras = pickle.load(backInFile)
        backInFile.close()

if CAMERA_STREAM and OCV_OMNIDIR_CALIB:
    
    top_stream = cv.VideoCapture(0)
    bot_stream = cv.VideoCapture(2)

    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    #out = cv.VideoWriter('disparity_test.avi', fourcc, 20.0, (640,640), 0)

    if not top_stream.isOpened():
        sys.exit("ERROR: Unable to open top_stream")
    if not bot_stream.isOpened():
        sys.exit("ERROR: Unable to open bot_stream")

    flipkey = 0
    while flipkey == 0:
        ret_top, top_frame = top_stream.read()
        ret_bot, bot_frame = bot_stream.read()

        msg = "Flip Cameras? (y/n)"
        textOrigin = (10, 50)
        top_frame_with_text = top_frame
        bot_frame_with_text = bot_frame
        cv.putText(top_frame_with_text, msg, textOrigin, cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))
        cv.putText(bot_frame_with_text, msg, textOrigin, cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))

        cv.imshow("Top", top_frame_with_text)
        cv.imshow("Bottom", bot_frame_with_text)
        flipkey = cv.waitKey(0)

        if flipkey == ord('n'):
            cv.destroyWindow("Top")
            cv.destroyWindow("Bottom")
            break

        if flipkey == ord('y'):
            top_stream.release()
            bot_stream.release()
            top_cap = cv.VideoCapture(4)
            bot_cap = cv.VideoCapture(2)
            cv.destroyWindow("Top")
            cv.destroyWindow("Bottom")
            break

    height, width, channels = bot_frame.shape
    bot_front = bot_frame[0:int(height - 80), 0:int(width / 2)]
    bot_back = bot_frame[0:int(height - 80), int(width / 2):width]
    top_front = top_frame[0:int(height - 80), 0:int(width / 2)]
    top_back = top_frame[0:int(height - 80), int(width / 2):width]

    image_center = (320, 320)

    tf_rot_mat = cv.getRotationMatrix2D(image_center, -110, 1.0)  # -90
    bf_rot_mat = cv.getRotationMatrix2D(image_center, 70, 1.0)  # 90
    tb_rot_mat = cv.getRotationMatrix2D(image_center, 110, 1.0)  # 90
    bb_rot_mat = cv.getRotationMatrix2D(image_center, -70, 1.0)  # -90

    knew = np.array([[640 / (np.pi * 19 / 18), 0, 0], [0, 640 / (np.pi * 19 / 18), 0], [0, 0, 1]], np.double)

    R1F, R2F = cv.omnidir.stereoRectify(frontCameras.rvec, frontCameras.tvec)
    R1B, R2B = cv.omnidir.stereoRectify(backCameras.rvec, backCameras.tvec)

    mapTF1, mapTF2 = cv.omnidir.initUndistortRectifyMap(frontCameras.K1, frontCameras.D1, frontCameras.xiT,
                                                        R1F, knew, (640, 640), cv.CV_32FC1,
                                                        cv.omnidir.RECTIFY_LONGLATI) # width, height, cv.CV_16SC2
    mapBF1, mapBF2 = cv.omnidir.initUndistortRectifyMap(frontCameras.K2, frontCameras.D2, frontCameras.xiB,
                                                        R2F, knew, (640, 640), cv.CV_32FC1,
                                                        cv.omnidir.RECTIFY_LONGLATI)
    mapTB1, mapTB2 = cv.omnidir.initUndistortRectifyMap(backCameras.K1, backCameras.D1, backCameras.xiT,
                                                        R1B, knew, (640, 640), cv.CV_32FC1,
                                                        cv.omnidir.RECTIFY_LONGLATI)
    mapBB1, mapBB2 = cv.omnidir.initUndistortRectifyMap(backCameras.K2, backCameras.D2, backCameras.xiB,
                                                        R2B, knew, (640, 640), cv.CV_32FC1,
                                                        cv.omnidir.RECTIFY_LONGLATI)

    cuda_mapTF1 = cv.cuda_GpuMat(mapTF1.astype(np.float32))
    cuda_mapTF2 = cv.cuda_GpuMat(mapTF2.astype(np.float32))
    cuda_mapBF1 = cv.cuda_GpuMat(mapBF1.astype(np.float32))
    cuda_mapBF2 = cv.cuda_GpuMat(mapBF2.astype(np.float32))
    cuda_mapTB1 = cv.cuda_GpuMat(mapTB1.astype(np.float32))
    cuda_mapTB2 = cv.cuda_GpuMat(mapTB2.astype(np.float32))
    cuda_mapBB1 = cv.cuda_GpuMat(mapBB1.astype(np.float32))
    cuda_mapBB2 = cv.cuda_GpuMat(mapBB2.astype(np.float32))

    stream_key = 0

    if OCV_SGBM:

        sgbm = cv.StereoSGBM_create(minDisparity=0,
                                    numDisparities=16 * 8,
                                    blockSize=3,
                                    P1=8 * 3 * 3 * 3,
                                    P2=64 * 3 * 3 * 3,
                                    disp12MaxDiff=160,
                                    preFilterCap=32,
                                    uniquenessRatio=8,
                                    speckleWindowSize=200,
                                    speckleRange=2)

        if LEFTRIGHTLEFT:

            left_matcher = sgbm
            right_matcher = cv.ximgproc.createRightMatcher(left_matcher)

            wls_lmbda = 800
            wls_sigma = 1.2

            wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
            wls_filter.setLambda(wls_lmbda)
            wls_filter.setSigmaColor(wls_sigma)

    if OCV_CUDA_SGM:

        sgbm_cuda = cv.cuda.createStereoSGM(minDisparity=0,
                                    numDisparities=128,
                                    P1=8 * 3 * 3 * 3,
                                    P2=64 * 3 * 3 * 3,
                                    uniquenessRatio=8)

    if PSM:
        # padding will need to be added to image if a new image resolution is used
        model_path = "C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\PSMNet\\pretrained_model_KITTI2015.tar"
        model = stackhourglass(192)
        model = nn.DataParallel(model, device_ids=[0])
        model.cuda()
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict['state_dict'])
        model.eval()

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        infer_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=mean, std=std)])


    gpu_left_frame_front = cv.cuda_GpuMat()
    gpu_right_frame_front = cv.cuda_GpuMat()
    gpu_left_frame_back = cv.cuda_GpuMat()
    gpu_right_frame_back = cv.cuda_GpuMat()


    if TRACKBAR:

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
        if LEFTRIGHTLEFT:
            cv.createTrackbar("WLS Filter Lambda: ", 'controls', wls_lmbda, 10000, on_trackbar_set_wlsLmbda)
            cv.createTrackbar("WLS Filter Sigma Color (x 0.1): ", 'controls', math.ceil(wls_sigma / 0.1), 50, on_trackbar_set_wlsSigmaColor)


    prev_frame_time = 0
    new_frame_time = 0
    avg_fps = 0
    fps_counter = 0

    while stream_key == 0:
        ret_top, top_frame = top_stream.read()
        ret_bot, bot_frame = bot_stream.read()

        if not top_stream.isOpened():
            sys.exit("ERROR: Unable to open top_stream")
        if not bot_stream.isOpened():
            sys.exit("ERROR: Unable to open bot_stream")


        height, width, channels = bot_frame.shape
        bot_front = bot_frame[0:int(height - 80), 0:int(width / 2)]
        bot_back = bot_frame[0:int(height - 80), int(width / 2):width]
        top_front = top_frame[0:int(height - 80), 0:int(width / 2)]
        top_back = top_frame[0:int(height - 80), int(width / 2):width]

        gpu_left_frame_back.upload(top_back)
        gpu_right_frame_back.upload(bot_back)
        gpu_left_frame_front.upload(top_front)
        gpu_right_frame_front.upload(bot_front)

        gpu_left_frame_front = cv.cuda.warpAffine(gpu_left_frame_front, tf_rot_mat, (height-80, int(width/2)), flags=cv.INTER_LINEAR)
        gpu_right_frame_front = cv.cuda.warpAffine(gpu_right_frame_front, bf_rot_mat, (height - 80, int(width / 2)), flags=cv.INTER_LINEAR)
        gpu_left_frame_back = cv.cuda.warpAffine(gpu_left_frame_back, tb_rot_mat, (height - 80, int(width / 2)), flags=cv.INTER_LINEAR)
        gpu_right_frame_back = cv.cuda.warpAffine(gpu_right_frame_back, bb_rot_mat, (height - 80, int(width / 2)), flags=cv.INTER_LINEAR)

        gpu_left_frame_front = cv.cuda.remap(gpu_left_frame_front, cuda_mapTF1, cuda_mapTF2, interpolation=cv.INTER_LINEAR, borderMode = cv.BORDER_CONSTANT)
        gpu_left_frame_back = cv.cuda.remap(gpu_left_frame_back, cuda_mapTB1, cuda_mapTB2, interpolation=cv.INTER_LINEAR, borderMode = cv.BORDER_CONSTANT)
        gpu_right_frame_front = cv.cuda.remap(gpu_right_frame_front, cuda_mapBF1, cuda_mapBF2, interpolation=cv.INTER_LINEAR, borderMode = cv.BORDER_CONSTANT)
        gpu_right_frame_back = cv.cuda.remap(gpu_right_frame_back, cuda_mapBB1, cuda_mapBB2, interpolation=cv.INTER_LINEAR, borderMode = cv.BORDER_CONSTANT)

        if OCV_SGBM:

            if LEFTRIGHTLEFT:

                rect_top_back = gpu_left_frame_back.download()
                rect_bot_back = gpu_right_frame_back.download()
                displ_back = left_matcher.compute(cv.UMat(rect_top_back), cv.UMat(rect_bot_back))
                dispr_back = right_matcher.compute(cv.UMat(rect_bot_back), cv.UMat(rect_top_back))
                displ_back = np.int16(cv.UMat.get(displ_back))
                dispr_back = np.int16(cv.UMat.get(dispr_back))
                disparity_back = wls_filter.filter(displ_back, rect_top_back, None, dispr_back)
                _, disparity_back = cv.threshold(disparity_back, 0, 128, cv.THRESH_TOZERO)
                disparity_back = (disparity_back / 16.).astype(np.uint8)

                rect_top_front = gpu_left_frame_front.download()
                rect_bot_front = gpu_right_frame_front.download()
                displ_front = left_matcher.compute(cv.UMat(rect_top_front), cv.UMat(rect_bot_front))
                dispr_front = right_matcher.compute(cv.UMat(rect_bot_front), cv.UMat(rect_top_front))
                displ_front = np.int16(cv.UMat.get(displ_front))
                dispr_front = np.int16(cv.UMat.get(dispr_front))
                disparity_front = wls_filter.filter(displ_front, rect_top_front, None, dispr_front)
                _, disparity_front = cv.threshold(disparity_front, 0, 128, cv.THRESH_TOZERO)
                disparity_front = (disparity_front / 16.).astype(np.uint8)

            else:

                rect_top_back = gpu_left_frame_back.download()
                rect_bot_back = gpu_right_frame_back.download()
                disparity_back = sgbm.compute(rect_top_back, rect_bot_back).astype(np.float32) / 16.0

                rect_top_front = gpu_left_frame_front.download()
                rect_bot_front = gpu_right_frame_front.download()
                disparity_front = sgbm.compute(rect_top_front, rect_bot_front).astype(np.float32) / 16.0

        if OCV_CUDA_SGM:

            gpu_left_frame_back = cv.cuda.cvtColor(gpu_left_frame_back, cv.COLOR_BGR2GRAY)
            gpu_right_frame_back = cv.cuda.cvtColor(gpu_right_frame_back, cv.COLOR_BGR2GRAY)
            disparity_back = sgbm_cuda.compute(gpu_left_frame_back, gpu_right_frame_back)
            disparity_back = disparity_back.download().astype(np.int16) / 16.0

            gpu_left_frame_front = cv.cuda.cvtColor(gpu_left_frame_front, cv.COLOR_BGR2GRAY)
            gpu_right_frame_front = cv.cuda.cvtColor(gpu_right_frame_front, cv.COLOR_BGR2GRAY)
            disparity_front = sgbm_cuda.compute(gpu_left_frame_front, gpu_right_frame_front)
            disparity_front = disparity_front.download().astype(np.int16) / 16.0

            # cv.filterSpeckles(dispB, 0, 40, 128)

        if PSM:

            rect_top_back = gpu_left_frame_back.download()
            rect_bot_back = gpu_right_frame_back.download()
            rect_top_back = infer_transform(rect_top_back).unsqueeze(0)
            rect_bot_back = infer_transform(rect_bot_back).unsqueeze(0)
            disparity_back = (PSM_disp(rect_top_back, rect_bot_back) * 256).astype('uint16')

            rect_top_front = gpu_left_frame_front.download()
            rect_bot_front = gpu_right_frame_front.download()
            rect_top_front = infer_transform(rect_top_front).unsqueeze(0)
            rect_bot_front = infer_transform(rect_bot_front).unsqueeze(0)
            disparity_front = (PSM_disp(rect_top_front, rect_bot_front) * 256).astype('uint16')

        cv.imshow('Disparity Back', np.uint8(disparity_back))
        cv.imshow('Disparty Front', np.uint8(disparity_front))

        new_frame_time = time.time()
        fps = 1/(new_frame_time - prev_frame_time)
        avg_fps = ((fps_counter * avg_fps) + fps)/(fps_counter + 1)
        fps_counter += 1
        prev_frame_time = new_frame_time
        print(avg_fps)

        #out.write(np.uint8(disparity_front))
        if cv.waitKey(1) == ord('q'):
            #out.release()
            top_stream.release()
            bot_stream.release()
            break

if CAMERA_STREAM and IMPROVEDOCAMCALIB:

    top_stream = cv.VideoCapture(0)
    bot_stream = cv.VideoCapture(2)

    ret_top, top_frame = top_stream.read()
    ret_bot, bot_frame = bot_stream.read()

    if not top_stream.isOpened():
        sys.exit("ERROR: Unable to open top_stream")
    if not bot_stream.isOpened():
        sys.exit("ERROR: Unable to open bot_stream")

    flipkey = 0
    while flipkey == 0:
        ret_top, top_frame = top_stream.read()
        ret_bot, bot_frame = bot_stream.read()

        msg = "Flip Cameras? (y/n)"
        textOrigin = (10, 50)
        top_frame_with_text = top_frame
        bot_frame_with_text = bot_frame
        cv.putText(top_frame_with_text, msg, textOrigin, cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))
        cv.putText(bot_frame_with_text, msg, textOrigin, cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))

        cv.imshow("Top", top_frame_with_text)
        cv.imshow("Bottom", bot_frame_with_text)
        flipkey = cv.waitKey(0)

        if flipkey == ord('n'):
            cv.destroyWindow("Top")
            cv.destroyWindow("Bottom")
            break

        if flipkey == ord('y'):
            top_stream.release()
            bot_stream.release()
            top_cap = cv.VideoCapture(2)
            bot_cap = cv.VideoCapture(4)
            cv.destroyWindow("Top")
            cv.destroyWindow("Bottom")
            break

    height, width, channels = bot_frame.shape
    bot_front = bot_frame[0:int(height - 80), 0:int(width / 2)]
    bot_back = bot_frame[0:int(height - 80), int(width / 2):width]
    top_front = top_frame[0:int(height - 80), 0:int(width / 2)]
    top_back = top_frame[0:int(height - 80), int(width / 2):width]

    OCam_top_front = OCamCalib_model()
    OCam_top_back = OCamCalib_model()
    OCam_bot_front = OCamCalib_model()
    OCam_bot_back = OCamCalib_model()

    OCam_top_front.readOCamFile(OCamCalibDir + OCamCalibTopFront)
    OCam_top_back.readOCamFile(OCamCalibDir + OCamCalibTopBack)
    OCam_bot_front.readOCamFile(OCamCalibDir + OCamCalibBotFront)
    OCam_bot_back.readOCamFile(OCamCalibDir + OCamCalibBotBack)

    tf_mapx, tf_mapy = perspective_undistortion_LUT(OCam_top_front, scale_factor, 640, 640, 640, 640)
    tb_mapx, tb_mapy = perspective_undistortion_LUT(OCam_top_back, scale_factor, 640, 640, 640, 640)
    bf_mapx, bf_mapy = perspective_undistortion_LUT(OCam_bot_front, scale_factor, 640, 640, 640, 640)
    bb_mapx, bb_mapy = perspective_undistortion_LUT(OCam_bot_back, scale_factor, 640, 640, 640, 640)

    cuda_tf_mapx = cv.cuda_GpuMat(tf_mapx.astype(np.float32))
    cuda_tf_mapy = cv.cuda_GpuMat(tf_mapy.astype(np.float32))
    cuda_tb_mapx = cv.cuda_GpuMat(tb_mapx.astype(np.float32))
    cuda_tb_mapy = cv.cuda_GpuMat(tb_mapy.astype(np.float32))
    cuda_bf_mapx = cv.cuda_GpuMat(bf_mapx.astype(np.float32))
    cuda_bf_mapy = cv.cuda_GpuMat(bf_mapy.astype(np.float32))
    cuda_bb_mapx = cv.cuda_GpuMat(bb_mapx.astype(np.float32))
    cuda_bb_mapy = cv.cuda_GpuMat(bb_mapy.astype(np.float32))

    image_center = (640, 640)

    tf_rot_mat = cv.getRotationMatrix2D(image_center, -110, 1.0)  # -90
    bf_rot_mat = cv.getRotationMatrix2D(image_center, 70, 1.0)  # 90
    tb_rot_mat = cv.getRotationMatrix2D(image_center, 110, 1.0)  # 90
    bb_rot_mat = cv.getRotationMatrix2D(image_center, -70, 1.0)  # -90

    R1F, R2F, P1F, P2F, QF, validPix1F, validPix2F = cv.stereoRectify(frontCameras.K1, frontCameras.D1, frontCameras.K2,
                                                                      frontCameras.D2, (1280, 1280), frontCameras.rvec,
                                                                      frontCameras.tvec)

    R1B, R2B, P1B, P2B, QB, validPix1B, validPix2B = cv.stereoRectify(backCameras.K1, backCameras.D1, backCameras.K2,
                                                                      backCameras.D2, (1280, 1280), backCameras.rvec,
                                                                      backCameras.tvec)

    mapTF1, mapTF2 = cv.initUndistortRectifyMap(frontCameras.K1, frontCameras.D1, R1F, None, (1280, 1280), cv.CV_32FC1)

    mapBF1, mapBF2 = cv.initUndistortRectifyMap(frontCameras.K2, frontCameras.D2, R2F, None, (1280, 1280), cv.CV_32FC1)

    mapTB1, mapTB2 = cv.initUndistortRectifyMap(backCameras.K1, backCameras.D1, R1B, None, (1280, 1280), cv.CV_32FC1)

    mapBB1, mapBB2 = cv.initUndistortRectifyMap(backCameras.K2, backCameras.D2, R2B, None, (1280, 1280), cv.CV_32FC1)

    cuda_mapTF1 = cv.cuda_GpuMat(mapTF1.astype(np.float32))
    cuda_mapTF2 = cv.cuda_GpuMat(mapTF2.astype(np.float32))
    cuda_mapBF1 = cv.cuda_GpuMat(mapBF1.astype(np.float32))
    cuda_mapBF2 = cv.cuda_GpuMat(mapBF2.astype(np.float32))
    cuda_mapTB1 = cv.cuda_GpuMat(mapTB1.astype(np.float32))
    cuda_mapTB2 = cv.cuda_GpuMat(mapTB2.astype(np.float32))
    cuda_mapBB1 = cv.cuda_GpuMat(mapBB1.astype(np.float32))
    cuda_mapBB2 = cv.cuda_GpuMat(mapBB2.astype(np.float32))


    if OCV_SGBM:

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

        if LEFTRIGHTLEFT:
            left_matcher = sgbm
            right_matcher = cv.ximgproc.createRightMatcher(left_matcher)

            wls_lmbda = 800
            wls_sigma = 1.2

            wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
            wls_filter.setLambda(wls_lmbda)
            wls_filter.setSigmaColor(wls_sigma)

    if OCV_CUDA_SGM:

        sgbm_cuda = cv.cuda.createStereoSGM(minDisparity=0,
                                            numDisparities=128,
                                            P1=8 * 3 * 3 * 3,
                                            P2=64 * 3 * 3 * 3,
                                            uniquenessRatio=8)

    if PSM:
        # padding will need to be added to image if a new image resolution is used
        model_path = "C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\PSMNet\\pretrained_model_KITTI2015.tar"
        model = stackhourglass(192)
        model = nn.DataParallel(model, device_ids=[0])
        model.cuda()
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict['state_dict'])
        model.eval()

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        infer_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=mean, std=std)])

    gpu_left_frame_front = cv.cuda_GpuMat()
    gpu_right_frame_front = cv.cuda_GpuMat()
    gpu_left_frame_back = cv.cuda_GpuMat()
    gpu_right_frame_back = cv.cuda_GpuMat()

    if TRACKBAR:

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
        if LEFTRIGHTLEFT:
            cv.createTrackbar("WLS Filter Lambda: ", 'controls', wls_lmbda, 10000, on_trackbar_set_wlsLmbda)
            cv.createTrackbar("WLS Filter Sigma Color (x 0.1): ", 'controls', math.ceil(wls_sigma / 0.1), 50, on_trackbar_set_wlsSigmaColor)

    stream_key = 0

    prev_frame_time = 0
    new_frame_time = 0
    avg_fps = 0
    fps_counter = 0

    while stream_key == 0:
        ret_top, top_frame = top_stream.read()
        ret_bot, bot_frame = bot_stream.read()

        if not top_stream.isOpened():
            sys.exit("ERROR: Unable to open top_stream")
        if not bot_stream.isOpened():
            sys.exit("ERROR: Unable to open bot_stream")

        height, width, channels = bot_frame.shape
        bot_front = bot_frame[0:int(height - 80), 0:int(width / 2)]
        bot_back = bot_frame[0:int(height - 80), int(width / 2):width]
        top_front = top_frame[0:int(height - 80), 0:int(width / 2)]
        top_back = top_frame[0:int(height - 80), int(width / 2):width]

        # Pad images

        top_front = pad_image(top_front, 1280)
        bot_front = pad_image(bot_front, 1280)
        top_back = pad_image(top_back, 1280)
        bot_back = pad_image(bot_back, 1280)

        gpu_left_frame_front.upload(top_front)
        gpu_right_frame_front.upload(bot_front)
        gpu_left_frame_back.upload(top_back)
        gpu_right_frame_back.upload(bot_back)

        # Rotate

        gpu_left_frame_front = cv.cuda.warpAffine(gpu_left_frame_front, tf_rot_mat, (height - 80, int(width / 2)),
                                                  flags=cv.INTER_LINEAR)
        gpu_right_frame_front = cv.cuda.warpAffine(gpu_right_frame_front, bf_rot_mat, (height - 80, int(width / 2)),
                                                   flags=cv.INTER_LINEAR)
        gpu_left_frame_back = cv.cuda.warpAffine(gpu_left_frame_back, tb_rot_mat, (height - 80, int(width / 2)),
                                                 flags=cv.INTER_LINEAR)
        gpu_right_frame_back = cv.cuda.warpAffine(gpu_right_frame_back, bb_rot_mat, (height - 80, int(width / 2)),
                                                  flags=cv.INTER_LINEAR)

        # Undistort using ImprovedOCamCalib

        gpu_left_frame_front = cv.cuda.remap(gpu_left_frame_front, cuda_tf_mapx, cuda_tf_mapy,
                                             interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
        gpu_right_frame_front = cv.cuda.remap(gpu_right_frame_front, cuda_bf_mapx, cuda_bf_mapy,
                                             interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
        gpu_left_frame_back = cv.cuda.remap(gpu_left_frame_back, cuda_tb_mapx, cuda_tb_mapy,
                                             interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
        gpu_right_frame_back = cv.cuda.remap(gpu_right_frame_back, cuda_bb_mapx, cuda_bb_mapy,
                                             interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)

        # Rectify Images

        gpu_left_frame_front = cv.cuda.remap(gpu_left_frame_front, cuda_mapTF1, cuda_mapTF2,
                                             interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
        gpu_left_frame_back = cv.cuda.remap(gpu_left_frame_back, cuda_mapTB1, cuda_mapTB2,
                                            interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
        gpu_right_frame_front = cv.cuda.remap(gpu_right_frame_front, cuda_mapBF1, cuda_mapBF2,
                                              interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
        gpu_right_frame_back = cv.cuda.remap(gpu_right_frame_back, cuda_mapBB1, cuda_mapBB2,
                                             interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)

        if OCV_SGBM:

            if LEFTRIGHTLEFT:

                rect_top_back = gpu_left_frame_back.download()
                rect_bot_back = gpu_right_frame_back.download()
                displ_back = left_matcher.compute(cv.UMat(rect_top_back), cv.UMat(rect_bot_back))
                dispr_back = right_matcher.compute(cv.UMat(rect_bot_back), cv.UMat(rect_top_back))
                displ_back = np.int16(cv.UMat.get(displ_back))
                dispr_back = np.int16(cv.UMat.get(dispr_back))
                disparity_back = wls_filter.filter(displ_back, rect_top_back, None, dispr_back)
                _, disparity_back = cv.threshold(disparity_back, 0, 128, cv.THRESH_TOZERO)
                disparity_back = (disparity_back / 16.).astype(np.uint8)

                rect_top_front = gpu_left_frame_front.download()
                rect_bot_front = gpu_right_frame_front.download()
                displ_front = left_matcher.compute(cv.UMat(rect_top_front), cv.UMat(rect_bot_front))
                dispr_front = right_matcher.compute(cv.UMat(rect_bot_front), cv.UMat(rect_top_front))
                displ_front = np.int16(cv.UMat.get(displ_front))
                dispr_front = np.int16(cv.UMat.get(dispr_front))
                disparity_front = wls_filter.filter(displ_front, rect_top_front, None, dispr_front)
                _, disparity_front = cv.threshold(disparity_front, 0, 128, cv.THRESH_TOZERO)
                disparity_front = (disparity_front / 16.).astype(np.uint8)

            else:

                rect_top_back = gpu_left_frame_back.download()
                rect_bot_back = gpu_right_frame_back.download()
                disparity_back = sgbm.compute(rect_top_back, rect_bot_back).astype(np.float32) / 16.0

                rect_top_front = gpu_left_frame_front.download()
                rect_bot_front = gpu_right_frame_front.download()
                disparity_front = sgbm.compute(rect_top_front, rect_bot_front).astype(np.float32) / 16.0

        if OCV_CUDA_SGM:

            gpu_left_frame_back = cv.cuda.cvtColor(gpu_left_frame_back, cv.COLOR_BGR2GRAY)
            gpu_right_frame_back = cv.cuda.cvtColor(gpu_right_frame_back, cv.COLOR_BGR2GRAY)
            disparity_back = sgbm_cuda.compute(gpu_left_frame_back, gpu_right_frame_back)
            disparity_back = disparity_back.download().astype(np.int16) / 16.0

            gpu_left_frame_front = cv.cuda.cvtColor(gpu_left_frame_front, cv.COLOR_BGR2GRAY)
            gpu_right_frame_front = cv.cuda.cvtColor(gpu_right_frame_front, cv.COLOR_BGR2GRAY)
            disparity_front = sgbm_cuda.compute(gpu_left_frame_front, gpu_right_frame_front)
            disparity_front = disparity_front.download().astype(np.int16) / 16.0

            # cv.filterSpeckles(dispB, 0, 40, 128)

        if PSM:

            rect_top_back = gpu_left_frame_back.download()
            rect_bot_back = gpu_right_frame_back.download()
            rect_top_back = rect_top_back[0: 1280, 400: 800]
            rect_bot_back = rect_bot_back[0: 1280, 400: 800]
            rect_top_back = infer_transform(rect_top_back).unsqueeze(0)
            rect_bot_back = infer_transform(rect_bot_back).unsqueeze(0)
            disparity_back = (PSM_disp(rect_top_back, rect_bot_back) * 256).astype('uint16')

            rect_top_front = gpu_left_frame_front.download()
            rect_bot_front = gpu_right_frame_front.download()
            rect_top_front = rect_top_front[0: 1280, 400: 800]
            rect_bot_front = rect_bot_front[0: 1280, 400: 800]
            rect_top_front = infer_transform(rect_top_front).unsqueeze(0)
            rect_bot_front = infer_transform(rect_bot_front).unsqueeze(0)
            disparity_front = (PSM_disp(rect_top_front, rect_bot_front) * 256).astype('uint16')

        #cv.imshow('Disparity Back', np.uint8(disparity_back))
        #cv.imshow('Disparty Front', np.uint8(disparity_front))

        if cv.waitKey(1) == ord('q'):
            top_stream.release()
            bot_stream.release()
            break

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        avg_fps = ((fps_counter * avg_fps) + fps) / (fps_counter + 1)
        fps_counter += 1
        prev_frame_time = new_frame_time
        print(avg_fps)

def pad_image(img, pad):

    h, w, c = np.shape(img)
    result = np.full((pad, pad, c), (0, 0, 0), dtype=np.uint8)

    pad_h = (pad - h) // 2
    pad_w = (pad - w) // 2

    print("padh", pad_h)
    print("Padw", pad_w)

    result[pad_h:pad_h + h, pad_w:pad_w + w] = img

    return result