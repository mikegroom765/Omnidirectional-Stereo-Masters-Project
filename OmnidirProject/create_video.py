import numpy as np
import cv2 as cv
import pickle
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from PSMNet.models import *
from improvedOCamCalib import *
from perspective_undistortion_LUT_OCamCalib import *

scale_factor = 5.0

OCamCalibDir = "C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\data\\calibration_parameters_OCamCalib"
OCamCalibFrontOut = "\\front_camera_params_OCam.xml"
OCamCalibBackOut = "\\back_camera_params_Ocam.xml"
OCamCalibTopFront = "\\calib_results_tf.txt"
OCamCalibTopBack = "\\calib_results_tb.txt"
OCamCalibBotFront = "\\calib_results_bf.txt"
OCamCalibBotBack = "\\calib_results_bb.txt"

def PSM_disp(imgL, imgR):

    imgL = imgL.cuda()
    imgR = imgR.cuda()

    with torch.no_grad():
        disp = model(imgL, imgR)

    disp = torch.squeeze(disp)
    pred_disp = disp.data.cpu().numpy()

    return pred_disp

def pad_image(img, pad):

    h, w, c = np.shape(img)
    result = np.full((pad, pad, c), (0, 0, 0), dtype=np.uint8)

    pad_h = (pad - h) // 2
    pad_w = (pad - w) // 2

    result[pad_h:pad_h + h, pad_w:pad_w + w] = img

    return result

top_cap = cv.VideoCapture('C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\Experiment\\top_video.mp4')
bot_cap = cv.VideoCapture('C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\Experiment\\bottom_video.mp4')

fourcc = cv.VideoWriter_fourcc(*'DIVX')
#front_rectified = cv.VideoWriter('front_SGBM_WLS_SF2.5_OCam.mp4', fourcc, 14.985, (2560, 1280))
#back_rectified = cv.VideoWriter('back_SGBM_WLS_SF2.5_OCam.mp4', fourcc, 14.985, (2560, 1280))

front_disparity = cv.VideoWriter('front_SGBM_SF5.0_OCam.mp4', fourcc, 14.985, (1280, 1280))
back_disparity = cv.VideoWriter('back_SGBM_SF5.0_OCam.mp4', fourcc, 14.985, (1280, 1280))

# Load previous front camera calibration
frontInFile = open('C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\data\\calibration_parameters_OCamCalib\\front_camera_params_OCam.xml', 'rb')
frontCameras = pickle.load(frontInFile)
frontInFile.close()
# Load previous back camera caibration
backInFile = open('C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\data\\calibration_parameters_OCamCalib\\back_camera_params_OCam.xml', 'rb')
backCameras = pickle.load(backInFile)
backInFile.close()

print("Creating LUT's..")

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

print("Done!")

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

left_matcher = sgbm
right_matcher = cv.ximgproc.createRightMatcher(left_matcher)

wls_lmbda = 800
wls_sigma = 1.2

wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(wls_lmbda)
wls_filter.setSigmaColor(wls_sigma)

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

image_center = (320, 320)

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

gpu_left_frame_front = cv.cuda_GpuMat()
gpu_right_frame_front = cv.cuda_GpuMat()
gpu_left_frame_back = cv.cuda_GpuMat()
gpu_right_frame_back = cv.cuda_GpuMat()

front_rectified_frame = np.full((1280, 2560, 3), (0, 0, 0), dtype=np.uint8)
back_rectified_frame = np.full((1280, 2560, 3), (0, 0, 0), dtype=np.uint8)

while top_cap.isOpened() and bot_cap.isOpened():
    ret, top_frame = top_cap.read()
    ret, bot_frame = bot_cap.read()

    height, width, channels = bot_frame.shape
    bot_front = bot_frame[0:int(height - 80), 0:int(width / 2)]
    bot_back = bot_frame[0:int(height - 80), int(width / 2):width]
    top_front = top_frame[0:int(height - 80), 0:int(width / 2)]
    top_back = top_frame[0:int(height - 80), int(width / 2):width]

    gpu_left_frame_front.upload(top_front)
    gpu_right_frame_front.upload(bot_front)
    gpu_left_frame_back.upload(top_back)
    gpu_right_frame_back.upload(bot_back)

    #rotate

    gpu_left_frame_front = cv.cuda.warpAffine(gpu_left_frame_front, tf_rot_mat, (height - 80, int(width / 2)),
                                              flags=cv.INTER_LINEAR)
    gpu_right_frame_front = cv.cuda.warpAffine(gpu_right_frame_front, bf_rot_mat, (height - 80, int(width / 2)),
                                               flags=cv.INTER_LINEAR)
    gpu_left_frame_back = cv.cuda.warpAffine(gpu_left_frame_back, tb_rot_mat, (height - 80, int(width / 2)),
                                             flags=cv.INTER_LINEAR)
    gpu_right_frame_back = cv.cuda.warpAffine(gpu_right_frame_back, bb_rot_mat, (height - 80, int(width / 2)),
                                              flags=cv.INTER_LINEAR)

    top_back = gpu_left_frame_back.download()
    bot_back = gpu_right_frame_back.download()
    top_front = gpu_left_frame_front.download()
    bot_front = gpu_right_frame_front.download()

    top_front = pad_image(top_front, 1280)
    bot_front = pad_image(bot_front, 1280)
    top_back = pad_image(top_back, 1280)
    bot_back = pad_image(bot_back, 1280)

    gpu_left_frame_front.upload(top_front)
    gpu_right_frame_front.upload(bot_front)
    gpu_left_frame_back.upload(top_back)
    gpu_right_frame_back.upload(bot_back)

    #undistort

    gpu_left_frame_front = cv.cuda.remap(gpu_left_frame_front, cuda_tf_mapx, cuda_tf_mapy,
                                         interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    gpu_right_frame_front = cv.cuda.remap(gpu_right_frame_front, cuda_bf_mapx, cuda_bf_mapy,
                                          interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    gpu_left_frame_back = cv.cuda.remap(gpu_left_frame_back, cuda_tb_mapx, cuda_tb_mapy,
                                        interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    gpu_right_frame_back = cv.cuda.remap(gpu_right_frame_back, cuda_bb_mapx, cuda_bb_mapy,
                                         interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)

    #rectify

    gpu_left_frame_front = cv.cuda.remap(gpu_left_frame_front, cuda_mapTF1, cuda_mapTF2, interpolation=cv.INTER_LINEAR,
                                         borderMode=cv.BORDER_CONSTANT)
    gpu_left_frame_back = cv.cuda.remap(gpu_left_frame_back, cuda_mapTB1, cuda_mapTB2, interpolation=cv.INTER_LINEAR,
                                        borderMode=cv.BORDER_CONSTANT)
    gpu_right_frame_front = cv.cuda.remap(gpu_right_frame_front, cuda_mapBF1, cuda_mapBF2,
                                          interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    gpu_right_frame_back = cv.cuda.remap(gpu_right_frame_back, cuda_mapBB1, cuda_mapBB2, interpolation=cv.INTER_LINEAR,
                                         borderMode=cv.BORDER_CONSTANT)

    rect_top_back = gpu_left_frame_back.download()
    rect_bot_back = gpu_right_frame_back.download()
    rect_top_front = gpu_left_frame_front.download()
    rect_bot_front = gpu_right_frame_front.download()

    rect_top_back = cv.transpose(rect_top_back)
    rect_top_back = cv.flip(rect_top_back, 0)
    rect_bot_back = cv.transpose(rect_bot_back)
    rect_bot_back = cv.flip(rect_bot_back, 0)

    rect_top_front = cv.transpose(rect_top_front)
    rect_top_front = cv.flip(rect_top_front, 0)
    rect_bot_front = cv.transpose(rect_bot_front)
    rect_bot_front = cv.flip(rect_bot_front, 0)

    #SGBM

    #disparity_front_sgbm = sgbm.compute(rect_top_front, rect_bot_front)
    #_, disparity_front_sgbm = cv.threshold(disparity_front_sgbm, 0, 128, cv.THRESH_TOZERO)
    #disparity_scaled_front_sgbm = (disparity_front_sgbm / 16.).astype(np.uint8)
    #disparity_to_display_front_sgbm = (disparity_scaled_front_sgbm * (256. / 128)).astype(np.uint8)

    #disparity_back_sgbm = sgbm.compute(rect_top_back, rect_bot_back)
    #_, disparity_back_sgbm = cv.threshold(disparity_back_sgbm, 0, 128, cv.THRESH_TOZERO)
    #disparity_scaled_back_sgbm = (disparity_back_sgbm / 16.).astype(np.uint8)
    #disparity_to_display_back_sgbm = (disparity_scaled_back_sgbm * (256. / 128)).astype(np.uint8)

    #SGBM_WLS

    grayL = cv.cvtColor(rect_top_front, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(rect_bot_front, cv.COLOR_BGR2GRAY)
    displ = left_matcher.compute(cv.UMat(grayL), cv.UMat(grayR))  # .astype(np.int16) # tf_img, bf_img
    dispr = right_matcher.compute(cv.UMat(grayR), cv.UMat(grayL))  # .astype(np.int16)
    displ = np.int16(cv.UMat.get(displ))
    dispr = np.int16(cv.UMat.get(dispr))
    disparity_front_wls = wls_filter.filter(displ, grayL, None, dispr)
    _, disparity_front_wls = cv.threshold(disparity_front_wls, 0, 128, cv.THRESH_TOZERO)
    disparity_scaled_front_wls = (disparity_front_wls / 16.).astype(np.uint8)
    disparity_to_display_front_wls = (disparity_scaled_front_wls * (256. / 128)).astype(np.uint8)

    grayL = cv.cvtColor(rect_top_back, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(rect_bot_back, cv.COLOR_BGR2GRAY)
    displ = left_matcher.compute(cv.UMat(grayL), cv.UMat(grayR))  # .astype(np.int16) # tf_img, bf_img
    dispr = right_matcher.compute(cv.UMat(grayR), cv.UMat(grayL))  # .astype(np.int16)
    displ = np.int16(cv.UMat.get(displ))
    dispr = np.int16(cv.UMat.get(dispr))
    disparity_back_wls = wls_filter.filter(displ, grayL, None, dispr)
    _, disparity_back_wls = cv.threshold(disparity_back_wls, 0, 128, cv.THRESH_TOZERO)
    disparity_scaled_back_wls = (disparity_back_wls / 16.).astype(np.uint8)
    disparity_to_display_back_wls = (disparity_scaled_back_wls * (256. / 128)).astype(np.uint8)

    # PSMNet

    #rect_top_front = cv.cvtColor(rect_top_front, cv.COLOR_BGR2RGB)
    #rect_bot_front = cv.cvtColor(rect_bot_front, cv.COLOR_BGR2RGB)
    #rect_top_front = infer_transform(rect_top_front)
    #rect_bot_front = infer_transform(rect_bot_front)

    #rect_top_back = cv.cvtColor(rect_top_back, cv.COLOR_BGR2RGB)
    #rect_bot_back = cv.cvtColor(rect_bot_back, cv.COLOR_BGR2RGB)
    #rect_top_back = infer_transform(rect_top_back)
    #rect_bot_back = infer_transform(rect_bot_back)

    #if rect_top_front.shape[1] % 16 != 0:
        #times = rect_top_front.shape[1] // 16
        #top_pad = (times + 1) * 16 - rect_top_front.shape[1]
   # else:
        #top_pad = 0

    #if rect_top_front.shape[2] % 16 != 0:
        #times = rect_top_front.shape[2] // 16
        #right_pad = (times + 1) * 16 - rect_top_front.shape[2]

   # else:
      #  right_pad = 0

    #rect_top_front = F.pad(rect_top_front, (0, right_pad, top_pad, 0)).unsqueeze(0)
   # rect_bot_front = F.pad(rect_bot_front, (0, right_pad, top_pad, 0)).unsqueeze(0)

    #if rect_top_back.shape[1] % 16 != 0:
     #   times = rect_top_back.shape[1] // 16
     #   top_pad = (times + 1) * 16 - rect_top_back.shape[1]
   # else:
    #    top_pad = 0

    #if rect_top_back.shape[2] % 16 != 0:
       # times = rect_top_back.shape[2] // 16
     #  right_pad = (times + 1) * 16 - rect_top_back.shape[2]

    #else:
      # right_pad = 0

    #rect_top_back = F.pad(rect_top_back, (0, right_pad, top_pad, 0)).unsqueeze(0)
    #rect_bot_back = F.pad(rect_bot_back, (0, right_pad, top_pad, 0)).unsqueeze(0)

    #pred_disp = PSM_disp(rect_top_front, rect_bot_front)
    #disparity_to_display_front_PSMNet = (pred_disp * (256 / 192)).astype('uint8')

    #pred_disp = PSM_disp(rect_top_back, rect_bot_back)
    #disparity_to_display_back_PSMNet = (pred_disp * (256 / 192)).astype('uint8')

    #Rectified

    #front_rectified_frame[:, 0:1280] = rect_top_front
    #front_rectified_frame[:, 1280:2560] = rect_bot_front
    #back_rectified_frame[:, 0:1280] = rect_top_back
    #back_rectified_frame[:, 1280:2560] = rect_bot_back

    #cv.imshow("test", front_rectified_frame)

    #front_rectified.write(front_rectified_frame)
    #back_rectified.write(back_rectified_frame)

    #disparity_to_display_front_wls = cv.transpose(disparity_to_display_front_wls)
    #disparity_to_display_front_wls = cv.flip(disparity_to_display_front_wls, 1)
    #disparity_to_display_back_wls = cv.transpose(disparity_to_display_back_wls)
    #disparity_to_display_back_wls = cv.flip(disparity_to_display_back_wls, 1)

    disparity_to_display_front = cv.applyColorMap(disparity_to_display_front_wls, cv.COLORMAP_PLASMA)
    disparity_to_display_back = cv.applyColorMap(disparity_to_display_back_wls, cv.COLORMAP_PLASMA)

    front_disparity.write(disparity_to_display_front)
    back_disparity.write(disparity_to_display_back)

    if cv.waitKey(10) & 0xFF == ord('q'):
        break

top_cap.release()
bot_cap.release()
front_disparity.release()
back_disparity.release()
cv.destroyAllWindows()


