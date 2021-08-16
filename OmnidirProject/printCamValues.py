import pickle
import cv2 as cv
import numpy as np

# Load previous front camera calibration (OpenCV omnidir)
frontInFile = open(
    "C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\data\\calibration_parameters_OCamCalib\\auto_calibs\\front_camera_params_OCam16.xml", 'rb')
frontCameras = pickle.load(frontInFile)
frontInFile.close()

# Load previous back camera caibration (OpenCV omnidir)
backInFile = open(
    "C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\data\\calibration_parameters_OCamCalib\\auto_calibs\\back_camera_params_OCam16.xml",
    'rb')
backCameras = pickle.load(backInFile)
backInFile.close()

print(frontCameras.tvec)
print(backCameras.tvec)

print(frontCameras.rvec)
print(backCameras.rvec)

print(backCameras.K1)
print(backCameras.K2)
print(frontCameras.K1)
print(frontCameras.K2)

tf_undistorted_image = cv.imread(
    "C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\data\\calibration_images\\OCamCalib_images\\tf\\tf_0.bmp")
bf_undistorted_image = cv.imread(
    "C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\data\\calibration_images\\OCamCalib_images\\bf\\bf_0.bmp")
tb_undistorted_image = cv.imread(
    "C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\data\\calibration_images\\OCamCalib_images\\tb\\tb_0.bmp")
bb_undistorted_image = cv.imread(
    "C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\data\\calibration_images\\OCamCalib_images\\bb\\bb_0.bmp")

R1F, R2F, P1F, P2F, QF, validPix1F, validPix2F = cv.stereoRectify(frontCameras.K1, frontCameras.D1,
                                                                                  frontCameras.K2,
                                                                                  frontCameras.D2, (1280, 1280),
                                                                                  frontCameras.rvec,
                                                                                  frontCameras.tvec,
                                                                                  alpha=0)

R1B, R2B, P1B, P2B, QB, validPix1B, validPix2B = cv.stereoRectify(backCameras.K1, backCameras.D1,
                                                                                  backCameras.K2,
                                                                                  backCameras.D2, (1280, 1280),
                                                                                  backCameras.rvec,
                                                                                  backCameras.tvec)

print("projection matrix")
print(P1F)

mapTF1, mapTF2 = cv.initUndistortRectifyMap(frontCameras.K1, frontCameras.D1, R1F, P1F, (1280, 1280),
                                            cv.CV_32FC1)

mapBF1, mapBF2 = cv.initUndistortRectifyMap(frontCameras.K2, frontCameras.D2, R2F, P2F, (1280, 1280),
                                                            cv.CV_32FC1)

mapTB1, mapTB2 = cv.initUndistortRectifyMap(backCameras.K1, backCameras.D1, R1B, P1B, (1280, 1280),
                                                            cv.CV_32FC1)

mapBB1, mapBB2 = cv.initUndistortRectifyMap(backCameras.K2, backCameras.D2, R2B, P2B, (1280, 1280),
                                                            cv.CV_32FC1)

tf_rectified_img = cv.remap(tf_undistorted_image, mapTF1, mapTF2, cv.INTER_LINEAR)
bf_rectified_img = cv.remap(bf_undistorted_image, mapBF1, mapBF2, cv.INTER_LINEAR)
tb_rectified_img = cv.remap(tb_undistorted_image, mapTB1, mapTB2, cv.INTER_LINEAR)
bb_rectified_img = cv.remap(bb_undistorted_image, mapBB1, mapBB2, cv.INTER_LINEAR)

front_rectified = np.full((1280, 2560, 3), (0, 0, 0), dtype=np.uint8)
back_rectified = np.full((1280, 2560, 3), (0, 0, 0), dtype=np.uint8)

tf_rectified_img = cv.transpose(tf_rectified_img)
tf_rectified_img = cv.flip(tf_rectified_img, 0)
bf_rectified_img = cv.transpose(bf_rectified_img)
bf_rectified_img = cv.flip(bf_rectified_img, 0)
tb_rectified_img = cv.transpose(tb_rectified_img)
tb_rectified_img = cv.flip(tb_rectified_img, 0)
bb_rectified_img = cv.transpose(bb_rectified_img)
bb_rectified_img = cv.flip(bb_rectified_img, 0)

front_rectified[0:1280, 0:1280] = tf_rectified_img
front_rectified[0:1280, 1280:2560] = bf_rectified_img
back_rectified[0:1280, 0:1280] = tb_rectified_img
back_rectified[0:1280, 1280:2560] = bb_rectified_img

for lines in range(7):
    lines += 1
    cv.line(front_rectified, (0, (160 * lines)), (2560, (160 * lines)), (0, 0, 255))
    cv.line(back_rectified, (0, (160 * lines)), (2560, (160 * lines)), (0, 0, 255))
    
cv.waitKey(1000)

cv.namedWindow('front', cv.WINDOW_NORMAL)
cv.imshow("front", front_rectified)
cv.namedWindow('back', cv.WINDOW_NORMAL)
cv.imshow("back", back_rectified)

cv.waitKey(100000)