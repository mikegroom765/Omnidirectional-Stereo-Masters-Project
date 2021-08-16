import numpy as np
import cv2 as cv

def setK_11(value):
    global K_11
    K_11 = value

def setK_12(value):
    global K_12
    K_12 = value

def setK_13(value):
    global K_13
    K_13 = value

def setK_22(value):
    global K_22
    K_22 = value

def setK_23(value):
    global K_23
    K_23 = value

def setD1(value):
    global D_1
    D_1 = (value - 50)/100

def setD2(value):
    global D_2
    D_2 = (value - 50)/100

def setD3(value):
    global D_3
    D_3= (value - 50)/100

def setD4(value):
    global D_4
    D_4 = (value - 50)/100

def setXi(value):
    global xi_1
    xi_1 = (value - 50)/100

bb_frame = cv.imread("C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\data\\calibration_images\\bottom_back\\bb_0.bmp")
tb_frame = cv.imread("C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\data\\calibration_images\\top_back\\tb_0.bmp")
frame_h, frame_w, channels = bb_frame.shape

K_11 = 320
K_12 = 7
K_13 = 320
K_22 = 328
K_23 = 333

D_1 = -0.3
D_2 = 0.1
D_3 = -0.01
D_4 = -0.005

xi_1 = 0.74

knew = np.array([[frame_w / (np.pi * 19 / 18), 0, 0], [0, frame_h / (np.pi * 19 / 18), 0], [0, 0, 1]], np.double)

cv.namedWindow('controls', cv.WINDOW_NORMAL)
cv.createTrackbar("K_11: ", 'controls', 0, 1000, setK_11)
cv.createTrackbar("K_12: ", 'controls', 0, 1000, setK_12)
cv.createTrackbar("K_13: ", 'controls', 0, 1000, setK_13)
cv.createTrackbar("K_22: ", 'controls', 0, 1000, setK_22)
cv.createTrackbar("K_23: ", 'controls', 0, 1000, setK_23)
cv.createTrackbar("D_1: ", 'controls', 0, 100, setD1)
cv.createTrackbar("D_2: ", 'controls', 0, 100, setD2)
cv.createTrackbar("D_3: ", 'controls', 0, 100, setD3)
cv.createTrackbar("D_4: ", 'controls', 0, 100, setD4)
cv.createTrackbar("xi: ", 'controls', 0, 200, setXi)

while True:
    K = np.array([[K_11, K_12, K_13], [0, K_22, K_23], [0, 0, 1]], np.double)

    D = np.array([D_1, D_2, D_3, D_4], np.double)

    xi = np.array([xi_1], np.double)

    cv.imshow("bb_frame", bb_frame)
    cv.imshow("tb_frame", tb_frame)

    bbuFrame = cv.omnidir.undistortImage(bb_frame, K, D, xi, cv.omnidir.RECTIFY_LONGLATI, None, knew)
    cv.imshow("bbuFrame", bbuFrame)

    tbuFrame = cv.omnidir.undistortImage(tb_frame, K, D, xi, cv.omnidir.RECTIFY_LONGLATI, None, knew)
    cv.imshow("tbuFrame", tbuFrame)

    cv.waitKey(5)
