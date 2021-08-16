import numpy as np
import cv2 as cv

class StereoCam:
    def __init__(self):
        self.K1 = cv.UMat()
        self.K2 = cv.UMat()
        self.D1 = cv.UMat()
        self.D2 = cv.UMat()
        self.xiT = np.float_()
        self.xiB = np.float_()
        self.rvec = cv.UMat()
        self.tvec = cv.UMat()
        self.flags = np.int
        self.rms = np.float_
        self.idx = cv.UMat()

        self.imgSize_1 = np.empty
        self.imgSize_2 = np.empty

        self.detect_list_1 = []
        self.detect_list_2 = []
        self.detect_list = []
        self.imgPoints_1 = np.empty
        self.imgPoints_2 = np.empty
        self.objPoints = np.empty
        self.rvecs = cv.UMat()
        self.tvecs = cv.UMat()
        self.reprojError = []

    def objPointsChangeSize(self, num_of_detected_images, n):
        #  n = boardsize.width * boardsize.height
        self.objPoints = np.empty((num_of_detected_images, n, 3), np.float32) # _ instead of 32 for omnidir

    def OCamCalibSave(self):
        # This doesn't actually save anything, just fills all the uninitialised cv.UMat fields so we can pickle them
        self.rvecs = np.zeros([1])
        self.tvecs = np.zeros([1])
        self.idx = np.zeros([1])

    #  def imgPoints1ChangeSize(self):
