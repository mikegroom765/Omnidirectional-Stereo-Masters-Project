import cv2 as cv
import numpy as np
from Size import *


def calcChessboardCorners(boardsize, square_width, square_height):

    # n is the total number of corners in one image
    n = boardsize.width * boardsize.height
    #  print(n)
    corners = np.empty((n,3), np.float32)  # Creates an empty list of size n x 1
    #corners[:, :2] = np.mgrid[0:boardsize.width, 0:boardsize.height].T.reshape(-1, 2)
    #corners *= square_width

    #print("corners 1")
    #print(corners)
    for i in range(boardsize.height):
        for j in range(boardsize.width):
            corners[i*boardsize.width + j] = [j*square_width, i*square_height, 0.0]

    #print("corners 2")
    print(corners)
    return corners
#  print(np.shape(calcChessboardCorners(Size(4,5),37,37)))
#  print(calcChessboardCorners(Size(4,5),37,37))