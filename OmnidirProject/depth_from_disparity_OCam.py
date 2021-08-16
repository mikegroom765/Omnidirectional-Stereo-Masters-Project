import cv2 as cv
import numpy as np

def depth_from_disp_OCam(img, tvec, fx):

    #depth = np.full((1280, 1280), 0, dtype=np.float32)

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    _, mask = cv.threshold(img, 0, 1, cv.THRESH_BINARY_INV)
    # mask[:, 0:120] = 0
    img = cv.inpaint(img, mask, 2, cv.INPAINT_NS)

    b = np.sqrt(tvec[0] ** 2 + tvec[1] ** 2 + tvec[2] ** 2)
    b = b / 1000

    depth = (fx * b[0]) / img

    return depth

