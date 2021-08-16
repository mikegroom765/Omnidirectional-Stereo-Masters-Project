import cv2 as cv
import glob

"C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\calib_pics_backup\\rotating\\tf\\tf_*.bmp"

tfImgList = sorted(glob.glob("C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\calib_pics_backup\\rotating\\tf\\tf_*.bmp"))
tbImgList = sorted(glob.glob("C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\calib_pics_backup\\rotating\\tb\\tb_*.bmp"))
bfImgList = sorted(glob.glob("C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\calib_pics_backup\\rotating\\bf\\bf_*.bmp"))
bbImgList = sorted(glob.glob("C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\calib_pics_backup\\rotating\\bb\\bb_*.bmp"))

tf_rot_mat = cv.getRotationMatrix2D((320, 320), -20, 1.0)
tb_rot_mat = cv.getRotationMatrix2D((320, 320), 20, 1.0)

for idx, value in enumerate(tfImgList):
    img = cv.imread(value)
    img_rot = cv.warpAffine(img, tf_rot_mat, (640, 640), flags=cv.INTER_LINEAR)
    cv.imwrite(value, img_rot)

for idx, value in enumerate(tbImgList):
    img = cv.imread(value)
    img_rot = cv.warpAffine(img, tb_rot_mat, (640, 640), flags=cv.INTER_LINEAR)
    cv.imwrite(value, img_rot)

for idx, value in enumerate(bfImgList):
    img = cv.imread(value)
    img_rot = cv.warpAffine(img, tf_rot_mat, (640, 640), flags=cv.INTER_LINEAR)
    cv.imwrite(value, img_rot)

for idx, value in enumerate(bbImgList):
    img = cv.imread(value)
    img_rot = cv.warpAffine(img, tb_rot_mat, (640, 640), flags=cv.INTER_LINEAR)
    cv.imwrite(value, img_rot)