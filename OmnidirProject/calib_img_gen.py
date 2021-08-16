import cv2 as cv
import numpy as np
from OCamCalib_model import *
from perspective_undistortion_LUT_OCamCalib import *

def calib_img_gen(calibImgDir, imgTF, imgTB, imgBF, imgBB, imgFormat, OCamCalibDir, OCamCalibTF, OCamCalibTB, OCamCalibBF, OCamCalibBB, sf):
    # Opens the two cameras, gives the option to flip the two cameras (top/bottom)
    top_cap = cv.VideoCapture(0)
    bot_cap = cv.VideoCapture(2)

    if not top_cap.isOpened() or not bot_cap.isOpened():
        print("ERROR! Unable to open camera(s)\n")
        return -1

    flipkey = 0

    while not flipkey == 'n' and not flipkey == 'y':
        ret_top, top_frame = top_cap.read()
        ret_bot, bot_frame = bot_cap.read()

        msg = "Flip Cameras? (y/n)"
        textOrigin = (10,50)
        top_frame_with_text = top_frame
        bot_frame_with_text = bot_frame
        cv.putText(top_frame_with_text, msg, textOrigin, cv.FONT_HERSHEY_COMPLEX, 1, (0 ,255, 0))
        cv.putText(bot_frame_with_text, msg, textOrigin, cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))

        cv.imshow("Top", top_frame_with_text)
        cv.imshow("Bottom", bot_frame_with_text)
        flipkey = cv.waitKey(5)

        if flipkey == ord('n'):
            cv.destroyWindow("Top")
            cv.destroyWindow("Bottom")
            break

        if flipkey == ord('y'):
            top_cap.release()
            bot_cap.release()
            top_cap = cv.VideoCapture(3)
            bot_cap = cv.VideoCapture(2)
            cv.destroyWindow("Top")
            cv.destroyWindow("Bottom")
            break

    key = 0
    ocam_key = 0
    frame_dir = bool(0)  # 0 - front, 1 - back
    frame_num = 0

    print("Reading the calib_result.txt files for the cameras...")

    tf_OCam = OCamCalib_model()
    tb_OCam = OCamCalib_model()
    bf_OCam = OCamCalib_model()
    bb_OCam = OCamCalib_model()

    tf_OCam.readOCamFile(OCamCalibDir + OCamCalibTF)
    tb_OCam.readOCamFile(OCamCalibDir + OCamCalibTB)
    bf_OCam.readOCamFile(OCamCalibDir + OCamCalibBF)
    bb_OCam.readOCamFile(OCamCalibDir + OCamCalibBB)

    print("Generating ImprovedOCamCalib LUT's...")

    tf_mapx, tf_mapy = perspective_undistortion_LUT(tf_OCam, sf, 640, 640, 640, 640)
    tb_mapx, tb_mapy = perspective_undistortion_LUT(tb_OCam, sf, 640, 640, 640, 640)
    bf_mapx, bf_mapy = perspective_undistortion_LUT(bf_OCam, sf, 640, 640, 640, 640)
    bb_mapx, bb_mapy = perspective_undistortion_LUT(bb_OCam, sf, 640, 640, 640, 640)

    print("Press 'c' on the image window to capture")
    print("Press 'f' to flip active camera capture side (default on front side)")
    print("Press 'i' to toggle ImprovedOCamCalib preview")
    print("Press 'q' to exit")

    while not key == 27:
        ret_top, top_frame = top_cap.read()
        ret_bot, bot_frame = bot_cap.read()

        # Splits the raw images into the two halves (back and front)
        height, width, channels = bot_frame.shape
        bot_front = bot_frame[0:int(height-80), 0:int(width/2)]
        bot_back = bot_frame[0:int(height-80), int(width/2):width]
        top_front = top_frame[0:int(height - 80), 0:int(width / 2)]
        top_back = top_frame[0:int(height - 80), int(width / 2):width]

        if key == ord('i'):
            ocam_key = not ocam_key
            if ocam_key:
                print("ImprovedOCamCalib preview toggled on")
                cv.destroyAllWindows()
            if not ocam_key:
                print("ImprovedOCamCalib preview toggled off")
                cv.destroyAllWindows()

        image_center = (320, 320)

        tf_rot_mat = cv.getRotationMatrix2D(image_center, -110, 1.0) #  -90
        bf_rot_mat = cv.getRotationMatrix2D(image_center, 70, 1.0) #  90
        tb_rot_mat = cv.getRotationMatrix2D(image_center, 110, 1.0) # 90
        bb_rot_mat = cv.getRotationMatrix2D(image_center, -70, 1.0) # -90

        tf_rot = cv.warpAffine(top_front, tf_rot_mat, (height - 80, int(width / 2)), flags=cv.INTER_LINEAR)
        bf_rot = cv.warpAffine(bot_front, bf_rot_mat, (height - 80, int(width / 2)), flags=cv.INTER_LINEAR)
        tb_rot = cv.warpAffine(top_back, tb_rot_mat, (height - 80, int(width / 2)), flags=cv.INTER_LINEAR)
        bb_rot = cv.warpAffine(bot_back, bb_rot_mat, (height - 80, int(width / 2)), flags=cv.INTER_LINEAR)

        if ocam_key:
            bot_front_padded = pad_image(bf_rot, 1280)
            bot_back_padded = pad_image(bb_rot, 1280)
            top_front_padded = pad_image(tf_rot, 1280)
            top_back_padded = pad_image(tb_rot, 1280)

            bot_front_undistorted = cv.remap(bot_front_padded, bf_mapx, bf_mapy, cv.INTER_LINEAR)
            bot_back_undistorted = cv.remap(bot_back_padded, bb_mapx, bb_mapy, cv.INTER_LINEAR)
            top_front_undistorted = cv.remap(top_front_padded, tf_mapx, tf_mapy, cv.INTER_LINEAR)
            top_back_undistorted = cv.remap(top_back_padded, tb_mapx, tb_mapy, cv.INTER_LINEAR)

            cv.imshow("Top Front ImprovedOCamCalib Preview", top_front_undistorted)
            cv.imshow("Top Back ImprovedOCamCalib Preview", top_back_undistorted)
            cv.imshow("Bottom Front ImprovedOCamCalib Preview", bot_front_undistorted)
            cv.imshow("Bottom Back ImprovedOCamCalib Preview", bot_back_undistorted)
        else:
            cv.imshow("Top Front", tf_rot)
            cv.imshow("Top Back", tb_rot)
            cv.imshow("Bottom Front", bf_rot)
            cv.imshow("Bottom Back", bb_rot)

        key = cv.waitKey(20)

        if key == ord('c'):
            if not frame_dir:  # if front
                top_name = calibImgDir + imgTF + str(frame_num) + imgFormat
                bot_name = calibImgDir + imgBF + str(frame_num) + imgFormat
                cv.imwrite(top_name, tf_rot)
                cv.imwrite(bot_name, bf_rot)
                print(top_name + ", " + bot_name + " saved.")
                frame_num += 1
            else:  # if back
                top_name = calibImgDir + imgTB + str(frame_num) + imgFormat
                bot_name = calibImgDir + imgBB + str(frame_num) + imgFormat
                cv.imwrite(top_name, tb_rot)
                cv.imwrite(bot_name, bb_rot)
                print(top_name + ", " + bot_name + " saved.")
                frame_num += 1
        elif key == ord('f'):
            frame_num = 0
            frame_dir = not frame_dir
            if frame_dir:
                print("Camera direction flipped - backwards")
            else:
                print("Camera direction flipped - forwards")
        elif key == ord('q'):
            break
    return 0

def pad_image(img, pad):

    h, w, c = np.shape(img)
    result = np.full((pad, pad, c), (0, 0, 0), dtype=np.uint8)

    pad_h = (pad - h) // 2
    pad_w = (pad - w) // 2

    result[pad_h:pad_h + h, pad_w:pad_w + w] = img

    return result
