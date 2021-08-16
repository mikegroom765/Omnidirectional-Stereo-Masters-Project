import cv2 as cv
import glob
from detectChessboardCorners import detectChessboardConrner
from stereocam import *
import pickle
from Size import *
import time
import csv

frontCameras = StereoCam()
backCameras = StereoCam()

def omnidirCalib(OCVcalibImgDir, imgFormat, boardSize, imgTF, imgTB, imgBF, imgBB, square_width, square_height, SHOW_CALIB_IMG):
    calib_key = 0  # true if not enough chessboard corners are detected
    tfImgList = sorted(glob.glob(OCVcalibImgDir + imgTF + "*" + imgFormat))
    tbImgList = sorted(glob.glob(OCVcalibImgDir + imgTB + "*" + imgFormat))
    bfImgList = sorted(glob.glob(OCVcalibImgDir + imgBF + "*" + imgFormat))
    bbImgList = sorted(glob.glob(OCVcalibImgDir + imgBB + "*" + imgFormat))

    if not (tfImgList or tbImgList or bfImgList or bbImgList):
        print("Images not found in directory :" + OCVcalibImgDir)
    else:
        print("Image lists generated!")

    print("Detecting chessboard corners ...")

    result_front = detectChessboardConrner(tfImgList, frontCameras.detect_list_1, bfImgList, frontCameras.detect_list_2,
                                           tuple(boardSize.size()), frontCameras.imgSize_1, frontCameras.imgSize_2,
                                           SHOW_CALIB_IMG, 0)

    result_back = detectChessboardConrner(tbImgList, backCameras.detect_list_1, bbImgList, backCameras.detect_list_2,
                                          tuple(boardSize.size()), backCameras.imgSize_1, backCameras.imgSize_2,
                                          SHOW_CALIB_IMG, 1)

    print(np.shape(frontCameras.detect_list_1))

    if result_front[4] == 0:
        print("Not enough corner detected front images\n")
        calib_key = 1
    else:
        frontCameras.imgPoints_1 = result_front[0]
        frontCameras.imgPoints_2 = result_front[1]
        frontCameras.imgSize_1 = result_front[2]
        frontCameras.imgSize_2 = result_front[3]

    if result_back[4] == 0:
        print("Not enough corner detected front images\n")
        calib_key = 1
    else:
        backCameras.imgPoints_1 = result_back[0]
        backCameras.imgPoints_2 = result_back[1]
        backCameras.imgSize_1 = result_back[2]
        backCameras.imgSize_2 = result_back[3]

    if not calib_key:
        num_detected_images_front_top = int(np.shape(frontCameras.detect_list_1)[0])

        num_detected_images_front_bot = int(np.shape(frontCameras.detect_list_2)[0])

        num_detected_images_back_top = int(np.shape(backCameras.detect_list_1)[0])

        num_detected_images_back_bot = int(np.shape(backCameras.detect_list_2)[0])

        print(str("Top Front Camera: " + str(num_detected_images_front_top) + "/" + str(
            int(np.shape(tfImgList)[0])) + " Detected"))  # Prints the number of detected images over number of images
        print(str("Bottom Front Camera: " + str(num_detected_images_front_bot) + "/" + str(
            int(np.shape(bfImgList)[0])) + " Detected"))
        print(str("Top Back Camera: " + str(num_detected_images_back_top) + "/" + str(
            int(np.shape(tbImgList)[0])) + " Detected"))
        print(str("Bottom Back Camera: " + str(num_detected_images_back_bot) + "/" + str(
            int(np.shape(bbImgList)[0])) + " Detected"))

        print("Chessboard corner detection finished!\n")
        print("Calculating object coordinates of chessboard corners ...\n")

        objP = np.empty((boardSize.width * boardSize.height, 3), np.float32)
        for i in range(boardSize.height):
            for j in range(boardSize.width):
                objP[i * boardSize.width + j] = [j * square_width, i * square_height, 0.0]

        frontCameras.objPointsChangeSize(num_detected_images_front_top, (boardSize.width * boardSize.height))
        backCameras.objPointsChangeSize(num_detected_images_back_top, (boardSize.width * boardSize.height))

        for idx, value in enumerate(frontCameras.detect_list_1):
            frontCameras.objPoints[idx, :, :] = objP
        for idx, value in enumerate(backCameras.detect_list_1):
            backCameras.objPoints[idx, :, :] = objP

        print("Finished calculating object coordinates of chessboard corners.\n")

        _xiTF = np.empty(1)
        _xiBF = np.empty(1)
        _xiTB = np.empty(1)
        _xiBB = np.empty(1)

        retval1 = None
        retval2 = None

        # Need to resize objPoints from (num_detected_images, num_chessboard_squares, 3) to (num_detected_images, 1, num_chessboard_squares, 3) to avoid an error in omnidir.stereoCalibrate
        frontCameras.objPoints = np.reshape(frontCameras.objPoints,
                                            (num_detected_images_front_top, 1, boardSize.height * boardSize.width, 3))

        backCameras.objPoints = np.reshape(backCameras.objPoints,
                                           (num_detected_images_back_top, 1, boardSize.height * boardSize.width, 3))

        # Need to resize imgPoints from (num_detected_images, num_chessboard_squares, 2) to (num_detected_images, 1, num_chessboard_squares, 2) for same reason
        frontCameras.imgPoints_1 = np.reshape(frontCameras.imgPoints_1,
                                              (num_detected_images_front_top, 1, boardSize.height * boardSize.width, 2))

        frontCameras.imgPoints_2 = np.reshape(frontCameras.imgPoints_2,
                                              (num_detected_images_front_top, 1, boardSize.height * boardSize.width, 2))

        backCameras.imgPoints_1 = np.reshape(backCameras.imgPoints_1,
                                             (num_detected_images_back_top, 1, boardSize.height * boardSize.width, 2))

        backCameras.imgPoints_2 = np.reshape(backCameras.imgPoints_2,
                                             (num_detected_images_back_top, 1, boardSize.height * boardSize.width, 2))

        # Termination criteria for stereoCalibrate # cv.TermCriteria_EPS + cv.TERM_CRITERIA_MAX_ITER
        print("Starting calibrations!")
        calib_number = 0
        error_file = []


        for eps_iter in range(2):
            for num_iterations_iter in range(4):
                calib_number += 1

                print("Starting calibration number: " + str(calib_number))

                print("Front camera calibration ...")

                num_iterations = 100 * (num_iterations_iter + 1)
                eps = 0.000001 / (10 ** eps_iter)

                term_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, num_iterations, eps)  # 1, 0.001

                calib_flags = 0 # cv.omnidir.CALIB_USE_GUESS

                K1 = np.array([[700, 1, 320], [0, 700, 320], [0, 0, 1]], np.double)
                D1 = np.array([[-0.305, 0.0695, -0.0025, -0.0035]], np.double)
                xi_1 = np.array([[2.9355]], np.double)

                K2 = np.array([[700, 1, 320], [0, 700, 320], [0, 0, 1]], np.double)
                D2 = np.array([[-0.3025, 0.067, 0, -0.002]], np.double)
                xi_2 = np.array([[2.991]], np.double)

                retval1, frontCameras.objPoints, frontCameras.imgPoints_1, frontCameras.imgPoints_2, frontCameras.K1, \
                frontCameras.xiT, frontCameras.D1, frontCameras.K2, frontCameras.xiB, frontCameras.D2, frontCameras.rvec, \
                frontCameras.tvec, frontCameras.rvecs, \
                frontCameras.tvecs, frontCameras.idx = cv.omnidir.stereoCalibrate(frontCameras.objPoints,
                                                                                  frontCameras.imgPoints_1,
                                                                                  frontCameras.imgPoints_2,
                                                                                  frontCameras.imgSize_1,
                                                                                  frontCameras.imgSize_2,
                                                                                  None, None, None, None, None, None, #K1, xi_1, D1, K1, xi_2, D2,
                                                                                  calib_flags, term_criteria)

                print("Back camera calibration ...")

                retval2, backCameras.objPoints, backCameras.imgPoints_1, backCameras.imgPoints_2, backCameras.K1, \
                backCameras.xiT, backCameras.D1, backCameras.K2, backCameras.xiB, backCameras.D2, backCameras.rvec, \
                backCameras.tvec, backCameras.rvecs, \
                backCameras.tvecs, backCameras.idx = cv.omnidir.stereoCalibrate(backCameras.objPoints,
                                                                                backCameras.imgPoints_1,
                                                                                backCameras.imgPoints_2,
                                                                                backCameras.imgSize_1,
                                                                                backCameras.imgSize_2,
                                                                                None, None, None, None, None, None, #K1, xi_1, D1, K1, xi_2, D2,
                                                                                calib_flags, term_criteria)

                print("Done!")

                print("Calculating Front Camera Reprojection Error")

                frontCam_used_img_list = []
                average_repoj_error_top = 0
                average_repoj_error_bot = 0
                rms_error_bot = 0
                rms_error_top = 0

                #  idx is the indices of image pairs that pass initialization, so the ones actually used in the calibration
                #  rvecs and tvecs are the same size as idx
                #print(frontCameras.idx[0])
                for idx, value in enumerate(frontCameras.idx[0]):
                    ProjectedImagePointsT, Jacobian_projpT = cv.omnidir.projectPoints(frontCameras.objPoints[idx],
                                                                                      frontCameras.rvecs[idx],
                                                                                      frontCameras.tvecs[idx],
                                                                                      frontCameras.K1,
                                                                                      frontCameras.xiT[0, 0],
                                                                                      frontCameras.D1)
                    # Rodrigues takes a rotation vector and produces a rotation matrix and vice-versa
                    camera_rot_matrix = np.asarray(cv.Rodrigues(frontCameras.rvec)[0])  # _R (3x3)
                    image_rot_matrix = np.asarray(cv.Rodrigues(frontCameras.rvecs[idx])[0])  # _RL (3x3)
                    translation_vector = frontCameras.tvecs[idx].reshape(3, 1)  # _TL (3x1)
                    _RR = np.matmul(camera_rot_matrix, image_rot_matrix)  # (3x3) x (3x3) = (3x3)
                    _TR = np.matmul(camera_rot_matrix,
                                    translation_vector) + frontCameras.tvec  # (3x3) x (3x1) + (3x1) = (3x1)
                    _omR = cv.Rodrigues(_RR)[0]  # (3x1)

                    ProjectedImagePointsB, Jacobian_projpB = cv.omnidir.projectPoints(frontCameras.objPoints[idx], _omR,
                                                                                      _TR,
                                                                                      frontCameras.K2,
                                                                                      frontCameras.xiB[0, 0],
                                                                                      frontCameras.D2)

                    imgPointsTop = frontCameras.imgPoints_1[value].reshape(np.shape(frontCameras.imgPoints_1[value])[0]
                                                                           * np.shape(frontCameras.imgPoints_1[value])[
                                                                               1], 2)
                    imgPointsBot = frontCameras.imgPoints_2[value].reshape(np.shape(frontCameras.imgPoints_2[value])[0]
                                                                           * np.shape(frontCameras.imgPoints_2[value])[
                                                                               1], 2)
                    projPointsTop = ProjectedImagePointsT.reshape(np.shape(ProjectedImagePointsT)[0] *
                                                                  np.shape(ProjectedImagePointsT)[1], 2)
                    projPointsBot = ProjectedImagePointsB.reshape(np.shape(ProjectedImagePointsB)[0] *
                                                                  np.shape(ProjectedImagePointsB)[1], 2)
                    errorTop = imgPointsTop - projPointsTop
                    errorBot = imgPointsBot - projPointsBot

                    reprojErrorT = 0
                    reprojErrorB = 0

                    for i in range(np.shape(errorTop)[0]):
                        reprojErrorT += np.sqrt((errorTop[i, 0]) ** 2 + (errorTop[i, 1]) ** 2)
                        total = i

                        rms_error_top += errorTop[i, 0] ** 2 + errorTop[i, 1] ** 2

                    reprojErrorT /= (total + 1)
                    #print(idx, "Top", reprojErrorT)
                    average_repoj_error_top += reprojErrorT

                    for i in range(np.shape(errorBot)[0]):
                        reprojErrorB += np.sqrt((errorBot[i, 0]) ** 2 + (errorBot[i, 1]) ** 2)
                        total = i

                        rms_error_bot += errorBot[i, 0] ** 2 + errorBot[i, 1] ** 2

                    reprojErrorB /= (total + 1)
                    #print(idx, "Bot", reprojErrorB)
                    average_repoj_error_bot += reprojErrorB

                    reprojError = (reprojErrorB + reprojErrorT) / 2

                    frontCameras.reprojError.append(reprojError)

                    frontCam_used_img_list.append(frontCameras.detect_list_1[value])

                    number_of_images = idx + 1

                average_repoj_error_top /= number_of_images
                average_repoj_error_bot /= number_of_images

                rms_error_top = np.sqrt(rms_error_top) / np.sqrt(2 * (total + 1) * number_of_images)
                rms_error_bot = np.sqrt(rms_error_bot) / np.sqrt(2 * (total + 1) * number_of_images)

                tf_average_reproj_error = average_repoj_error_top
                bf_average_reproj_error = average_repoj_error_bot
                tf_rms_reproj_error = rms_error_top
                bf_rms_reproj_error = rms_error_bot

                #print("Average reprojection error top front: ", average_repoj_error_top)
                #print("Average reprojection error bottom front: ", average_repoj_error_bot)
                #print("RMS Error top: ", rms_error_top)
                #print("RMS Error bot: ", rms_error_bot)

                backCam_used_img_list = []
                average_repoj_error_top = 0
                average_repoj_error_bot = 0
                rms_error_bot = 0
                rms_error_top = 0

                print("Calculating Back Camera Reprojection Error\n")

                for idx, value in enumerate(backCameras.idx[0]):
                    ProjectedImagePointsT, Jacobian_projpT = cv.omnidir.projectPoints(backCameras.objPoints[idx],
                                                                                      backCameras.rvecs[idx],
                                                                                      backCameras.tvecs[idx],
                                                                                      backCameras.K1,
                                                                                      backCameras.xiT[0, 0],
                                                                                      backCameras.D1)
                    # Rodrigues takes a rotation vector and produces a rotation matrix and vice-versa
                    camera_rot_matrix = np.asarray(cv.Rodrigues(backCameras.rvec)[0])  # _R (3x3)
                    image_rot_matrix = np.asarray(cv.Rodrigues(backCameras.rvecs[idx])[0])  # _RL (3x3)
                    translation_vector = backCameras.tvecs[idx].reshape(3, 1)  # _TL (3x1)
                    _RR = np.matmul(camera_rot_matrix, image_rot_matrix)  # (3x3) x (3x3) = (3x3)
                    _TR = np.matmul(camera_rot_matrix,
                                    translation_vector) + backCameras.tvec  # (3x3) x (3x1) + (3x1) = (3x1)
                    _omR = cv.Rodrigues(_RR)[0]  # (3x1)

                    ProjectedImagePointsB, Jacobian_projpB = cv.omnidir.projectPoints(backCameras.objPoints[idx], _omR,
                                                                                      _TR,
                                                                                      backCameras.K2,
                                                                                      backCameras.xiB[0, 0],
                                                                                      backCameras.D2)

                    imgPointsTop = backCameras.imgPoints_1[value].reshape(np.shape(backCameras.imgPoints_1[value])[0]
                                                                          * np.shape(backCameras.imgPoints_1[value])[1],
                                                                          2)
                    imgPointsBot = backCameras.imgPoints_2[value].reshape(np.shape(backCameras.imgPoints_2[value])[0]
                                                                          * np.shape(backCameras.imgPoints_2[value])[1],
                                                                          2)
                    projPointsTop = ProjectedImagePointsT.reshape(np.shape(ProjectedImagePointsT)[0] *
                                                                  np.shape(ProjectedImagePointsT)[1], 2)
                    projPointsBot = ProjectedImagePointsB.reshape(np.shape(ProjectedImagePointsB)[0] *
                                                                  np.shape(ProjectedImagePointsB)[1], 2)
                    errorTop = imgPointsTop - projPointsTop
                    errorBot = imgPointsBot - projPointsBot
                    reprojErrorT = 0
                    reprojErrorB = 0

                    for i in range(np.shape(errorTop)[0]):
                        reprojErrorT += np.sqrt((errorTop[i, 0]) ** 2 + (errorTop[i, 1]) ** 2)
                        total = i

                        rms_error_top += errorTop[i, 0] ** 2 + errorTop[i, 1] ** 2

                    reprojErrorT /= (total + 1)
                    average_repoj_error_top += reprojErrorT

                    for i in range(np.shape(errorBot)[0]):
                        reprojErrorB += np.sqrt((errorBot[i, 0]) ** 2 + (errorBot[i, 1]) ** 2)
                        total = i

                        rms_error_bot += errorBot[i, 0] ** 2 + errorBot[i, 1] ** 2

                    reprojErrorB /= (total + 1)
                    average_repoj_error_bot += reprojErrorB

                    reprojError = (reprojErrorB + reprojErrorT) / 2

                    backCameras.reprojError.append(reprojError)

                    backCam_used_img_list.append(backCameras.detect_list_1[value])

                    number_of_images = idx + 1

                average_repoj_error_top /= number_of_images
                average_repoj_error_bot /= number_of_images

                rms_error_top = np.sqrt(rms_error_top) / np.sqrt(2 * (total + 1) * number_of_images)
                rms_error_bot = np.sqrt(rms_error_bot) / np.sqrt(2 * (total + 1) * number_of_images)

                tb_average_reproj_error = average_repoj_error_top
                bb_average_reproj_error = average_repoj_error_bot
                tb_rms_reproj_error = rms_error_top
                bb_rms_reproj_error = rms_error_bot

                #print("Average reprojection error top back: ", average_repoj_error_top)
                #print("Average reprojection error bottom back", average_repoj_error_bot)
                #print("RMS Error top", rms_error_top)
                #print("RMS Error bot", rms_error_bot)

                R1F, R2F = cv.omnidir.stereoRectify(frontCameras.rvec, frontCameras.tvec)
                R1B, R2B = cv.omnidir.stereoRectify(backCameras.rvec, backCameras.tvec)

                knew = np.array([[640 / (np.pi * 19 / 18), 0, 0], [0, 640 / (np.pi * 19 / 18), 0], [0, 0, 1]],
                                np.double)

                mapTF1, mapTF2 = cv.omnidir.initUndistortRectifyMap(frontCameras.K1, frontCameras.D1, frontCameras.xiT,
                                                                    R1F, knew, (640, 640), cv.CV_32FC1,
                                                                    cv.omnidir.RECTIFY_LONGLATI)  # width, height, cv.CV_16SC2
                mapBF1, mapBF2 = cv.omnidir.initUndistortRectifyMap(frontCameras.K2, frontCameras.D2, frontCameras.xiB,
                                                                    R2F, knew, (640, 640), cv.CV_32FC1,
                                                                    cv.omnidir.RECTIFY_LONGLATI)
                mapTB1, mapTB2 = cv.omnidir.initUndistortRectifyMap(backCameras.K1, backCameras.D1, backCameras.xiT,
                                                                    R1B, knew, (640, 640), cv.CV_32FC1,
                                                                    cv.omnidir.RECTIFY_LONGLATI)
                mapBB1, mapBB2 = cv.omnidir.initUndistortRectifyMap(backCameras.K2, backCameras.D2, backCameras.xiB,
                                                                    R2B, knew, (640, 640), cv.CV_32FC1,
                                                                    cv.omnidir.RECTIFY_LONGLATI)

                tf_img = cv.imread(
                    "C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\data\\calibration_images\\top_front\\tf_0.bmp")
                bf_img = cv.imread(
                    "C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\data\\calibration_images\\bottom_front\\bf_0.bmp")
                tb_img = cv.imread(
                    "C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\data\\calibration_images\\top_back\\tb_0.bmp")
                bb_img = cv.imread(
                    "C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\data\\calibration_images\\bottom_back\\bb_0.bmp")

                undistorted_tf = cv.remap(tf_img, mapTF1, mapTF2, cv.INTER_LINEAR,
                                          borderMode=cv.BORDER_WRAP + cv.BORDER_CONSTANT)
                undistorted_bf = cv.remap(bf_img, mapBF1, mapBF2, cv.INTER_LINEAR,
                                          borderMode=cv.BORDER_WRAP + cv.BORDER_CONSTANT)
                undistorted_tb = cv.remap(tb_img, mapTB1, mapTB2, cv.INTER_LINEAR,
                                          borderMode=cv.BORDER_WRAP + cv.BORDER_CONSTANT)
                undistorted_bb = cv.remap(bb_img, mapBB1, mapBB2, cv.INTER_LINEAR,
                                          borderMode=cv.BORDER_WRAP + cv.BORDER_CONSTANT)

                front_rectified = np.full((640, 1280, 3), (0, 0, 0), dtype=np.uint8)
                back_rectified = np.full((640, 1280, 3), (0, 0, 0), dtype=np.uint8)

                front_rectified[0:640, 0:640] = undistorted_tf
                front_rectified[0:640, 640:1280] = undistorted_bf
                back_rectified[0:640, 0:640] = undistorted_tb
                back_rectified[0:640, 640:1280] = undistorted_bb

                for lines in range(7):
                    lines += 1
                    cv.line(front_rectified, (0, (80 * lines)), (1280, (80 * lines)), (0, 0, 255))
                    cv.line(back_rectified, (0, (80 * lines)), (1280, (80 * lines)), (0, 0, 255))

                cv.imwrite(
                    "C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\data\\calibration_images\\rectified_images\\auto_calibs\\omnidir_front_" + str(
                        calib_number) + ".bmp", front_rectified)
                cv.imwrite(
                    "C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\data\\calibration_images\\rectified_images\\auto_calibs\\omnidir_back_" + str(
                        calib_number) + ".bmp", back_rectified)

                error = [calib_number, tf_average_reproj_error, tf_rms_reproj_error, bf_average_reproj_error, bf_rms_reproj_error,
                 tb_average_reproj_error, tb_rms_reproj_error, bb_average_reproj_error, bb_rms_reproj_error, num_iterations, eps]

                error_file.append(error)

                # Save front camera object as a .xml file
                frontOutFile = open(
                    "C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\data\\calibration_parameters_OCV\\auto_calibs\\front_camera_params_OCV" + str(
                        calib_number) + ".xml", 'wb')
                pickle.dump(frontCameras, frontOutFile)
                frontOutFile.close()
                # Save back camera object as a .xml file
                backOutFile = open(
                    "C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\data\\calibration_parameters_OCV\\auto_calibs\\back_camera_params_OCV" + str(
                        calib_number) + ".xml", 'wb')
                pickle.dump(backCameras, backOutFile)
                backOutFile.close()

        fields = ['Calib number', 'tf reproj error', 'tf reproj rms', 'bf reproj error', 'bf reproj rms',
                  'tb reproj error', 'tb reproj rms', 'bb reproj error', 'bb reproj rms', 'num iterations', 'eps']
        with open('C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\data\\calibration_parameters_OCV\\calib_errors.csv', 'w', newline='') as f:

            write = csv.writer(f)

            write.writerow(fields)
            write.writerows(error_file)

        print(str("Top Front Camera: " + str(num_detected_images_front_top) + "/" + str(
            int(np.shape(tfImgList)[0])) + " Detected"))  # Prints the number of detected images over number of images
        print(str("Bottom Front Camera: " + str(num_detected_images_front_bot) + "/" + str(
            int(np.shape(bfImgList)[0])) + " Detected"))
        print(str("Top Back Camera: " + str(num_detected_images_back_top) + "/" + str(
            int(np.shape(tbImgList)[0])) + " Detected"))
        print(str("Bottom Back Camera: " + str(num_detected_images_back_bot) + "/" + str(
            int(np.shape(bbImgList)[0])) + " Detected"))


      #  save_long_lat_param = input("Save long-lat projection of TB calibration images? (y/n): ")
      #  save_long_lat_flag = 0
      #  while save_long_lat_flag == 0:
        #    if save_long_lat_param == "y":
            #    for idx, value in enumerate(backCameras.idx[0]):
                  #  frame = cv.imread(backCameras.detect_list_1[value])
                    #frame_h, frame_w, channels = frame.shape
                    #knew = np.array(
                       # [[frame_w / (np.pi * 19 / 18), 0, 0], [0, frame_h / (np.pi * 19 / 18), 0], [0, 0, 1]],
                        #np.double)
                    #uFrame = cv.omnidir.undistortImage(frame, backCameras.K1, backCameras.D1, backCameras.xiT,
                       #                                cv.omnidir.RECTIFY_LONGLATI, None, knew)
                    # cv.imshow("BackCamera1", uFrame)
                    #fname = str(OCVcalibImgDir + "\\rectified_images" + imgTB + str(value) + imgFormat)
                    #print(fname)
                    #cv.imwrite(fname, uFrame)
                #save_long_lat_flag = 1
           # elif save_long_lat_param == "n":
              #  save_long_lat_flag = 1
           # else:
               # save_long_lat_param = input("Please select y or n to save long-lat projections: ")

       # save_long_lat_param = input("Save long-lat projection of BB calibration images? (y/n): ")
       # save_long_lat_flag = 0
       # while save_long_lat_flag == 0:
        #    if save_long_lat_param == "y":
              #  for idx, value in enumerate(backCameras.idx[1]):
                  #  frame = cv.imread(backCameras.detect_list_2[value])
                   # frame_h, frame_w, channels = frame.shape
                  #  knew = np.array(
                   #     [[frame_w / (np.pi * 19 / 18), 0, 0], [0, frame_h / (np.pi * 19 / 18), 0], [0, 0, 1]],
                    #    np.double)
                   # uFrame = cv.omnidir.undistortImage(frame, backCameras.K2, backCameras.D2, backCameras.xiB,
                   #                                    cv.omnidir.RECTIFY_LONGLATI, None, knew)
                   # cv.imshow("Wooooo", uFrame)
                   # fname = str(OCVcalibImgDir + "\\rectified_images" + imgBB + str(value) + imgFormat)
                    #cv.imwrite(fname, uFrame)
                    #cv.waitKey(1000)
                #save_long_lat_flag = 1
           # elif save_long_lat_param == "n":
                #save_long_lat_flag = 1
            #else:
                #save_long_lat_param = input("Please select y or n to save long-lat projections: ")


    #return frontCameras, backCameras