import glob
from detectChessboardCorners import detectChessboardConrner
from stereocam import *
from OCamCalib_model import *
from perspective_undistortion_LUT_OCamCalib import *
import time
import pickle
import csv

frontCameras = StereoCam()
backCameras = StereoCam()
calib_flags = cv.CALIB_USE_INTRINSIC_GUESS
 #  cv.CALIB_FIX_PRINCIPAL_POINT  cv.CALIB_USE_INTRINSIC_GUESS

#  This is the final image width/height after adding padding! (square images only!)
pad = 1280

image_w = 640
image_h = 640

def improvedOCamCalib(calibImgDir, imgFormat, boardSize, imgTF, imgTB, imgBF, imgBB, square_width, square_height, SHOW_CALIB_IMG, OCamCalibDir, OCamCalibTF, OCamCalibTB, OCamCalibBF, OCamCalibBB, sf):
    calib_key = 0

    #list of directories of calib images that haven't been distorted
    tfImgList = sorted(glob.glob(calibImgDir + imgTF + "*" + imgFormat))
    tbImgList = sorted(glob.glob(calibImgDir + imgTB + "*" + imgFormat))
    bfImgList = sorted(glob.glob(calibImgDir + imgBF + "*" + imgFormat))
    bbImgList = sorted(glob.glob(calibImgDir + imgBB + "*" + imgFormat))

    if not (tfImgList or tbImgList or bfImgList or bbImgList):
        print("Images not found in directory :" + calibImgDir)
    else:
        print("Image lists generated!")

    print("Reading the calib_result.txt files for the cameras...")

    tf_OCam = OCamCalib_model()
    tb_OCam = OCamCalib_model()
    bf_OCam = OCamCalib_model()
    bb_OCam = OCamCalib_model()


    tf_OCam.readOCamFile(OCamCalibDir + OCamCalibTF)
    tb_OCam.readOCamFile(OCamCalibDir + OCamCalibTB)
    bf_OCam.readOCamFile(OCamCalibDir + OCamCalibBF)
    bb_OCam.readOCamFile(OCamCalibDir + OCamCalibBB)

    print("Undistorting calib images...")

    padx = pad - image_w
    pady = pad - image_h

    tf_mapx, tf_mapy = perspective_undistortion_LUT(tf_OCam, sf, image_h, image_w, padx, pady)
    tb_mapx, tb_mapy = perspective_undistortion_LUT(tb_OCam, sf, image_h, image_w, padx, pady)
    bf_mapx, bf_mapy = perspective_undistortion_LUT(bf_OCam, sf, image_h, image_w, padx, pady)
    bb_mapx, bb_mapy = perspective_undistortion_LUT(bb_OCam, sf, image_h, image_w, padx, pady)

    for idx, value in enumerate(tfImgList):
        image = cv.imread(value)
        image = pad_image(image, pad)
        undistorted_image = cv.remap(image, tf_mapx, tf_mapy, cv.INTER_LINEAR)
        cv.imwrite("C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\data\\calibration_images\\OCamCalib_images\\tf\\tf_" + str(idx) + imgFormat, undistorted_image)
    for idx, value in enumerate(tbImgList):
        image = cv.imread(value)
        image = pad_image(image, pad)
        undistorted_image = cv.remap(image, tb_mapx, tb_mapy, cv.INTER_LINEAR)
        cv.imwrite("C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\data\\calibration_images\\OCamCalib_images\\tb\\tb_" + str(idx) + imgFormat, undistorted_image)
    for idx, value in enumerate(bfImgList):
        image = cv.imread(value)
        image = pad_image(image, pad)
        undistorted_image = cv.remap(image, bf_mapx, bf_mapy, cv.INTER_LINEAR)
        cv.imwrite("C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\data\\calibration_images\\OCamCalib_images\\bf\\bf_" + str(idx) + imgFormat, undistorted_image)
    for idx, value in enumerate(bbImgList):
        image = cv.imread(value)
        image = pad_image(image, pad)
        undistorted_image = cv.remap(image, bb_mapx, bb_mapy, cv.INTER_LINEAR)
        cv.imwrite("C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\data\\calibration_images\\OCamCalib_images\\bb\\bb_" + str(idx) + imgFormat, undistorted_image)

    print("Detecting chessboard corners ...")

    tfImgList = sorted(glob.glob("C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\data\\calibration_images\\OCamCalib_images\\tf\\tf_" + "*" + imgFormat))
    tbImgList = sorted(glob.glob("C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\data\\calibration_images\\OCamCalib_images\\tb\\tb_" + "*" + imgFormat))
    bfImgList = sorted(glob.glob("C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\data\\calibration_images\\OCamCalib_images\\bf\\bf_" + "*" + imgFormat))
    bbImgList = sorted(glob.glob("C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\data\\calibration_images\\OCamCalib_images\\bb\\bb_" + "*" + imgFormat))

    result_front = detectChessboardConrner(tfImgList, frontCameras.detect_list_1, bfImgList, frontCameras.detect_list_2,
                                           tuple(boardSize.size()), frontCameras.imgSize_1, frontCameras.imgSize_2,
                                           SHOW_CALIB_IMG, 0)

    result_back = detectChessboardConrner(tbImgList, backCameras.detect_list_1, bbImgList, backCameras.detect_list_2,
                                          tuple(boardSize.size()), backCameras.imgSize_1, backCameras.imgSize_2,
                                          SHOW_CALIB_IMG, 1)

    if result_front[4] == 0:
        print("Not enough corner detected front images\n")
        calib_key = 1
    else:
        frontCameras.imgPoints_1 = np.float32(result_front[0])
        frontCameras.imgPoints_2 = np.float32(result_front[1])
        frontCameras.imgSize_1 = result_front[2]
        frontCameras.imgSize_2 = result_front[3]

    if result_back[4] == 0:
        print("Not enough corner detected front images\n")
        calib_key = 1
    else:
        backCameras.imgPoints_1 = np.float32(result_back[0])
        backCameras.imgPoints_2 = np.float32(result_back[1])
        backCameras.imgSize_1 = result_back[2]
        backCameras.imgSize_2 = result_back[3]

    if not calib_key:
        num_detected_images_front_top = int(np.shape(frontCameras.detect_list_1)[0])

        num_detected_images_front_bot = int(np.shape(frontCameras.detect_list_2)[0])

        num_detected_images_back_top = int(np.shape(backCameras.detect_list_1)[0])

        num_detected_images_back_bot = int(np.shape(backCameras.detect_list_2)[0])

        print(str("Top Front Camera: " + str(num_detected_images_front_top) + "/" + str(
                int(np.shape(tfImgList)[
                        0])) + " Detected"))  # Prints the number of detected images over number of images
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

        print("Starting calibrations!")
        calib_number = 0
        error_file = []

        cv.destroyAllWindows()

        for eps_iter in range(2):
            for num_iterations_iter in range(4):
                calib_number += 1

                print("Starting calibration number: " + str(calib_number))

                num_iterations = 100 * (num_iterations_iter + 1)
                eps = 0.000001 / (10 ** eps_iter)

                term_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, num_iterations, eps)  # 1, 0.001

                calib_flags = 0 # cv.CALIB_USE_INTRINSIC_GUESS

                print("Front camera calibration ...\n")

                retval, frontCameras.K1, frontCameras.D1, rvecs_f1, tvecs_f1 = cv.calibrateCamera(
                    frontCameras.objPoints,
                    frontCameras.imgPoints_1,
                    (pad, pad), None, None)

                retval, frontCameras.K2, frontCameras.D2, rvecs_f2, tvecs_f2 = cv.calibrateCamera(
                    frontCameras.objPoints,
                    frontCameras.imgPoints_2,
                    (pad, pad), None, None)

                K_guess = np.array([[127.39151821, 0, 639.48517708],[0, 127.58784237, 639.53524461],[0, 0, 1]], np.double)
                D_guess = np.array([[-9.21032160e-04, 7.47354769e-05, -4.91948667e-04, -2.93645279e-04, -3.03004692e-06]],np.double)

                retval1, frontCameras.K1, frontCameras.D1, frontCameras.K2, frontCameras.D2, \
                frontCameras.rvec, frontCameras.tvec, E, F = cv.stereoCalibrate(frontCameras.objPoints,
                                                                                frontCameras.imgPoints_1,
                                                                                frontCameras.imgPoints_2,
                                                                                K_guess, #frontCameras.K1,
                                                                                D_guess, K_guess, # frontCameras.D1, frontCameras.K2,
                                                                                D_guess, #frontCameras.D2,
                                                                                (pad, pad), flags=calib_flags,
                                                                                criteria=term_criteria)

                print("Back camera calibration ...\n")

                retval, backCameras.K1, backCameras.D1, rvecs_b1, tvecs_b1 = cv.calibrateCamera(backCameras.objPoints,
                                                                                                backCameras.imgPoints_1,
                                                                                                (pad, pad), None, None)

                retval, backCameras.K2, backCameras.D2, rvecs_b2, tvecs_b2 = cv.calibrateCamera(backCameras.objPoints,
                                                                                                backCameras.imgPoints_2,
                                                                                                (pad, pad), None, None)

                retval2, backCameras.K1, backCameras.D1, backCameras.K2, backCameras.D2, \
                backCameras.rvec, backCameras.tvec, E, F = cv.stereoCalibrate(backCameras.objPoints,
                                                                              backCameras.imgPoints_1,
                                                                              backCameras.imgPoints_2, K_guess, #backCameras.K1,
                                                                              D_guess, K_guess, # backCameras.D1, backCameras.K2,
                                                                              D_guess,
                                                                              (pad, pad), flags=calib_flags,
                                                                              criteria=term_criteria)

                print("Done!\n")

                frontCameras.OCamCalibSave()
                backCameras.OCamCalibSave()

                average_repoj_error_top = 0
                average_repoj_error_bot = 0
                rms_error_bot = 0
                rms_error_top = 0

                print("Calculating Front Camera Reprojection Error\n")

                for idx, value in enumerate(frontCameras.detect_list_1):

                    ProjectedImagePointsT, Jacobian_projpT = cv.projectPoints(frontCameras.objPoints[idx],
                                                                              rvecs_f1[idx], tvecs_f1[idx],
                                                                              frontCameras.K1, frontCameras.D1)

                    # camera_rot_matrix = np.asarray(cv.Rodrigues(frontCameras.rvec)[0])  # _R (3x3)
                    # image_rot_matrix = np.asarray(cv.Rodrigues(frontCameras.rvecs[idx])[0])  # _RL (3x3)
                    # translation_vector = frontCameras.tvecs[idx].reshape(3, 1)  # _TL (3x1)
                    # _RR = np.matmul(camera_rot_matrix, image_rot_matrix)  # (3x3) x (3x3) = (3x3)
                    # _TR = np.matmul(camera_rot_matrix, translation_vector) + frontCameras.tvec  # (3x3) x (3x1) + (3x1) = (3x1)
                    # _omR = cv.Rodrigues(_RR)[0]  # (3x1)

                    ProjectedImagePointsB, Jacobian_projpB = cv.projectPoints(frontCameras.objPoints[idx],
                                                                              rvecs_f2[idx], tvecs_f2[idx],
                                                                              frontCameras.K2, frontCameras.D2)

                    imgPointsTop = frontCameras.imgPoints_1[idx]
                    imgPointsBot = frontCameras.imgPoints_2[idx]
                    projPointsTop = ProjectedImagePointsT[:, 0, :]
                    projPointsBot = ProjectedImagePointsB[:, 0, :]
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

                    number_of_images = idx + 1

                average_repoj_error_top /= number_of_images
                average_repoj_error_bot /= number_of_images

                rms_error_top = np.sqrt(rms_error_top) / np.sqrt(2 * (total + 1) * number_of_images)
                rms_error_bot = np.sqrt(rms_error_bot) / np.sqrt(2 * (total + 1) * number_of_images)

                #print("Average reprojection error top front: ", average_repoj_error_top)
                #print("Average reprojection error bottom front: ", average_repoj_error_bot)
                #print("RMS Error top: ", rms_error_top)
                #print("RMS Error bot: ", rms_error_bot)

                tf_average_reproj_error = average_repoj_error_top
                bf_average_reproj_error = average_repoj_error_bot
                tf_rms_reproj_error = rms_error_top
                bf_rms_reproj_error = rms_error_bot

                average_repoj_error_top = 0
                average_repoj_error_bot = 0
                rms_error_bot = 0
                rms_error_top = 0

                print("Calculating Back Camera Reprojection Error\n")

                for idx, value in enumerate(backCameras.detect_list_1):

                    ProjectedImagePointsT, Jacobian_projpT = cv.projectPoints(backCameras.objPoints[idx],
                                                                              rvecs_b1[idx], tvecs_b1[idx],
                                                                              backCameras.K1, backCameras.D1)

                    # camera_rot_matrix = np.asarray(cv.Rodrigues(backCameras.rvec)[0])  # _R (3x3)
                    # image_rot_matrix = np.asarray(cv.Rodrigues(backCameras.rvecs[idx])[0])  # _RL (3x3)
                    # translation_vector = backCameras.tvecs[idx].reshape(3, 1)  # _TL (3x1)
                    # _RR = np.matmul(camera_rot_matrix, image_rot_matrix)  # (3x3) x (3x3) = (3x3)
                    # _TR = np.matmul(camera_rot_matrix, translation_vector) + backCameras.tvec  # (3x3) x (3x1) + (3x1) = (3x1)
                    # _omR = cv.Rodrigues(_RR)[0]  # (3x1)

                    ProjectedImagePointsB, Jacobian_projpB = cv.projectPoints(backCameras.objPoints[idx],
                                                                              rvecs_b2[idx], tvecs_b2[idx],
                                                                              backCameras.K2, backCameras.D2)
                    imgPointsTop = backCameras.imgPoints_1[idx]
                    imgPointsBot = backCameras.imgPoints_2[idx]
                    projPointsTop = ProjectedImagePointsT[:, 0, :]
                    projPointsBot = ProjectedImagePointsB[:, 0, :]
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

                    number_of_images = idx + 1

                average_repoj_error_top /= number_of_images
                average_repoj_error_bot /= number_of_images

                rms_error_top = np.sqrt(rms_error_top) / np.sqrt(2 * (total + 1) * number_of_images)
                rms_error_bot = np.sqrt(rms_error_bot) / np.sqrt(2 * (total + 1) * number_of_images)

                #print("Average reprojection error top back: ", average_repoj_error_top)
                #print("Average reprojection error bottom back", average_repoj_error_bot)
                #print("RMS Error top", rms_error_top)
                #print("RMS Error bot", rms_error_bot)

                tb_average_reproj_error = average_repoj_error_top
                bb_average_reproj_error = average_repoj_error_bot
                tb_rms_reproj_error = rms_error_top
                bb_rms_reproj_error = rms_error_bot

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
                                                                                  frontCameras.tvec)

                R1B, R2B, P1B, P2B, QB, validPix1B, validPix2B = cv.stereoRectify(backCameras.K1, backCameras.D1,
                                                                                  backCameras.K2,
                                                                                  backCameras.D2, (1280, 1280),
                                                                                  backCameras.rvec,
                                                                                  backCameras.tvec)

                mapTF1, mapTF2 = cv.initUndistortRectifyMap(frontCameras.K1, frontCameras.D1, R1F, P1F, (1280, 1280),
                                                            cv.CV_32FC1)

                mapBF1, mapBF2 = cv.initUndistortRectifyMap(frontCameras.K2, frontCameras.D2, R2F, P2F, (1280, 1280),
                                                            cv.CV_32FC1)

                mapTB1, mapTB2 = cv.initUndistortRectifyMap(backCameras.K1, backCameras.D1, R1B, P1B, (1280, 1280),
                                                            cv.CV_32FC1)

                mapBB1, mapBB2 = cv.initUndistortRectifyMap(backCameras.K2, backCameras.D2, R2B, P2B, (1280, 1280),
                                                            cv.CV_32FC1)

                tf_rectified_img = cv.remap(tf_undistorted_image, mapTF1, mapTF2, cv.INTER_LINEAR,
                                            borderMode=cv.BORDER_CONSTANT)
                bf_rectified_img = cv.remap(bf_undistorted_image, mapBF1, mapBF2, cv.INTER_LINEAR,
                                            borderMode=cv.BORDER_CONSTANT)
                tb_rectified_img = cv.remap(tb_undistorted_image, mapTB1, mapTB2, cv.INTER_LINEAR,
                                            borderMode=cv.BORDER_CONSTANT)
                bb_rectified_img = cv.remap(bb_undistorted_image, mapBB1, mapBB2, cv.INTER_LINEAR,
                                            borderMode=cv.BORDER_CONSTANT)

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

                cv.imwrite(
                    "C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\data\\calibration_images\\rectified_images\\auto_calibs\\ocam_front_" + str(
                        calib_number) + ".bmp", front_rectified)
                cv.imwrite(
                    "C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\data\\calibration_images\\rectified_images\\auto_calibs\\ocam_back_" + str(
                        calib_number) + ".bmp", back_rectified)

                error = [calib_number, tf_average_reproj_error, tf_rms_reproj_error, bf_average_reproj_error,
                                   bf_rms_reproj_error,
                                   tb_average_reproj_error, tb_rms_reproj_error, bb_average_reproj_error,
                                   bb_rms_reproj_error, num_iterations, eps]

                error_file.append(error)

                # Save front camera object as a .xml file
                frontOutFile = open(
                    "C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\data\\calibration_parameters_OCamCalib\\auto_calibs\\front_camera_params_OCam" + str(
                        calib_number) + ".xml", 'wb')
                pickle.dump(frontCameras, frontOutFile)
                frontOutFile.close()
                # Save back camera object as a .xml file
                backOutFile = open(
                    "C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\data\\calibration_parameters_OCamCalib\\auto_calibs\\back_camera_params_OCam" + str(
                        calib_number) + ".xml", 'wb')
                pickle.dump(backCameras, backOutFile)
                backOutFile.close()

        fields = ['Calib number', 'tf reproj error', 'tf reproj rms', 'bf reproj error', 'bf reproj rms',
                  'tb reproj error', 'tb reproj rms', 'bb reproj error', 'bb reproj rms', 'num iterations', 'eps']
        with open('C:\\Users\\Len\\PycharmProjects\\OmnidirProject\\data\\calibration_parameters_OCamCalib\\calib_errors.csv', 'w',  newline='') as f:

            write = csv.writer(f)

            write.writerow(fields)
            write.writerows(error_file)

        print(str("Top Front Camera: " + str(num_detected_images_front_top) + "/" + str(
            int(np.shape(tfImgList)[
                    0])) + " Detected"))  # Prints the number of detected images over number of images
        print(str("Bottom Front Camera: " + str(num_detected_images_front_bot) + "/" + str(
            int(np.shape(bfImgList)[0])) + " Detected"))
        print(str("Top Back Camera: " + str(num_detected_images_back_top) + "/" + str(
            int(np.shape(tbImgList)[0])) + " Detected"))
        print(str("Bottom Back Camera: " + str(num_detected_images_back_bot) + "/" + str(
            int(np.shape(bbImgList)[0])) + " Detected"))

def pad_image(img, pad):

    h, w, c = np.shape(img)
    result = np.full((pad, pad, c), (0, 0, 0), dtype=np.uint8)

    pad_h = (pad - h) // 2
    pad_w = (pad - w) // 2

    result[pad_h:pad_h + h, pad_w:pad_w + w] = img

    return result