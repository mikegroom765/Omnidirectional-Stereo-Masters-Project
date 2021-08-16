import cv2 as cv
import numpy as np

#  list1 - globVector of strings with image directories for top camera
#  list_detected_1 - takes a detected_list_1 of a stereoCam object as input, is a list of all images from top camera with detected chessboard corners
#  list2 - globVector of strings with image directories for bottom camera
#  list_detected_2 - takes a detected_list_2 of a stereoCam object as input, is a list of all images from bottom camera with detected chessboard corners
#  image_points_1 - list of lists of all detected points on chessboard from top camera (from stereoCam object)
#  image_points_2 - list of lists of all detected points on chessboard from bottom camera (from stereoCam object)
#  board_size - number of inner corners on chessboard
#  image_size_1 - passes imgsize variable from the a stereoCam object (top camera)
#  image_size_2 - passes imgsize variable from the a stereoCam object (bottom camera)
#  show_img - if true chessboard corners are drawn on image

def detectChessboardConrner(list1, list_detected_1, list2, list_detected_2, board_size, image_size_1, image_size_2, show_img, test):

    image_points_1_list = []
    image_points_2_list = []
    not_found_bottom = []
    not_found_top = []
    count = 0
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    #  num_not_detected = 0

    for idx, value in enumerate(list1):
        img_top_colour = cv.imread(list1[idx])
        img_bot_colour = cv.imread(list2[idx])
        img_top = cv.cvtColor(img_top_colour, cv.COLOR_BGR2GRAY)
        img_bot = cv.cvtColor(img_bot_colour, cv.COLOR_BGR2GRAY)

        image_size_1 = img_top.shape[::-1]
        image_size_2 = img_bot.shape[::-1]

        found_top, points_top = cv.findChessboardCorners(img_top, board_size)
        found_bot, points_bot = cv.findChessboardCorners(img_bot, board_size) # cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_FILTER_QUADS

        print("Found top: " + str(found_top))
        print("Found bottom:" + str(found_bot))

        if found_top and found_bot:
            key = None
            cv.cornerSubPix(img_top, points_top, (5, 5), (-1, -1), criteria)
            cv.cornerSubPix(img_bot, points_bot, (5, 5), (-1, -1), criteria)
            if show_img:
                cv.drawChessboardCorners(img_top, board_size, points_top, found_top)
                cv.drawChessboardCorners(img_bot, board_size, points_bot, found_bot)
                cv.imshow("Top", img_top)
                cv.imshow("Bottom", img_bot)
                key = cv.waitKey(0)
            else:
                key = ord('y')
            if key == ord('y'):
                if not type(points_top) == cv.CV_64F:
                    points_top = np.double(points_top)
                if not type(points_bot) == cv.CV_64F:
                    points_bot = np.double(points_bot)

                points_top = points_top[:, 0, :]
                points_bot = points_bot[:, 0, :]

                image_points_1_list.insert(count, points_top)  # Creates a list of all corners in all detected images
                image_points_2_list.insert(count, points_bot)

                list_detected_1.append(list1[idx])  # Creates list of all detected images
                list_detected_2.append(list2[idx])
                count += 1
        else:
            print(list1[idx])
            if found_top:
                not_found_bottom.append(list2[idx])
            if found_bot:
                not_found_top.append(list1[idx])

    print("not found bottom", not_found_bottom)
    print("not found top", not_found_top)
    image_points_1 = np.array(image_points_1_list)
    image_points_2 = np.array(image_points_2_list)

    if found_top and found_bot:
        cv.destroyWindow("Top")
        cv.destroyWindow("Bottom")

    if np.size(image_points_1) <= 3:
        #  print(np.size(image_points_1))
        return [0, 0, 0, 0, 0]
    else:
        return [image_points_1, image_points_2, image_size_1, image_size_2, 1]

