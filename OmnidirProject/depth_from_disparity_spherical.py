import cv2 as cv
import numpy as np

def depth_from_disp_spherical(img, tvec):

    f_s = 640 / (190 * (2 * np.pi / 360))

    depth = np.full((640, 640), 0, dtype=np.float32)

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    for j in range(img.shape[0]): # i
        for i in range(img.shape[1]): # j

            if not img[j, i] == 0:
                theta_t = j / f_s
                phi_t = i / f_s
                phi_b = (i - img[j, i]) / f_s
                rad_disp = img[j, i] / f_s
                b = np.sqrt(tvec[0] ** 2 + tvec[1] ** 2 + tvec[2] ** 2)
                b = b / 1000
                rho_t = b[0] * (np.sin(phi_b) / np.sin(rad_disp))
                # z = rho_t * np.sin(phi_t) * np.sin(theta_t)
                z = rho_t

                depth[j, i] = z.astype('float32')

    return depth
