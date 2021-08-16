from OCamCalib_model import *
import numpy as np
import cv2 as cv

def world2cam(point3D, ocam_model, padx, pady):
    invpol = ocam_model.invPolyCoefs
    invpol_length = len(invpol)
    xc = ocam_model.centerCords[0] + (padx / 2)
    yc = ocam_model.centerCords[1] + (pady / 2)
    c = ocam_model.affineCoefs[0]
    d = ocam_model.affineCoefs[1]
    e = ocam_model.affineCoefs[2]
    width = ocam_model.imSize[1]
    height = ocam_model.imSize[0]
    norm = np.sqrt(point3D[0]*point3D[0] + point3D[1]*point3D[1])
    if norm == 0:
        theta = np.pi/2
    else:
        theta = np.arctan(point3D[2]/norm)
    point2D = [None, None]

    if norm != 0:
        invnorm = 1/norm
        t = theta
        rho = invpol[0]
        t_i = 1

        for i in range(invpol_length):
            if i != 0:
                t_i *= t
                rho += t_i*invpol[i]

        x = point3D[0]*invnorm*rho
        y = point3D[1]*invnorm*rho

        point2D[0] = x*c + y*d + xc
        point2D[1] = x*e + y + yc
    else:
        point2D[0] = xc
        point2D[1] = yc

    return point2D

#height and width of source image
#padx = width of padded image - width of source image
#pady = height of padded image - height of source image
def perspective_undistortion_LUT(ocam_model, sf, height, width, padx, pady):

    mapx = np.zeros((height + pady, width + padx), dtype=np.float32)
    mapy = np.zeros((height + pady, width + padx), dtype=np.float32)
    Nxc = (height + pady) / 2 # (height + pady) / 2
    Nyc = (width + padx) / 2 # (width + padx) / 2
    Nz = -(width + padx) / sf # -(width + padx) / sf
    M = [None, None, None]

    scale_x = (width + padx) / (width)
    scale_y = (height + pady) / (height)

    for i in range(int(height + pady)): # height + pady
        for j in range(int(width + padx)): # width + padx
            M[0] = (i - Nxc) * scale_x
            M[1] = (j - Nyc) * scale_y
            M[2] = Nz
            m = world2cam(M, ocam_model, padx, pady)
            mapx[i, j] = m[1]
            mapy[i, j] = m[0]

    return mapx, mapy

