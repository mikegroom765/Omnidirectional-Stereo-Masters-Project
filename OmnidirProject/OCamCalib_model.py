#filepath should include filename!

import os
import numpy as np

class OCamCalib_model:
    def __init__(self):
        self.polyCoefs = []
        self.invPolyCoefs = []
        self.centerCords = []
        self.affineCoefs = []
        self.imSize = []

    def readOCamFile(self, filepath):
        if os.path.isfile(filepath):
            file = open(filepath, "r")

            for idx, line in enumerate(file):
                #  Read polynomial coefficients
                if idx == 2:
                    polyCoefs_temp = line.split()
                    #  For some reason they are saved as the wrong sign (+ve should be -ve and vice versa)
                    for idx, value in enumerate(polyCoefs_temp):
                        if not idx == 0:
                            self.polyCoefs.append(float(value))

                #  Read inverse polynomial coefficients
                if idx == 6:
                    invPolyCoefs_temp = line.split()
                    #  Might also be inverted? Not sure if i'll need these
                    for idx, value in enumerate(invPolyCoefs_temp):
                        if not idx == 0:
                            self.invPolyCoefs.append(float(value))

                #  Read center coordinates
                if idx == 10:
                    self.centerCords = line.split()
                    self.centerCords = [float(self.centerCords[0]), float(self.centerCords[1])]

                #  Read affine coefficients
                if idx == 14:
                    self.affineCoefs = line.split()
                    self.affineCoefs = [float(self.affineCoefs[0]), float(self.affineCoefs[1]), float(self.affineCoefs[2])]

                #  Read image size
                if idx == 18:
                    self.imSize = line.split()
                    self.imSize = [int(self.imSize[0]), int(self.imSize[1])]

            file.close()

        else:
            print("The OCamCalib text file at " + str(filepath) + " could not be opened!")
            raise Exception()