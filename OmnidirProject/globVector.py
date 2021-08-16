import glob
import numpy as np

# Takes a pathname and returns a vector called files of pathnames with the a similar pattern

def globVector(pathname):
    glob_result = glob.glob(pathname)
    glob_result = sorted(glob_result)
    print(glob_result)

    return glob_result