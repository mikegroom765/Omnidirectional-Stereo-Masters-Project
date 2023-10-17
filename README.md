# Omnidirectional-Stereo-Masters-Project
This repository contains the results of my Final Year Research & Development Project at Durham University.
There are a few files/folders missing because they are larger than the file size limit of 25MB contain too many files. 
The most important missing folder is named data, located in the OmnidirProject folder, to see the file structure i've recreated it in a google drive: 

https://drive.google.com/drive/folders/1jPppsRbSW1FJ_pvVocgB9Cv6IUbYQ_hv?usp=sharing

Omnidirectional Stereo Vision (2021) - Durham University
Author: Michael Groom
Supervised by: Prof. Toby Breckon
Date: 15/06/2021
Contact: mikegroom765@gmail.com

############################# Introduction #####################################
This program was part of a Final Year Research & Development Project at Durham
University. The research was based upon a previous Final Year Project conducted 
by Kaiwen Lin and supervised by Prof. Toby Breckon in 2017 which implemented a
omnidirectional stereo vision system using two Ricoh Theta S cameras set up in a
top-bottom topology. During this project the stereo rig system was redesigned for 
more stability, placing the cameras in a new orientation. The previous C++ 
implementation was reimplemented in Python, along with additional features such 
as a new camera calibration technique and new stereo matching algorithms. 

It is used to:
   - Generate Images for Omnidirectional Stereo Calibration
   - Calibrate the Omnidirection Stereo rig using two available methods
	- OpenCV omnidir
	- ImprovedOCamCalib
   - Compute Disparity Maps using
	- OpenCV SGBM
	- OpenCV CUDA SGM
	- OpenCV SGBM Left-Right Right-Left WLS
	- PSMNet

############################# Requirements ####################################
- Python3
- OpenCV 4.5.1 (with CUDA and ccalib)
- PyTorch (1.6.0+)
- 2 Ricoh Theta S Cameras (Run in Live Mode) connected on the same USB peripheral

I used Pycharm with Anaconda on Windows 10. The code assumes the project is located 
at the directory below. To use you will either have to create the below directory
and place the project folder there or go thorugh the code and change each file location.

C:\Users\Len\PycharmProjects\OmnidirProject

################################# Run #########################################

Spherical Camera Model (OCV):

When calibrating the cameras using this code it will run through the calibration
several times using different termintation criteria on the Levenberg-Marquardt
optimization algorithm as when using it i've found this process to optimization
to be quite unstable. The results of the calibrations are saved to a csv file 
and rectified images are produced to visually check if the horizontal scanlines 
in the two images line up nicely. Good reprojection error results don't always
translate to good rectifications. The chosen calibrations will have to be moved
to the caibration_parameters_OCV folder and renamed, removing the caibration
number from the filename so the two .xml files are names "back_camera_params_OCV.xml"
and "front_camera_params_OCV.xml".

Planar Camera Model (Scaramuzza's Model/OCamCalib):

To calibrate cameras using this method individual cameras first need to calibrated
using Scaramuzza's omnidirectional camera calibration toolbox for Matlab.

Link: https://sites.google.com/site/scarabotix/ocamcalib-toolbox

Calibration results can be saved to text files using this toolbox, which need to be 
renamed to "calib_results_NAME.txt" where NAME is the name of the camera, either bb,
bb, tb or tf. For example, the calibration results of the top back camera should be
saved as "calib_results_tb.txt". 
Once all 4 cameras are calibrated using this toolbox they then need to be calibrated
using the OCamCalib calibration method in the main code. This will produced .xml files
named "back_camera_params_OCam.xml" and "front_camera_params_OCam.xml" containing 
camera parameters. The 4 text files produced from the OCamCalib Matlab toolbox and these
two .xml files should be placed in the calibration_paramters_OCamCalib folder. 

########################### Paper ###############################################

The paper we published from this project is available here: https://ieeexplore.ieee.org/document/9956197
A free version is also available here: https://breckon.org/toby/publications/papers/groom22omnidirectional.pdf

################################ Notes ########################################

4point_test.py is the code used to sample disparity maps that is mentioned in the 
report. 
