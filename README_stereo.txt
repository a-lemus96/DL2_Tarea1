MPI-Sintel, Stereo Training Data
================================

Copyright (c) 2015 Jonas Wulff, Michael Black.
Max Planck Institute for Intelligent Systems, Tuebingen

************************************
* NOTE:
* This is the beta version of the dataset, and might change
* in the near future. We will keep registered users up-to-date,
* but please make sure to check www.mpi-sintel.de for the latest
* version.
************************************

For questions and comments please contact sintel@tue.mpg.de.


CHANGELOG
=========
2015/02/03      First version of the archives



INTRODUCTION
============

This archive contains the beta version of the Sintel-stereo training set.
For each of the training sequences, it contains:
- The "clean" and "final" passes rendered from two cameras with a baseline of
  10 cm apart.
- The negative disparities.
- A mask showing the occluded pixels (which are visible in the left frame, but
  occluded in the right frame).
- A mask showing the out-of-frame pixels, which are visible in the left frame,
  but leave the visible area in the right frame. These pixels are not taken
  into account for the evaluation.

For all data, the left frame is the reference frame.

Note that due to the non-physicality of the atmospheric effects, these were
removed from the final pass. Therefore, the final pass images are different
from the ones that are published as part of the optical flow benchmark. 


DIRECTORY STRUCTURE
===================

training/
    Training set.

training/clean_left/
training/clean_right/
    The "clean" pass images, rendered from the left and right camera.

training/final_left/
training/final_left/
    The "final" pass images, rendered from the left and right camera.

training/disparities/
    The negative disparities, 16 bit encoded in PNG files.

training/disparities_viz/
    Visualization of the disparities.

training/occlusions/
    Occluded pixels

training/outofframe/
    Pixel that are outside of the frame in the right view.

sdk/
    I/O scripts for MATLAB and Python.



DATA FORMAT
===========

Images:
    Standard 8 bit RGB PNG images.

Masks (occlusions, outofframe):
    Given as binary, single-channel PNG files, where a pixel is set to 1 if it
    is not visible in the right view.

Disparities:
    The disparities are given with 16 bit accuracy, for better compatibility
    encoded in the R and G channels of 8-bit PNG images. This introduces a
    slight error, which never exceeds 0.016 pixel over the whole Sintel
    dataset. We believe this accuracy is sufficient for disparities.

    Since the images in the dataset have a width of 1024 pixel, we cap the disparities at this value.
    Given a disparity map DISP and the channels R and G of an image, the conversion is:

    R = floor(DISP/4)
    G = floor(DISP * (2^6) % 256)

    and

    DISP = R * 4 + G / (2^6)


Example I/O scripts for Python and MATLAB are included in the sdk/ folder. The
function names are formed as disparity_{read/write}.

To read a disparity image into Python, for example, use

>> import sintel_io
>> disparity = sintel_io.disparity_read('/some/filename/frame_0001.png')



FURTHER INFORMATION
===================

More information and data can be obtained from http://sintel.is.tue.mpg.de.

The original dataset is published as
    Butler, D., Wulff, J., Stanley, G., Black, M.:
    "A naturalistic open source movie for optical flow evaluation", ECCV 2012
    
A more technical account can be found in
    Wulff, J., Butler, D., Stanley, G., Black, M.:
    "Lessons and insights from creating a synthetic optical flow benchmark",
    ECCV 2012, Workshop on Unsolved Problems in Optical Flow and Stereo 
    Estimation
    
We are currently preparing a comprehensive journal paper. If you use this work
before then, please cite:

@inproceedings{Butler:ECCV:2012,
  title = {A naturalistic open source movie for optical flow evaluation},
  author = {Butler, D. J. and Wulff, J. and Stanley, G. B. and Black, M. J.},
  booktitle = {European Conf. on Computer Vision (ECCV)},
  editor = {{A. Fitzgibbon et al. (Eds.)}},
  publisher = {Springer-Verlag},
  series = {Part IV, LNCS 7577},
  month = {oct},
  pages = {611--625},
  year = {2012}
}

