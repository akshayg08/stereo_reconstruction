# stereo_reconstruction

## Overview 

This GitHub repository contains an implementation of 3D reconstruction using Binocular Stereo. The project incorporates several key algorithms to achieve accurate reconstruction from binocular images. The implemented algorithms include the 8-point algorithm, epipolar correspondences, triangulation, RANSAC, and bundle adjustment.

## Algorithms

### 8-point Algorithm

The 8-point algorithm is used for computing the essential matrix from corresponding points in two images. It plays a crucial role in establishing the geometric relationship between the two camera views.

### Epipolar Correspondences

Epipolar correspondences are essential for understanding the relationship between points in one image and their corresponding epipolar lines in the other image.

### Triangulation 

Triangulation is the process of determining the 3D coordinates of a point by intersecting the lines of sight from two camera views. It is a fundamental step in reconstructing the 3D structure of a scene.

### Bundle Adjustment

Bundle Adjustment optimizes the entire system by refining the camera parameters and 3D coordinates to minimize the reprojection error. It enhances the accuracy of the 3D reconstruction.




