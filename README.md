# TRACKING PEOPLE IN A VIDEO SURVEILLANCE AREA

## CONTENTS
1. Introfuction
2. Related works
3. Approach
4. Results
5. Conclusions
6. References

## 1.INTRODUCTION
The aim of our work is to detect and track people in a stream taken by a camera. For this we decided to use a Tracking-by-Detection pipeline: first we find the people in the scene, and then we perform the tracking between frames. We chose this approach because we saw that the frames had a good enough quality to allow us to recognize people inside the scene. Moreover, recent developments in Convolutional Neural Network allow to perform accurate detections of persons insidean image.
For the detection phase we chose to useYOLO v3 [1]. This CNN is trained over the COCO dataset and we employ it to performpedestrians detection.
For the tracking phase we decided touse two approaches: Simple Online and Real-time Tracking, shortened in SORT, which is a tracking method based on the Kalman Filter with a constant velocity model, and an approach developed by us that we called Multiple Criteria Tracking, based on SIFT and on the trajectory of the pedestrians.
We will also provide a comparison between those two approaches.

## 2.RELATED WORKS
In order to achieve our goal we exploited some algorithms and deep learning architectures that we will list and then present:
 • YOLO v3: CNN architecture to perform the detection in images;
 • SIFT: algorithm to extract scale invariant features of objects, we used it to identify people in different frames; 
 • SORT: algorithm used to perform the tracking of pedestrians between frames.

## 2.1.YOLO v3
YOLO v3 is a Convolutional Neural Network which performs object segmentation inside an image. YOLO stands for You Only Look Once and is one of the faster detection methods available to this day, which is one of the reasons why we decided to use it. YOLO divides the input image into a grid of 7x7 cells. Within each grid cell it regress from the base boxes to a final box with 5 numbers that represents the confidence and the bounding box (dx, dy, dw, dh), and predicts scores for each classes, including the background as a class. The output is a volume of bounding boxes, score and confidence. 

## 2.2.SIFT
The SIFT (Scale invariant Feature Transform) function is a feature detector that solves the problem of matching features with       scaling, rotation and luminance. As shown in Figure 1, the algorithm allows to find points in common between two entities and, once A certain threshold is reached, determine if these are the same identity but in two different representations. 

..................

The algorithm is divided intoadetector part and a descriptor part. It starts from the creation of a space scale of images, usingthe Gaussian function and, after getting a progressively Gaussian blurred images, it calculates the Difference of Gaussian (DoG) pyramid of octaves, iterating the procedure [2]. 
