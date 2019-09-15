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
