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
The SIFT (Scale invariant Feature Transform) function is a feature detector that solves the problem of matching features with      scaling, rotation and luminance. As shown in Figure 1, the algorithm allows to find points in common between two entities and, once A certain threshold is reached, determine if these are the same identity but in two different representations. 

..................

The algorithm is divided intoadetector part and a descriptor part. It starts from the creation of a space scale of images, usingthe Gaussian function and, after getting a progressively Gaussian blurred images, it calculates the Difference of Gaussian (DoG) pyramid of octaves, iterating the procedure [2]. 


## 2.3 SIMPLE ONLINE AND REAL-TIME TRACKING (SORT)
SORT is an algorithm used to solve the problem of Multiple Objects Tracking (MOT) targeted to online and real-time tracking [3]. 
 The MOT problem can be viewed as a data association problem where the aim is to associate detections across frames in a video. 
 SORT assumes that the detection phase is robust since it does not perform any error correction. It focuses on efficient and reliable handling of common frame-to-frame associations. To handle motion prediction and data association it uses a Kalman Filter and the Hungarian algorithm. 
  The inter-frame displacement of each object is approximated using a linear constant velocity model which is indipendent from other objects and the camera motion. The state of each target is:
*   x = [u v s r ũ ṽ š ] 
where u and v represent the horizontal and vertical pixel location of the center of the target, s represents the scale of the target’s bounding box, r represents the ratio of the target’s bounding box. When a detection is associated to a target, the detected bounding box is used to update the state of the target where the velocity components are solved optimally with a Kalman Filter. If no detection is associated to the target, its state is simply predicted without correction using the linear velocity model.
 In assigning detections to existing targets, each target’s bounding box is estimated by predicting its new location. The assignment cost matrix is computed as the Intersection-over-Union (IoU) distance between each detection and all predicted bounding boxes from existing targets. The assignment is solved optimally using the Hungarian algorithm. Additionally, a minimum IoU is imposed to reject assignments where the detection to target overlap is less then IoUmin . This implicitly handles the short-term occlusion problem. 
  When objects enters or leave the scene, unique identifiers need to be created or destroyed accordingly. For creating trackers, SORT considers any detection lower than IoUmin to represent an untracked object. The tracker is initialized with the geometry of its bounding box, a velocity equals to 0 and a high covariance for velocity to represent its uncertainity. The target then undergoes a probationary period where it needs to be associated with enough evidences to prove that it’s not a false positive. Trackers are terminated if they are not detected for Tlost frames.
  
## 3. APPROACH
As we mentioned before, we used two types of approaches that we will explain. The detection phase is in common for both of them and it’s done with YOLO.

### 3.1 MULTIPLE CRITERIA TRACKING
The base of this algorithm is the use of a Person Class and the creation of a list of Person that will be updated with each new frame. The Person Class is made of: 
*• an Identifier;
*• a Bounding Box;
*• a list of Ground Points that models the history of positions in the current and previous frames. A Ground Point is simply the midpoint of the lower side of the Bounding Box, that we assumed being the point of contact between the person and the ground; 
*• a list of SIFT descriptors and keypoints.
 For each frame, the algorithm analyzes every detection that YOLO provides. For each one of them a new Person is instantiated, then keypoints and descriptors are extracted from the area withing the Bounding Box using the SIFT algorithm. The figure 5 shows an example of such keypoints. 
 
 Figure 5
 
 Once we have created a list of all the Persons found in the current frame, we need to compare them with the list of Persons coming from the previous frames: Old Persons.
 We used a voting system to determine the best match between the Person detected in the new frame and the ones in Old Persons. For each candidate couple we calculate some scores based on different criteria. Each criterion can then be used to vote for the candidate with the highest score. The best match is defined as the candidate with the most votes. 
 The used criteria are: 
* • The Euclidean distance between the two Ground Points of the entities, which have been transformed to be considered in a new “Birdeye” plane. The transformation matrix has been calculated using our knowledge of the scene, in particular the real-world distance between some chosen points on the ground, and the position of the corresponding points in the frame. The smallest calculated distance receives the vote. Distances exceeding a chosen threshold receive a huge negative vote, essentially imposing a veto on the decision. 
*• The number of matching SIFT keypoints. The candidate with the most matches receives the vote. 
*• The angular distance between the average Movement Vectors. These vectors are calculated using the coordinates of the latest K Birdeye Ground Points, assuming a rectilinear trajectory in the short term. The distances with absolute value closest to zero receives a vote. Distances exceeding a chosen threshold receive a negative vote, discouraging the assignment of the match. 
 As said, persons belonging to the candidate couple with the most positive votes are identified as the same one. 
 If, for a given new detection, there are no positive votes with any of the remaining candidates in Old Persons, we declare that it must be a new Person unknown before. 
 If, for a given old person, there are no positive votes with any of the new detections, we do not immediatly remove it from the list of existing persons. Instead, since it can be temporarely undetected or hidden by something in front of it, we mark it as hidden and we try to guess its new position and update its informations accordingly. The new position is calculated using the latest average Movement Vector, in a similar manner to what previously explained. When the hidden-counter of a person reacheas a chosen threshold, we remove it from the Old Person list, assuming it is no longer contained in the frame. 
 Detections are displayed as a colored bounding box around the person. Different colors indicate different IDs. Bright colors indicate a real detection provided by YOLO, darker colors indicate a predicted detection.

Figure 6

