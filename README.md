# TRACKING PEOPLE IN A VIDEO SURVEILLANCE AREA

## CONTENTS
1. Introfuction
2. Related works
3. Approach
4. Results
5. Conclusions
6. References

## CO-WORKERS
* https://github.com/eMDi94 
* https://github.com/damianocaprari

# 1.INTRODUCTION
The aim of our work is to detect and track people in a stream taken by a camera. For this we decided to use a Tracking-by-Detection pipeline: first we find the people in the scene, and then we perform the tracking between frames. We chose this approach because we saw that the frames had a good enough quality to allow us to recognize people inside the scene. Moreover, recent developments in Convolutional Neural Network allow to perform accurate detections of persons insidean image.
For the detection phase we chose to useYOLO v3 [1]. This CNN is trained over the COCO dataset and we employ it to performpedestrians detection.
For the tracking phase we decided touse two approaches: Simple Online and Real-time Tracking, shortened in SORT, which is a tracking method based on the Kalman Filter with a constant velocity model, and an approach developed by us that we called Multiple Criteria Tracking, based on SIFT and on the trajectory of the pedestrians.
We will also provide a comparison between those two approaches.

# 2.RELATED WORKS
In order to achieve our goal we exploited some algorithms and deep learning architectures that we will list and then present:
* YOLO v3: CNN architecture to perform the detection in images;
* SIFT: algorithm to extract scale invariant features of objects, we used it to identify people in different frames; 
* SORT: algorithm used to perform the tracking of pedestrians between frames.

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
  
# 3. APPROACH
As we mentioned before, we used two types of approaches that we will explain. The detection phase is in common for both of them and it’s done with YOLO.

### 3.1 MULTIPLE CRITERIA TRACKING (MCT)
The base of this algorithm is the use of a Person Class and the creation of a list of Person that will be updated with each new frame. The Person Class is made of: 
* an Identifier;
* a Bounding Box;
* a list of Ground Points that models the history of positions in the current and previous frames. A Ground Point is simply the midpoint of the lower side of the Bounding Box, that we assumed being the point of contact between the person and the ground; 
* a list of SIFT descriptors and keypoints.

For each frame, the algorithm analyzes every detection that YOLO provides. For each one of them a new Person is instantiated, then keypoints and descriptors are extracted from the area withing the Bounding Box using the SIFT algorithm. The figure 5 shows an example of such keypoints. 
 
 Figure 5
 
 Once we have created a list of all the Persons found in the current frame, we need to compare them with the list of Persons coming from the previous frames: Old Persons.

We used a voting system to determine the best match between the Person detected in the new frame and the ones in Old Persons. For each candidate couple we calculate some scores based on different criteria. Each criterion can then be used to vote for the candidate with the highest score. The best match is defined as the candidate with the most votes. 
 The used criteria are: 
* The Euclidean distance between the two Ground Points of the entities, which have been transformed to be considered in a new “Birdeye” plane. The transformation matrix has been calculated using our knowledge of the scene, in particular the real-world distance between some chosen points on the ground, and the position of the corresponding points in the frame. The smallest calculated distance receives the vote. Distances exceeding a chosen threshold receive a huge negative vote, essentially imposing a veto on the decision. 
* The number of matching SIFT keypoints. The candidate with the most matches receives the vote. 
* The angular distance between the average Movement Vectors. These vectors are calculated using the coordinates of the latest K Birdeye
Ground Points, assuming a rectilinear trajectory in the short term. The distances with absolute value closest to zero receives a vote. Distances exceeding a chosen threshold receive a negative vote, discouraging the assignment of the match. 

As said, persons belonging to the candidate couple with the most positive votes are identified as the same one. 

If, for a given new detection, there are no positive votes with any of the remaining candidates in Old Persons, we declare that it must be a new Person unknown before. 

If, for a given old person, there are no positive votes with any of the new detections, we do not immediatly remove it from the list of existing persons. Instead, since it can be temporarely undetected or hidden by something in front of it, we mark it as hidden and we try to guess its new position and update its informations accordingly. The new position is calculated using the latest average Movement Vector, in a similar manner to what previously explained. When the hidden-counter of a person reacheas a chosen threshold, we remove it from the Old Person list, assuming it is no longer contained in the frame. 

Detections are displayed as a colored bounding box around the person. Different colors indicate different IDs. Bright colors indicate a real detection provided by YOLO, darker colors indicate a predicted detection.

Figure 6

## 3.2 SORT
After the detection has been made, the list of objects detected with the surrounding bounding boxes and confidence are passed to the SORT algorithm. Applying all the steps explained in section 2.3, SORT searches for the best associations and outputs the list of identifiers that are currently detected and successfully associated to previous detections.

It can be noted, also, that identities are appropriately mantained and assigned even when a subject has not been detected for a short amount of time.

# 4. EXPERIMENTS AND RESULTS
We quantitatively measured the performances of the two algorithms on two of the most complex videos we had available. The chosen metric is the Multiple Object Tracking Accuracy, MOTA in short, defined as:

MOTA=---------------

where fni, fpi and fai are, respectively, the number of misses, false positives and mismatches for i-th frame. GTi is the number of ground truth objects present at time i.

For each frame, we examined those values and used them to finally calculate the MOTA for the two algorithms. The green and red colors in Table 1 represents which algorithm worked better or worse in the corresponding category.

Qualitatively we abserved that SORT has a reduced number of false assignments, and does not produce false positives, since it is dependent on YOLO's detections. MCT, on the other hand, produced a lower number of false negatives, thanks to the attempts at prediction when detections are not provided by YOLO.

It has to be said, though, that a different implementation of the SORT algorithm could be able to produce better results displaying its predictions and thus reducing the number of false negatives.

---table 1
link

# 5. IMPROVEMENTS AND FUTURE WORKS
Recently, a new algorithm has been proposed for Tracking-by-Detection based on SORT called DEEP-SORT [4]. This works performs better than SORT, but we thought that it’s heavier in terms of computation. Our aim is to provide a framework for real-time detection, YOLO is already heavy to run on a CPU. If the camera is connected to a dedicated server which runs the application on a GPU, then it could be possible to implement also DEEP-SORT.

For MCT, it could be interesting to use Machine Learning algorithms to improve the voting phase, for example by assigning different weights to different criteria.

A further consideration to improve the algorithm consists in modifying the criterion concerning ground points, to manage the cases in which YOLO detections do not identify the whole body, but only a part of it like the bust. In these cases, in fact, the Ground Point is in the wrong position with respect to reality. To improve this aspect, more information could be used from the Bounding Boxes.

# 6.CONCLUSIONS
Comparing the two algorithms we found that both are interesting Multiple Object Tracking. The implementation we have chosen for SORT manages to maintain the same id for the same person, but if in some frames there is no detection, the prediction is not shown on the screen. In contrast, MCT manages to reduce fn at the cost of increasing the fp with the prediction technique based on past ground points. This feature can be considered positive in the case of a video surveillance problem.

# REFERENCES
* [1] github.com/eriklindernoren/PyTorch-YOLOv3
* [2] Wang, Guohui & Rister, Blaine & Cavallaro,
Joseph. (2013). “Workload Analysis and Ef icient
OpenCL-based Implementation of SIFT Algorithm
on a Smartphone” 2013 IEEE Global Conference
on Signal and Information Processing, GlobalSIP
2013 - Proceedings.
* [3] A. Bewley, Z. Ge, L. Ott, F. Ramos and B.
Upcroft, "Simple online and realtime tracking,"
2016 IEEE International Conference on Image
Processing (ICIP), Phoenix, AZ, 2016, pp.
3464-3468.
* [4] N. Wojke, A. Bewley and D. Paulus, "Simple
online and realtime tracking with a deep
association metric," 2017 IEEE International
Conference on Image Processing (ICIP), Beijing,
2017, pp. 3645-3649.
