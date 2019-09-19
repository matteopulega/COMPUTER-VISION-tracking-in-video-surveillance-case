import numpy as np


class Parameters:

    MAX_GHOST_DETECTION = 10  # 5 # 3
    NUMBER_OF_POINTS_CALC_GHOST = 5
    MAX_DISTANCE_FOR_CALC_GHOST = 7

    class CUDA:
        DEVICE = 'cuda:0'
        IMG_SIZE = 416

    class CPU:
        DEVICE = 'cpu'
        IMG_SIZE = 160

    class DARKNET:
        CONF_THS = 0.8
        NMS_THS = 0.4

    class VIDEOWRITER:
        FORMAT = 'XVID'
        FPS = 20.0
        SIZE = (640, 480)
        COLOR = True

    class LIKELIHOOD:
        DISTANCE_THS = 50       # 100
        DIRECTION_THS = (np.pi * 2) / 3
        DIRECTION_CONTRIB = 18     # more means less
        WEIGHTS = [1,       # distance
                   1,       # sift
                   1,       # direction contrib
                   # ... altro ...
                  ]

    class HOMOGRAPHY:
        MAT = np.float32(
            [[-1.64925245e+02, - 2.18596944e+02, 6.45620565e+04],
             [-7.16313524e+01, - 4.35765696e+02, 9.04273357e+04],
             [-8.93452882e-02, - 3.09985829e-01, 1.00000000e+00]])

    COLORS = [[255, 0, 0],
              [0, 255, 0],
              [0, 0, 255],
              [0, 255, 255],
              [255, 0, 255],
              [255, 255, 0],
              [255, 255, 255],
              [0, 0, 0],
              [48, 27, 155],
              [198, 214, 245],
              [133, 154, 250],
              [54, 62, 90],
              [120, 91, 206],
              [25, 129, 224],
              [124, 75, 42],
              [132, 114, 87],
              [20, 103, 249],
              [54, 78, 38],
              [190, 224, 243],
              [62, 41, 42],
              [153, 156, 159],
              [58, 123, 121],
              [50, 65, 221],
              [98, 214, 255],
              [206, 181, 232],
              [214, 234, 240],
              [80, 85, 97],
              [58, 61, 189],
              [127, 234, 241],
              [201, 158, 190],
              [109, 110, 0],
              [103, 81, 72],
              [190, 188, 188],
              [79, 117, 169],
              [84, 219, 236],
              [145, 165, 0],
              [49, 91, 107],
              [227, 222, 149],
              [93, 93, 173],
              [81, 110, 0],
              [60, 224, 250],
              [172, 184, 69],
              [68, 93, 225],
              [60, 36, 188],
              ]