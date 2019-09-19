import numpy as np
import cv2

from common.parameters import Parameters as P


def from_camera_to_birdeye(pts):
    """
    :param pts: list of 2D points
    :return:  np.array of mapped 2D points
    """
    return np.rint(cv2.perspectiveTransform(np.array([pts]), P.HOMOGRAPHY.MAT)[0])


def from_birdeye_to_camera(pts):
    """
    :param pts: list of 2D points
    :return:  np.array of mapped 2D points
    """
    return np.rint(cv2.perspectiveTransform(np.array([pts]), np.linalg.inv(P.HOMOGRAPHY.MAT))[0])
