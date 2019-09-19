import numpy as np
import cv2

from common.models import BasePersonTrack
from common.parameters import Parameters as P


class MCTPersonTracker(BasePersonTrack):

    def __init__(self, coordinates):
        super(MCTPersonTracker, self).__init__()
        self._id = 0
        self.p1 = np.round(coordinates[:2]).astype(np.int)
        self.p2 = np.round(coordinates[2:]).astype(np.int)
        self.centroid_past = [np.mean([self.p2, self.p1], axis=0)]
        self.ground_point_past = [np.array((self.centroid[0], self.p2[1]), dtype=np.int)]
        self.centroid_future = (0, 0)
        self.sift_kp = []
        self.sift_descriptors = []  # list of arrays, 1 row for each kp
        self.ghost_detection_count = 0

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, new_id):
        self._id = new_id

    @property
    def bbox(self):
        return np.vstack((self.p1, self.p2))

    @property
    def h(self):
        return self.p2[1] - self.p1[1]

    @property
    def w(self):
        return self.p2[0] - self.p1[0]

    @property
    def centroid(self):
        return self.centroid_past[0]

    @property
    def ground_point(self):
        return self.ground_point_past[0]

    @property
    def color_dmm(self):
        c = P.COLORS[self.id % (len(P.COLORS))].copy()
        if self.ghost_detection_count is not 0:
            c[0] = c[0] / 2;
            c[1] = c[1] / 2;
            c[2] = c[2] / 2
        return c

    def draw_bounding_box_on_img(self, img):
        img = cv2.rectangle(img, tuple(self.p1), tuple(self.p2), self.color_dmm)
        img = cv2.putText(img, 'ID: ' + str(self.id), tuple(self.p1), cv2.FONT_HERSHEY_PLAIN, 0.8, self.color_dmm)
        return img

    def update_past(self, id, centroid_past, ground_point_past):
        self.id = id
        self.centroid_past.extend(centroid_past)
        self.ground_point_past.extend(ground_point_past)

    def follow_moving_ground_point(self, ground_point):
        h = self.h
        w = self.w
        self.p1 = np.array([ground_point[0] - w / 2, ground_point[1] - h]).astype(np.int)
        self.p2 = np.array([ground_point[0] + w / 2, ground_point[1]]).astype(np.int)
        self.centroid_past.insert(0, np.array([ground_point[0], ground_point[1] - h / 2]).astype(np.int))
        self.ground_point_past.insert(0, ground_point)


def find_closest_person(current_person, persons):
    current_centroid = current_person.centroid
    others_centroid = np.empty(shape=(len(persons), 2), dtype=np.float)
    for idx, p in enumerate(persons):
        others_centroid[idx] = p.centroid
    distances = np.sum(np.square(others_centroid - current_centroid), axis=1)
    return np.argmin(distances)


def set_sift_keypoints(img, person):
    sift = cv2.xfeatures2d.SIFT_create()
    x11, y11 = person.p1
    x12, y12 = person.p2
    x11 = max(0, x11)
    y11 = max(0, y11)
    x12 = max(0, x12)
    y12 = max(0, y12)
    crop_img = img[y11:y12, x11:x12]
    gray_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    kp = sift.detect(gray_crop, None)
    kp, des1 = sift.compute(gray_crop, kp)
    if des1 is not None:
        person.sift_kp = kp
        for r in range(des1.shape[0]):
            person.sift_descriptors.append(des1[r, :])
    return person
