import random
import cv2

from .colors import ColorGenerator


class BasePersonTrack(object):

    color_generator = ColorGenerator()

    def __init__(self):
        self.color = self.color_generator.generate_color()

    @property
    def id(self):
        raise NotImplementedError('Implement the id property')

    @property
    def bbox(self):
        raise NotImplementedError('Implement the bbox property')

    def draw_bbox_on_image(self, img):
        bbox = self.bbox
        p1 = tuple(bbox[:2])
        p2 = tuple(bbox[2:])
        img = cv2.rectangle(img, p1, p2, self.color, 2)
        img = cv2.putText(img, 'ID: ' + str(self.id), p1, cv2.FONT_HERSHEY_PLAIN, 0.8, self.color)
        return img
