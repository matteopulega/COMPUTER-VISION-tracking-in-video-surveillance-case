import cv2

from common.utils import cv2_img_to_torch_tensor


class VideoDataLoader(object):

    def __init__(self, video_path, img_size):
        self.video = cv2.VideoCapture(video_path)
        if not self.video.isOpened():
            raise ValueError('The path to video is wrong.')
        self.img_size = img_size

    def __iter__(self):
        return self

    def __len__(self):
        return int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

    def __next__(self):
        ret, frame = self.video.read()
        if ret is False:
            raise StopIteration()

        torch_img = cv2_img_to_torch_tensor(frame, self.img_size)
        return frame, torch_img

    def __del__(self):
        self.video.release()
