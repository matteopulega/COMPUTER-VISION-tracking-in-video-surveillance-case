import cv2
import torchvision.transforms as T

from yolo_v3 import pad_to_square, resize


def cv2_img_to_torch_tensor(img, img_size):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    torch_img = T.ToTensor()(img)
    torch_img, _ = pad_to_square(torch_img, 0)
    torch_img = resize(torch_img, img_size)
    return torch_img
