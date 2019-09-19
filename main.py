from tqdm import tqdm
import numpy as np
import cv2
import torch

from common.data import VideoDataLoader
from yolo_v3 import create_darknet_instance, rescale_boxes
from sort import SORT
from mct.mct_person_tracker import MCTPersonTracker, set_sift_keypoints
from mct.birdeye import from_camera_to_birdeye
from mct.functional import calc_ghost_point, update_persons_DICT
from common.parameters import Parameters as P


def get_current_parameters():
    if torch.cuda.is_available():
        return P.CUDA.DEVICE, P.CUDA.IMG_SIZE, torch.cuda.FloatTensor
    else:
        return P.CPU.DEVICE, P.CPU.IMG_SIZE, torch.FloatTensor


"""
*
* 
*
* SORT main implementation
*
*
*
*
"""


def main_sort(video_path, output_video):
    device, img_size, Tensor = get_current_parameters()
    device = torch.device(device)

    loader = VideoDataLoader(video_path, img_size)
    net = create_darknet_instance(img_size, device, P.DARKNET.CONF_THS, P.DARKNET.NMS_THS)
    sort_tracker = SORT(max_age=10, min_hits=3)
    fourcc = cv2.VideoWriter_fourcc(*P.VIDEOWRITER.FORMAT)
    frame, _ = next(loader)
    video_writer = cv2.VideoWriter(output_video + '.avi', fourcc, P.VIDEOWRITER.FPS, frame.shape[:2][::-1])
    for idx, (img, torch_img) in tqdm(enumerate(loader), unit=' processed frames'):
        if img is None or torch_img is None:
            raise RuntimeError('There was an error somewhere.')
        torch_img = torch_img.type(Tensor).to(device)
        detections = net.detect(torch_img)[0]
        if detections is not None:
            detections = detections[detections[:, -1] == 0.]
            detections = detections[detections[:, -1] == 0.]
            detections = rescale_boxes(detections, img_size, img.shape[:2])
            d1 = detections[:, :4]
            d2 = detections[:, 5].view(-1, 1)
            detections = torch.cat((d1, d2), dim=1)
            sure_trks = sort_tracker.update(detections.cpu().numpy())
            for sure_trks_idx in sure_trks:
                person = sort_tracker.trackers[sure_trks_idx]
                img = person.draw_bbox_on_image(img)
        video_writer.write(img)

    video_writer.release()


"""
*
*
*
*
* DMM main implementation
*
*
*
*
"""


def draw_points_in_birdeye(z_img, persons):
    for p in persons:
        a = np.float32([p.ground_point])
        pt = from_camera_to_birdeye(a)[0]
        cv2.circle(z_img, (pt[0], pt[1]), 3, p.color_dmm, -1)
    return z_img


def main_mct(video_path, output_video):
    device, img_size, Tensor = get_current_parameters()
    device = torch.device(device)
    print("device:",device)

    net = create_darknet_instance(img_size, device, P.DARKNET.CONF_THS, P.DARKNET.NMS_THS)
    loader = VideoDataLoader(video_path, img_size)
    fourcc = cv2.VideoWriter_fourcc(*P.VIDEOWRITER.FORMAT)
    frame, _ = next(loader)
    writer = cv2.VideoWriter(output_video+'.avi', fourcc, P.VIDEOWRITER.FPS, frame.shape[:2][::-1])

    persons_old = []
    max_used_id = 0

    z_img = cv2.imread('./mct/aligned.png')
    contatore = 0
    for img, torch_img in tqdm(loader, unit=' processed frames'):
        if img is None or torch_img is None:
            continue
        torch_img = torch_img.type(Tensor).to(device)

        detections = net.detect(torch_img)[0]
        if detections is not None:
            detections = detections[detections[:, -1] == 0.]
            detections = rescale_boxes(detections, img_size, img.shape[:2])
            # detections = detections.cpu()
            persons_detected = []
            tmp_new_id = 100
            for i, detection in enumerate(detections):
                # person = DmmPersonTracker(detection[:4].numpy())
                person = MCTPersonTracker(detection[:4].cpu().numpy())

                person.id = tmp_new_id
                tmp_new_id += 1

                sift_img = np.copy(img)
                person = set_sift_keypoints(sift_img, person)
                persons_detected.append(person)

            persons_old, max_used_id = update_persons_DICT(persons_detected, persons_old, max_used_id)

            for p in persons_old:
                p.draw_bounding_box_on_img(img)
                #p.draw_bbox_on_image(img)
                cv2.circle(img, (p.centroid[0].astype(np.int), p.centroid[1].astype(np.int)), 1, p.color_dmm, -1)
                cv2.circle(img, (p.ground_point[0], p.ground_point[1]), 3, p.color_dmm, -1)
        else:
            persons_old_tmp = []
            for p in persons_old:
                p.ghost_detection_count += 1
                if p.ghost_detection_count < P.MAX_GHOST_DETECTION:
                    new_ghost_point = calc_ghost_point(p)
                    p.follow_moving_ground_point(new_ghost_point)
                    p.draw_bounding_box_on_img(img)
                    #p.draw_bbox_on_image(img)
                    cv2.circle(img, (p.centroid[0].astype(np.int), p.centroid[1].astype(np.int)), 1, p.color_dmm, -1)
                    cv2.circle(img, (p.ground_point[0], p.ground_point[1]), 3, p.color_dmm, -1)
                    persons_old_tmp.append(p)

            persons_old = persons_old_tmp

        z_img = draw_points_in_birdeye(z_img, persons_old)


        contatore += 1

        # nuovaaaaaaaaaaaaaaa
        writer.write(img)

    birdeyename = 'birdeye_view.png'
    cv2.imwrite(birdeyename, z_img)
    writer.release()
