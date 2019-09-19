import numpy as np
import cv2

from .birdeye import from_camera_to_birdeye
from common.parameters import Parameters as P


def calc_ghost_point(p, mode='camera'):
    assert mode == 'camera' or mode == 'birdeye', "mode can only be 'camera' or 'birdeye'"

    if mode == 'birdeye':
        if len(p.ground_point_past) >= 2:
            last_pts = p.ground_point_past[p.ghost_detection_count: p.ghost_detection_count + P.NUMBER_OF_POINTS_CALC_GHOST]
            if len(last_pts) >= 2:
                last_pts = from_camera_to_birdeye(np.array(last_pts))
                new_point = np.add(p.ground_point, np.divide(np.subtract(last_pts[0], last_pts[-1]), len(last_pts) - 1))
                return new_point.astype(np.int)
        return from_camera_to_birdeye(np.reshape(p.ground_point, (1, 2)).astype(np.float32))

    else:   # if mode == 'camera':
        if len(p.ground_point_past) >= 2:
            last_pts = p.ground_point_past[p.ghost_detection_count: p.ghost_detection_count + P.NUMBER_OF_POINTS_CALC_GHOST]
            if len(last_pts) >= 2:
                pt = np.divide(np.subtract(last_pts[0], last_pts[-1]), len(last_pts) - 1)
                dist = np.sqrt(pt[0]**2 + pt[1]**2)
                if dist > P.MAX_DISTANCE_FOR_CALC_GHOST:
                    pt = pt * P.MAX_DISTANCE_FOR_CALC_GHOST / dist
                return np.add(p.ground_point, pt).astype(np.int)
        return p.ground_point


def euclidean_distance(p1, p2):
    #print(p1, p2)
    return np.sqrt(np.sum(np.square(p1 - p2), axis=0)).astype(np.float)


def sift_contrib(person, other):
    bf = cv2.BFMatcher()
    des1 = np.asarray(person.sift_descriptors)
    des2 = np.asarray(other.sift_descriptors)
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for i in matches:
        if len(i) <= 1:
            return 0
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return len(good)


def direction_contrib(this, other):
    l_other = other.ground_point_past[0:P.NUMBER_OF_POINTS_CALC_GHOST]
    if len(l_other) >= 2:
        l = [this.ground_point]
        l.extend(l_other)
        other_vector = np.divide(np.subtract(l_other[0], l_other[-1]), len(l_other)-1)
        this_vector = np.divide(np.subtract(l[0], l[-1]), len(l)-1)
        other_angle = np.arctan2(other_vector[1], other_vector[0])
        this_angle = np.arctan2(this_vector[1], this_vector[0])
        return min(np.abs(this_angle - other_angle), np.abs(other_angle - this_angle))

    else:
        return 999


def match_likelihood_DICT(this, other):
    """
    :param this: person
    :param other:  person
    :return: likelihood value
    """
    # -- DICTIONARY --
    """
    likel_dict
        0: distance
        1: sift
        2: direction
        3: other.id 
    """
    likel_dict = np.zeros((4), dtype=np.float)
    likel_dict[3] = other.id

    # find points in birdeye view
    pt_this_z = from_camera_to_birdeye(np.float32([this.ground_point]))[0]
    pt_other_z = from_camera_to_birdeye(np.float32([other.ground_point]))[0]

    # -- distance
    likel_dict[0] = euclidean_distance(pt_this_z, pt_other_z)
    # -- sift
    likel_dict[1] = sift_contrib(this, other)              # return number of matches
    # -- direction
    likel_dict[2] = direction_contrib(this, other)    # return angular distance
    return likel_dict


def return_scores(likelihoods):
    idx_dist_min = np.argmin(likelihoods[:, 0])
    idx_sift_max = np.argmax(likelihoods[:, 1])
    idx_dir_min = np.argmin(likelihoods[:, 2])

    scoreboard = np.zeros_like((likelihoods), dtype=np.int)
    scoreboard[idx_dist_min, 0] = 1
    scoreboard[likelihoods[:, 0] >P.LIKELIHOOD.DISTANCE_THS, 0] = -100

    scoreboard[idx_sift_max, 1] = 1

    scoreboard[idx_dir_min, 2] = 1
    scoreboard[likelihoods[:, 2] > P.LIKELIHOOD.DIRECTION_THS , 2] = -1
    scoreboard[likelihoods[:, 2] == 999, 2] = 0

    ret_scores = np.sum(scoreboard, axis=1)
    idx = np.argmax(ret_scores)
    return idx, ret_scores[idx]


def update_persons_DICT(persons_detected, persons_old, max_used_id):
    persons_tmp = []
    if persons_old:     # forse inutile dato che ho gia fatto il controllo fuori dalla funzione
        if len(persons_detected) <= len(persons_old):
            remaining = persons_old.copy()
            for p in persons_detected:
                # print("person detected ", p.id)

                likelihoods = np.array( list(map(lambda x: match_likelihood_DICT(p, x), remaining)) )
                idx, score = return_scores(likelihoods)

                if score <= 0:
                    # no matches found
                    # add to person_old, tramite person_tmp
                    max_used_id += 1
                    p.id = max_used_id
                else:
                    person_matching = remaining.pop(idx)
                    # p.centroid_past.extend(person_matching.centroid_past)
                    p.update_past(person_matching.id, person_matching.centroid_past, person_matching.ground_point_past)
                    # p.id = person_matching.id
                persons_tmp.append(p)

            # prima di aggiornare person_tmp aggiungendo remaining...
            # io prendo questi "fantasmi" e aggiorno i valori, soprattuto il flag real_detection
            for p in remaining:
                if p.ghost_detection_count < P.MAX_GHOST_DETECTION:
                    p.ghost_detection_count += 1
                    new_ghost_point = calc_ghost_point(p)
                    p.follow_moving_ground_point(new_ghost_point)
                    persons_tmp.append(p)

        else:  # len(persons_detected) > len(persons_old):
            remaining = persons_detected
            for p in persons_old:
                likelihoods = np.array(list(map(lambda x: match_likelihood_DICT(x, p), remaining)))

                idx, score = return_scores(likelihoods)

                if score <= 0:
                    # no matches found
                    # la persona old e' USCITA oppure NASCOSTA
                    if p.ghost_detection_count < P.MAX_GHOST_DETECTION:
                        p.ghost_detection_count += 1
                        new_ghost_point = calc_ghost_point(p)
                        p.follow_moving_ground_point(new_ghost_point)
                        persons_tmp.append(p)
                else:
                    person_matching = remaining.pop(idx)
                    # person_matching.centroid_past.extend(p.centroid_past)
                    person_matching.update_past(p.id, p.centroid_past, p.ground_point_past)
                    # person_matching.id = p.id
                    persons_tmp.append(person_matching)

            for p in remaining:
                max_used_id += 1
                p.id = max_used_id
                persons_tmp.append(p)

    else:  # persons_old is empty
        for p in persons_detected:
            max_used_id += 1
            p.id = max_used_id
            persons_tmp.append(p)

    return persons_tmp, max_used_id
