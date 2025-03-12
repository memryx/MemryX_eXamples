from scipy.spatial import distance as dist
import pandas as pd
import numpy as np
import cv2

classes = {
    0: 'F',
    1: 'K',
    2: 'U',
    3: 'W',
    4: 'N',
    5: 'S',
    6: 'D',
    7: 'V',
    8: 'A',
    9: 'J',
    10: 'E',
    11: 'Z',
    12: 'M',
    13: 'T',
    14: 'O',
    15: 'X',
    16: 'R',
    17: 'Y',
    18: 'Q',
    19: 'L',
    20: 'G',
    21: 'I',
    22: 'B',
    23: 'H',
    24: 'P',
    25: 'C'
}

#        8   12  16  20
#        |   |   |   |
#        7   11  15  19
#    4   |   |   |   |
#    |   6   10  14  18
#    3   |   |   |   |
#    |   5---9---13--17
#    2    \         /
#     \    \       /
#      1    \     /
#       \    \   /
#        ------0-


def calc_distances(joints):
    """
    Calculate euclidean distances between given two pairs of 2-D coordinates
    :param joints: predicted joints by regressor model (21, 2) array
    :return: list with 9 distances
    """
    dist_20_0 = dist.euclidean(joints[20], joints[0])
    dist_16_0 = dist.euclidean(joints[16], joints[0])
    dist_12_0 = dist.euclidean(joints[12], joints[0])
    dist_8_0 = dist.euclidean(joints[8], joints[0])
    dist_4_0 = dist.euclidean(joints[4], joints[0])
    dist_20_16 = dist.euclidean(joints[20], joints[16])
    dist_16_12 = dist.euclidean(joints[16], joints[12])
    dist_12_8 = dist.euclidean(joints[12], joints[8])
    dist_8_4 = dist.euclidean(joints[8], joints[4])

    return [dist_20_0, dist_16_0, dist_12_0, dist_8_0, dist_4_0,
            dist_20_16, dist_16_12, dist_12_8, dist_8_4]


def predict_sign(joints, gesture_clf, classes):
    """
    Function for predicting a shown sign by passing as input array of 9 euclidean distances and get letter from
    classes dictionary for visualising.
    :param joints: (21, 2) array of predicted coordinates of joints
    :param gesture_clf: loaded Bayesian classifier model
    :param classes: dictionary of mapped labels {int: str}
    :return: string, predicted letter that represents a sign gesture
    """
    distances = calc_distances(joints)
    normalized = [float(i)/sum(distances) for i in distances]
    distances = np.expand_dims(normalized, axis=0)
    pred = gesture_clf.predict(distances)[0]
    sign = classes[pred]

    return sign