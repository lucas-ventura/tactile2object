import numpy as np
import pandas as pd
import os
from pyquaternion import Quaternion

from collections import OrderedDict

# Lengths of the Distal Proximal
tips_lengths = OrderedDict()
tips_lengths['LeftFirst'] = np.array([0, 0.02762, 0])
tips_lengths['LeftSecond'] = np.array([0, 0.018028, 0])
tips_lengths['LeftThird'] = np.array([0, 0.018916, 0])
tips_lengths['LeftFourth'] = np.array([0, 0.019094, 0])
tips_lengths['LeftFifth'] = np.array([0, 0.018384, 0])

tips_lengths['RightFirst'] = np.array([0., -0.02762, 0.])
tips_lengths['RightSecond'] = np.array([0., -0.018028, 0.])
tips_lengths['RightThird'] = np.array([0., -0.018916, 0.])
tips_lengths['RightFourth'] = np.array([0., -0.019094, 0.])
tips_lengths['RightFifth'] = np.array([0., -0.018384, 0.])


class Data:
    def __init__(self, xlsx_pth):

        # Get labels
        labels_l = pd.read_excel(xlsx_pth, sheet_name="fingerTrackingSegmentsLeft").to_numpy().T[0]
        labels_r = pd.read_excel(xlsx_pth, sheet_name="fingerTrackingSegmentsRight").to_numpy().T[0]

        labels_positions_l = []
        labels_orientations_l = []

        for label_l in labels_l:
            labels_positions_l.extend([label_l + "_x", label_l + "_y", label_l + "_z"])
            labels_orientations_l.extend([label_l + "_0", label_l + "_1", label_l + "_2", label_l + "_3"])

        labels_positions_r = []
        labels_orientations_r = []

        for label_r in labels_r:
            labels_positions_r.extend([label_r + "_x", label_r + "_y", label_r + "_z"])
            labels_orientations_r.extend([label_r + "_0", label_r + "_1", label_r + "_2", label_r + "_3"])
            pass

            # Get positions
        positions_l = pd.read_excel(xlsx_pth, sheet_name="positionFingersLeft")
        positions_r = pd.read_excel(xlsx_pth, sheet_name="positionFingersRight")
        positions_l.columns = labels_positions_l
        positions_r.columns = labels_positions_r

        # Get orientations
        orientations_l = pd.read_excel(xlsx_pth, sheet_name="orientationFingersLeft")
        orientations_r = pd.read_excel(xlsx_pth, sheet_name="orientationFingersRight")
        orientations_l.columns = labels_orientations_l
        orientations_r.columns = labels_orientations_r

        # Merge
        self.positions_xyz = pd.concat([positions_l, positions_r], axis=1)
        self.orientations_xyz = pd.concat([orientations_l, orientations_r], axis=1)
        self.labels = np.concatenate((labels_l, labels_r))

        # Merge xyz in one column
        self.positions = pd.DataFrame(columns=self.labels)
        for label in self.labels:
            self.positions[label] = self.positions_xyz.apply(lambda x: np.array(
                [x[label + '_x'],
                 x[label + '_y'],
                 x[label + '_z']]), axis=1)
        self.orientations = pd.DataFrame(columns=self.labels)
        for label in self.labels:
            self.orientations[label] = self.orientations_xyz.apply(lambda x: np.array(
                [x[label + '_0'],
                 x[label + '_1'],
                 x[label + '_2'],
                 x[label + '_3']]), axis=1)

        self.positions_FT = None


    def getFingertips(self):
        """
        Computes the fingertip position from the position and orientation of the distal proximal
        """
        all_data = pd.concat([self.positions, self.orientations], axis=1)
        all_data.columns = np.concatenate((self.labels + '_position', self.labels + '_orientation'))

        for finger, length in tips_lengths.items():
            distal_position = finger + 'DP_position'
            distal_orientation = finger + 'DP_orientation'
            all_data[finger + 'FT_position'] = all_data.apply(
                lambda x: x[distal_position] + Quaternion(x[distal_orientation]).rotate(length), axis=1)

        positions_FT = all_data.drop(self.labels + '_orientation', axis=1)
        labels_FT = [position[:-9] for position in positions_FT.columns]
        positions_FT.columns = labels_FT

        self.positions_FT = positions_FT

        return positions_FT