from utils import *

import os
import argparse
import pickle
import pandas as pd
import numpy as np

alignment_idxs = [0,5,9,13,17]
alignment_points_r = np.array([[ 94.6605,   1.4790,   3.3575],
                               [  6.5632,  -3.7214,  24.0435],
                               [  0.0000,   0.0000,   0.0000],
                               [ 12.9249,  -2.4785, -23.3157],
                               [ 25.8735,  -8.4614, -39.8518]])

alignment_points_l = np.array([[-94.6605,   1.4790,   3.3575],
                               [ -6.5632,  -3.7214,  24.0435],
                               [  0.0000,   0.0000,   0.0000],
                               [-12.9249,  -2.4785, -23.3157],
                               [-25.8735,  -8.4614, -39.8518]])

positions_order_l = [
    "LeftCarpus",

    "LeftFirstMC",
    "LeftFirstPP",
    "LeftFirstDP",
    "LeftFirstFT",

    "LeftSecondPP",
    "LeftSecondMP",
    "LeftSecondDP",
    "LeftSecondFT",

    "LeftThirdPP",
    "LeftThirdMP",
    "LeftThirdDP",
    "LeftThirdFT",

    "LeftFourthPP",
    "LeftFourthMP",
    "LeftFourthDP",
    "LeftFourthFT",

    "LeftFifthPP",
    "LeftFifthMP",
    "LeftFifthDP",
    "LeftFifthFT"
]

positions_order_r = [position.replace("Left", "Right") for position in positions_order_l]

parser = argparse.ArgumentParser(description='Prepare data')
parser.add_argument('--xslx_pth', type=str,
                    default='data/recorded_data.xlsx',
                    help="Path excel with the recorded data")


def align_points_l(positions, frame):
    keypoints_l = np.zeros((len(positions_order_l), 3))

    for i, position in enumerate(positions_order_l):
        keypoints_l[i, :] = positions.loc[frame, position] * 1000

    A = keypoints_l[alignment_idxs].T
    B = alignment_points_l.T
    R, t = rigid_transform_3D(A, B)

    keypoints_l_rt = np.matmul(keypoints_l, R.T) + t.T

    return keypoints_l_rt

def align_points_r(positions, frame):
    keypoints_r = np.zeros((len(positions_order_r), 3))

    for i, position in enumerate(positions_order_r):
        keypoints_r[i, :] = positions.loc[frame, position] * 1000

    A = keypoints_r[alignment_idxs].T
    B = alignment_points_r.T
    R, t = rigid_transform_3D(A, B)

    keypoints_r_rt = np.matmul(keypoints_r, R.T) + t.T

    return keypoints_r_rt

def main(args):
    positions_pth = args.xslx_pth.replace(".xlsx", "_positions.p")
    if not os.path.exists(positions_pth):
        data = Data(args.xslx_pth)
        # Compute finger tips positions
        positions = data.getFingertips()
        # Save positions
        pickle.dump(positions, open(positions_pth, "wb"))
    else:
        positions = pickle.load(open(positions_pth, "rb"))

    target_pts_l_pth = args.xslx_pth.replace(".xlsx", "_target_pts_l.p")
    target_pts_r_pth = args.xslx_pth.replace(".xlsx", "_target_pts_r.p")
    if not os.path.exists(target_pts_l_pth):
        frames = positions.shape[0]
        target_pts_l = np.zeros((frames, len(positions_order_l), 3))
        target_pts_r = np.zeros((frames, len(positions_order_r), 3))

        for frame in range(frames):
            target_pts_l[frame, :, :] = align_points_l(positions, frame=frame)
            target_pts_r[frame, :, :] = align_points_r(positions, frame=frame)

        pickle.dump(target_pts_l, open(target_pts_l_pth, "wb"))
        pickle.dump(target_pts_r, open(target_pts_r_pth, "wb"))


if __name__ == '__main__':
    main(parser.parse_args())