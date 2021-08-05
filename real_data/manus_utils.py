import os
import pickle
import numpy as np
import pandas as pd
from pyquaternion import Quaternion
from collections import OrderedDict
import sys
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
import open3d as o3d

from utils import rigid_transform_3D


class Keypoints:
    """
    Keypoints from the object

    Args:
        hand (str): Left ("l") or right ("r") hand

    Attributes:
        fingers (list): List of all the names of the keypoints
    """
    def __init__(self, manus_pth, hand="l"):
        self.fingers = ['LeftCarpus',
                        'LeftFirstMC', 'LeftFirstPP', 'LeftFirstDP', 'LeftFirstFT',
                        'LeftSecondPP', 'LeftSecondMP', 'LeftSecondDP', 'LeftSecondFT',
                        'LeftThirdPP', 'LeftThirdMP', 'LeftThirdDP', 'LeftThirdFT',
                        'LeftFourthPP', 'LeftFourthMP', 'LeftFourthDP', 'LeftFourthFT',
                        'LeftFifthPP', 'LeftFifthMP', 'LeftFifthDP', 'LeftFifthFT'
                        ]

        if hand == "r":
            tips_lengths = OrderedDict()
            tips_lengths['RightFirst'] = np.array([0., -0.02762, 0.])
            tips_lengths['RightSecond'] = np.array([0., -0.018028, 0.])
            tips_lengths['RightThird'] = np.array([0., -0.018916, 0.])
            tips_lengths['RightFourth'] = np.array([0., -0.019094, 0.])
            tips_lengths['RightFifth'] = np.array([0., -0.018384, 0.])

            self.fingers = [finger.replace("Left", "Right") for finger in self.fingers]

            labels = pd.read_excel(manus_pth, sheet_name="fingerTrackingSegmentsRight").to_numpy().T[0]
            positions = pd.read_excel(manus_pth, sheet_name="positionFingersRight")
            orientations = pd.read_excel(manus_pth, sheet_name="orientationFingersRight")

        elif hand == 'l':
            tips_lengths = OrderedDict()
            tips_lengths['LeftFirst'] = np.array([0, 0.02762, 0])
            tips_lengths['LeftSecond'] = np.array([0, 0.018028, 0])
            tips_lengths['LeftThird'] = np.array([0, 0.018916, 0])
            tips_lengths['LeftFourth'] = np.array([0, 0.019094, 0])
            tips_lengths['LeftFifth'] = np.array([0, 0.018384, 0])

            labels = pd.read_excel(manus_pth, sheet_name="fingerTrackingSegmentsLeft").to_numpy().T[0]
            positions = pd.read_excel(manus_pth, sheet_name="positionFingersLeft")
            orientations = pd.read_excel(manus_pth, sheet_name="orientationFingersLeft")

        labels_positions = []
        labels_orientations = []

        for label in labels:
            labels_positions.extend([label + "_x", label + "_y", label + "_z"])
            labels_orientations.extend([label + "_0", label + "_1", label + "_2", label + "_3"])

        positions.columns = labels_positions
        orientations.columns = labels_orientations

        positions_xyz = positions
        orientations_xyz = orientations

        positions = pd.DataFrame(columns=labels)

        for label in labels:
            positions[label] = positions_xyz.apply(lambda x: np.array(
                [x[label + '_x'],
                 x[label + '_y'],
                 x[label + '_z']]), axis=1)

        orientations = pd.DataFrame(columns=labels)
        for label in labels:
            orientations[label] = orientations_xyz.apply(lambda x: np.array(
                [x[label + '_0'],
                 x[label + '_1'],
                 x[label + '_2'],
                 x[label + '_3']]), axis=1)

        all_data = pd.concat([positions, orientations], axis=1)
        all_data.columns = np.concatenate((labels + '_position', labels + '_orientation'))

        for finger, length in tips_lengths.items():
            distal_position = finger + 'DP_position'
            distal_orientation = finger + 'DP_orientation'
            all_data[finger + 'FT_position'] = all_data.apply(
                lambda x: x[distal_position] + Quaternion(x[distal_orientation]).rotate(length), axis=1)

        self.positions_FT = all_data.drop(labels + '_orientation', axis=1)
        labels_FT = [position[:-9] for position in self.positions_FT.columns]
        self.positions_FT.columns = labels_FT

    def __len__(self):
        return self.positions_FT.shape[0]

    def from_frame(self, frame):
        """
        Return keypoints from a specific frame

        Parameters
        ----------
            frame (int): Frame

        Returns
        -------
            keypoints (np.array): N_keypoints x 3 which contains the world position of the keypoints

        """
        all_fingers = []

        for finger in self.fingers:
            all_fingers.append(self.positions_FT.loc[frame, finger])

        return np.vstack(all_fingers) * 1000


class ManusData:
    def __init__(self, manus_pth, manopth_pth, mano_scale=0.95, ts_offset=5*3600):
        """

        Parameters
        ----------
            ts (numpy): Start timestamp
            manus_pth (str): Path to fbx file
            manopth_pth (str): manopth path
            mano_scale (float): Manus scale so keypoints have a similar bone length.
                                This will lead to better MANO pose estimation.
        """
        # Timestamp from mvnx is in milliseconds, we first need to convert it to seconds.
        # For some reason it has a 5h offset
        self.ts = pd.read_excel(manus_pth, sheet_name="recDateMSecsSinceEpoch", header=None)[0][0] / 1_000 - ts_offset

        pkl_pth = manus_pth.replace(".xlsx", ".p")
        if os.path.exists(pkl_pth):
            self.hand_verts, self.hand_joints, self.hand_faces = pickle.load(open(pkl_pth, "rb"))

        else:
            keypoints = Keypoints(manus_pth)

            all_keypoints = np.zeros((len(keypoints), len(keypoints.fingers), 3))

            for frame in range(len(keypoints)):
                all_keypoints[frame, :, :] = keypoints.from_frame(frame)

            # Convert keypoints to MANO model
            sys.path.insert(1, manopth_pth)
            from manus.manus_to_mano import get_MANO_params

            mano_root = os.path.join(manopth_pth, "mano/models")
            hand_verts, hand_joints, hand_faces = get_MANO_params(all_keypoints * mano_scale, mano_root=mano_root)
            self.hand_verts = hand_verts / mano_scale
            self.hand_joints = hand_joints / mano_scale
            self.hand_faces = hand_faces
            pickle.dump((self.hand_verts, self.hand_joints, self.hand_faces), open(pkl_pth, "wb"))

    def __len__(self):
        return self.hand_joints.shape[0]

    def to_MANOs(self):
        """Returns hand verts, hand joints and hand faces"""
        return self.hand_verts, self.hand_joints, self.hand_faces

    def to_MANO(self, idx):
        """Returns hand verts, hand joints and hand faces from a specific idx"""
        return self.hand_verts[idx, :, :], self.hand_joints[idx, :, :], self.hand_faces

    def get_grasps_values(self, dist_open_hand=163):
        """
        A grasp is measured as the average distance of the fingers from a fully opened hand pose.
        e.g. A value close to 0 means that the hand is open.

        Parameters
        ----------
            dist_open_hand: distance when the hand is fully open (no grasp).

        Returns
        -------
            Indices: Indices of grasp peaks

        """
        # Distances for each frame
        grasp_values = []

        for hand_joints_frame in self.hand_joints:
            # Distances from Carpus to Distal Phalanx
            d1 = np.linalg.norm(hand_joints_frame[0] - hand_joints_frame[8])
            d2 = np.linalg.norm(hand_joints_frame[0] - hand_joints_frame[12])
            d3 = np.linalg.norm(hand_joints_frame[0] - hand_joints_frame[16])
            d4 = np.linalg.norm(hand_joints_frame[0] - hand_joints_frame[20])

            d = (d1 + d2 + d3 + d4) / 4
            grasp_values.append(dist_open_hand - d)

        return np.array(grasp_values)

    def get_grasps(self, height=5, distance=15):
        """
        Find peaks of grasps.

        Parameters
        ----------
        height: Required height of peaks.
        distance: Required minimal horizontal distance in samples between neighbouring peaks.

        Returns
        -------

        """
        grasp_values = self.get_grasps_values()

        # Find peaks
        peak_indices, peak_heights = find_peaks(grasp_values, height=height, distance=distance)

        return peak_indices

    def plot_grasps(self, height=5, distance=15):
        frames_m = np.arange(0, len(self))
        grasp_values = self.get_grasps_values()
        peak_indices = self.get_grasps(height=height, distance=distance)

        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot()

        # Plot pressure
        ax.plot(frames_m, grasp_values)
        ax.plot(peak_indices, grasp_values[peak_indices], 'o', color='orange')

        ax.set_ylabel("Grasp value")
        ax.set_xlabel("Manus frames")
        plt.show()

    def get_ts(self, fps=60):
        """
        It returns an array of timestamps starting once it runs. We will find the start timestamp by finding the offset

        Parameters
        ----------
            fps: Fps of the current manus recording
            ts_start: Start timestamp

        Returns
        -------
            ts (np.array): Array of timestamp
        """
        ts = self.ts + np.arange(0, len(self)) / fps

        return ts

    def from_corners(self, frame_m, corners_w):
        # World positions of AprilTag bottom corners
        pts_corners = corners_w[4:, :]
        # World positions of MANO verts touching AprilTag
        pts_m_verts = self.hand_verts[frame_m, :, :][[204, 229, 144, 183], :] / 1000

        R, t = rigid_transform_3D(pts_m_verts.T, pts_corners.T)

        mesh_hand = o3d.geometry.TriangleMesh()
        mesh_hand.vertices = o3d.utility.Vector3dVector(self.hand_verts[frame_m, :, :])
        mesh_hand.triangles = o3d.utility.Vector3iVector(self.hand_faces)
        mesh_hand.scale(1 / 1000, center=(0, 0, 0))
        mesh_hand.rotate(R, center=(0, 0, 0))
        mesh_hand.translate(t)
        mesh_hand.compute_vertex_normals()
        mesh_hand.paint_uniform_color([141 / 255, 184 / 255, 226 / 255])

        pcd_joints = o3d.geometry.PointCloud()
        pcd_joints.points = o3d.utility.Vector3dVector(self.hand_joints[frame_m, :, :])
        pcd_joints.scale(1 / 1000, center=(0, 0, 0))
        pcd_joints.rotate(R, center=(0, 0, 0))
        pcd_joints.translate(t)
        pcd_joints.paint_uniform_color([73 / 255, 84 / 255, 94 / 255])

        return mesh_hand

    def hand_verts_from_corners(self, frame_m, corners_w):
        """
        Return hand_verts oriented with the AprilTag corners

        Parameters
        ----------
            frame_m (int): Manus frame
            corners_w (np.array): Corners of the AprilTag

        Returns
        -------
            hand_verts (np.array): Hand verts rotated according to the AprilTag location
        """
        # World positions of AprilTag bottom corners
        pts_corners = corners_w[4:, :]

        # Scale hand_verts
        scaled_hand_verts = self.hand_verts[frame_m, :, :] / 1000

        # World positions of MANO verts touching AprilTag
        pts_m_verts = scaled_hand_verts[[204, 229, 144, 183], :]

        R, t = rigid_transform_3D(pts_m_verts.T, pts_corners.T)

        return np.matmul(scaled_hand_verts, R.T) + t.T