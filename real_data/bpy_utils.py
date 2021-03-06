import bpy
import math
import numpy as np
import sys
import os
from datetime import datetime
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
import open3d as o3d
import pickle

from utils import rigid_transform_3D


def load_fbx(fbx_pth, scale=1000):
    # remove mesh Cube
    if "Cube" in bpy.data.meshes:
        mesh = bpy.data.meshes["Cube"]
        bpy.data.meshes.remove(mesh)

    # Remove all objects
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj)

    # Import FBX
    bpy.ops.import_scene.fbx(filepath=fbx_pth, force_connect_children=True, automatic_bone_orientation=True, global_scale=scale)


def get_keyframes():
    """
    Returns a list of all the frames where the object is being animated
    """
    # get all selected objects
    selection = bpy.context.selected_objects

    # Raise exception if selection is empty
    if not selection:
        raise Exception('No object selected')

    keyframes = []
    anim = selection[0].animation_data
    if anim is not None and anim.action is not None:
        # We select any of the fcurves
        fcu = anim.action.fcurves[0]

        for keyframe in fcu.keyframe_points:
            x, y = keyframe.co
            if x not in keyframes:
                keyframes.append((math.ceil(x)))

    return keyframes


class Keypoints:
    """
    Keypoints from the object

    Args:
        hand (str): Left ("l") or right ("r") hand

    Attributes:
        armature (bpy object): Object we are taking the keypoints from
        fingers (list): List of all the names of the keypoints
    """
    def __init__(self, hand="l"):
        self.armature = bpy.context.scene.objects['Dongle_1B5E4344']
        self.fingers = ["root", "thumb_01_l", "thumb_02_l", "thumb_03_l", "thumb_tip_l",
                        "index_01_l", "index_02_l", "index_03_l", "index_tip_l",
                        "middle_01_l", "middle_02_l", "middle_03_l", "middle_tip_l",
                        "ring_01_l", "ring_02_l", "ring_03_l", "ring_tip_l",
                        "pinky_01_l", "pinky_02_l", "pinky_03_l", "pinky_tip_l"]

        if hand == "r":
            self.fingers = [finger.replace("_l", "_r") for finger in self.fingers]

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
        bpy.context.scene.frame_set(frame)

        all_fingers = []

        for finger in self.fingers:
            bone = self.armature.pose.bones[finger].head
            bonePos = self.armature.matrix_world @ bone

            all_fingers.append(np.array(bonePos))

        return np.vstack(all_fingers)


def get_manus_data(fbx_pth, manopth_pth, mano_scale=0.77):
    """

    Parameters
    ----------
        fbx_pth (str): Path to fbx file
        manopth_pth (str): manopth path
        mano_scale (float): Manus scale so keypoints have a similar bone length.
                             This will lead to better MANO pose estimation.

    Returns
    -------
        hand_verts, hand_joints, hand_faces

    """
    # Obtain keypoints from fbx file
    load_fbx(fbx_pth)

    keyframes = get_keyframes()
    keypoints = Keypoints()

    all_keypoints = np.zeros((len(keyframes), len(keypoints.fingers), 3))

    for idx, frame in enumerate(keyframes):
        all_keypoints[idx, :, :] = keypoints.from_frame(frame)

    # Convert keypoints to MANO model
    sys.path.insert(1, manopth_pth)
    from manus.manus_to_mano import get_MANO_params

    mano_root = os.path.join(manopth_pth, "mano/models")
    hand_verts, hand_joints, hand_faces = get_MANO_params(all_keypoints * mano_scale, mano_root=mano_root)

    return hand_verts / mano_scale, hand_joints / mano_scale, hand_faces

class ManusData:
    def __init__(self, fbx_pth, manopth_pth, mano_scale=0.77):
        """

        Parameters
        ----------
            fbx_pth (str): Path to fbx file
            manopth_pth (str): manopth path
            mano_scale (float): Manus scale so keypoints have a similar bone length.
                                This will lead to better MANO pose estimation.
        """
        pkl_pth = fbx_pth.replace(".fbx", ".p")
        if os.path.exists(pkl_pth):
            self.hand_verts, self.hand_joints, self.hand_faces = pickle.load(open(pkl_pth, "rb"))

        else:
            # Obtain keypoints from fbx file
            load_fbx(fbx_pth)

            keyframes = get_keyframes()
            keypoints = Keypoints()

            all_keypoints = np.zeros((len(keyframes), len(keypoints.fingers), 3))

            for idx, frame in enumerate(keyframes):
                all_keypoints[idx, :, :] = keypoints.from_frame(frame)

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

    def get_ts(self, fps=20, ts_start=None):
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
        if ts_start is None:
            now = datetime.now()
            ts_start = datetime.timestamp(now)

        ts = ts_start + np.arange(0, len(self)) / fps

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


def get_fbx_creation_time(fbx_pth, pyfbx_i42_pth, offset="+0000"):
    """
    Get creation time from fbx file.

    Download parser for binary FBX: https://github.com/ideasman42/pyfbx_i42

    Time zone offset indicating a positive or negative time difference from UTC/GMT of the form +HHMM or -HHMM,
    where H represents decimal hour digits and M represents decimal minute digits [-23:59, +23:59].
    """
    sys.path.append(pyfbx_i42_pth)
    import pyfbx.parse_bin
    fbx_root_elem, _ = pyfbx.parse_bin.parse(fbx_pth, use_namedtuple=True)
    creation_data = fbx_root_elem.elems[2]

    assert creation_data.id == b'CreationTime'

    # Get creation data
    date_string = creation_data.props[0].decode("utf-8") + f" {offset}"
    date_string_format = "%Y-%m-%d %H:%M:%S:%f %z"

    date_and_time = datetime.strptime(date_string, date_string_format)
    timestamp = datetime.timestamp(date_and_time)

    return timestamp