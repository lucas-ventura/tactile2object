import bpy
import math
import numpy as np
import sys
import os
from datetime import datetime
from scipy.signal import find_peaks
from matplotlib import pyplot as plt

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


def get_manus_data(fbx_pth, manopth_pth):
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
    hand_verts, hand_joints, hand_faces = get_MANO_params(all_keypoints, mano_root=mano_root)

    return hand_verts, hand_joints, hand_faces

class Manus_data:
    def __init__(self, fbx_pth, manopth_pth):
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
        self.hand_verts, self.hand_joints, self.hand_faces = get_MANO_params(all_keypoints, mano_root=mano_root)


    def __len__(self):
        return self.hand_joints.shape[0]

    def to_MANOs(self):
        """Returns hand verts, hand joints and hand faces"""
        return self.hand_verts, self.hand_joints, self.hand_faces

    def to_MANO(self, idx):
        """Returns hand verts, hand joints and hand faces from a specific idx"""
        return self.hand_verts[idx,:,:], self.hand_joints[idx,:,:], self.hand_faces[idx,:,:]

    def get_grasps(self, height=80, distance=15):
        """
        Find peaks of grasps. A grasp is measured as the average distance of the fingers from a fully opened hand pose.
        e.g. A value close to 0 means that the hand is open.

        Parameters
        ----------
            height: Required height of peaks.
            distance: Required minimal horizontal distance in samples between neighbouring peaks.

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
            # 163 is the distance when the hand is fully open (no grasp).
            grasp_values.append(163 - d)

        # Find peaks
        peak_indices, peak_heights = find_peaks(grasp_values, height=height, distance=distance)

        return np.array(grasp_values), peak_indices

    def plot_grasps(self, height=80, distance=15):
        frames_m = np.arange(0, len(self))
        grasp_values, peak_indices = self.get_grasps(height=height, distance=distance)

        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot()

        # Plot pressure
        ax.plot(frames_m, grasp_values)
        ax.plot(peak_indices, grasp_values[peak_indices], 'o', color='orange')

        ax.set_ylabel("Hand pressure")
        ax.set_xlabel("Pressure frames")
        plt.show()

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