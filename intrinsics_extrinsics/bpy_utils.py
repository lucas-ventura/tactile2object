import bpy
import math
import numpy as np
import sys
import os
from datetime import datetime

def load_fbx(fbx_pth, scale=1000):
    # remove mesh Cube
    if "Cube" in bpy.data.meshes:
        mesh = bpy.data.meshes["Cube"]
        bpy.data.meshes.remove(mesh)

    # remove Camera
    if "Camera" in bpy.data.objects:
        mesh = bpy.data.objects["Camera"]
        bpy.data.objects.remove(mesh)

    # remove Light
    if "Light" in bpy.data.objects:
        mesh = bpy.data.objects["Light"]
        bpy.data.objects.remove(mesh)

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