import bpy
import math
import numpy as np


def load_fbx(fbx_pth):
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
    bpy.ops.import_scene.fbx(filepath=fbx_pth, force_connect_children=True, automatic_bone_orientation=True)


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
        self.fingers = ["thumb_01_l", "thumb_02_l", "thumb_03_l", "thumb_tip_l",
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
