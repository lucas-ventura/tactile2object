import bpy
import math


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