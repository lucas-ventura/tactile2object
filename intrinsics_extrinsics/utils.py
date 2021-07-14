
import xml.etree.ElementTree as ET
from xml.etree import cElementTree as ElementTree
import numpy as np
import os
import open3d as o3d
import copy
import matplotlib.image as mpimg
import re

class XmlListConfig(list):
    # https://stackoverflow.com/a/5807028
    def __init__(self, aList):
        for element in aList:
            if element:
                # treat like dict
                if len(element) == 1 or element[0].tag != element[1].tag:
                    self.append(XmlDictConfig(element))
                # treat like list
                elif element[0].tag == element[1].tag:
                    self.append(XmlListConfig(element))
            elif element.text:
                text = element.text.strip()
                if text:
                    self.append(text)


class XmlDictConfig(dict):
    # https://stackoverflow.com/a/5807028
    '''
    Example usage:

    >>> tree = ElementTree.parse('your_file.xml')
    >>> root = tree.getroot()
    >>> xmldict = XmlDictConfig(root)

    Or, if you want to use an XML string:

    >>> root = ElementTree.XML(xml_string)
    >>> xmldict = XmlDictConfig(root)

    And then use xmldict for what it is... a dict.
    '''
    def __init__(self, parent_element):
        if parent_element.items():
            self.update(dict(parent_element.items()))
        for element in parent_element:
            if element:
                # treat like dict - we assume that if the first two tags
                # in a series are different, then they are all different.
                if len(element) == 1 or element[0].tag != element[1].tag:
                    aDict = XmlDictConfig(element)
                # treat like list - we assume that if the first two tags
                # in a series are the same, then the rest are the same.
                else:
                    # here, we put the list in dictionary; the key is the
                    # tag name the list elements all share in common, and
                    # the value is the list itself
                    aDict = {element[0].tag: XmlListConfig(element)}
                # if the tag has attributes, add those to the dict
                if element.items():
                    aDict.update(dict(element.items()))
                self.update({element.tag: aDict})
            # this assumes that if you've got an attribute in a tag,
            # you won't be having any text. This may or may not be a
            # good idea -- time will tell. It works for the way we are
            # currently doing XML configuration files...
            elif element.items():
                self.update({element.tag: dict(element.items())})
            # finally, if there are no child tags and no attributes, extract
            # the text
            else:
                self.update({element.tag: element.text})


def readXML(xml_pth):
    tree = ElementTree.parse(xml_pth)
    root = tree.getroot()

    return XmlDictConfig(root)


class Intrinsics:
    def __init__(self, xml_dir, use_txt=True):
        self.xml_dir = xml_dir

        if use_txt:
            self.camera_intrinsics = self.read_txt()
        else:
            self.camera_intrinsics = {}

    def read_txt(self):
        txt_pth = os.path.join(self.xml_dir, "intrinsics_640x480.txt")

        with open(txt_pth, "r") as file_handle:
            file_contents = file_handle.read()

        camera_intrinsics = {}

        for camera_info in file_contents.split("\n\n")[:-1]:
            *_, camera, _, intrinsics = camera_info.split("\n")

            p = re.search(r"p\[([0-9]*[.][0-9]*)\s([0-9]*[.][0-9]*)\]", intrinsics)
            f = re.search(r"f\[([0-9]*[.][0-9]*)\s([0-9]*[.][0-9]*)\]", intrinsics)

            cx, cy = p.group(1), p.group(2)
            fx, fy = f.group(1), f.group(2)

            camera_intrinsics[camera] = float(fx), float(fy), float(cx), float(cy)

        return camera_intrinsics

    def read_xml(self, camera="020122061233", W=640, H=480):
        intrinsic_pth = os.path.join(self.xml_dir, f"{camera}.xml")
        xmldict = readXML(intrinsic_pth)
        params = [float(param) for param in xmldict['camera']['camera_model']['params'][3:-3].split(";")]

        fx_old, fy_old, cx, cy, *_ = params

        dimx_old = int(xmldict['camera']['camera_model']['width'])
        dimy_old = int(xmldict['camera']['camera_model']['height'])
        dimx_new, dimy_new = W, H

        scale = dimy_new / dimy_old
        # print('scale intrinsics', scale)

        fx_new = fx_old * scale
        fy_new = fy_old * scale
        cx_new = cx / dimx_old * dimx_new
        cy_new = cy / dimy_old * dimy_new

        return fx_new, fy_new, cx_new, cy_new

    def from_camera(self, camera="020122061233", W=640, H=480):
        try:
            fx, fy, cx, cy = self.camera_intrinsics[camera]
        except:
            fx, fy, cx, cy = self.read_xml(camera, W, H)
            self.camera_intrinsics[camera] = fx, fy, cx, cy

        intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

        return intrinsic

    def params_from_camera(self, camera="020122061233", W=640, H=480):
        try:
            fx, fy, cx, cy = self.camera_intrinsics[camera]
        except:
            fx, fy, cx, cy = self.read_xml(camera, W, H)
            self.camera_intrinsics[camera] = fx, fy, cx, cy

        return fx, fy, cx, cy

class Extrinsics:
    def __init__(self, xml_dir, default_camera="020122061233"):
        self.xml_dir = xml_dir
        self.default_camera = default_camera

    def from_camera(self, camera="020122061233"):
        if camera == self.default_camera:
            return np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])

        extrinsic_pth = os.path.join(self.xml_dir, f"{self.default_camera}-{camera}.xml")
        xmldict = readXML(extrinsic_pth)
        T_wc = [param.split(",") for param in xmldict['camera']['pose']['T_wc'][3:-3].split(";")]
        T_wc.append([0., 0., 0., 1.])
        T_wc = np.array(T_wc)
        T_wc = T_wc.astype(np.float64)

        return np.linalg.inv(T_wc)


def get_rgbd(color_pth, depth_pth, cam_scale=1):
    depth = o3d.io.read_image(depth_pth)
    color = o3d.io.read_image(color_pth)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, convert_rgb_to_intensity=False)

    return rgbd


class RGBD:
    def __init__(self, recordings_dir, recording="20210709_004128"):
        self.recordings_dir = recordings_dir
        self.recording = recording

    def from_camera(self, camera="020122061233", idx="000000"):
        color_name = f"color_{idx}.jpg"
        depth_name = f"aligned_depth_to_color_{idx}.png"

        color_pth = os.path.join(self.recordings_dir, self.recording, camera, color_name)
        depth_pth = os.path.join(self.recordings_dir, self.recording, camera, depth_name)

        rgbd = get_rgbd(color_pth, depth_pth)

        return rgbd


def crop_geometry(pcd):
    # ICP Registration open3D
    print(
        "1) Press 'Y' twice to align geometry with negative direction of y-axis"
    )
    print("2) Press 'K' to lock screen and to switch to selection mode")
    print("3) Drag for rectangle selection,")
    print("   or use ctrl + left click for polygon selection")
    print("4) Press 'C' to get a selected geometry and to save it")
    print("5) Press 'F' to switch to freeview mode")

    o3d.visualization.draw_geometries_with_editing([pcd])


def draw_registration_result(source, target, transformation):
    # ICP Registration open3D
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def pick_points(pcd):
    # ICP Registration open3D
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


def manual_registration(source, target):
    # ICP Registration open3D
    print("Manual ICP")
    print("Visualization of two point clouds before manual alignment")
    draw_registration_result(source, target, np.identity(4))

    # pick points from two point clouds and builds correspondences
    picked_id_source = pick_points(source)
    picked_id_target = pick_points(target)
    assert (len(picked_id_source) >= 3 and len(picked_id_target) >= 3)
    assert (len(picked_id_source) == len(picked_id_target))
    corr = np.zeros((len(picked_id_source), 2))
    corr[:, 0] = picked_id_source
    corr[:, 1] = picked_id_target

    # estimate rough transformation using correspondences
    print("Compute a rough transform using the correspondences given by user")
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    trans_init = p2p.compute_transformation(source, target,
                                            o3d.utility.Vector2iVector(corr))

    # point-to-point ICP for refinement
    print("Perform point-to-point ICP refinement")
    threshold = 0.03  # 3cm distance threshold
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    draw_registration_result(source, target, reg_p2p.transformation)
    print("")

    return reg_p2p.transformation

