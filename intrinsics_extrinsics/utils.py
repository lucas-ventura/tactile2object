
import xml.etree.ElementTree as ET
from xml.etree import cElementTree as ElementTree
import numpy as np
import os
import open3d as o3d
import matplotlib.image as mpimg

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
    def __init__(self, xml_dir):
        self.xml_dir = xml_dir

    def from_camera(self, camera="020122061233", H=640, W=480):
        intrinsic_pth = os.path.join(self.xml_dir, f"{camera}.xml")
        xmldict = readXML(intrinsic_pth)
        params = [float(param) for param in xmldict['camera']['camera_model']['params'][3:-3].split(";")]

        fx_old, fy_old, cx, cy, *_ = params

        dimx_old = int(xmldict['camera']['camera_model']['width'])
        dimy_old = int(xmldict['camera']['camera_model']['height'])
        dimx_new, dimy_new = H, W

        fx_new = (dimx_new / dimx_old) * fx_old
        fy_new = (dimy_new / dimy_old) * fy_old

        intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx_new, fy_new, cx, cy)

        return intrinsic


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

        return T_wc




def get_rgbd(color_pth, depth_pth, cam_scale=1):
    depth = mpimg.imread(depth_pth)
    color = mpimg.imread(color_pth)

    img = o3d.geometry.Image(color.astype(np.uint8))
    depth = np.asarray(depth).astype(np.float32) / cam_scale
    depth = o3d.geometry.Image(depth)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img, depth)

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