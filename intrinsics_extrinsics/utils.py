
import xml.etree.ElementTree as ET
from xml.etree import cElementTree as ElementTree
import numpy as np
import os
import open3d as o3d
import copy
import matplotlib.image as mpimg
import re
import cv2
import pupil_apriltags as apriltag
from matplotlib import pyplot as plt


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

    tree = ElementTree.parse('your_file.xml')
    root = tree.getroot()
    xmldict = XmlDictConfig(root)

    Or, if you want to use an XML string:

    root = ElementTree.XML(xml_string)
    xmldict = XmlDictConfig(root)

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


def read_xml(xml_pth):
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
        xmldict = read_xml(intrinsic_pth)
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

    def from_camera(self, camera="020122061233", inverse=True):
        if camera == self.default_camera:
            return np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])

        extrinsic_pth = os.path.join(self.xml_dir, f"{self.default_camera}-{camera}.xml")
        xmldict = read_xml(extrinsic_pth)
        T_wc = [param.split(",") for param in xmldict['camera']['pose']['T_wc'][3:-3].split(";")]
        T_wc.append([0., 0., 0., 1.])
        T_wc = np.array(T_wc)
        T_wc = T_wc.astype(np.float64)

        return np.linalg.inv(T_wc) if inverse else T_wc


def get_rgbd(color_pth, depth_pth):
    depth = o3d.io.read_image(depth_pth)
    color = o3d.io.read_image(color_pth)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, convert_rgb_to_intensity=False)

    return rgbd


class RGBD:
    def __init__(self, xml_dir, recording="recording_wAprilTag/20210714_002709/"):
        self.xml_dir = xml_dir
        self.recording = recording

    def from_camera(self, camera="020122061233", idx="000000"):
        color_name = f"color_{idx}.jpg"
        depth_name = f"aligned_depth_to_color_{idx}.png"

        color_pth = os.path.join(self.xml_dir, self.recording, camera, color_name)
        depth_pth = os.path.join(self.xml_dir, self.recording, camera, depth_name)

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


def pick_points_location(pcd, picked_ids=None):
    if picked_ids is None:
        picked_ids = pick_points(pcd)

    point_cloud = np.asarray(pcd.points)
    touch_locations = []
    for point_id in picked_ids:
        point_in_world = point_cloud[point_id, :]
        touch_locations.append([float(point_in_world[0]), float(point_in_world[1]), float(point_in_world[2])])

    return np.array(touch_locations)


def manual_registration(source, target):
    """
    Returns transformation between two point clouds.
    1. Plots the two point clouds
    2. Pick points from two point clouds and builds correspondences
    3. Estimate rough transformation using correspondences
    4. point-to-point ICP for refinement

    Parameters
    ----------
        source (o3d PointCloud): Source point cloud
        target (o3d PointCloud): Target point cloud

    Returns
    -------
        transformation (
    """
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


def find_object_transformation(pcd, obj_pth, return_obj_mesh=True, color=[1, 0.706, 0]):
    """
    Function to estimate object transformation with initial transformation with correspondences and then ICP.

    Parameters
    ----------
        pcd (o3d PointCloud): Point cloud with the target object.
        obj_pth (str): Path to the object file (ply)
        return_obj_mesh (boolean): If we want to return the transformed mesh
        color (list): Color of the returned mesh

    Returns
    -------
        Transformation (4x4 np.array): Transformation from the loaded object to the target point cloud.
        obj_mesh_t (o3d TriangleMesh): Object mesh with the transformation.

    """
    obj_pcd = o3d.io.read_point_cloud(obj_pth)
    obj_mesh = o3d.io.read_triangle_mesh(obj_pth)

    obj_pcd.scale(0.001, center=(0, 0, 0))
    obj_mesh.scale(0.001, center=(0, 0, 0))

    transformation = manual_registration(obj_pcd, pcd)

    if not return_obj_mesh:
        return transformation
    else:
        obj_mesh_t = copy.deepcopy(obj_mesh).transform(transformation)
        obj_mesh_t.compute_vertex_normals()
        obj_mesh_t.paint_uniform_color(color)

        return transformation, obj_mesh_t


class Stitching_pcds:
    """
    Stitches the point clouds from all the cameras with the intrinsic and extrinsic parameters.
    """
    def __init__(self, intrinsics, extrinsics, rgbds,
                 cameras=["020122061233", "821312060044", "020122061651", "821312062243"]):
        self.cameras = cameras
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.rgbds = rgbds

    def __getitem__(self, idx):
        """ Returns the stitched point cloud given an index."""
        stitched_pcd = o3d.geometry.PointCloud()

        for camera in self.cameras:
            rgbd = self.rgbds.from_camera(camera, idx=idx)
            intrinsic = self.intrinsics.from_camera(camera)
            extrinsic = self.extrinsics.from_camera(camera)
            stitched_pcd += o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, extrinsic)

        return stitched_pcd


class AprilTag:
    def __init__(self, img_pth, detector, intrinsic_params, extrinsic_mat=np.eye(4), tag_size=0.039):
        fx, fy, cx, cy = intrinsic_params
        self.image = cv2.imread(img_pth)
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        self.results = detector.detect(gray, estimate_tag_pose=True, camera_params=[fx, fy, cx, cy], tag_size=tag_size)

        # No results
        if len(self.results) == 0:
            self.pose_err = None
            self.center_p = None
            self.corners_p = None
            self.center_w = None
            self.corners_w = None
            self.R = None
            self.t = None

        # AprilTag found
        else:
            # Pose error of the AprilTag found
            self.pose_err = self.results[0].pose_err

            # Center of AprilTag in pixel coordinates
            self.center_p = [int(num) for num in self.results[0].center]

            # Corners of AprilTag in pixel coordinates
            self.corners_p = []
            for corner in self.results[0].corners:
                corner = [int(num) for num in corner]
                self.corners_p.append(corner)
            self.corners_p = np.array(self.corners_p)

            # Rotation and translation
            self.R = self.results[0].pose_R
            self.t = self.results[0].pose_t

            # Computing world AprilTag world positions
            tag_pos = np.array([[0, 0, 0],                           # Center AprilTag
                                [-tag_size / 2, tag_size / 2, 0],    # First corner AprilTag
                                [tag_size / 2, tag_size / 2, 0],     # Second corner AprilTag
                                [tag_size / 2, -tag_size / 2, 0],    # Third corner AprilTag
                                [-tag_size / 2, -tag_size / 2, 0]    # Fourth corner AprilTag
                                ])
            # Position with only intrinsic matrix
            tags_w_i = (self.R @ tag_pos.T + self.t).T

            # Compute extrinsic info
            tags_w_1 = np.ones((5, 4))
            tags_w_1[:, :3] = tags_w_i
            tags_w_e = extrinsic_mat @ tags_w_1.T

            # Center of AprilTag in world coordinates
            self.center_w = tags_w_e[:3, 0].T
            # Corners of AprilTag tag in world coordinates
            self.corners_w_t = tags_w_e[:3, 1:5].T

            # Normal of the AprilTag plane
            n = np.cross(self.corners_w_t[3] - self.corners_w_t[0], self.corners_w_t[1] - self.corners_w_t[0])
            n = n / np.sqrt(np.sum(n ** 2))
            # Corners of AprilTag in world coordinates projected to the hand
            self.corners_w_h = self.corners_w_t + n*0.02
            # All AprilTag corners (corners_w_t + corners_w_h)
            self.corners_w = np.concatenate((self.corners_w_t, self.corners_w_h))

    def get_image(self, radius=1, thickness=2, show=True):
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        image = cv2.circle(image, center=self.center_p, radius=radius, color=[255, 0, 0], thickness=thickness)

        if self.corners_p is not None:
            cv2.line(image, self.corners_p[0], self.corners_p[1], (0, 255, 0), 2)
            cv2.line(image, self.corners_p[1], self.corners_p[2], (0, 255, 0), 2)
            cv2.line(image, self.corners_p[2], self.corners_p[3], (0, 255, 0), 2)
            cv2.line(image, self.corners_p[3], self.corners_p[0], (0, 255, 0), 2)

            for corner in self.corners_p:
                image = cv2.circle(image, center=corner, radius=radius, color=[255, 0, 0], thickness=thickness)
        elif show:
            print("No AprilTag detected!")

        if show:
            plt.imshow(image)
            plt.axis('off')
            plt.show()
        else:
            return image

    def __repr__(self):
        return f"AprilTag with \ncenter: {self.center_w}\ncorners: {self.corners_w}\npose_R: {self.R}\npose_t: {self.t}"


class AprilTags:
    """
    Get corner pixel location from camera and index.
    """
    def __init__(self, xml_dir, intrinsics, extrinsics,
                 recording="recording_wAprilTag/20210714_002709/",
                 cameras=["020122061233", "821312060044", "020122061651", "821312062243"]):
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.recording_dir = os.path.join(xml_dir, recording)
        self.cameras = cameras
        self.detector = apriltag.Detector(families="tag36h11")

    def from_idx_camera(self, idx, camera):
        img_pth = os.path.join(self.recording_dir, camera, f"color_{idx}.jpg")
        intrinsic_params = self.intrinsics.params_from_camera(camera)
        extrinsic_mat = self.extrinsics.from_camera(camera, inverse=False)
        single_apriltag = AprilTag(img_pth, self.detector, intrinsic_params, extrinsic_mat)

        return single_apriltag

    def corners_w(self, idx, camera=None, pose_err_thld=1*10**-6):
        """
        Apriltag corners

        Parameters
        ----------
        idx: Image index
        camera: If camera is None fins corners in all cameras
        pose_err_thld: Pose error threshold from AprilTag detection

        Returns
        -------
        Pixels coordinates of apriltag corners.
        """
        # If camera is passed, return corners from camera
        if camera is not None:
            return self.from_idx_camera(idx, camera).corners_w

        # If camera is not passed, check all the cameras
        all_corners_w = []
        for camera in self.cameras:
            apriltag_camera = self.from_idx_camera(idx, camera)
            corners_w = apriltag_camera.corners_w
            pose_err = apriltag_camera.pose_err

            # AprilTag found and pose error is less than pose error threshold
            if corners_w is not None and pose_err < pose_err_thld:
                all_corners_w.append(corners_w)

        # Return None if it does not find the AprilTag. If it finds one or more, do average
        return None if len(all_corners_w) == 0 else np.mean(all_corners_w, axis=0)

    def image(self, idx, camera, radius=1, thickness=2, show=True):
        img_pth = os.path.join(self.recording_dir, camera, f"color_{idx}.jpg")
        intrinsic_params = self.intrinsics.params_from_camera(camera)
        single_apriltag = AprilTag(img_pth, self.detector, intrinsic_params)

        return single_apriltag.get_image(radius=radius, thickness=thickness, show=show)


def save_view_point(pcd, viewpoint_file="data/viewpoint.json"):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(viewpoint_file, param)
    vis.destroy_window()


def load_view_point(pcd, viewpoint_file="data/viewpoint.json"):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(viewpoint_file)
    if isinstance(pcd, list):
        for item in pcd:
            vis.add_geometry(item)
            ctr.convert_from_pinhole_camera_parameters(param)
    else:
        vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    vis.destroy_window()


def save_draw_geometries(pcd, filename, viewpoint_file="data/viewpoint.json"):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(viewpoint_file)
    if isinstance(pcd, list):
        for item in pcd:
            vis.add_geometry(item)
            vis.update_geometry(item)
    else:
        vis.add_geometry(pcd)
        vis.update_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(filename)
    vis.destroy_window()


def get_balls_from_corners(corners_w, color=[0, 1, 0]):
    """
    Get o3d balls at the corner locations.

    Parameters
    ----------
        corners_w (nx3 np.array): Location of the balls
        color (3x1 list): Balls color

    Returns
    -------
        corners_balls (list of o3d TriangleMesh): Balls at the specified locations
    """
    corners_balls = []

    for corner_w in corners_w:
        corner_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        corner_mesh.paint_uniform_color(color)
        corner_mesh.translate(corner_w)

        if not np.all((corner_w == 0)):
            corners_balls.append(corner_mesh)

    return corners_balls


def get_rectangle_from_corners(corners_w, color=[0.7, 0, 0]):
    """
    Get o3d rectangle from 8 corners of the AprilTag.

    Parameters
    ----------
        corners_w (8x3 np.array): Corners of the AprilTag.
        color (3x1 list): Color of the rectangle

    Returns
    -------
        rectangle (o3d TriangleMesh): Rectangle mesh
    """
    assert corners_w.shape == (8, 3)

    triangles = np.array(([4, 6, 5],
                          [4, 7, 6],
                          [0, 3, 4],
                          [3, 7, 4],
                          [0, 1, 3],
                          [1, 2, 3],
                          [1, 5, 6],
                          [1, 6, 2],
                          [3, 2, 6],
                          [3, 6, 7],
                          [0, 4, 1],
                          [1, 4, 5]))

    rectangle = o3d.geometry.TriangleMesh()
    rectangle.vertices = o3d.utility.Vector3dVector(corners_w)
    rectangle.triangles = o3d.utility.Vector3iVector(triangles)
    rectangle.compute_vertex_normals()
    rectangle.paint_uniform_color(color)

    return rectangle


def rigid_transform_3D(A, B, transform_source=False):
    """
    Finds the rigit transformation between two point clouds

    Code adapted from https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py

    Parameters
    ----------
        A (np.array): Source point cloud
        B (np.array): Target point cloud

    Returns
    -------
        R (np.array): Rotation (3x3)
        t (np.array): translation (3x1)
    """
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    if transform_source:
        A_rt = np.matmul(A.T, R.T) + t.T
        return R, t, A_rt
    else:
        return R, t
