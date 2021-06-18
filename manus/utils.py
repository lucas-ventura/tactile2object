import numpy as np
import pandas as pd
import os
from pyquaternion import Quaternion
from collections import OrderedDict
import open3d as o3d

# Lengths of the Distal Proximal
tips_lengths = OrderedDict()
tips_lengths['LeftFirst'] = np.array([0, 0.02762, 0])
tips_lengths['LeftSecond'] = np.array([0, 0.018028, 0])
tips_lengths['LeftThird'] = np.array([0, 0.018916, 0])
tips_lengths['LeftFourth'] = np.array([0, 0.019094, 0])
tips_lengths['LeftFifth'] = np.array([0, 0.018384, 0])

tips_lengths['RightFirst'] = np.array([0., -0.02762, 0.])
tips_lengths['RightSecond'] = np.array([0., -0.018028, 0.])
tips_lengths['RightThird'] = np.array([0., -0.018916, 0.])
tips_lengths['RightFourth'] = np.array([0., -0.019094, 0.])
tips_lengths['RightFifth'] = np.array([0., -0.018384, 0.])


class Data:
    """
    Import data from the excel file.

    Attributes
    ----------
    positions_xyz: pandas.DataFrame
        Data positions with x y z in separate columns
    orientations_xyz: pandas.DataFrame
        Data orientations with x y z in separate columns
    labels: np.array
        Labels of the above data
    positions: pandas.DataFrame
        Data positions with xyz in the same column
    orientations: pandas.DataFrame
        Data orientations with xyz in the same column
    positions_FT: pandas.DataFrame
        Data positions with fingertip positions

    Methods
    -------
    getFingertips
        Computes the fingertip positions
    """
    def __init__(self, xlsx_pth):


        # Get labels
        labels_l = pd.read_excel(xlsx_pth, sheet_name="fingerTrackingSegmentsLeft").to_numpy().T[0]
        labels_r = pd.read_excel(xlsx_pth, sheet_name="fingerTrackingSegmentsRight").to_numpy().T[0]

        labels_positions_l = []
        labels_orientations_l = []

        for label_l in labels_l:
            labels_positions_l.extend([label_l + "_x", label_l + "_y", label_l + "_z"])
            labels_orientations_l.extend([label_l + "_0", label_l + "_1", label_l + "_2", label_l + "_3"])

        labels_positions_r = []
        labels_orientations_r = []

        for label_r in labels_r:
            labels_positions_r.extend([label_r + "_x", label_r + "_y", label_r + "_z"])
            labels_orientations_r.extend([label_r + "_0", label_r + "_1", label_r + "_2", label_r + "_3"])
            pass

            # Get positions
        positions_l = pd.read_excel(xlsx_pth, sheet_name="positionFingersLeft")
        positions_r = pd.read_excel(xlsx_pth, sheet_name="positionFingersRight")
        positions_l.columns = labels_positions_l
        positions_r.columns = labels_positions_r

        # Get orientations
        orientations_l = pd.read_excel(xlsx_pth, sheet_name="orientationFingersLeft")
        orientations_r = pd.read_excel(xlsx_pth, sheet_name="orientationFingersRight")
        orientations_l.columns = labels_orientations_l
        orientations_r.columns = labels_orientations_r

        # Merge
        self.positions_xyz = pd.concat([positions_l, positions_r], axis=1)
        self.orientations_xyz = pd.concat([orientations_l, orientations_r], axis=1)
        self.labels = np.concatenate((labels_l, labels_r))

        # Merge xyz in one column
        self.positions = pd.DataFrame(columns=self.labels)
        for label in self.labels:
            self.positions[label] = self.positions_xyz.apply(lambda x: np.array(
                [x[label + '_x'],
                 x[label + '_y'],
                 x[label + '_z']]), axis=1)
        self.orientations = pd.DataFrame(columns=self.labels)
        for label in self.labels:
            self.orientations[label] = self.orientations_xyz.apply(lambda x: np.array(
                [x[label + '_0'],
                 x[label + '_1'],
                 x[label + '_2'],
                 x[label + '_3']]), axis=1)

        self.positions_FT = None


    def getFingertips(self):
        """
        Computes the fingertip position from the position and orientation of the distal proximal
        """
        all_data = pd.concat([self.positions, self.orientations], axis=1)
        all_data.columns = np.concatenate((self.labels + '_position', self.labels + '_orientation'))

        for finger, length in tips_lengths.items():
            distal_position = finger + 'DP_position'
            distal_orientation = finger + 'DP_orientation'
            all_data[finger + 'FT_position'] = all_data.apply(
                lambda x: x[distal_position] + Quaternion(x[distal_orientation]).rotate(length), axis=1)

        positions_FT = all_data.drop(self.labels + '_orientation', axis=1)
        labels_FT = [position[:-9] for position in positions_FT.columns]
        positions_FT.columns = labels_FT

        self.positions_FT = positions_FT

        return positions_FT


def pick_points(pcd):
    """
    http://www.open3d.org/docs/tutorial/Advanced/interactive_visualization.html
    :param pcd: open3d point cloud data
    :return: None
    """
    print("")
    print("1) Please pick at least three correspondences using [shift + left click]")
    print("   Press [shift + right click] to undo point picking")
    print("2) Afther picking points, press q for close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()

def load_point_cloud(mesh_path: str):
    """
    Currently o3d doesn't support obj file, thus we need some
    hacky solution here
    :param mesh_path: Full path to an ply/obj file
    :return:
    """
    if mesh_path.endswith('.ply'):
        return o3d.io.read_point_cloud(mesh_path)
    elif mesh_path.endswith('.obj'):
        raise NotImplementedError()
    else:
        raise RuntimeError('Unknown data format')


def get_touch_locations_real(positions, bone, frames, scale=1000):
    """Get xyz positions from the real data

    Parameters
    ----------
    positions: pandas.DataFrame
        DataFrame containing the bone and frames information
    bone: str
        Bone that uses to touch
    frames: list
        Frames which are touching the object
    scale:
        Scale real data to match simulated data
    """

    touch_locations = np.zeros((len(frames), 3))

    for i, frame in enumerate(frames):
        touch_locations[i, :] = positions.loc[frame, bone] * scale

    return touch_locations


def get_touch_locations_simu(mesh_pth, scale=1., picked_ids=None):
    """ Get xyz positions from the simulated data

    Parameters
    ----------
    mesh_pth: str
        Path to mesh file
    scale: float
        Scale to apply to the real object to match the real data
    picked_ids: list
        If the points are know, you can input them
    """

    assert os.path.exists(mesh_pth)
    pcd = load_point_cloud(mesh_pth)
    pcd.paint_uniform_color([0.3, 0.3, 0.3])

    # Scale pointcloud
    pcd.scale(scale, center=pcd.get_center())  # center=(0, 0, 0) pcd.get_center()

    if picked_ids == None:
        picked_ids = pick_points(pcd)

    # Extract the keypoint in world
    point_cloud = np.asarray(pcd.points)
    touch_locations = []
    for point_id in picked_ids:
        point_in_world = point_cloud[point_id, :]
        touch_locations.append([float(point_in_world[0]), float(point_in_world[1]), float(point_in_world[2])])

    return np.array(touch_locations), pcd


def get_lines(points1, points2):
    """Creates lines that connect points1 with points2
    """
    line_points = []
    line_lines = []

    for i in range(points1.shape[0]):
        line_points.append(points1[i, :])
        line_points.append(points2[i, :])

        line_lines.append([i * 2, i * 2 + 1])

    line_points = np.array(line_points)
    line_lines = np.array(line_lines)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(line_lines)
    colors = [[0, 0, 1] for i in range(len(line_lines))]
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def get_balls(points, color=[1, 0, 0], radius=3.):
    """Given some points, it returns balls in open3d in that position"""
    balls = []

    for xyz in points:
        ball = o3d.geometry.TriangleMesh.create_sphere(radius=radius).translate((xyz))
        ball.paint_uniform_color(color)
        balls.append(ball)

    return balls