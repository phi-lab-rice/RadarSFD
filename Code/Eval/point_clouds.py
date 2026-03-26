import cv2
import numpy as np
from scipy.spatial import KDTree


class PointClouds:
    def __init__(
        self, RMAX=10.8, RBINS=256, ABINS=512, MIN_THRESHOLD=1, MAX_THRESHOLD=255
    ):
        self.RMAX = RMAX
        self.RBINS = RBINS
        self.ABINS = ABINS
        self.MIN_THRESHOLD = MIN_THRESHOLD
        self.MAX_THRESHOLD = MAX_THRESHOLD
        self.agrid = np.linspace(-90, 90, self.ABINS)
        self.rgrid = np.linspace(0, self.RMAX, self.RBINS)
        self.cosgrid = np.cos(self.agrid * np.pi / 180)
        self.singrid = np.sin(self.agrid * np.pi / 180)
        self.sine_theta, self.range_d = np.meshgrid(self.singrid, self.rgrid)
        self.cos_theta = np.sqrt(np.maximum(0, 1 - self.sine_theta**2))
        self.x_axis = np.multiply(self.range_d, self.cos_theta)
        self.y_axis = np.multiply(self.range_d, self.sine_theta)
        self.x_axis_grid = np.linspace(0, self.RMAX, self.RBINS)
        self.y_axis_grid = np.linspace(-self.RMAX, self.RMAX, self.ABINS)

    def convert_pol2cart(self, a):
        b = np.zeros((self.RBINS, self.ABINS))
        loc = np.argwhere(a > 0)
        if loc.size > 0:
            xloc = loc[:, 0]
            yloc = loc[:, 1]
            x = self.x_axis[xloc, yloc]
            y = self.y_axis[xloc, yloc]
            new_xloc = [np.argmax(self.x_axis_grid >= x[i]) for i in range(len(x))]
            new_yloc = [np.argmax(self.y_axis_grid >= y[i]) for i in range(len(y))]
            b[new_xloc, new_yloc] = a[xloc, yloc]
        return b

    def convert2pcd_from_array(self, cart_img):
        if cart_img.dtype != np.uint8:
            cart_img = cart_img.astype(np.uint8)
        ret, thresh_img = cv2.threshold(
            cart_img, self.MIN_THRESHOLD, self.MAX_THRESHOLD, cv2.THRESH_TOZERO
        )
        location = np.squeeze(cv2.findNonZero(thresh_img))

        if location.size == 1:
            dummy = np.column_stack((np.array([0]), np.array([0]), np.array([0])))
            return dummy
        elif location.size == 2:
            dummy = np.column_stack((np.array([0]), np.array([0]), np.array([0])))
            return dummy
        else:
            y_location = self.y_axis_grid[location[:, 0]]
            x_location = self.x_axis_grid[location[:, 1]]
            point_loc_3d = np.column_stack(
                (x_location, y_location, np.zeros(location.shape[0]))
            )
            return point_loc_3d

    def extract_point_cloud_from_polar_image(self, polar_img):
        cart_img = self.convert_pol2cart(polar_img)
        cart_img = cart_img.astype(np.uint8)
        points = self.convert2pcd_from_array(cart_img)
        return points

    def compute_chamfer(self, point_cloud_1, point_cloud_2):
        if point_cloud_1.size == 0 and point_cloud_2.size == 0:
            return 0.0
        elif point_cloud_1.size == 0 or point_cloud_2.size == 0:
            return float("inf")

        if point_cloud_1.ndim == 1:
            point_cloud_1 = point_cloud_1.reshape(1, -1)
        if point_cloud_2.ndim == 1:
            point_cloud_2 = point_cloud_2.reshape(1, -1)

        kdtree_2 = KDTree(point_cloud_2)
        kdtree_1 = KDTree(point_cloud_1)

        distances_1_to_2, _ = kdtree_2.query(point_cloud_1, k=1, p=2)
        term1 = np.sum(distances_1_to_2) / len(point_cloud_1)
        distances_2_to_1, _ = kdtree_1.query(point_cloud_2, k=1, p=2)
        term2 = np.sum(distances_2_to_1) / len(point_cloud_2)

        cd = (term1 + term2) / 2
        return cd

    def compute_modified_hausdorff_distance(self, point_cloud_1, point_cloud_2):
        if point_cloud_1.size == 0 and point_cloud_2.size == 0:
            return 0.0
        elif point_cloud_1.size == 0 or point_cloud_2.size == 0:
            return float("inf")

        if point_cloud_1.ndim == 1:
            point_cloud_1 = point_cloud_1.reshape(1, -1)
        if point_cloud_2.ndim == 1:
            point_cloud_2 = point_cloud_2.reshape(1, -1)

        kdtree_2 = KDTree(point_cloud_2)
        kdtree_1 = KDTree(point_cloud_1)
        distances_1_to_2, _ = kdtree_2.query(point_cloud_1, k=1, p=2)
        median_dist_1_to_2 = np.median(distances_1_to_2)
        distances_2_to_1, _ = kdtree_1.query(point_cloud_2, k=1, p=2)
        median_dist_2_to_1 = np.median(distances_2_to_1)

        mhd = max(median_dist_1_to_2, median_dist_2_to_1)
        return mhd
