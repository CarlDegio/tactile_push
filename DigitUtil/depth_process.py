import cv2
import numpy as np


class DepthKit:
    def __init__(self, depth_poisson=np.zeros((160, 120)), z_range=0.002):
        self.z_range = z_range
        self.depth = np.clip(depth_poisson / self.z_range, 0, 1)

    def update_depth(self, depth_poisson):
        self.depth = np.clip(depth_poisson / self.z_range, 0, 1)

    def check_contact(self):
        if np.max(self.depth) > 0:
            return True
        else:
            return False

    def calc_center(self):  # TODO: center of circle maybe better
        center = np.mean(np.argwhere(self.depth > 0), axis=0)
        return center

    def calc_total(self):
        return np.sum(self.depth)

    def calc_mean(self):
        return self.calc_total() / (160 * 120)

    def show(self):
        cv2.imshow("depth", self.depth)
        cv2.waitKey(0)


if __name__ == "__main__":
    depth = cv2.imread("../Test/test_figure/depth_feature_0.png", cv2.IMREAD_GRAYSCALE)
    depth = np.asarray(depth * 0.002 / 255, dtype=np.float64)
    depth_kit = DepthKit(depth)
    depth_kit.show()
    print("have contact:", depth_kit.check_contact())
    print("center:", depth_kit.calc_center())
    print("sum:", depth_kit.calc_total())
    print("mean:", depth_kit.calc_mean())
