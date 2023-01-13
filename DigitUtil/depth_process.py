import cv2
import numpy as np


class DepthKit:
    def __init__(self, depth_255=np.zeros((160,120))):
        self.depth = depth_255

    def update_depth(self, depth_255):
        self.depth = depth_255

    def check_contact(self):
        if np.max(self.depth) > 0:
            return True
        else:
            return False

    def calc_center(self):  # TODO: center of circle maybe better
        center = np.mean(np.argwhere(self.depth > 0), axis=0)
        return center

    def calc_total(self):
        return np.sum(self.depth / 255)

    def show(self):
        cv2.imshow("depth", self.depth)
        cv2.waitKey(0)


if __name__ == "__main__":
    depth = cv2.imread("../Test/test_figure/depth_feature_0.png", cv2.IMREAD_GRAYSCALE)
    depth_kit = DepthKit(depth)
    depth_kit.show()
    print(depth_kit.check_contact())
    print(depth_kit.calc_center())
    print(depth_kit.calc_total())
