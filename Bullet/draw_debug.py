import pybullet as p


def draw_frame(link_index):
    p.addUserDebugLine((0, 0, 0), (1, 0, 0), lineColorRGB=(1.0, 0, 0),parentObjectUniqueId=1,parentLinkIndex=7)
    p.addUserDebugLine((0, 0, 0), (0, 1, 0), lineColorRGB=(0, 1.0, 0), parentObjectUniqueId=1, parentLinkIndex=7)
    p.addUserDebugLine((0, 0, 0), (0, 0, 1), lineColorRGB=(0, 0, 1.0), parentObjectUniqueId=1, parentLinkIndex=7)

