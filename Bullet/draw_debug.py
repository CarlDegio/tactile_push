import pybullet as p


def draw_frame(link_index):
    p.addUserDebugLine((0, 0, 0), (1, 0, 0), lineColorRGB=(1.0, 0, 0), parentObjectUniqueId=1,
                       parentLinkIndex=link_index)
    p.addUserDebugLine((0, 0, 0), (0, 1, 0), lineColorRGB=(0, 1.0, 0), parentObjectUniqueId=1,
                       parentLinkIndex=link_index)
    p.addUserDebugLine((0, 0, 0), (0, 0, 1), lineColorRGB=(0, 0, 1.0), parentObjectUniqueId=1,
                       parentLinkIndex=link_index)


def draw_area(size, position, quaternion=None):
    if quaternion is None:
        quaternion = [0, 0, 0, 1]
    visual_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=[1, 0, 0, 0.5])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id,
                      basePosition=position, baseOrientation=quaternion)
