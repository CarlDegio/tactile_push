import pybullet as p


def add_box(size, position):
    visual_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=[0, 0, 1, 1])
    collision_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_shape_id, baseVisualShapeIndex=visual_shape_id,
                      basePosition=position)

def add_walls():
    add_box([0.4, 0.02, 0.05], [0.4, 0.7, 0.05])
    add_box([0.4, 0.02, 0.05], [0.4, -0.7, 0.05])
    add_box([0.02, 0.7, 0.05], [0.8, 0.0, 0.05])