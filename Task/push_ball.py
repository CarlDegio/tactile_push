
from Bullet import draw_debug
import pybullet as p
import pybulletX as px
import numpy as np
import tacto

px.init(mode=p.GUI)
robot = px.Robot("../Meshes/ur10_tactile.urdf", use_fixed_base=True)
sphere = px.Body("../Meshes/sphere_small/sphere_small.urdf", base_position=[0.4, 0.5, 0.1], use_fixed_base=False)
digits = tacto.Sensor()
digits.add_camera(robot.id, robot.get_joint_index_by_name("digit_joint"))
digits.add_body(sphere)

desire_pos = np.array([0.0, 0.5, 0.01])
desire_quaternion = np.array([0, 0, 0, 1])
draw_debug.draw_frame(robot.get_joint_index_by_name("digit_joint"))

while True:
    desire_pos[0] += 0.0009

    desired_joint_positions = p.calculateInverseKinematics(
        robot.id, robot.get_joint_index_by_name("digit_joint"), desire_pos, desire_quaternion,
    )
    p.setJointMotorControlArray(
        bodyIndex=robot.id,
        jointIndices=robot.free_joint_indices,
        controlMode=p.POSITION_CONTROL,
        targetPositions=desired_joint_positions
    )

    color, depth = digits.render()
    digits.updateGUI(color, depth)
    p.stepSimulation()