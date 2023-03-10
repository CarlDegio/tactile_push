from Bullet import draw_debug
from Bullet import reset
import pybullet as p
import pybulletX as px
import numpy as np
import tacto

px.init(mode=p.GUI)
robot = px.Robot("../Meshes/ur10_tactile.urdf", use_fixed_base=True, flags=1)
sphere = px.Body("../Meshes/sphere_small/sphere_small.urdf", base_position=[0.4, 0, 0.1], use_fixed_base=False,
                 flags=1)
digits = tacto.Sensor()
digits.add_camera(robot.id, robot.get_joint_index_by_name("digit_joint"))
digits.add_body(sphere)

desire_pos = np.array([0.3, 0.0, 0.01])
desire_quaternion = np.array([0, 0, 0, 1])
desired_joint_positions = p.calculateInverseKinematics(
    robot.id, robot.get_joint_index_by_name("digit_joint"), desire_pos, desire_quaternion,
)
draw_debug.draw_frame(robot.get_joint_index_by_name("digit_joint"))
draw_debug.draw_area(size=[0.05, 0.1, 0.05], position=[0.8, 0.0, 0])
reset.reset_ur10(robot, desired_joint_positions=desired_joint_positions)

tick = 0

while True:
    if tick < 240 * 2:
        desire_pos[0] += 0.0005  # 不超过0.0005
    if tick > 240 * 6:
        break
    desired_joint_positions = p.calculateInverseKinematics(
        robot.id, robot.get_joint_index_by_name("digit_joint"), desire_pos, desire_quaternion,
    )
    p.setJointMotorControlArray(
        bodyIndex=robot.id,
        jointIndices=robot.free_joint_indices,
        controlMode=p.POSITION_CONTROL,
        targetPositions=desired_joint_positions
    )

    real_ee_position, real_ee_orientation = \
        p.getLinkState(robot.id, robot.get_joint_index_by_name("ee_fixed_joint"))[0:2]

    real_ball_position, real_ball_orientation = sphere.get_base_pose()

    if tick % 60 == 0:  # low frequency
        color, depth = digits.render()
        digits.updateGUI(color, depth)
        print(np.sum(depth))  # z_range=0.002
    p.stepSimulation()
    tick += 1
