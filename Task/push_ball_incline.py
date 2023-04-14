from Bullet import draw_debug
from Bullet import reset
from Bullet import add_wall
from Bullet import rotate_mapping
import pybullet as p
import pybulletX as px
import numpy as np
import tacto

from DigitUtil import depth_process

px.init(mode=p.GUI)
robot = px.Robot("../Meshes/ur10_tactile.urdf", use_fixed_base=True, flags=1)
sphere = px.Body("../Meshes/sphere_small/sphere_small.urdf", base_position=[0.27, 0, 0.03], use_fixed_base=False,
                 flags=1)
digits = tacto.Sensor()
digits.add_camera(robot.id, robot.get_joint_index_by_name("digit_joint"))
digits.add_body(sphere)

incline_rad = np.pi / 180 * 10
desire_plane_pos = np.array([0.25, 0.0, 0.01])
desire_real_pos = rotate_mapping.xoy_point_rotate_y_axis(desire_plane_pos, theta=incline_rad)
print(desire_real_pos)
desire_quaternion = p.getQuaternionFromEuler([0, -incline_rad, 0])
draw_debug.draw_frame(robot.get_joint_index_by_name("digit_joint"))
draw_debug.draw_area(size=[0.05, 0.1, 0.05], position=rotate_mapping.xoy_point_rotate_y_axis([0.6, 0, 0], theta=incline_rad),
                     quaternion=desire_quaternion)
reset.reset_ball_pos(sphere, desired_position=rotate_mapping.xoy_point_rotate_y_axis([0.32, 0.0, 0.03], theta=incline_rad))
reset.reset_ur10_cartesian(robot, desire_real_pos, desire_quaternion)
add_wall.add_walls_incline(incline_rad)

depth_kit = depth_process.DepthKit()

tick = 0

while True:
    if tick < 240 * 2:
        desire_plane_pos[0] += 0.0005  # 不超过0.0005
    if tick > 240 * 6:
        pass
    desired_joint_positions = p.calculateInverseKinematics(
        robot.id, robot.get_joint_index_by_name("digit_joint"),
        rotate_mapping.xoy_point_rotate_y_axis(desire_plane_pos, theta=incline_rad), desire_quaternion,
    )
    p.setJointMotorControlArray(
        bodyIndex=robot.id,
        jointIndices=robot.free_joint_indices,
        controlMode=p.POSITION_CONTROL,
        targetPositions=desired_joint_positions
    )

    real_ee_position, real_ee_orientation = \
        p.getLinkState(robot.id, robot.get_joint_index_by_name("digit_joint"))[0:2]

    real_ball_position, real_ball_orientation = sphere.get_base_pose()

    if tick % 24 == 0:  # low frequency
        color, depth = digits.render()
        digits.updateGUI(color, depth)
        depth_kit.update_depth(depth[0])
        print("mean:", depth_kit.calc_center()[1], "---total:", depth_kit.calc_total())
    p.stepSimulation()
    tick += 1
