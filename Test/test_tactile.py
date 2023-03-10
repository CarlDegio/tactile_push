def test_tactile_contact():
    from Bullet import draw_debug
    import Bullet.reset as reset
    import pybullet as p
    import pybulletX as px
    import numpy as np
    import tacto

    px.init(mode=p.DIRECT)
    robot = px.Robot("../Meshes/ur10_tactile.urdf", use_fixed_base=True)
    sphere = px.Body("../Meshes/sphere_small/sphere_small.urdf", base_position=[0.6, 0.0, 0.5], use_fixed_base=True)
    digits = tacto.Sensor()
    digits.add_camera(robot.id, robot.get_joint_index_by_name("digit_joint"))
    digits.add_body(sphere)

    desire_pos = np.array([0.4, 0.0, 0.5])
    desire_quaternion = np.array([0, 0, 0, 1])
    draw_debug.draw_frame(robot.get_joint_index_by_name("digit_joint"))
    desired_joint_positions = p.calculateInverseKinematics(
        robot.id, robot.get_joint_index_by_name("digit_joint"), desire_pos, desire_quaternion,
    )
    reset.reset_ur10_joints(robot, desired_joint_positions=desired_joint_positions)
    step = 0
    while True:
        desire_pos[0] += 0.001

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
        if depth[0].max() > 0.001:  # z_range=0.002 default
            break
        p.stepSimulation()
        step += 1
        if step > 1000:
            print("no contact information")
            assert False


def test_tactile_contact_control():
    from Bullet import draw_debug
    from Bullet import reset
    import pybullet as p
    import pybulletX as px
    import numpy as np
    import tacto

    px.init(mode=p.GUI)
    robot = px.Robot("../Meshes/ur10_tactile.urdf", use_fixed_base=True)
    sphere = px.Body("../Meshes/sphere_small/sphere_small.urdf", base_position=[0.6, 0.0, 0.5], use_fixed_base=True)
    digits = tacto.Sensor()
    digits.add_camera(robot.id, robot.get_joint_index_by_name("digit_joint"))
    digits.add_body(sphere)

    desire_pos = np.array([0.5, 0.0, 0.5])
    desire_quaternion = np.array([0, 0, 0, 1])
    draw_debug.draw_frame(robot.get_joint_index_by_name("digit_joint"))

    desired_joint_positions = p.calculateInverseKinematics(
        robot.id, robot.get_joint_index_by_name("digit_joint"), desire_pos, desire_quaternion,
    )
    reset.reset_ur10_joints(robot, desired_joint_positions=desired_joint_positions)

    step = 0
    control_velocity = 0.0001
    p_coeff = 0.1
    d_coeff = 0.01
    last_delta=0
    while True:
        desire_pos[0] += control_velocity

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

        delta=0.001-depth[0].max()
        control_velocity=p_coeff*delta+d_coeff*(delta-last_delta)
        last_delta=delta

        p.stepSimulation()
        step += 1
        if step > 2000:
            print("no contact information")
            assert False


def test_depth_feature():
    import cv2
    import numpy as np
    from DigitUtil.depth_process import DepthKit
    kit = DepthKit()
    for i in range(3):
        depth = cv2.imread('test_figure/depth_feature_' + str(i) + '.png', cv2.IMREAD_GRAYSCALE)
        kit.update_depth(depth/255*0.002) # input depth range is 0~0.002
        assert kit.check_contact() == (np.max(depth) > 0)
        if kit.check_contact():
            assert np.all(kit.calc_center() == np.mean(np.argwhere(depth > 0), axis=0))
        assert kit.calc_total() == np.sum(depth/255) # depth max is 255, while image in kit max is 1
