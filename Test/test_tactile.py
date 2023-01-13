def test_tactile_contact():
    from Bullet import draw_debug
    import pybullet as p
    import pybulletX as px
    import numpy as np
    import tacto

    px.init(mode=p.DIRECT)
    robot = px.Robot("../Meshes/ur10_tactile.urdf", use_fixed_base=True)
    sphere = px.Body("../Meshes/sphere_small/sphere_small.urdf", base_position=[0.7, 0.50, 1.0], use_fixed_base=True)
    digits = tacto.Sensor()
    digits.add_camera(robot.id, robot.get_joint_index_by_name("digit_joint"))
    digits.add_body(sphere)

    desire_pos = np.array([0.5, 0.5, 1])
    desire_quaternion = np.array([0, 0, 0, 1])
    draw_debug.draw_frame(robot.get_joint_index_by_name("digit_joint"))

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


def test_depth_feature():
    import cv2
    import numpy as np
    from DigitUtil.depth_process import DepthKit
    kit = DepthKit()
    for i in range(3):
        depth = cv2.imread('test_figure/depth_feature_' + str(i) + '.png', cv2.IMREAD_GRAYSCALE)
        kit.update_depth(depth)
        assert kit.check_contact() == (np.max(depth) > 0)
        if kit.check_contact():
            assert np.all(kit.calc_center() == np.mean(np.argwhere(depth > 0), axis=0))
        assert kit.calc_total() == np.sum(depth / 255)
