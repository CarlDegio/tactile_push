def test_p2p():
    import pybullet as p
    import pybulletX as px
    import Bullet.draw_debug as draw_debug
    import Bullet.reset as reset
    import numpy as np

    px.init(mode=p.DIRECT)  # mode=p.DIRECT/p.GUI
    robot = px.Robot("../Meshes/ur10_origin.urdf", use_fixed_base=True)
    step = 0
    reset.reset_ur10_joints(robot)
    desire_pos = np.array([0.5, 0.5, 0.1])
    desire_quaternion = np.array([0.707106, 0, -0.707106, 0])
    draw_debug.draw_frame(robot.get_joint_index_by_name("ee_fixed_joint"))
    while step < 100:
        # time.sleep(0.01) #240hz default,not same as this

        desired_joint_positions = p.calculateInverseKinematics(
            robot.id, robot.get_joint_index_by_name("ee_fixed_joint"), desire_pos, desire_quaternion,
        )
        # endEffectorLinkIndex, ee_link == ee_fixed_joint
        # global targetPosition, targetOrientation
        p.setJointMotorControlArray(
            bodyIndex=robot.id,
            jointIndices=robot.free_joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=desired_joint_positions
        )
        p.stepSimulation()
        step += 1
    real_ee_position, real_ee_orientation = \
        p.getLinkState(robot.id, robot.get_joint_index_by_name("ee_fixed_joint"))[0:2]
    assert np.allclose(real_ee_position, desire_pos, atol=0.001)
    assert np.allclose(real_ee_orientation, desire_quaternion, atol=0.001)
