
def test_p2p():

    import time
    import pybullet as p
    import pybulletX as px
    import Bullet.draw_debug as draw_debug

    px.init()
    robot = px.Robot("../Meshes/ur10_origin.urdf", use_fixed_base=True)

    while True:
        time.sleep(0.1)

        desired_joint_positions = p.calculateInverseKinematics(
            robot.id, robot.get_joint_index_by_name("ee_fixed_joint"), [0.5, 0.5, 1] ,[0, 0, 0, 1],
            )
        # endEffectorLinkIndex, ee_link == ee_fixed_joint
        # global targetPosition, targetOrientation
        draw_debug.draw_frame(robot.get_joint_index_by_name("ee_fixed_joint"))
        p.setJointMotorControlArray(
            bodyIndex=robot.id,
            jointIndices=robot.free_joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=desired_joint_positions
        )
        p.stepSimulation()
