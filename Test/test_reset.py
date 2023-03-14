def test_tactile_contact():
    from Bullet import draw_debug
    import pybullet as p
    import pybulletX as px
    import numpy as np
    import Bullet.reset as reset

    px.init(mode=p.DIRECT)
    robot = px.Robot("../Meshes/ur10_tactile.urdf", use_fixed_base=True)
    sphere = px.Body("../Meshes/sphere_small/sphere_small.urdf", base_position=[0.7, 0.50, 1.0], use_fixed_base=True)
    reset.reset_ur10_joints(robot)

    desire_pos = np.array([0.9, 0.0, 0.1])
    desire_quaternion = np.array([0, 0, 0, 1])
    draw_debug.draw_frame(robot.get_joint_index_by_name("digit_joint"))

    step = 0
    while True:
        reset.reset_ur10_cartesian(robot,desired_position=desire_pos,desired_orientation=desire_quaternion)
        p.resetBasePositionAndOrientation(sphere.id, [0.7, 0, 0.5], [0, 0, 0, 1])

        ball_pos = sphere.get_base_pose()[0]
        real_ee_position = p.getLinkState(robot.id, robot.get_joint_index_by_name("digit_joint"))[0]
        if step == 20:
            assert np.allclose(ball_pos, [0.7, 0, 0.5])
            assert np.allclose(desire_pos, real_ee_position, atol=0.01)
            break

        p.stepSimulation()
        step += 1
