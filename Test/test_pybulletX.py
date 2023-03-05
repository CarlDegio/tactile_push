def pybulletx_panel():
    import pybullet as p
    import pybulletX as px
    import time

    px.init(mode=p.GUI)
    robot = px.Robot("../Meshes/ur10_tactile.urdf", use_fixed_base=True)
    panel = px.gui.RobotControlPanel(robot)
    panel.start()

    while True:
        time.sleep(0.01)
        p.stepSimulation()
