from controller import Robot, Motor, Camera, LED, Keyboard, InertialUnit, Gyro

def moters_init(robot:Robot):
    motors = [robot.getDevice(name) for name in [
        "front left shoulder abduction motor", "front left shoulder rotation motor", "front left elbow motor", # 0 1 2 
        "front right shoulder abduction motor", "front right shoulder rotation motor", "front right elbow motor", # 3 4 5 
        "rear left shoulder abduction motor", "rear left shoulder rotation motor", "rear left elbow motor",# 6 7 8 
        "rear right shoulder abduction motor", "rear right shoulder rotation motor", "rear right elbow motor" # 9 10 11
    ]]
    return motors

def cameras_init(robot:Robot, timestep):
    cameras = [robot.getDevice(name) for name in [
        "left head camera", "right head camera", "left flank camera",
        "right flank camera", "rear camera"
    ]]
    cameras[0].enable(timestep)
    cameras[1].enable(timestep) # 前两个摄像头初始化
    return cameras

def leds_init(robot:Robot):
    leds = [robot.getDevice(name) for name in [
        "left top led", "left middle up led", "left middle down led",
        "left bottom led", "right top led", "right middle up led",
        "right middle down led", "right bottom led"
    ]]
    return leds

def gyro_init(robot:Robot, timestep):
    gyro = robot.getDevice("gyro")
    gyro.enable(timestep)
    return gyro

def init_all_device(robot:Robot, timestep):
    motors = moters_init(robot)
    cameras = cameras_init(robot, timestep)
    leds = leds_init(robot)
    gyro = gyro_init(robot, timestep)

    keys = Keyboard(timestep)

    return motors, cameras, leds, gyro, keys