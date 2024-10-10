from controller import Robot, Motor, Camera, LED, Keyboard, InertialUnit, Gyro, Supervisor
from device_init import *
import math
# from cobot import Cobot
NUMBER_OF_LEDS = 8
NUMBER_OF_JOINTS = 12
NUMBER_OF_CAMERAS = 5


robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
motors, cameras, leds, gyro, keys= init_all_device(robot=robot, timestep=timestep)




# ########################################## #
# 这种分解式的动作执行方法，每次都会完全阻塞进程，无法被打断，需要进行修改
# ########################################## #
def step():
    if robot.step(timestep) == -1:
        exit(0)
def movement_decomposition(target, duration): #  这个控制的是输入的 position
    n_steps_to_achieve_target = int(duration * 1000 / timestep)
    step_difference = [0] * NUMBER_OF_JOINTS
    current_position = [0] * NUMBER_OF_JOINTS

    for i in range(NUMBER_OF_JOINTS):
        current_position[i] = motors[i].getTargetPosition()
        step_difference[i] = (target[i] - current_position[i]) / n_steps_to_achieve_target

    for i in range(n_steps_to_achieve_target):
        for j in range(NUMBER_OF_JOINTS):
            current_position[j] += step_difference[j]
            motors[j].setPosition(current_position[j])
        step()
def movement_decomposition_emo_change(target, duration, cobot): #  这个控制的是输入的 position
    temp_emo = cobot.check_emo_queue()
    n_steps_to_achieve_target = int(duration * 1000 / timestep)
    step_difference = [0] * NUMBER_OF_JOINTS
    current_position = [0] * NUMBER_OF_JOINTS

    for i in range(NUMBER_OF_JOINTS):
        current_position[i] = motors[i].getTargetPosition()
        step_difference[i] = (target[i] - current_position[i]) / n_steps_to_achieve_target

    for i in range(n_steps_to_achieve_target):
        for j in range(NUMBER_OF_JOINTS):
            if(cobot.check_emo_queue() != temp_emo): # 如果emo 发生变化 则退出当前动作
                return
            current_position[j] += step_difference[j]
            motors[j].setPosition(current_position[j])
        step()

def lie_down_emo_change(duration, cobot):
    motors_target_pos = [-0.40, -0.99, 1.59,   # Front left leg
                         0.40, -0.99, 1.59,   # Front right leg
                         -0.40, -0.99, 1.59,  # Rear left leg
                         0.40, -0.99, 1.59]   # Rear right leg
    movement_decomposition_emo_change(motors_target_pos, duration, cobot)

def lie_down(duration):
    motors_target_pos = [-0.40, -0.99, 1.59,   # Front left leg
                         0.40, -0.99, 1.59,   # Front right leg
                         -0.40, -0.99, 1.59,  # Rear left leg
                         0.40, -0.99, 1.59]   # Rear right leg
    movement_decomposition(motors_target_pos, duration)
def stand_up(duration):
    motors_target_pos = [-0.1, 0.0, 0.0,   # Front left leg
                         0.1,  0.0, 0.0,   # Front right leg
                         -0.1, 0.0, 0.0,   # Rear left leg
                         0.1,  0.0, 0.0]   # Rear right leg
    movement_decomposition(motors_target_pos, duration)
def sit_down(duration):
    motors_target_pos = [-0.20, -0.40, -0.19,  # Front left leg
                         0.20, -0.40, -0.19,  # Front right leg
                         -0.40, -0.90, 1.18,  # Rear left leg
                         0.40, -0.90, 1.18]   # Rear right leg
    movement_decomposition(motors_target_pos, duration)

def sit_down_emo_change(duration):
    motors_target_pos = [-0.20, -0.40, -0.19,  # Front left leg
                         0.20, -0.40, -0.19,  # Front right leg
                         -0.40, -0.90, 1.18,  # Rear left leg
                         0.40, -0.90, 1.18]   # Rear right leg
    movement_decomposition_emo_change(motors_target_pos, duration, cobot)

def give_paw_emo_change(duration, cobot):
    # Stabilize posture
    temp_emo = cobot.check_emo_queue()
    motors_target_pos_1 = [-0.20, -0.30, 0.05,  # Front left leg
                           0.20, -0.40, -0.19,  # Front right leg
                           -0.40, -0.90, 1.18,  # Rear left leg
                           0.49, -0.90, 0.80]   # Rear right leg

    movement_decomposition_emo_change(motors_target_pos_1, 2, cobot)

    initial_time = robot.getTime()
    while robot.getTime() - initial_time < duration:
        if(cobot.check_emo_queue() != temp_emo): # 如果emo 发生变化 则退出当前动作
            return
        motors[4].setPosition(0.2 * math.sin(2 * robot.getTime()) + 0.6)  # Upperarm movement
        motors[5].setPosition(0.4 * math.sin(2 * robot.getTime()))        # Forearm movement
        step()
    # Get back in sitting posture
    motors_target_pos_2 = [-0.20, -0.40, -0.19,  # Front left leg
                           0.20, -0.40, -0.19,  # Front right leg
                           -0.40, -0.90, 1.18,  # Rear left leg
                           0.40, -0.90, 1.18]   # Rear right leg

    movement_decomposition_emo_change(motors_target_pos_2, 2, cobot)
def give_paw(duration):
    # Stabilize posture

    motors_target_pos_1 = [-0.20, -0.30, 0.05,  # Front left leg
                           0.20, -0.40, -0.19,  # Front right leg
                           -0.40, -0.90, 1.18,  # Rear left leg
                           0.49, -0.90, 0.80]   # Rear right leg

    movement_decomposition(motors_target_pos_1, 2)

    initial_time = robot.getTime()
    while robot.getTime() - initial_time < duration:

        motors[4].setPosition(0.2 * math.sin(2 * robot.getTime()) + 0.6)  # Upperarm movement
        motors[5].setPosition(0.4 * math.sin(2 * robot.getTime()))        # Forearm movement
        step()
    # Get back in sitting posture
    motors_target_pos_2 = [-0.20, -0.40, -0.19,  # Front left leg
                           0.20, -0.40, -0.19,  # Front right leg
                           -0.40, -0.90, 1.18,  # Rear left leg
                           0.40, -0.90, 1.18]   # Rear right leg

    movement_decomposition(motors_target_pos_2, 2)

def deg_2_rad(deg):
    return deg * (math.pi / 180.0)

def go_back(duration):
    # Initial positions for the joints
    motors_target_pos = [-0.1, 0.0, 0.0,   # Front left leg
                          0.1, 0.0, 0.0,   # Front right leg
                         -0.1, 0.0, 0.0,   # Rear left leg
                          0.1, 0.0, 0.0]   # Rear right leg

    movement_decomposition(motors_target_pos, 1.0) # 先stand_up
    Ah = 10
    Ak = 8
    initial_time = robot.getTime()
    t = 0
    T = 1
    while t < duration:
        t = robot.getTime() - initial_time

        lf_hip_pos = Ah * math.sin(2 * math.pi / T * t - math.pi / 2)
        lf_knee_pos = Ak * math.sin(2 * math.pi / T * t + math.pi)
        lb_hip_pos = Ah * math.sin(2 * math.pi / T * t + math.pi / 2)
        lb_knee_pos = Ak * math.sin(2 * math.pi / T * t)
        rf_hip_pos = Ah * math.sin(2 * math.pi / T * t + math.pi / 2)
        rf_knee_pos = Ak * math.sin(2 * math.pi / T * t)
        rb_hip_pos = Ah * math.sin(2 * math.pi / T * t - math.pi / 2)
        rb_knee_pos = Ak * math.sin(2 * math.pi / T * t + math.pi)

        motors[1].setPosition(deg_2_rad(lb_knee_pos))
        motors[2].setPosition(deg_2_rad(rb_knee_pos))
        motors[4].setPosition(deg_2_rad(lf_knee_pos))
        motors[5].setPosition(deg_2_rad(rf_knee_pos))
        motors[7].setPosition(deg_2_rad(lb_hip_pos))
        motors[8].setPosition(deg_2_rad(rb_hip_pos))
        motors[10].setPosition( deg_2_rad(lf_hip_pos))
        motors[11].setPosition(deg_2_rad(rf_hip_pos))

        t += timestep / 1000.0
        step()
def go_ahead(duration):
    # Initial positions for the joints
    motors_target_pos = [-0.1, 0.0, 0.0,   # Front left leg
                          0.1, 0.0, 0.0,   # Front right leg
                         -0.1, 0.0, 0.0,   # Rear left leg
                          0.1, 0.0, 0.0]   # Rear right leg

    movement_decomposition(motors_target_pos, 1.0) # 先stand_up
    Ah = 14
    Ak = 10
    initial_time = robot.getTime()
    t = 0
    T = 1
    while t < duration:
        t = robot.getTime() - initial_time

        lf_hip_pos = Ah * math.sin(2 * math.pi / T * t)
        lf_knee_pos = Ak * math.sin(2 * math.pi / T * t - math.pi / 2)
        lb_hip_pos = Ah * math.sin(2 * math.pi / T * t)
        lb_knee_pos = Ak * math.sin(2 * math.pi / T * t + math.pi / 2)

        rf_hip_pos = Ah * math.sin(2 * math.pi / T * t + math.pi)
        rf_knee_pos = Ak * math.sin(2 * math.pi / T * t - math.pi / 2)
        rb_hip_pos = Ah * math.sin(2 * math.pi / T * t + math.pi)
        rb_knee_pos = Ak * math.sin(2 * math.pi / T * t + math.pi / 2)

        motors[1].setPosition(deg_2_rad(lb_knee_pos))
        motors[2].setPosition(deg_2_rad(rb_knee_pos))
        motors[4].setPosition(deg_2_rad(lf_knee_pos))
        motors[5].setPosition(deg_2_rad(rf_knee_pos))
        motors[7].setPosition(deg_2_rad(lb_hip_pos))
        motors[8].setPosition(deg_2_rad(rb_hip_pos))
        motors[10].setPosition( deg_2_rad(lf_hip_pos))
        motors[11].setPosition(deg_2_rad(rf_hip_pos))

        t += timestep / 1000.0
        step()

