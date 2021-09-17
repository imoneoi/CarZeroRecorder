#!/usr/bin/env python

import os
from multiprocessing import Process, Event

import numpy as np
import onnxruntime
import cv2
import time

import rospy
from dynamic_reconfigure.server import Server
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

from carzero.cfg import RLPlannerConfig
from shared_array import SharedArray


# Global Variables & Callback

GLOBAL_ODOM = Odometry()
GLOBAL_CONFIG = None


def odom_callback(data: Odometry):
    global GLOBAL_ODOM
    GLOBAL_ODOM = data


def reconfigure_callback(config: RLPlannerConfig, level: int):
    global GLOBAL_CONFIG
    GLOBAL_CONFIG = config
    return config


# Main


def visualizer_worker_(vis_dict: dict, event: Event, terminate: Event, control_rate: int, record_path: str):
    vis_dict = {k: v.get() for k, v in vis_dict.items()}
    for k in vis_dict.keys():
        cv2.namedWindow(k, cv2.WINDOW_NORMAL)

    video_writers = {}
    if record_path:
        save_path = os.path.join(record_path, time.strftime("%H-%M-%S %m-%d-%Y"))
        os.makedirs(save_path, exist_ok=True)

        video_writers = {
            k: cv2.VideoWriter(os.path.join(save_path, k + ".mp4"), cv2.VideoWriter_fourcc(*"mp4v"), control_rate,
                               v.shape[1::-1])
            for k, v in vis_dict.items()}

    while not terminate.is_set():
        event.wait()
        event.clear()
        for k, v in vis_dict.items():
            image = cv2.cvtColor(v, cv2.COLOR_RGB2BGR)
            cv2.imshow(k, image)
            if record_path:
                video_writers[k].write(image)

        cv2.waitKey(1)

    for v in video_writers.values():
        v.release()


def read_image_size(res: str):
    return list(map(int, res.split("x")))


def center_crop(img: np.ndarray, size: list):
    img_h, img_w, _ = img.shape
    x = (img_w - size[0]) // 2
    y = (img_h - size[1]) // 2
    assert (x >= 0) and (y >= 0)
    return img[y: y + size[1], x: x + size[0]]


def run_semantic_segmentation(img: np.ndarray, model: onnxruntime.InferenceSession):
    dtype = np.float32
    # to float & normalize
    x = np.array(img, dtype=dtype) / 255
    x = (x - np.array([0.485, 0.456, 0.406], dtype=dtype)) / np.array([0.229, 0.224, 0.225], dtype=dtype)
    # to NCHW
    x = np.expand_dims(x, 0).transpose((0, 3, 1, 2))
    # inference
    logprob = model.run(None, {"input": x})[0]
    label = np.argmax(logprob, 1)
    # get drivable (ignore 0: non drivable)
    return np.concatenate([[label == i] for i in range(1, logprob.shape[1])], axis=1).astype(np.uint8) * 255


def main():
    # init node and comm
    rospy.init_node("rl_planner")
    odom_sub = rospy.Subscriber("odom", Odometry, odom_callback)
    twist_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
    reconf_server = Server(RLPlannerConfig, reconfigure_callback)

    # read models
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))
    model_semseg = onnxruntime.InferenceSession(os.path.join(model_path, rospy.get_param("model_semseg", "ep_44_iou_0.9343.checkpoint.onnx")))
    model_policy = onnxruntime.InferenceSession(os.path.join(model_path, rospy.get_param("model_policy", "569979_deploy_v0.1.onnx")))

    # read parameters
    param_loop_rate = rospy.get_param("loop_rate", 10)

    param_cam = rospy.get_param("camera", 0)
    param_cam_fps = rospy.get_param("camera_fps", 30)
    param_cap_fourcc = rospy.get_param("camera_fourcc", "YUYV")

    param_size_cam = read_image_size(rospy.get_param("size_camera", "640x480"))
    param_size_seg = read_image_size(rospy.get_param("size_seg", "640x320"))
    param_size_policy = read_image_size(rospy.get_param("size_policy", "160x80"))

    param_record_path = rospy.get_param("record_path", "Videos/carzero")

    # visualizer process
    shared_images_arr = {
        "crop": SharedArray((*param_size_seg[::-1], 3), np.uint8),
        "seg": SharedArray((*param_size_seg[::-1], 3), np.uint8),
        "policy": SharedArray((*param_size_policy[::-1], 3), np.uint8)
    }
    shared_images = {k: v.get() for k, v in shared_images_arr.items()}
    shared_event = Event()
    terminate_event = Event()

    vis_process = Process(target=visualizer_worker_, args=(shared_images_arr, shared_event, terminate_event,
                                                           param_loop_rate, param_record_path))
    vis_process.start()

    # control loop
    rate = rospy.Rate(param_loop_rate)

    cap = cv2.VideoCapture(param_cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, param_size_cam[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, param_size_cam[1])
    cap.set(cv2.CAP_PROP_FPS, param_cam_fps)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*param_cap_fourcc))

    twist = Twist()
    dt = 1.0 / param_loop_rate
    while not rospy.is_shutdown():
        time_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # color & crop
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = center_crop(frame, param_size_seg)
        np.copyto(shared_images["crop"], frame)

        # semantic segmentation
        seg = run_semantic_segmentation(frame, model_semseg)
        # add speed layer
        global GLOBAL_ODOM, GLOBAL_CONFIG
        speed = GLOBAL_ODOM.twist.twist.linear
        speed = np.linalg.norm(np.array([speed.x, speed.y, speed.z]))
        speed = np.clip(speed / GLOBAL_CONFIG.max_vel * 255, 0, 255).astype(np.uint8)
        seg = np.concatenate([np.broadcast_to(speed, (1, 1, *seg.shape[2:])), seg], axis=1)

        seg = seg[0].transpose(1, 2, 0)
        np.copyto(shared_images["seg"], seg)
        # policy inference
        policy_in = cv2.resize(seg, param_size_policy)
        np.copyto(shared_images["policy"], policy_in)

        policy = model_policy.run(None, {"obs": np.expand_dims(policy_in.transpose(2, 0, 1), [0, 1])})[0]
        accel, steer = policy[0]
        print(policy)
        # output
        if accel < 0:
            converted_accel = GLOBAL_CONFIG.max_brake * accel - GLOBAL_CONFIG.vel_decay
        else:
            converted_accel = GLOBAL_CONFIG.max_accel * accel - GLOBAL_CONFIG.vel_decay

        twist.angular.z = GLOBAL_CONFIG.max_steer * -steer  # steer > 0 is turn right
        twist.linear.x = np.clip(twist.linear.x + converted_accel * dt, 0, GLOBAL_CONFIG.max_vel)
        twist_pub.publish(twist)

        # loop
        # print("Latency: {:.2f} ms".format((time.time() - time_start) * 1000))
        shared_event.set()
        rate.sleep()

    # stop visualizer
    terminate_event.set()
    shared_event.set()


if __name__ == "__main__":
    main()
