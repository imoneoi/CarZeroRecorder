#!/usr/bin/env python

import time
import pprint

import numpy as np
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from std_msgs.msg import Int64

from movavg import MovAvg


# global variables
GLOBAL_METRICS = {
    "meters": 0.0,
    "infractions": 0,
    "steer_avg": MovAvg(int(1e5)),
    "v_avg": MovAvg(int(1e5))
}
GLOBAL_VEHICLE_MODE = 0
GLOBAL_LAST_VEHICLE_MODE = 0
GLOBAL_LAST_ODOM = Odometry()


def to_numpy(coord: Point):
    return np.array([coord.x, coord.y, coord.z])


def vehicle_mode_callback(data: Int64):
    global GLOBAL_VEHICLE_MODE
    GLOBAL_VEHICLE_MODE = data.data


def odom_callback(data: Odometry):
    global GLOBAL_METRICS
    global GLOBAL_VEHICLE_MODE, GLOBAL_LAST_VEHICLE_MODE
    global GLOBAL_LAST_ODOM

    # count infractions
    if GLOBAL_LAST_VEHICLE_MODE != GLOBAL_VEHICLE_MODE:
        GLOBAL_LAST_VEHICLE_MODE = GLOBAL_VEHICLE_MODE
        if GLOBAL_VEHICLE_MODE == 0x00:
            GLOBAL_METRICS["infractions"] += 1

    # accumulate in auto mode only
    if GLOBAL_VEHICLE_MODE == 0x01:
        GLOBAL_METRICS["meters"] += np.linalg.norm(
            to_numpy(data.pose.pose.position) - to_numpy(GLOBAL_LAST_ODOM.pose.pose.position))
        GLOBAL_METRICS["steer_avg"].push(data.twist.twist.angular.z)
        GLOBAL_METRICS["v_avg"].push(data.twist.twist.linear.x)

    GLOBAL_LAST_ODOM = data


def main():
    rospy.init_node("metric_node")
    vehicle_mode_sub = rospy.Subscriber("vehicle_mode", Int64, vehicle_mode_callback, queue_size=10)
    odom_sub = rospy.Subscriber("odom", Odometry, odom_callback, queue_size=10)

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        # clear screen
        print("\033c")

        # print metric
        global GLOBAL_METRICS, GLOBAL_VEHICLE_MODE
        print("Mode: {}".format(GLOBAL_VEHICLE_MODE))
        pprint.pprint(GLOBAL_METRICS)
        rate.sleep()


if __name__ == "__main__":
    main()
