#!/usr/bin/env python3

# TODO: add author and license

# Usage:
# ros2 launch velma_moveit_config planning_only.launch.py
# ros2 run tamp_htn_stream_examples moveit_planning_test.py

import argparse
import sys

import random
import math
import time
import copy

import rclpy
from rclpy.utilities import remove_ros_args

import PyKDL

from tamp_htn_stream_velma.moveit_interface import load_named_states_from_srdf, deg2rad,\
    kdl_frame_to_pose, PlannerClient, overwrite_configuration, cPoseStamped, cHeader,\
    get_trajectory_last_point


def main(argv=None) -> int:
    rclpy.init(args=argv)

    # Parse only non-ROS args (so `--ros-args` doesn’t break argparse)
    args_no_ros = remove_ros_args(sys.argv)

    parser = argparse.ArgumentParser(description="Call MoveIt plan_kinematic_path (GetMotionPlan) for a joint-space goal.")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="Joint tolerance (+/-), default 1e-3")
    parser.add_argument("--time", type=float, default=5.0, help="Allowed planning time [s]")
    parser.add_argument("--attempts", type=int, default=1, help="Number of planning attempts")
    parser.add_argument("--pipeline", default="", help="Planning pipeline id (empty = default)")
    parser.add_argument("--planner", default="", help="Planner id (empty = default)")
    parser.add_argument("--vel", type=float, default=1.0, help="Max velocity scaling [0..1]")
    parser.add_argument("--acc", type=float, default=1.0, help="Max acceleration scaling [0..1]")
    ns = parser.parse_args(args_no_ros[1:])

    node = PlannerClient()

    print('named states:')
    named_states = load_named_states_from_srdf(node)
    for group_name in named_states:
        for pose_name in named_states[group_name]:
            print('  group: {}, pose: {}'.format(group_name, pose_name))

    node.wait_for_service()

    f_l = PyKDL.Frame(
        PyKDL.Rotation.RotZ(math.radians(180)),
        PyKDL.Vector(0.52332, 0.42315, 1.165))

    f_r = PyKDL.Frame(
        PyKDL.Rotation(),#.RotZ(math.radians(180)),
        PyKDL.Vector(0.52332, -0.42315, 1.165))

    # Tryby działania planera:
    # plan_kinematic_path - planowanie trajektorii od q_1 do q_2
    # plan_to_ee_pose - planowanie trajektorii od q_1 do p_ee
    # ## compute_ik - wyznaczenie IK dla p_ee, iteracyjnie, startując od q_1
    # plan_to_ee_pose_ik - planowanie trajektorii od q_1 do p_ee, z konfiguracji dla p_ee, iteracyjnie, zaczynając od q_2
    # compute_cartesian_path - planowanie trajektorii prostoliniowej od p_ee_1 do p_ee_2

    node.add_obstacle()
    main_group_name = 'arms_torso'

    commands = [
        ('set_pose', named_states[main_group_name]['zero']),
        ('plan_kinematic_path', (main_group_name, named_states[main_group_name]['rest'])),
        ('plan_to_ee_pose', ('left_arm', kdl_frame_to_pose(f_l))),
        ('plan_to_ee_pose', ('right_arm', kdl_frame_to_pose(f_r))),
        ('plan_to_ee_pose_ik', ('left_arm', kdl_frame_to_pose(f_l), named_states[main_group_name]['work_02'])),
        ('plan_to_ee_pose_ik', ('left_arm', kdl_frame_to_pose(f_l), named_states[main_group_name]['work_01'])),
        ('plan_to_ee_pose_ik', ('right_arm', kdl_frame_to_pose(f_r), named_states[main_group_name]['work_02'])),
        ('plan_to_ee_pose_ik', ('right_arm', kdl_frame_to_pose(f_r), named_states[main_group_name]['work_01'])),
        ('plan_kinematic_path', (main_group_name, named_states[main_group_name]['ready'])),
        ('plan_kinematic_path', (main_group_name, named_states[main_group_name]['rest'])),
        ('compute_cartesian_path', ('left_arm', kdl_frame_to_pose(f_l))),
        ('compute_cartesian_path', ('right_arm', kdl_frame_to_pose(f_r))),
        ('plan_kinematic_path', (main_group_name, named_states[main_group_name]['rest'])),
        ]

    try:
        q_current = None
        for command_name, command_data in commands:
            if command_name == 'set_pose':
                q_current = command_data
            elif command_name == 'plan_kinematic_path':
                group_name, q_dest = command_data
                resp = node.plan_joint_goal(
                    group=group_name,
                    start_state=q_current,
                    joint_goal=q_dest,
                    tolerance=ns.tolerance,
                    allowed_time=ns.time,
                    num_attempts=ns.attempts,
                    pipeline_id=ns.pipeline,
                    planner_id=ns.planner,
                    vel_scale=ns.vel,
                    acc_scale=ns.acc,
                )

                mpr = resp.motion_plan_response
                err = mpr.error_code.val
                
                if mpr.error_code.val == 1:  # SUCCESS w MoveIt (MoveItErrorCodes.SUCCESS)
                    # node.publish_traj_rviz(mpr.trajectory)
                    node.publish_to_rviz2(mpr.trajectory_start, mpr.trajectory)
                    q_current = overwrite_configuration(q_current, q_dest)
                else:
                    node.get_logger().warn("Planning failed, not publishing trajectory.")
                    break

            elif command_name == 'plan_to_ee_pose':
                group_name, target_frame = command_data

                resp = node.plan_to_ee_pose(
                    group=group_name,
                    ee_link='{}_7_link'.format(group_name),
                    start_state=q_current,
                    target=cPoseStamped(cHeader(node.get_clock().now().to_msg(), 'world'), target_frame),
                    )
                mpr = resp.motion_plan_response

                if mpr.error_code.val == 1:  # SUCCESS w MoveIt (MoveItErrorCodes.SUCCESS)
                    # node.publish_traj_rviz(mpr.trajectory)
                    node.publish_to_rviz2(mpr.trajectory_start, mpr.trajectory)
                    q_current = overwrite_configuration(q_current, get_trajectory_last_point(mpr.trajectory.joint_trajectory))
                else:
                    node.get_logger().warn("Planning failed, not publishing trajectory.")
                    break

            elif command_name == 'plan_to_ee_pose_ik':
                group_name, target_frame, q_search = command_data

                resp = node.calculate_ik(group_name, '{}_7_link'.format(group_name), q_search,
                    cPoseStamped(cHeader(node.get_clock().now().to_msg(), 'world'), target_frame))
                if resp.error_code.val == 1:  # SUCCESS w MoveIt (MoveItErrorCodes.SUCCESS)
                    q_dest = {}
                    for joint_name, pos in zip(resp.solution.joint_state.name, resp.solution.joint_state.position):
                        q_dest[joint_name] = pos
                else:
                    node.get_logger().warn("ik failed, err_code: {}".format(resp.error_code.val))
                    break

                resp = node.plan_joint_goal(
                    group=group_name,
                    start_state=q_current,
                    joint_goal=q_dest,
                    tolerance=ns.tolerance,
                    allowed_time=ns.time,
                    num_attempts=ns.attempts,
                    pipeline_id=ns.pipeline,
                    planner_id=ns.planner,
                    vel_scale=ns.vel,
                    acc_scale=ns.acc,
                )

                mpr = resp.motion_plan_response
                err = mpr.error_code.val
                
                if mpr.error_code.val == 1:  # SUCCESS w MoveIt (MoveItErrorCodes.SUCCESS)
                    # node.publish_traj_rviz(mpr.trajectory)
                    node.publish_to_rviz2(mpr.trajectory_start, mpr.trajectory)
                    q_current = overwrite_configuration(q_current, get_trajectory_last_point(mpr.trajectory.joint_trajectory))
                else:
                    node.get_logger().warn("Planning failed, not publishing trajectory.")
                    break

            elif command_name == 'compute_cartesian_path':
                group_name, p_ee = command_data
                resp = node.compute_cartesian_path(group=group_name,
                    ee_link='{}_7_link'.format(group_name), start_state=q_current,
                    waypoints=[p_ee], frame_id='world')

                if mpr.error_code.val == 1:  # SUCCESS w MoveIt (MoveItErrorCodes.SUCCESS)
                    # node.publish_traj_rviz(resp.solution)
                    node.publish_to_rviz2(mpr.trajectory_start, mpr.trajectory)
                    q_current = overwrite_configuration(q_current, get_trajectory_last_point(resp.solution.joint_trajectory))
                else:
                    node.get_logger().warn("Planning failed, not publishing trajectory.")
                    break

                
    finally:
        node.destroy_node()
        rclpy.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
