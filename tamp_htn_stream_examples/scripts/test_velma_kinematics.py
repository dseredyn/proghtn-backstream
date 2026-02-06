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

from rclpy.utilities import remove_ros_args

import PyKDL

from tamp_htn_stream_velma.velma_kinematics import KinematicsSolverVelma, KinematicsSolverLWR4

from tamp_htn_stream_velma.moveit_interface import load_named_states_from_srdf, deg2rad,\
    kdl_frame_to_pose, PlannerClient, overwrite_configuration, cPoseStamped, cHeader,\
    get_trajectory_last_point

import numpy as np
import rclpy
from rclpy.node import Node

from visualization_msgs.msg import *
from sensor_msgs.msg import JointState

from tamp_htn_stream_velma.conversions import cMarkerFrame, cMarkerArrow
from tamp_htn_stream_velma.data_types import KdlFrame, KdlVector

def strFrame(T):
    qx, qy, qz, qw = T.M.GetQuaternion()
    return 'PyKDL.Frame(PyKDL.Rotation.Quaternion({}, {}, {}, {}), PyKDL.Vector({}, {}, {}))'.format(qx, qy, qz, qw, T.p.x(), T.p.y(), T.p.z())

def printFrame(T):
    print(strFrame(T))

def generateTestQ():
    limits = [[-2.96, 2.96],
                [-2.09, 2.09],
                [-2.96, 2.96],
                [-2.095, 2.095],
                [-2.96, 2.96],
                [-2.09, 2.09],
                [-2.96, 2.96],]

    result = []
    samples = 4
    for q0 in np.linspace(limits[0][0], limits[0][1], samples, endpoint=True):
        for q1 in np.linspace(limits[1][0], limits[1][1], samples, endpoint=True):
            for q2 in np.linspace(limits[2][0], limits[2][1], samples, endpoint=True):
                for q3 in np.linspace(limits[3][0], limits[3][1], samples, endpoint=True):
                    for q4 in np.linspace(limits[4][0], limits[4][1], samples, endpoint=True):
                        for q5 in np.linspace(limits[5][0], limits[5][1], samples, endpoint=True):
                            for q6 in np.linspace(limits[6][0], limits[6][1], samples, endpoint=True):
                                result.append( (q0, q1, q2, q3, q4, q5, q6) )
    return result

class KinematicsTest(Node):
    def __init__(self):
        super().__init__("planner_client")
        self._js_pub = self.create_publisher(JointState, "/joint_states", 10)
        self._m_pub = self.create_publisher(MarkerArray, "/velma_ik_geom", 10)

    def testFK(self, arm_name):
        assert arm_name in ('right', 'left')

        for it in range(5):
            rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.1)

        solv = KinematicsSolverLWR4()
        vsolv = KinematicsSolverVelma()
        js_msg = JointState()
        for i in range(7):
            js_msg.name.append('{}_arm_{}_joint'.format(arm_name, i))
            js_msg.position.append(0.0)

        js_msg.name.append('torso_0_joint')
        js_msg.position.append(0.0)

        T_B_A0 = vsolv.getArmBaseFk(arm_name, 0.0)
        try:
            phase = 0.0
            while True: #not rospy.is_shutdown():
                q = [   math.sin(phase),
                        math.sin(phase*1.1),
                        math.sin(phase*1.2),
                        math.sin(phase*1.3),
                        math.sin(phase*1.4),
                        math.sin(phase*1.5),
                        math.sin(phase*1.6)]
                phase += 0.01

                T_A0_A7 = solv.calculateFk( q )
                m_id = 0
                markers = cMarkerFrame('frame', 0, T_B_A0*T_A0_A7, 0.3)

                m = MarkerArray()
                m.markers = markers
                self._m_pub.publish(m)

                js_msg.header.stamp = self.get_clock().now().to_msg()
                for i in range(7):
                    js_msg.position[i] = q[i]
                self._js_pub.publish(js_msg)

                rclpy.spin_once(self, timeout_sec=0.1)
                time.sleep(0.1)

        finally:
            self.destroy_node()
            rclpy.shutdown()
    


    def testIk1(self):
        solv = KinematicsSolverLWR4()

        samples = generateTestQ()
        samples_count = len(samples)
        print('Number of samples: {}'.format(samples_count))

        for sample_idx, sample in enumerate(samples):
            q = sample
            # Calculate FK
            print('sample: {} / {}'.format(sample_idx, samples_count))
            print(f'  q: {q}')
            T_A0_A7d = solv.calculateFk( q )
            #print('  T_A0_A7d: {}'.format(strFrame(T_A0_A7d)))

            for elbow_circle_angle in np.linspace(-math.pi, math.pi, 10, endpoint=True):
                #print('  elbow_circle_angle: {}'.format(elbow_circle_angle))
                iq = solv.calculateIk(T_A0_A7d, elbow_circle_angle, False, False, False)
                print(iq)
                T_A0_A7 = solv.calculateFk( iq )
                # compare results
                diff = PyKDL.diff(T_A0_A7, T_A0_A7d, 1.0)
                if diff.vel.Norm() > 0.00001 or diff.rot.Norm() > 0.00001:
                    print('ERROR: {}: {}, {}, {}, {}'.format(sample_idx, sample, elbow_circle_angle, strFrame(T_A0_A7d), strFrame(T_A0_A7)))
                    return

    def testIk2(self, arm_name, T_A0_A7d, ampl):
        assert arm_name in ('right', 'left')

        for it in range(5):
            rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.1)

        solv = KinematicsSolverLWR4()
        vsolv = KinematicsSolverVelma()

        torso_angle = 0.5

        js_msg = JointState()
        for i in range(7):
            js_msg.name.append('{}_arm_{}_joint'.format(arm_name, i))
            js_msg.position.append(0.0)

        js_msg.name.append('torso_0_joint')
        js_msg.position.append(torso_angle)

        base_link_name = 'calib_{}_arm_base_link'.format(arm_name)
        phase = 0.0
        try:
            while True:
                tx = ampl * math.sin(phase)
                ty = ampl * math.sin(phase*1.1)
                tz = ampl * math.sin(phase*1.2)
                T_A0_A7d2 = PyKDL.Frame(PyKDL.Rotation(), PyKDL.Vector(tx, ty, tz)) * T_A0_A7d
                elbow_circle_angle = phase*1.3
                phase += 0.01

                flip_elbow = (math.sin(phase*1.51) > 0)
                flip_ee = (math.sin(phase*1.93) > 0)
                flip_shoulder = (math.sin(phase*2.73) > 0)
                #q = solv.calculateIk(KdlFrame.fromKDL(T_A0_A7d2), elbow_circle_angle, flip_shoulder, flip_elbow, flip_ee)
                # pt_shoulder = solv.getDebug('pt_shoulder')
                # elbow_pt = solv.getDebug('elbow_pt')
                # ee_pt = solv.getDebug('ee_pt')
                ##T_A0_A6 = solv.getDebug('T_A0_A6')

                T_B_A0 = vsolv.getArmBaseFk(arm_name, torso_angle)
                T_B_W = T_B_A0 * KdlFrame.fromKDL(T_A0_A7d2)
                q = vsolv.calculateIkArm(arm_name, T_B_W, torso_angle, elbow_circle_angle, flip_shoulder, flip_elbow, flip_ee)
                pt_shoulder = None
                elbow_pt = None
                ee_pt = None

                m_id = 0
                markers = []
                if not pt_shoulder is None and not ee_pt is None:
                    markers.append( cMarkerArrow('arrow', m_id, pt_shoulder, ee_pt, 0.01, [1.0, 0.0, 0.0], frame_id=base_link_name) )
                    m_id = m_id + 1
                if not pt_shoulder is None and not elbow_pt is None:
                    markers.append( cMarkerArrow('arrow', m_id, pt_shoulder, elbow_pt, 0.01, [1.0, 0.0, 0.0], frame_id=base_link_name) )
                    m_id = m_id + 1

                markers += cMarkerFrame('frame', m_id, KdlFrame.fromKDL(T_A0_A7d2), 0.1, frame_id=base_link_name)
                m_id = m_id + 3

                m = MarkerArray()
                m.markers = markers
                self._m_pub.publish(m)


                # m_id = self._m_pub.publishFrameMarker(T_A0_A7d2, m_id, scale=0.1,
                #                                             frame=base_link_name, namespace='default')
                js_msg.header.stamp = self.get_clock().now().to_msg()
                for i in range(7):
                    if q[i] is None:
                        js_msg.position[i] = 0.0
                    else:
                        js_msg.position[i] = q[i]
                self._js_pub.publish(js_msg)

                rclpy.spin_once(self, timeout_sec=0.1)
                time.sleep(0.04)
        finally:
            self.destroy_node()
            rclpy.shutdown()


def randomOrientation():
    while True:
        qx = random.gauss(0.0, 1.0)
        qy = random.gauss(0.0, 1.0)
        qz = random.gauss(0.0, 1.0)
        qw = random.gauss(0.0, 1.0)
        q_len = math.sqrt( qx**2 + qy**2 + qz**2 + qw**2 )
        if q_len > 0.001:
            qx /= q_len
            qy /= q_len
            qz /= q_len
            qw /= q_len
            return PyKDL.Rotation.Quaternion(qx, qy, qz, qw)

def testIk3(arm_name):
    assert arm_name in ('right', 'left')

    m_pub = MarkerPublisher('velma_ik_geom')
    js_pub = rospy.Publisher('/joint_states', JointState, queue_size=1000)
    rospy.sleep(0.5)

    flips = []
    for flip_shoulder in (True, False):
        for flip_elbow in (True, False):
            for flip_ee in (True, False):
                flips.append( (flip_shoulder, flip_elbow, flip_ee) )
    solv = KinematicsSolverVelma()

    torso_angle = 0.0

    if arm_name == 'right':
        central_point = PyKDL.Vector( 0.7, -0.7, 1.4 )
    else:
        central_point = PyKDL.Vector( 0.7, 0.7, 1.4 )

    js_msg = JointState()
    for i in range(7):
        js_msg.name.append('{}_arm_{}_joint'.format(arm_name, i))
        js_msg.position.append(0.0)

    js_msg.name.append('torso_0_joint')
    js_msg.position.append(torso_angle)

    base_link_name = 'calib_{}_arm_base_link'.format(arm_name)
    phase = 0.0
    while not rospy.is_shutdown():
        # Get random pose
        T_B_A7d = PyKDL.Frame(randomOrientation(), central_point + PyKDL.Vector(random.uniform(-0.4, 0.4), random.uniform(-0.4, 0.4), random.uniform(-0.4, 0.4)))

        m_id = 0
        m_id = m_pub.publishFrameMarker(T_B_A7d, m_id, scale=0.1,
                                                    frame='world', namespace='default')
        
        for flip_shoulder, flip_elbow, flip_ee in flips:
            for elbow_circle_angle in np.linspace(-math.pi, math.pi, 20):
                arm_q = solv.calculateIkArm(arm_name, T_B_A7d, torso_angle, elbow_circle_angle, flip_shoulder, flip_elbow, flip_ee)

                if not arm_q[0] is None:
                    js_msg.header.stamp = self.get_clock().now().to_msg()
                    for i in range(7):
                        js_msg.position[i] = arm_q[i]
                    js_pub.publish(js_msg)
                    rospy.sleep(0.04)
                if rospy.is_shutdown():
                    break
            if rospy.is_shutdown():
                break
        rospy.sleep(0.04)


def testIk4flip(arm_name):
    assert arm_name in ('right', 'left')

    m_pub = MarkerPublisher('velma_ik_geom')
    js_pub = rospy.Publisher('/joint_states', JointState, queue_size=1000)
    rospy.sleep(0.5)

    flips = []
    for flip_shoulder in (True, False):
        for flip_elbow in (True, False):
            for flip_ee in (True, False):
                flips.append( (flip_shoulder, flip_elbow, flip_ee) )
    solv = KinematicsSolverVelma()

    torso_angle = 0.0

    if arm_name == 'right':
        central_point = PyKDL.Vector( 0.7, -0.7, 1.4 )
    else:
        central_point = PyKDL.Vector( 0.7, 0.7, 1.4 )

    js_msg = JointState()
    for i in range(7):
        js_msg.name.append('{}_arm_{}_joint'.format(arm_name, i))
        js_msg.position.append(0.0)

    js_msg.name.append('torso_0_joint')
    js_msg.position.append(torso_angle)

    base_link_name = 'calib_{}_arm_base_link'.format(arm_name)
    phase = 0.0
    while not rospy.is_shutdown():
        # Get random pose
        T_B_A7d = PyKDL.Frame(randomOrientation(), central_point + PyKDL.Vector(random.uniform(-0.4, 0.4), random.uniform(-0.4, 0.4), random.uniform(-0.4, 0.4)))

        m_id = 0
        m_id = m_pub.publishFrameMarker(T_B_A7d, m_id, scale=0.1,
                                                    frame='world', namespace='default')
        
        for flip_shoulder, flip_elbow, flip_ee in flips:
            elbow_circle_angle = 0.0
            #for elbow_circle_angle in np.linspace(-math.pi, math.pi, 20):
            arm_q = solv.calculateIkArm(arm_name, T_B_A7d, torso_angle, elbow_circle_angle, flip_shoulder, flip_elbow, flip_ee)

            if not arm_q[0] is None:
                js_msg.header.stamp = self.get_clock().now().to_msg()
                for i in range(7):
                    js_msg.position[i] = arm_q[i]
                js_pub.publish(js_msg)
                rospy.sleep(0.04)
                rospy.sleep(2.0)
            if rospy.is_shutdown():
                break
            # if rospy.is_shutdown():
            #     break
        rospy.sleep(0.04)

def main(argv=None) -> int:
    rclpy.init(args=argv)

    node = KinematicsTest()
    #node.testIk1()
    T_A0_A7d = PyKDL.Frame(PyKDL.Rotation.Quaternion(0, 0.00304150858481, 0.0910915674525, 0.995837876145), PyKDL.Vector(0.10227842037159, 0.2623692295165, 0.30143578700507))
    node.testIk2( 'left', T_A0_A7d, 0.5 )

    #node.testFK('right')
    return 0

    # Requires:
    # ros2 launch velma_description description_test_no_gui.launch.py

    v_solv = KinematicsSolverVelma()
    printFrame( v_solv.getLeftArmBaseFk(0.0) )
    printFrame( v_solv.getRightArmBaseFk(0.0) )
    printFrame( v_solv.getLeftArmFk(0.0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) )
    printFrame( v_solv.getRightArmFk(0.0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) )

    #testFK('right')
    #return 0

    #testIk3( 'left' )
    testIk4flip( 'left' )
    return 0




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
                    node.publish_traj_rviz(mpr.trajectory)
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
                    node.publish_traj_rviz(mpr.trajectory)
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
                    node.publish_traj_rviz(mpr.trajectory)
                    q_current = overwrite_configuration(q_current, get_trajectory_last_point(mpr.trajectory.joint_trajectory))
                else:
                    node.get_logger().warn("Planning failed, not publishing trajectory.")
                    break

            elif command_name == 'compute_cartesian_path':
                group_name, p_ee = command_data
                resp = node.compute_cartesian_path(group=group_name,
                    ee_link='{}_7_link'.format(group_name), start_state=q_current,
                    waypoints=[p_ee], frame_id='world')

                mpr = resp.motion_plan_response

                if mpr.error_code.val == 1:  # SUCCESS w MoveIt (MoveItErrorCodes.SUCCESS)
                    node.publish_traj_rviz(resp.solution)
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
