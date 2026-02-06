# Copyright (c) 2026 Dawid Seredyński

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Decription: this module contains interfaces for MoveIt and some helper functions and classes.

from __future__ import annotations

from typing import Dict, List

import math
import time
import copy

import rclpy
from rclpy.node import Node
from rclpy.utilities import remove_ros_args

from moveit_msgs.srv import GetPositionIK, GetPositionFK, GetMotionPlan, GetCartesianPath,\
    ApplyPlanningScene, GetMotionPlan_Response, ApplyPlanningScene_Response
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint,\
    JointConstraint, BoundingVolume, RobotTrajectory, DisplayTrajectory,\
    DisplayRobotState, RobotState, PlanningScene, CollisionObject

from sensor_msgs.msg import JointState

from geometry_msgs.msg import PoseStamped, Pose, Vector3
from shape_msgs.msg import SolidPrimitive

from trajectory_msgs.msg import JointTrajectory

from builtin_interfaces.msg import Duration

import xml.etree.ElementTree as ET
from typing import Dict

from rclpy.task import Future
from std_msgs.msg import Header

from rclpy.parameter_client import AsyncParameterClient

from .data_types import KdlVector, KdlRotation, KdlFrame

def getAllJointNames() -> list[str]:
    # TODO: read this from the URDF / SRDF
    # Update when URDF / SRDF changes.
    return [
            'gripper_x',
            'gripper_y',
            'gripper_z',
            'gripper_roll',
            'gripper_pitch',
            'gripper_yaw',
            'floating_HandFingerOneKnuckleOneJoint',
            'floating_HandFingerOneKnuckleTwoJoint',
            'floating_HandFingerOneKnuckleThreeJoint',
            'floating_HandFingerThreeKnuckleTwoJoint',
            'floating_HandFingerThreeKnuckleThreeJoint',
            'floating_HandFingerTwoKnuckleOneJoint',
            'floating_HandFingerTwoKnuckleTwoJoint',
            'floating_HandFingerTwoKnuckleThreeJoint',
            'torso_0_joint',
            'head_pan_joint',
            'head_tilt_joint',
            'left_arm_0_joint',
            'left_arm_1_joint',
            'left_arm_2_joint',
            'left_arm_3_joint',
            'left_arm_4_joint',
            'left_arm_5_joint',
            'left_arm_6_joint',
            'leftKeepUprightJoint0',
            'leftKeepUprightJoint1',
            'leftFtSensorJoint',
            'left_HandFingerOneKnuckleOneJoint',
            'left_HandFingerOneKnuckleTwoJoint',
            'left_HandFingerOneKnuckleThreeJoint',
            'left_HandFingerThreeKnuckleTwoJoint',
            'left_HandFingerThreeKnuckleThreeJoint',
            'left_HandFingerTwoKnuckleOneJoint',
            'left_HandFingerTwoKnuckleTwoJoint',
            'left_HandFingerTwoKnuckleThreeJoint',
            'right_arm_0_joint',
            'right_arm_1_joint',
            'right_arm_2_joint',
            'right_arm_3_joint',
            'right_arm_4_joint',
            'right_arm_5_joint',
            'right_arm_6_joint',
            'rightKeepUprightJoint0',
            'rightKeepUprightJoint1',
            'rightFtSensorJoint',
            'right_HandFingerOneKnuckleOneJoint',
            'right_HandFingerOneKnuckleTwoJoint',
            'right_HandFingerOneKnuckleThreeJoint',
            'right_HandFingerThreeKnuckleTwoJoint',
            'right_HandFingerThreeKnuckleThreeJoint',
            'right_HandFingerTwoKnuckleOneJoint',
            'right_HandFingerTwoKnuckleTwoJoint',
            'right_HandFingerTwoKnuckleThreeJoint',
        ]

def read_SRDF(node):
    target_node = '/move_group'
    client = AsyncParameterClient(node, target_node)

    # poczekaj aż serwis parametrów będzie dostępny
    if not client.wait_for_services(timeout_sec=5.0):
        raise RuntimeError(f"Parameter services not available on {target_node}")

    fut = client.get_parameters(["robot_description_semantic"])
    rclpy.spin_until_future_complete(node, fut, timeout_sec=5.0)
    resp = fut.result()
    assert not resp is None
    return resp.values[0].string_value


def load_named_states_from_srdf(node) -> dict[str, dict[str, dict[str, float]]]:
    data = read_SRDF(node)
    tree = ET.ElementTree(ET.fromstring(data))
    root = tree.getroot()
    assert not root is None

    result = {}
    for gs in root.findall("group_state"):
        group = gs.get("group")
        if not group in result:
            result[group] = {}

        state_name = gs.get("name")
        assert not state_name in result[group]
        result[group][state_name] = {}

        for j in gs.findall("joint"):
            jname = j.get("name")
            jval = j.get("value")
            if jname is None or jval is None:
                continue
            result[group][state_name][jname] = float(jval)
    return result


class PlannerClient(Node):
    _named_states: dict[str,dict[str, dict[str,float]]]
    def __init__(self):
        super().__init__("planner_client")
        # self._services_info = {
        #     'plan_kinematic_path':('/plan_kinematic_path', GetMotionPlan),
        #     'compute_cartesian_path':('/compute_cartesian_path', GetCartesianPath),
        #     'compute_ik':('/compute_ik', GetPositionIK),
        #     'apply_planning_scene':('/apply_planning_scene', ApplyPlanningScene),
        # }
        self._mv_clients = {
            'plan_kinematic_path':
                self.create_client(GetMotionPlan, '/plan_kinematic_path'),
            'compute_cartesian_path':
                self.create_client(GetCartesianPath, '/compute_cartesian_path'),
            'compute_ik':
                self.create_client(GetPositionIK, '/compute_ik'),
            'compute_fk':
                self.create_client(GetPositionFK, '/compute_fk'),
            'apply_planning_scene':
                self.create_client(ApplyPlanningScene, '/apply_planning_scene'),
        }
        # for service_name in self._services_info:
        #     service_path, service_type = self._services_info[service_name]
        #     self._services[service_name] = self.create_client(GetMotionPlan, self._service_name)
        # self._service_name = "/plan_kinematic_path"
        # self._cp_service_name = "/compute_cartesian_path"
        # self._client = self.create_client(GetMotionPlan, self._service_name)
        # self._cp_client = self.create_client(GetCartesianPath, self._cp_service_name)
        # self._ik_client = self.create_client(GetPositionIK, "/compute_ik")
        self._display_pub = self.create_publisher(DisplayTrajectory, "/display_planned_path", 10)
        self._display2_pub = self.create_publisher(DisplayRobotState, "/test_display_robot_state", 10)

        # WARNING: robot-specific code
        # TODO: use urdf_parser_py to parse the robot model and to get joint names for each group.
        self._group_joint_names_map = {
            'arms_torso': ['torso_0_joint', 'left_arm_0_joint', 'left_arm_1_joint',
                'left_arm_2_joint', 'left_arm_3_joint', 'left_arm_4_joint', 'left_arm_5_joint',
                'left_arm_6_joint', 'right_arm_0_joint', 'right_arm_1_joint', 'right_arm_2_joint',
                'right_arm_3_joint', 'right_arm_4_joint', 'right_arm_5_joint', 'right_arm_6_joint'],
            'left_arm': ['left_arm_0_joint', 'left_arm_1_joint',
                'left_arm_2_joint', 'left_arm_3_joint', 'left_arm_4_joint', 'left_arm_5_joint',
                'left_arm_6_joint'],
            'right_arm': ['right_arm_0_joint', 'right_arm_1_joint', 'right_arm_2_joint',
                'right_arm_3_joint', 'right_arm_4_joint', 'right_arm_5_joint', 'right_arm_6_joint'],
            'left_arm_torso': ['torso_0_joint', 'left_arm_0_joint', 'left_arm_1_joint',
                'left_arm_2_joint', 'left_arm_3_joint', 'left_arm_4_joint', 'left_arm_5_joint',
                'left_arm_6_joint'],
            'right_arm_torso': ['torso_0_joint', 'right_arm_0_joint', 'right_arm_1_joint', 'right_arm_2_joint',
                'right_arm_3_joint', 'right_arm_4_joint', 'right_arm_5_joint', 'right_arm_6_joint'],
            }
        

    def getNamedState(self, group_name, state_name):
        assert not self._named_states is None
        return self._named_states[group_name][state_name]

    def wait_for_service(self):
        timeout_sec = 0.5
        # kp_client_connected = False
        # cp_client_connected = False
        # ik_client_connected = False
        self._named_states = load_named_states_from_srdf(self)

        connected = set()
        for it in range(20):
            rclpy.spin_once(self, timeout_sec=0.1)
            for service_name, client in self._mv_clients.items():
                if not service_name in connected and client.wait_for_service(timeout_sec=timeout_sec):
                    self.get_logger().info(f"connected to service {client.service_name} after {it+1} tries")
                    connected.add(service_name)
            # if not kp_client_connected and self._client.wait_for_service(timeout_sec=timeout_sec):
            #     kp_client_connected = True
            #     self.get_logger().info(f"connected to service after {it+1} tries")
            # if not cp_client_connected and self._cp_client.wait_for_service(timeout_sec=timeout_sec):
            #     cp_client_connected = True
            #     self.get_logger().info(f"connected to service after {it+1} tries")
            # if not ik_client_connected and self._ik_client.wait_for_service(timeout_sec=timeout_sec):
            #     ik_client_connected = True
            #     self.get_logger().info(f"connected to service after {it+1} tries")
            # if kp_client_connected and cp_client_connected and ik_client_connected:
            #     return
        # Check if all clients are connected
        for service_name in self._mv_clients:
            if not service_name in connected:
                raise RuntimeError(
                    f"No {service_name} service found. "
                    "Check `ros2 node list` for /move_group and `ros2 service list | grep plan_kinematic_path`."
                )

    def publish_to_rviz2(self, trajectory_start, trajectory, model_id: str = "") -> None:
        msg = DisplayTrajectory()
        msg.model_id = model_id  # może być "", RViz zwykle i tak użyje robot_description
        msg.trajectory_start = trajectory_start
        # trajectory.joint_trajectory.header.stamp = self.get_clock().now().to_msg()
        msg.trajectory = [trajectory]  # lista RobotTrajectory
        self._display_pub.publish(msg)
        self.get_logger().info("Published DisplayTrajectory on /display_planned_path")

    def publish_traj_rviz(self, traj, model_id: str = "") -> None:
        assert isinstance(traj, RobotTrajectory)
        start_traj_time = time.time()+0.5
        for pt in traj.joint_trajectory.points:
            msg = DisplayRobotState()
            msg.state.joint_state.header
            msg.state.joint_state.name = traj.joint_trajectory.joint_names
            msg.state.joint_state.position = pt.positions

            self._display2_pub.publish(msg)

            traj_time = start_traj_time + duration_to_float(pt.time_from_start)

            while time.time() < traj_time:
                rclpy.spin_once(self, timeout_sec=0.1)
                time.sleep(0.1)

        self.get_logger().info("Published DisplayRobotState on /test_display_robot_state")


    def publish_conf_rviz(self, solution, model_id: str = "") -> None:
        assert isinstance(solution, JointState)

        msg = DisplayRobotState()
        msg.state.joint_state = solution
        # msg.state.joint_state.name = solution.trajectory.joint_trajectory.joint_names
        # msg.state.joint_state.position = pt.positions

        self._display2_pub.publish(msg)

        #for it in range(2):
        rclpy.spin_once(self, timeout_sec=0.1)
        #    time.sleep(0.1)

        self.get_logger().info("Published DisplayRobotState on /test_display_robot_state")


    def calculate_ik(self, group, ee_link, start_state: dict[str, float], target):
        # if self._ik_client is None:
        #     raise RuntimeError("Service client not ready. Call wait_for_service() first.")

        # Jedno rozwiązanie IK zależne od start_state
        req = GetPositionIK.Request()
        req.ik_request.group_name = group
        req.ik_request.avoid_collisions = True
        req.ik_request.ik_link_name = ee_link
        req.ik_request.pose_stamped = target
        req.ik_request.robot_state = PlannerClient._make_start_state(start_state)
        req.ik_request.robot_state.is_diff = False
        req.ik_request.timeout = Duration(sec=1, nanosec=int(0.2e9))

        #future = self._ik_client.call_async(req)
        future = self._mv_clients['compute_ik'].call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            raise RuntimeError(f"Service call failed: {future.exception()}")

        return future.result()

    def calculate_fk(self, group, ee_link, start_state):
        req = GetPositionFK.Request()
        req.fk_link_names.append(ee_link) # type: ignore
        req.robot_state = PlannerClient._make_start_state(start_state)
        req.robot_state.is_diff = False
        future = self._mv_clients['compute_fk'].call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            raise RuntimeError(f"Service call failed: {future.exception()}")
        return future.result()

    def plan_joint_goal(
        self,
        attached_collision_objects,
        group: str,
        start_state: Dict[str, float],
        joint_goal: Dict[str, float],
        tolerance: float,
        allowed_time: float,
        num_attempts: int,
        pipeline_id: str,
        planner_id: str,
        vel_scale: float,
        acc_scale: float,
    ):
        # if self._client is None:
        #     raise RuntimeError("Service client not ready. Call wait_for_service() first.")

        req = GetMotionPlan.Request()

        mpr = req.motion_plan_request
        mpr.group_name = group
        mpr.num_planning_attempts = int(num_attempts)
        mpr.allowed_planning_time = float(allowed_time)
        mpr.max_velocity_scaling_factor = float(vel_scale)
        mpr.max_acceleration_scaling_factor = float(acc_scale)

        # Use current robot state from PlanningSceneMonitor (requires /joint_states reaching move_group).
        # This avoids having to fill start_state manually.
        # mpr.start_state.is_diff = True
        mpr.start_state = PlannerClient._make_start_state(start_state)
        mpr.start_state.attached_collision_objects = attached_collision_objects

        if pipeline_id:
            mpr.pipeline_id = pipeline_id
        if planner_id:
            mpr.planner_id = planner_id

        goal = Constraints()
        for jname, jpos in joint_goal.items():
            if not jname in self._group_joint_names_map[group]:
                continue
            # else:
            jc = JointConstraint()
            jc.joint_name = jname
            jc.position = float(jpos)
            jc.tolerance_above = float(tolerance)
            jc.tolerance_below = float(tolerance)
            jc.weight = 1.0
            goal.joint_constraints.append(jc) # type: ignore

        mpr.goal_constraints = [goal]

        future = self._mv_clients['plan_kinematic_path'].call_async(req)
        rclpy.spin_until_future_complete(self, future)
        resp = future.result()
        if resp is None:
            raise RuntimeError(f"Service call failed: {future.exception()}")

        assert isinstance(resp, GetMotionPlan_Response)
        return resp.motion_plan_response

    @staticmethod
    # def _make_start_state(joint_positions: dict[str, float], *, diff: bool = False) -> RobotState:
    def _make_start_state(joint_positions: dict[str, float], diff: bool = False) -> RobotState:
        all_joint_names = getAllJointNames()

        rs = RobotState()
        rs.is_diff = bool(diff)
        js = JointState()
        js.name = list(joint_positions.keys())
        js.position = [float(joint_positions[n]) for n in js.name]

        # print('_make_start_state()')
        # print(f'Provided names:')
        # for name, pos in zip(js.name, js.position):
        #     print(f'  {name}: {pos}')
        # print('Adding joints positions:')
        for joint_name in all_joint_names:
            if not joint_name in js.name:
                js.name.append(joint_name)
                js.position.append( 0.0 )
                # print(f'  {joint_name}: {0.0}')

        # print(f'_make_start_state:')
        # for name, pos in zip(js.name, js.position):
        #     print(f'  {name}: {pos}')

        rs.joint_state = js
        return rs


    @staticmethod
    def _make_pose_goal_constraints(
        target: PoseStamped,
        ee_link: str,
        pos_tol_x: float,
        pos_tol_y: float,
        pos_tol_z: float,
        ori_tol: float|None,
        constraint_name: str = "ee_pose_goal",
    ) -> Constraints:
        """
        Buduje Constraints z:
          - PositionConstraint jako BOX o wymiarach 2*pos_tol wokół target.position
          - OrientationConstraint z tolerancjami osi (abs_*_axis_tolerance)
        """
        goal = Constraints()
        goal.name = constraint_name

        # --- PositionConstraint: mały box wokół pozycji ---
        pc = PositionConstraint()
        pc.header = target.header
        pc.link_name = ee_link
        pc.weight = 1.0

        bv = BoundingVolume()
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [2.0 * pos_tol_x, 2.0 * pos_tol_y, 2.0 * pos_tol_z]

        box_pose = Pose()
        box_pose.position = target.pose.position
        # orientacja boxa dowolna; zwykle identyczność wystarczy
        box_pose.orientation.w = 1.0

        bv.primitives = [box]
        bv.primitive_poses = [box_pose]
        pc.constraint_region = bv

        # --- OrientationConstraint ---
        if not ori_tol is None:
            oc = OrientationConstraint()
            oc.header = target.header
            oc.link_name = ee_link
            oc.orientation = target.pose.orientation
            oc.absolute_x_axis_tolerance = ori_tol
            oc.absolute_y_axis_tolerance = ori_tol
            oc.absolute_z_axis_tolerance = ori_tol
            oc.weight = 1.0
            goal.orientation_constraints = [oc]

        goal.position_constraints = [pc]
        return goal

    def plan_to_ee_pose(
        self,
        attached_collision_objects,
        *,
        group: str,
        ee_link: str,
        start_state: Dict[str, float],
        target: PoseStamped,
        pos_tol_x: float = 0.002,
        pos_tol_y: float = 0.002,
        pos_tol_z: float = 0.002,
        ori_tol: float|None = 0.05,
        allowed_time: float = 5.0,
        num_attempts: int = 1,
        pipeline_id: str = "",
        planner_id: str = "",
        vel_scale: float = 1.0,
        acc_scale: float = 1.0,
    ):
        """
        Plan do celu kartezjańskiego (PoseStamped) dla końcówki.
        Zwraca GetMotionPlan.Response.
        """
        # if self._client is None:
        #     raise RuntimeError("Service client not ready. Call wait_for_service() first.")

        req = GetMotionPlan.Request()
        mpr = req.motion_plan_request

        # mpr.workspace_parameters.header.frame_id = 'world'
        # mpr.workspace_parameters.min_corner = cVector3(-2, -2, 0)
        # mpr.workspace_parameters.max_corner = cVector3(2, 2, 2.5)

        mpr.group_name = group
        mpr.num_planning_attempts = int(num_attempts)
        mpr.allowed_planning_time = float(allowed_time)
        mpr.max_velocity_scaling_factor = float(vel_scale)
        mpr.max_acceleration_scaling_factor = float(acc_scale)

        # mpr.start_state.is_diff = True
        mpr.start_state = PlannerClient._make_start_state(start_state)
        mpr.start_state.attached_collision_objects = attached_collision_objects

        if pipeline_id:
            mpr.pipeline_id = pipeline_id
        if planner_id:
            mpr.planner_id = planner_id

        goal = self._make_pose_goal_constraints(
            target=target, ee_link=ee_link, pos_tol_x=pos_tol_x,
             pos_tol_y=pos_tol_y, pos_tol_z=pos_tol_z, ori_tol=ori_tol
        )
        mpr.goal_constraints = [goal]

        future = self._mv_clients['plan_kinematic_path'].call_async(req)
        rclpy.spin_until_future_complete(self, future)
        resp = future.result()
        if resp is None:
            raise RuntimeError(f"Service call failed: {future.exception()}")

        assert isinstance(resp, GetMotionPlan_Response)
        return resp.motion_plan_response

    def compute_cartesian_path(
        self,
        attached_collision_objects,
        # *,
        group: str,
        ee_link: str,
        start_state: Dict[str, float],
        waypoints: List[Pose],
        frame_id: str,
        max_step: float = 0.01,
        jump_threshold: float = 0.0,
        avoid_collisions: bool = True,
    ):
        """
        Prostoliniowy Cartesian path po waypointach (GetCartesianPath).
        Zwraca GetCartesianPath.Response (zwykle zawiera m.in. fraction + solution traj).
        """
        # if self._cp_client is None:
        #     raise RuntimeError("Service client not ready. Call wait_for_service() first.")

        req = GetCartesianPath.Request()
        req.header.frame_id = frame_id
        req.group_name = group
        req.link_name = ee_link
        req.waypoints = waypoints
        req.start_state = PlannerClient._make_start_state(start_state)
        req.start_state.attached_collision_objects = attached_collision_objects
        req.max_step = float(max_step)
        req.jump_threshold = float(jump_threshold)
        req.avoid_collisions = bool(avoid_collisions)

        # start_state: najlepiej bieżący, więc is_diff=True i resztę zostaw.
        req.start_state.is_diff = True

        fut = self._mv_clients['compute_cartesian_path'].call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        if fut.result() is None:
            raise RuntimeError(f"GetCartesianPath call failed: {fut.exception()}")
        return fut.result()

    def add_obstacle(self):
        scene = PlanningScene()
        scene.is_diff = True
        #scene.robot_state.is_diff = True

        obj = CollisionObject()
        obj.header = cHeader(self.get_clock().now().to_msg(), 'world')
        obj.pose = kdl_frame_to_pose(KdlFrame(KdlVector(0.75, 0.5, 0.4)))
        obj.id = 'table'

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.5, 1.4, 0.8]

        obj.primitives.append(box) # type: ignore
        obj.primitive_poses.append(kdl_frame_to_pose(KdlFrame())) # type: ignore
        scene.world.collision_objects.append(obj) # type: ignore

        req = ApplyPlanningScene.Request()
        req.scene = scene

        fut = self._mv_clients['apply_planning_scene'].call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        if fut.result() is None:
            raise RuntimeError(f"GetCartesianPath call failed: {fut.exception()}")
        return fut.result()

    def apply_planning_scene(self, planning_scene: PlanningScene) -> bool:
        req = ApplyPlanningScene.Request()
        req.scene = planning_scene

        fut = self._mv_clients['apply_planning_scene'].call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        resp = fut.result()
        if resp is None:
            raise RuntimeError(f"GetCartesianPath call failed: {fut.exception()}")
        assert isinstance(resp, ApplyPlanningScene_Response)
        return resp.success


def deg2rad(x):
    result = {}
    for name, pos in x.items():
        result[name] = math.radians(pos)
    return result

def duration_to_float(d):
    return d.sec + d.nanosec/1000000000.0

#
# Constructors for some ROS msgs
#
def cHeader(stamp=None, frame_id='world'):
    result = Header()
    if not stamp is None:
        result.stamp = stamp
    result.frame_id = frame_id
    return result

def cPoseStamped(header, pose):
    assert isinstance(header, Header)
    assert isinstance(pose, Pose)
    result = PoseStamped()
    result.header = header
    result.pose = pose
    return result

def cVector3(x, y, z):
    result = Vector3()
    result.x = x
    result.y = y
    result.z = z
    return result


def kdl_frame_to_pose(f: KdlFrame) -> Pose:
    pose = Pose()

    # translation
    pose.position.x = f.p.x()
    pose.position.y = f.p.y()
    pose.position.z = f.p.z()

    # rotation -> quaternion
    qx, qy, qz, qw = f.M.GetQuaternion()
    pose.orientation.x = qx
    pose.orientation.y = qy
    pose.orientation.z = qz
    pose.orientation.w = qw

    return pose

def pose_to_kdl_frame(pose: Pose) -> KdlFrame:
    M = KdlRotation.Quaternion(pose.orientation.x, pose.orientation.y,
                                        pose.orientation.z, pose.orientation.w)
    return KdlFrame(M,
        KdlVector(
        x=pose.position.x,
        y=pose.position.y,
        z=pose.position.z)
        )


def get_trajectory_first_point(traj) -> dict[str, float]:
    assert isinstance(traj, JointTrajectory)
    q_result = {}
    for joint_name, pos in zip(traj.joint_names, traj.points[0].positions): # type: ignore
        q_result[joint_name] = pos
    return q_result


def get_trajectory_last_point(traj) -> dict[str, float]:
    assert isinstance(traj, JointTrajectory)
    q_result = {}
    for joint_name, pos in zip(traj.joint_names, traj.points[-1].positions): # type: ignore
        q_result[joint_name] = pos
    return q_result


def overwrite_configuration(q_1, q_2):
    q_result = copy.copy(q_1)
    for joint_name, pos in q_2.items():
        q_result[joint_name] = pos
    return q_result

