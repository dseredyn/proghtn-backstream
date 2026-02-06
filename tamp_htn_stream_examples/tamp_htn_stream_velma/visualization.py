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


from __future__ import annotations

from typing import Any
import copy

from moveit_msgs.msg import RobotState, DisplayTrajectory, RobotTrajectory
from visualization_msgs.msg import Marker

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from tamp_htn_stream.core import TypedValue
from .moveit_interface import get_trajectory_first_point, get_trajectory_last_point, getAllJointNames

from .conversions import jsonToJsStr, armJointNameList,\
    cMarkerSphere, cMarkerCylinder, cMarkerFrame, modelToRvizMarkers, rosMsgToDict,\
    primitiveShapeToRvizMarker, cMarkerArrow, joinJsMap

from .data_types import KdlFrame, KdlVector, PoseWithFreeDOF, GraspTraj,\
    interpolateToJointTrajectory, add_duration_msgs, PlanningSceneCreator, PrimitiveShape,\
    ConfA, ConfT, ConfG, ConfH, Placement, Volume

from .plugin import getNodeVisualizationTemplateHtml, getPlanVisualizationTemplateHtml,\
    getStatesInterface

from .const import get_constants

from .velma_kinematics import KinematicsSolverVelma


def generate_node_visualization_html(state: dict[str, Any], task_network: dict[str, Any],
                                     generator_instance, verbosity_level) -> str:
    assert verbosity_level in ['low', 'medium', 'high']
    html = getNodeVisualizationTemplateHtml()

    replacements = generate_state_visualization(state, verbosity_level) +\
                    generate_generator_visualization(generator_instance, verbosity_level) +\
                    generate_task_network_visualization(task_network, verbosity_level)

    for str1, str2 in replacements:
        html = html.replace(str1, str2)
    
    return html

def hex_to_rgb01(hex_color: str) -> list[float]:
    """
    Convert HTML hex color '#aabbcc' (or 'aabbcc') to (r,g,b) in range [0, 1].
    """
    s = hex_color.strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) != 6:
        raise ValueError("Expected 6 hex digits in format '#aabbcc'.")

    r = int(s[0:2], 16) / 255.0
    g = int(s[2:4], 16) / 255.0
    b = int(s[4:6], 16) / 255.0
    return [r, g, b]

obj_colors_hex = {
    # 'table_1': 'ff049b',  # magenta
    # 'table_2': 'd149ff', # purple
    # 'table_3': '6e49ff',  # violet
    # 'cabinet_2': 'ff9d5c',  # orange
    # 'cabinet_1': 'ff5c5c',  # pink
    'table_1': 'c09090',  # gray
    'table_2': '90c090', # gray
    'table_3': '9090c0',  # gray
    'cabinet_2': 'c0c090',  # gray
    'cabinet_1': '90c0c0',  # gray
    'jar_1': '00be9e',  # dark cyan
    'bowl_1': '0083bf',  # blue
    'box_1': 'c5ff25',   # light green
    'box_2': '51ff12',   # lime
    'box_3': '45ff92',  # light green/blue
}
# 'ffdc5c',  # darker yellow/orange
# 'b47500',   # yellow
# '00be9e',  # dark cyan
# '0083bf',  # blue
obj_color_map = {obj_id: hex_to_rgb01(color) for obj_id, color in obj_colors_hex.items()}


def generate_state_visualization(state: dict[str, Any], verbosity_level: str) -> list[tuple[str, str]]:
    assert verbosity_level in ['low', 'medium', 'high']
    joint_names = []
    joint_positions = []
    all_markers = []

    if verbosity_level in ['medium', 'high']:
        # Full state of the robot
        sti = getStatesInterface()
        assert not sti is None
        joint_names, joint_positions = sti.getJointNamesAndPositions(state)

        # Visualize object with known exact pose
        obj_poses = sti.getObjectsAndExactPoses(state) +\
                                                sti.getGraspedObjectsAndExactPoses(state)

        env_markers = []
        marker_id = 0
        # collision_objects = []
        for obj_id_str, obj_pose in obj_poses:
            if obj_id_str in obj_color_map:
                obj_color = obj_color_map[obj_id_str]
            else:
                obj_color = [0.0, 1.0, 0.0]
            model = sti.getModelByObjectId(state, obj_id_str)
            obj_markers, marker_id = modelToRvizMarkers('env', marker_id, model, obj_pose, obj_color)
            env_markers = env_markers + obj_markers

        # Show poses of end-effectors
        # T_W_El = sti.fkArmTorso('left', joint_names, joint_positions)
        # assert not T_W_El is None
        # T_W_Er = sti.fkArmTorso('right', joint_names, joint_positions)
        # assert not T_W_Er is None
        # ee_markers = cMarkerFrame(f'ee_left', 0, T_W_El, 0.1) +\
        #             cMarkerFrame(f'ee_right', 0, T_W_Er, 0.1)

        # all_markers = ee_markers + env_markers
        all_markers = env_markers

    return [
        ('"<<<<joint_names>>>>"', jsonToJsStr(joint_names)),
        ('"<<<<joint_positions>>>>"', jsonToJsStr(joint_positions)),
        ('"<<<<markers_state>>>>"', jsonToJsStr(rosMsgToDict(all_markers))) #markers))
    ]

def generate_generator_visualization(generator_instance, verbosity_level: str) -> list[tuple[str, str]]:
    assert verbosity_level in ['low', 'medium', 'high']
    if not generator_instance is None and hasattr(generator_instance, 'can_visualize') and\
                generator_instance.can_visualize():
        # TODO: use verbosity_level
        generator_data, markers = generator_instance.get_visualization()
        return [
            ('<generator_data>', generator_data),
            ('"<<<<markers_generator>>>>"', jsonToJsStr(rosMsgToDict(markers)))
        ]
    # else:        
    return [
        ('<generator_data/>', ''),
        ('"<<<<markers_generator>>>>"', '[]')
    ]    

def generate_task_network_visualization(task_network: dict[str, Any], verbosity_level: str) -> list[tuple[str, str]]:
    assert verbosity_level in ['low', 'medium', 'high']
    html_task_network_parameters = ''
    button_id_list = []
    markers_list_list = []
    output_types_list = []
    param_vis_idx = 0
    for param_name, param_data in task_network['ParameterGroundings'].items():
        param_type = param_data.getType()
        param_value = param_data.getValue()
        visual = None
        if param_type == 'PoseWithFreeDOF' and verbosity_level in ['medium', 'high']:
            sub_markers = []
            pose = param_value
            assert isinstance(pose, PoseWithFreeDOF)
            T_W_AX, axis = pose.getAxis('vert_rot', {})
            ns = f'TaskNetwork.{param_name}'
            color = [0.0, 0.0, 1.0]
            sub_markers.append( cMarkerSphere(ns, 0, T_W_AX,
                0.025, color) )
            sub_markers.append( cMarkerArrow(ns, 1, T_W_AX.p, T_W_AX*(axis*0.1), 0.01, color) )

            visual = ('MarkerArray', sub_markers)
        elif param_type == 'Placement' and verbosity_level in ['medium', 'high']:
            sub_markers = []
            pose = param_value
            assert isinstance(pose, Placement)
            T_W_AX, axis = pose.T_W_F.getAxis('vert_rot', {})
            ns = f'TaskNetwork.{param_name}'
            color = [0.0, 0.0, 1.0]
            sub_markers.append( cMarkerSphere(ns, 0, T_W_AX,
                0.025, color) )
            sub_markers.append( cMarkerArrow(ns, 1, T_W_AX.p, T_W_AX*(axis*0.1), 0.01, color) )

            visual = ('MarkerArray', sub_markers)
        elif param_type == 'Pose' and verbosity_level in ['medium', 'high']:
            T = KdlFrame.fromDict(param_value)
            sub_markers = cMarkerFrame(f'TaskNetwork.{param_name}', 0, T, 0.1)
            visual = ('MarkerArray', sub_markers)

        elif param_type == 'ConfA' and verbosity_level in ['medium', 'high']:
            assert isinstance(param_value, ConfA)
            robot_state = RobotState()
            robot_state.is_diff = True
            robot_state.joint_state.name = armJointNameList(param_value.side)
            robot_state.joint_state.position = param_value.q

            visual = ('DisplayRobotState', robot_state)

        elif param_type == 'ConfT' and verbosity_level in ['medium', 'high']:
            robot_state = RobotState()
            robot_state.is_diff = True
            robot_state.joint_state.name = ['torso_0_joint']
            robot_state.joint_state.position = param_value.q

            visual = ('DisplayRobotState', robot_state)

        elif param_type == 'JointTraj' and verbosity_level in ['high']:
            traj = param_value
            if len(traj.points) > 0:
                robot_state = RobotState()
                robot_state.is_diff = True
                robot_state.joint_state.name = traj.joint_names
                js_map = get_trajectory_first_point(traj)
                robot_state.joint_state.position = [js_map[name] for name in robot_state.joint_state.name]

                vis_traj = DisplayTrajectory()
                vis_traj.trajectory_start = robot_state
                rob_traj = RobotTrajectory()
                rob_traj.joint_trajectory = traj
                vis_traj.trajectory = [rob_traj]

                visual = ('DisplayTrajectory', vis_traj)

        elif param_type == 'GraspTraj' and verbosity_level in ['high']:
            grasp_traj = param_value
            assert isinstance(grasp_traj, GraspTraj)
            jnt_traj_list = []
            for jnt_traj in grasp_traj.getArmTrajectories():
                if len(jnt_traj.points) > 0:
                    rob_traj = RobotTrajectory()
                    rob_traj.joint_trajectory = jnt_traj
                    jnt_traj_list.append( rob_traj )
                
            # for mov in grasp_traj['GraspTraj']:
            #     if mov['MovementType'] == 'JointTraj':
            #         traj = mov['JointTraj']
            #         if len(traj.points) > 0:
            #             rob_traj = RobotTrajectory()
            #             rob_traj.joint_trajectory = traj
            #             jnt_traj_list.append( rob_traj )

            if len(jnt_traj_list) > 0:
                robot_state = RobotState()
                robot_state.is_diff = True
                robot_state.joint_state.name = jnt_traj_list[0].joint_trajectory.joint_names
                js_map = get_trajectory_first_point(jnt_traj_list[0].joint_trajectory)
                robot_state.joint_state.position = [js_map[name] for name in robot_state.joint_state.name]

                vis_traj = DisplayTrajectory()
                vis_traj.trajectory_start = robot_state
                vis_traj.trajectory = jnt_traj_list

                visual = ('DisplayTrajectory', vis_traj)

        elif param_type == 'Volume' and verbosity_level in ['medium', 'high']:
            vol = param_value
            assert isinstance(vol, Volume)
            sub_markers = []
            for idx, shape in enumerate(vol.col_shapes):
                sub_markers.append( primitiveShapeToRvizMarker(f'volume_{param_name}', idx, shape, [1.0, 0.0, 0.0, 0.5]) )
            visual = ('MarkerArray', sub_markers)

        elif param_type == 'Approaches' and verbosity_level in ['high']:
            approaches = param_value
            pt = approaches.pt
            sub_markers = []
            for idx, (ap_score, ap_dir) in enumerate(approaches.ap):
                width = 0.05*ap_score
                sub_markers.append( cMarkerArrow('approaches', idx, pt-ap_dir*0.5, pt-ap_dir*0.1, width, [0.0, 0.0, 1.0]) )
            visual = ('MarkerArray', sub_markers)

        if not visual is None:
            button_id = f'btnVisualize{param_vis_idx}'
            html_task_network_parameters +=\
                f'<button id="{button_id}">{param_type}: {param_name}</button><br>'
            button_id_list.append( button_id )
            output_types_list.append( visual[0] )
            markers_list_list.append( visual[1] )
            param_vis_idx = param_vis_idx + 1

    return [
        ('<task_network_parameters/>', html_task_network_parameters),
        ('"<button_id_list>"', jsonToJsStr( button_id_list) ),
        ('"<markers_list_list>"', jsonToJsStr( rosMsgToDict(markers_list_list)) ),
        ('"<output_types_list>"', jsonToJsStr( output_types_list) ),
    ]


def format_as_text(param_data: TypedValue):
    if not isinstance(param_data, TypedValue):
        print(param_data)
        raise Exception()
    v_type = param_data.getType()
    v = param_data.getValue()
    if v_type == 'JointTraj':
        return f'Points: {len(v.points)}'
    elif v_type == 'GraspTraj':
        #print(v)
        return f'GraspTraj'
    elif v_type == 'Pose':
        return str(KdlFrame.fromDict(v))
    elif v_type in ['ConfA', 'ConfT', 'ConfG', 'ConfH']:
        # assert isinstance(v, ConfA)
        return str(v)
    elif v_type == 'Volume':
        return f'Volume: {len(v.col_shapes)}'

    elif v_type == 'Approaches':
        return f'Approaches: {len(v.ap)}'

    # Fallback
    return str(v)


def fillMissingJoints(traj: JointTrajectory, joint_names: list[str],
                      js_map: dict[str, float]) -> JointTrajectory:
    joint_name_to_idx = {joint_name: idx for idx, joint_name in enumerate(traj.joint_names)}

    result = JointTrajectory()
    result.joint_names = joint_names
    for point in traj.points:
        result_point = JointTrajectoryPoint()
        result_point.time_from_start = point.time_from_start
        for joint_name in joint_names:
            if joint_name in joint_name_to_idx:
                # Get current point in the input trajectory
                original_idx = joint_name_to_idx[joint_name]
                result_point.positions.append( point.positions[original_idx] )
            else:
                # Get constant position from the js_map
                result_point.positions.append( js_map[joint_name] )
        result.points.append( result_point ) # type: ignore
    return result

def concatenateTrajectories(traj1: JointTrajectory, traj2: JointTrajectory) -> JointTrajectory:
    assert len(traj1.joint_names) == len(traj2.joint_names)

    result = JointTrajectory()
    result.joint_names = traj1.joint_names

    for point in traj1.points:
        result.points.append( point ) # type: ignore

    if len(traj1.points) > 0:
        traj2_base_time = traj1.points[-1].time_from_start # type: ignore
    else:
        traj2_base_time = Duration()
    for point in traj2.points:
        added_point = copy.copy(point)
        added_point.time_from_start = add_duration_msgs(point.time_from_start, traj2_base_time)
        result.points.append( added_point ) # type: ignore
    return result


# def generate_plan_visualization(plan: list):
#     print('generate_plan_visualization()')

#     # TODO: get initial state
#     # TODO: publish environment
#     # TODO: attach / detach objects
#     # TODO: show change of environment state

#     # Build a very long joint trajectory for all joints
#     all_joint_names = getAllJointNames()
#     # all_joint_names = [
#     #         'torso_0_joint',
#     #         'head_pan_joint',
#     #         'head_tilt_joint',
#     #         'left_arm_0_joint',
#     #         'left_arm_1_joint',
#     #         'left_arm_2_joint',
#     #         'left_arm_3_joint',
#     #         'left_arm_4_joint',
#     #         'left_arm_5_joint',
#     #         'left_arm_6_joint',
#     # ]
#     # Initialize the current configuration to 0.0
#     # TODO: get it from the initial state
#     current_js_map = {joint_name: 0.0 for joint_name in all_joint_names}

#     full_traj = JointTrajectory()
#     full_traj.joint_names = all_joint_names

#     for task, param_value_list in plan:
#         param_types = [data['ValueType'] for data in param_value_list]
#         print(f'  ({task['Class']} {task['Args']}), types: {param_types}')

#         if task['Class'] == 'pMvBody':
#             traj = param_value_list[0].getValue()
#             assert isinstance(traj, JointTrajectory)
#             traj = fillMissingJoints(traj, all_joint_names, current_js_map)
#             full_traj = concatenateTrajectories(full_traj, traj)
#             current_js_map = get_trajectory_last_point(full_traj)

#         elif task['Class'] == 'pGrasp' or task['Class'] == 'pUngrasp':
#             grasp_traj = param_value_list[3].getValue()
#             assert isinstance(grasp_traj, GraspTraj)

#             # Before any movement, ensure fingers are properly placed (pregrasp / grasp)
#             qg_js = grasp_traj.getInitialFingersConf()
#             current_qg_js = {name: current_js_map[name] for name in qg_js}
#             traj = interpolateToJointTrajectory(current_qg_js, qg_js)
#             traj = fillMissingJoints(traj, all_joint_names, current_js_map)
#             full_traj = concatenateTrajectories(full_traj, traj)
#             current_js_map = get_trajectory_last_point(full_traj)

#             for mov in grasp_traj.getMovements():
#                 traj = fillMissingJoints(mov.traj, all_joint_names, current_js_map)
#                 full_traj = concatenateTrajectories(full_traj, traj)
#                 current_js_map = get_trajectory_last_point(full_traj)

#         elif task['Class'] == 'pMvHead':
#             qh1, qh2 = param_value_list
#             qh1_map = confToJsMap([qh1.getValue()])
#             qh2_map = confToJsMap([qh2.getValue()])
#             traj = interpolateToJointTrajectory(qh1_map, qh2_map)
#             traj = fillMissingJoints(traj, all_joint_names, current_js_map)
#             full_traj = concatenateTrajectories(full_traj, traj)
#             current_js_map = get_trajectory_last_point(full_traj)

#         elif task['Class'] == 'pCloseGr':
#             side = param_value_list[0].getValue()
#             qg = get_constants()['CLOSED_HAND'].getValue()
#             qg['Side'] = side
#             qg_js = confToJsMap( [qg] )
#             current_qg_js = {name: current_js_map[name] for name in qg_js}
#             traj = interpolateToJointTrajectory(current_qg_js, qg_js)
#             traj = fillMissingJoints(traj, all_joint_names, current_js_map)
#             full_traj = concatenateTrajectories(full_traj, traj)
#             current_js_map = get_trajectory_last_point(full_traj)

#             # for mov in grasp_traj['GraspTraj']:
#             #     if mov['MovementType'] == 'JointTraj':
#             #         traj = mov['JointTraj']
#             #         assert isinstance(traj, JointTrajectory)
#             #         traj = fillMissingJoints(traj, all_joint_names, current_js_map)
#             #         full_traj = concatenateTrajectories(full_traj, traj)
#             #     elif mov['MovementType'] == 'MoveFingers':

#             #         mov['q_start']
#             #         mov['q_end']
#             #         pass
#             #         # TODO



#             # elif param_type == 'JointTraj' and verbosity_level in ['high']:
#             # traj = param_value
#             # if len(traj.points) > 0:
#             #     robot_state = RobotState()
#             #     robot_state.is_diff = True
#             #     robot_state.joint_state.name = traj.joint_names
#             #     js_map = get_trajectory_first_point(traj)
#             #     robot_state.joint_state.position = [js_map[name] for name in robot_state.joint_state.name]

#             #     vis_traj = DisplayTrajectory()
#             #     vis_traj.trajectory_start = robot_state
#             #     rob_traj = RobotTrajectory()
#             #     rob_traj.joint_trajectory = traj
#             #     vis_traj.trajectory = [rob_traj]

#             #     visual = ('DisplayTrajectory', vis_traj)

#     robot_state = RobotState()
#     robot_state.is_diff = True
#     robot_state.joint_state.name = all_joint_names
#     robot_state.joint_state.position = [current_js_map[name] for name in robot_state.joint_state.name]
#     disp_traj = DisplayTrajectory()
#     disp_traj.trajectory_start = robot_state
#     rob_traj = RobotTrajectory()
#     rob_traj.joint_trajectory = full_traj
#     disp_traj.trajectory = [rob_traj]
#     print(f'  Final trajectory length: {len(full_traj.points)}')
#     html = getPlanVisualizationTemplateHtml()
#     html = html.replace('"<<<<marker_array>>>>"', '[]')
#     html = html.replace('"<<<<display_trajectory>>>>"', jsonToJsStr( rosMsgToDict(disp_traj)))
#     return html

def getActionTrajectory(task_class: str, param_value_list: list,
                        all_joint_names: list[str], current_js_map: dict[str, float]
                        ) -> tuple[JointTrajectory, dict[str, float]]:
    if task_class == 'pMvBody':
        traj = param_value_list[0].getValue()
        assert isinstance(traj, JointTrajectory)
        current_traj = fillMissingJoints(traj, all_joint_names, current_js_map)
        current_js_map = get_trajectory_last_point(current_traj)
        return current_traj, current_js_map

    elif task_class == 'pGrasp' or task_class == 'pUngrasp':
        grasp_traj = param_value_list[3].getValue()
        assert isinstance(grasp_traj, GraspTraj)

        # Before any movement, ensure fingers are properly placed (pregrasp / grasp)
        qg_js = grasp_traj.getInitialFingersConf()
        current_qg_js = {name: current_js_map[name] for name in qg_js}

        current_traj = interpolateToJointTrajectory(current_qg_js, qg_js)
        current_traj = fillMissingJoints(current_traj, all_joint_names, current_js_map)
        current_js_map = get_trajectory_last_point(current_traj)

        for mov in grasp_traj.getMovements():
            mov_traj = fillMissingJoints(mov.traj, all_joint_names, current_js_map)
            current_traj = concatenateTrajectories(current_traj, mov_traj)
            current_js_map = get_trajectory_last_point(current_traj)

        return current_traj, current_js_map

    elif task_class == 'pMvHead':
        qh1, qh2 = param_value_list
        qh1_map = qh1.getValue().toJsMap()
        qh2_map = qh2.getValue().toJsMap()

        current_traj = interpolateToJointTrajectory(qh1_map, qh2_map)
        current_traj = fillMissingJoints(current_traj, all_joint_names, current_js_map)
        current_js_map = get_trajectory_last_point(current_traj)

        return current_traj, current_js_map

    elif task_class == 'pCloseGr':
        side = param_value_list[0].getValue()
        qg = get_constants()['CLOSED_HAND'].getValue()
        qg = ConfG(side, qg.q)
        qg_js = qg.toJsMap()
        current_qg_js = {name: current_js_map[name] for name in qg_js}

        current_traj = interpolateToJointTrajectory(current_qg_js, qg_js)
        current_traj = fillMissingJoints(current_traj, all_joint_names, current_js_map)
        current_js_map = get_trajectory_last_point(current_traj)

        return current_traj, current_js_map

    raise Exception()

def getStepSetObjPose(env_col:dict[str, list[PrimitiveShape]], obj: str,
                      T_W_O: KdlFrame, obj_base_m_id: int):
    param_list = []
    m_id = obj_base_m_id
    col = env_col[obj]
    if obj in obj_color_map:
        obj_color = obj_color_map[obj]
    else:
        obj_color = [0.0, 1.0, 0.0]
    for shape in col:
        shape.setObjectPose(T_W_O, obj)
        T_W_S = shape.getShapePose(obj)
        px = T_W_S.p.x()
        py = T_W_S.p.y()
        pz = T_W_S.p.z()
        qx, qy, qz, qw = T_W_S.M.GetQuaternion()
        cx, cy, cz = obj_color
        ros_tp = None
        if shape.tp == 'box':
            ros_tp = Marker.CUBE
            sx = shape.size[0]
            sy = shape.size[1]
            sz = shape.size[2]
        elif shape.tp == 'cylinder':
            ros_tp = Marker.CYLINDER
            sx = shape.size[0]*2
            sy = shape.size[0]*2
            sz = shape.size[1]
        else:
            raise Exception(f'Cant visualize shape of type "{shape.tp}"')
        param_list.append( [m_id, ros_tp, px, py, pz, qx, qy, qz, qw, sx, sy, sz, cx, cy, cz] )
        m_id = m_id + 1
    return ['MarkerArray', 0, param_list]

def getSteps(traj: JointTrajectory, grasped_map: dict[str, tuple[str, str]],
             env_col:dict[str, list[PrimitiveShape]], obj_base_m_id_map: dict[str, int],
             init_state, velma_solv: KinematicsSolverVelma):

    sti = getStatesInterface()
    steps: list[list] = []
    for pt in traj.points:
        current_js_map = {name: pos for name, pos in zip(traj.joint_names, pt.positions)}
        for obj_id, (side, grasp_id) in grasped_map.items():
            obj_model = sti.getModelByObjectId(init_state, obj_id)
            grasp_def = sti.getGraspDef(init_state, obj_id, grasp_id)
            T_G_O = sti.getGraspPose(grasp_def)
            T_E_G = sti.getT_E_G(side)
            T_W_E = velma_solv.getArmFkJsMap(side, current_js_map)
            T_W_O = T_W_E * T_E_G * T_G_O
            steps.append( getStepSetObjPose(env_col, obj_id, T_W_O, obj_base_m_id_map[obj_id]) )
        steps.append( ['DisplayRobotState', 100, list(pt.positions)] )

    return steps

def generate_plan_visualization(init_state, plan: list):
    print('generate_plan_visualization()')

    velma_solv = KinematicsSolverVelma()
    sti = getStatesInterface()
    env_col = sti.getEnvironmentCollisionShapes(init_state, [], 0)

    grasped_objs = sti.getGraspedObjects(init_state)
    grasped_map: dict[str, tuple[str, str]] = {obj: (sd, gr) for obj, sd, gr in grasped_objs}

    steps = []

    param_list = []
    obj_base_m_id_map = {}
    m_id = 0
    for obj_id, col in env_col.items():
        if obj_id in obj_color_map:
            obj_color = obj_color_map[obj_id]
        else:
            obj_color = [0.0, 1.0, 0.0]

        if not obj_id in obj_base_m_id_map:
            obj_base_m_id_map[obj_id] = m_id
        for shape in col:
            T_W_O = shape.getShapePose('world')
            px = T_W_O.p.x()
            py = T_W_O.p.y()
            pz = T_W_O.p.z()
            qx, qy, qz, qw = T_W_O.M.GetQuaternion()
            cx, cy, cz = obj_color
            ros_tp = None
            if shape.tp == 'box':
                ros_tp = Marker.CUBE
                sx = shape.size[0]
                sy = shape.size[1]
                sz = shape.size[2]
            elif shape.tp == 'cylinder':
                ros_tp = Marker.CYLINDER
                sx = shape.size[0]*2
                sy = shape.size[0]*2
                sz = shape.size[1]
            else:
                raise Exception(f'Cant visualize shape of type "{shape.tp}"')
            param_list.append( [m_id, ros_tp, px, py, pz, qx, qy, qz, qw, sx, sy, sz, cx, cy, cz] )
            m_id = m_id + 1
    step = ['MarkerArray', 0, param_list]
    steps.append(step)


    # TODO: get initial state
    # TODO: publish environment
    # TODO: attach / detach objects
    # TODO: show change of environment state

    # Build a very long joint trajectory for all joints
    all_joint_names = getAllJointNames()

    # Initialize the current configuration to 0.0
    # TODO: get it from the initial state
    current_js_map = sti.getFullConfJs(init_state)
    for joint_name in all_joint_names:
        if not joint_name in current_js_map:
            current_js_map[joint_name] = 0.0

    full_traj = JointTrajectory()
    full_traj.joint_names = all_joint_names

    for task, param_value_list in plan:
        param_types = [data.getType() for data in param_value_list]
        # print(f'  ({task['Class']} {task['Args']}), types: {param_types}')

        current_traj = None
        if task['Class'] in ['pMvBody', 'pMvHead', 'pCloseGr']:
            current_traj, current_js_map = getActionTrajectory(task['Class'], param_value_list,
                                                                all_joint_names, current_js_map)
            # Add trajectory to steps
            steps += getSteps(current_traj, grasped_map, env_col, obj_base_m_id_map,
                              init_state, velma_solv)

        elif task['Class'] == 'pGrasp':
            current_traj, current_js_map = getActionTrajectory(task['Class'], param_value_list,
                                                                all_joint_names, current_js_map)

            obj = param_value_list[0].getValue()
            sd = param_value_list[1].getValue()
            gr = param_value_list[2].getValue()

            # First, add trajectory to steps, then add grasp event to steps
            steps += getSteps(current_traj, grasped_map, env_col, obj_base_m_id_map,
                              init_state, velma_solv)
            assert not obj in grasped_map
            grasped_map[obj] = (sd, gr)
            #steps += [getGraspStep()]
            
            
        elif task['Class'] == 'pUngrasp':
            current_traj, current_js_map = getActionTrajectory(task['Class'], param_value_list,
                                                                all_joint_names, current_js_map)

            obj = param_value_list[0].getValue()
            sd = param_value_list[1].getValue()
            gr = param_value_list[2].getValue()
            p_obj = param_value_list[10].getValue()
            # First, add ungrasp event to steps, then add trajectory to steps
            T_W_O = KdlFrame.fromDict(p_obj)
            #steps += [getStepSetObjPose(env_col, obj, T_W_O, obj_base_m_id_map[obj])]
            assert obj in grasped_map
            del grasped_map[obj]
            steps += getSteps(current_traj, grasped_map, env_col, obj_base_m_id_map,
                              init_state, velma_solv)

        if not current_traj is None:
            full_traj = concatenateTrajectories(full_traj, current_traj)

    html = getPlanVisualizationTemplateHtml()
    html = html.replace('"<<<<steps>>>>"', jsonToJsStr(steps, indent=None))
    html = html.replace('"<<<<joint_names>>>>"', jsonToJsStr( all_joint_names, indent=None))
    return html
    
