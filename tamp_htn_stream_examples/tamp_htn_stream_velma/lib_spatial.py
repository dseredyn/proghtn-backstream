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

from typing import Any, Dict
from tamp_htn_stream.core import getPredicateParameterValues, State
import math
import numpy as np
from trajectory_msgs.msg import JointTrajectory

from moveit_msgs.msg import PlanningScene

from tamp_htn_stream.core import TypedValue

from .moveit_interface import PlannerClient, pose_to_kdl_frame, cHeader, cPoseStamped,\
    kdl_frame_to_pose

from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from .conversions import is_value_equal, joinJsMap

from .velma_kinematics import KinematicsSolverBarrettHand, KinematicsSolverVelma

from .data_types import KdlVector, KdlRotation, KdlFrame, PrimitiveShape, ConfA, ConfT,\
    ConfG, ConfH, Volume, FlatBase, Hole

# TODO: make code more efficient by using data cache

class StatesInterface:
    def __init__(self, models_map: Dict[str, Any], grasps_map: Dict[str, Any],
                                                                    moveit_client: PlannerClient):
        self._models_map = models_map
        self._grasps_map = grasps_map
        self._moveit_client = moveit_client
        self._solv_bh = KinematicsSolverBarrettHand()
        self._velma_solv = KinematicsSolverVelma()

    def hasReservedSpace(self, state: State) -> bool:
        return len(self.getReservedSpace(state)) > 0

    def getT_E_G(self, side_str: str) -> KdlFrame:
        poses = {'right':
                KdlFrame( KdlRotation.RPY(0, math.pi/2, 0), KdlVector(0.235, 0, -0.078) ),
            'left':
                KdlFrame( KdlRotation.RPY(0, -math.pi/2, 0), KdlVector(-0.235, 0, -0.078) )}
        return poses[side_str]

    def getConfA(self, state: State, side: str) -> ConfA:
        assert side in ('left', 'right')
        param_value_list = [TypedValue('Side', side), None]
        value_list = getPredicateParameterValues(state, is_value_equal, 'AtConfA', param_value_list)
        val = value_list[0][0].getValue()
        if not isinstance(val, ConfA):
            print(val)
            raise Exception()
        # print(value_list[0][0])
        # result = ConfA(side, q)
        return val
        # q_dict = value_list[0][0]['Value']['ArmConfig']
        # return [float(q_dict[x]) for x in ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6']]
    
    def getConfT(self, state: State) -> ConfT:
        param_value_list = [None]
        value_list = getPredicateParameterValues(state, is_value_equal, 'AtConfT', param_value_list)
        val = value_list[0][0].getValue()
        if not isinstance(val, ConfT):
            print(val)
            raise Exception()
        return val

    def getConfH(self, state: State) -> ConfH:
        pred_name = 'AtConfH'
        param_value_list = [None]
        value_list = getPredicateParameterValues(state, is_value_equal, pred_name, param_value_list)
        val = value_list[0][0].getValue()
        assert isinstance(val, ConfH)
        return val
        # return [float(value_list[0][0]['Value']['hp']), float(value_list[0][0]['Value']['ht'])]

    def getConfG(self, state: State, side: str) -> ConfG:
        assert side in ('left', 'right')
        param_value_list = [TypedValue('Side', side), None]
        value_list = getPredicateParameterValues(state, is_value_equal, 'AtConfG', param_value_list)
        val = value_list[0][0].getValue()
        assert isinstance(val, ConfG)
        return val
        # q_dict = value_list[0][0]['Value']['HandConfig']
        # return [float(q_dict[x]) for x in ['sp', 'f0a', 'f0b', 'sp', 'f1a', 'f1b', 'f2a', 'f2b']]

    def getFullConfJs(self, state: State):
        conf_all = [self.getConfA(state, 'left').toJsMap(),
                    self.getConfA(state, 'right').toJsMap(),
                    self.getConfG(state, 'left').toJsMap(),
                    self.getConfG(state, 'right').toJsMap(),
                    self.getConfT(state).toJsMap(),
                    self.getConfH(state).toJsMap()]
        return joinJsMap(conf_all)
        
        
    # def getJsMapConfG(self, state: State, side: str) -> dict[str, float]:
    #     assert side in ('left', 'right')
    #     param_value_list = [{"ValueType": "Side","Value": side}, None]
    #     value_list = getPredicateParameterValues(state, is_value_equal, 'AtConfG', param_value_list)
    #     q_dict = value_list[0][0]['Value']['HandConfig']
    #     js_name_map = {
    #     f'{side}_HandFingerOneKnuckleOneJoint': 'sp',
    #     f'{side}_HandFingerOneKnuckleTwoJoint': 'f0a',
    #     f'{side}_HandFingerOneKnuckleThreeJoint': 'f0b',
    #     f'{side}_HandFingerTwoKnuckleOneJoint': 'sp',
    #     f'{side}_HandFingerTwoKnuckleTwoJoint': 'f1a',
    #     f'{side}_HandFingerTwoKnuckleThreeJoint': 'f1b',
    #     f'{side}_HandFingerThreeKnuckleTwoJoint': 'f2a',
    #     f'{side}_HandFingerThreeKnuckleThreeJoint': 'f2b',
    #     }
    #     return {name: float(q_dict[name2]) for name, name2 in js_name_map.items()}

    def getJointNamesAndPositions(self, state: State) -> tuple[list[str], list[float]]:

        conf_list = [self.getConfT(state),
            self.getConfA(state, 'right'),
            self.getConfA(state, 'left'),
            self.getConfH(state),
            self.getConfG(state, 'right'),
            self.getConfG(state, 'left')]
        joint_names = [name for conf in conf_list for name in conf.getJointNames()]
        joint_positions = [q for conf in conf_list for q in conf.q]

        # joint_names = ['torso_0_joint',
        #     'right_arm_0_joint', 'right_arm_1_joint', 'right_arm_2_joint',
        #     'right_arm_3_joint', 'right_arm_4_joint', 'right_arm_5_joint', 'right_arm_6_joint',
        #     'left_arm_0_joint', 'left_arm_1_joint', 'left_arm_2_joint',
        #     'left_arm_3_joint', 'left_arm_4_joint', 'left_arm_5_joint', 'left_arm_6_joint',
        #     'head_pan_joint', 'head_tilt_joint',
        #     'right_HandFingerOneKnuckleOneJoint',
        #     'right_HandFingerOneKnuckleTwoJoint',
        #     'right_HandFingerOneKnuckleThreeJoint',
        #     'right_HandFingerTwoKnuckleOneJoint',
        #     'right_HandFingerTwoKnuckleTwoJoint',
        #     'right_HandFingerTwoKnuckleThreeJoint',
        #     'right_HandFingerThreeKnuckleTwoJoint',
        #     'right_HandFingerThreeKnuckleThreeJoint',
        #     'left_HandFingerOneKnuckleOneJoint',
        #     'left_HandFingerOneKnuckleTwoJoint',
        #     'left_HandFingerOneKnuckleThreeJoint',
        #     'left_HandFingerTwoKnuckleOneJoint',
        #     'left_HandFingerTwoKnuckleTwoJoint',
        #     'left_HandFingerTwoKnuckleThreeJoint',
        #     'left_HandFingerThreeKnuckleTwoJoint',
        #     'left_HandFingerThreeKnuckleThreeJoint',
        # ]
        # joint_positions = [self.getConfT(state).q] +\
        #     self.getConfA(state, 'right').q +\
        #     self.getConfA(state, 'left').q +\
        #     self.getConfH(state).q +\
        #     self.getConfG(state, 'right').q +\
        #     self.getConfG(state, 'left').q
        return joint_names, joint_positions

    # def fkArmTorso(self, side: str, joint_names: list[str]|None = None,
    #                 joint_positions: list[float]|None = None,
    #                 pos_map: dict[str, float]|None = None) -> KdlFrame | None:
    #     if pos_map is None and not joint_names is None and not joint_positions is None:
    #         pos_map = {}
    #         for name, value in zip(joint_names, joint_positions):
    #             pos_map[name] = value
    #     assert not pos_map is None
    #     result = self._moveit_client.calculate_fk('arms_torso', f'{side}_arm_7_link', pos_map)
    #     if result is None or result.error_code.val != 1:
    #         print(f'fk error: {None if result is None else result.error_code.val}')
    #         return None
    #     return pose_to_kdl_frame( result.pose_stamped[0].pose )

    def ikArmTorso(self, side: str, qt: float, p_ee: KdlFrame,
                   start_pose: None | dict[str, float] = None) -> dict[str, float] | None:
        target_pose = cPoseStamped(cHeader(self._moveit_client.get_clock().now().to_msg(), 'world'), kdl_frame_to_pose(p_ee))
        result = None
        if start_pose is None:
            start_poses = [self._moveit_client.getNamedState('arms_torso', pose_name) for pose_name in ['work_01', 'work_02']]
        else:
            start_poses = [start_pose]
        for start_state in start_poses:
            # start_state = self._moveit_client.getNamedState('arms_torso', start_pose)
            start_state['torso_0_joint'] = qt
            # print(start_state)
            result = self._moveit_client.calculate_ik(f'{side}_arm', f'{side}_arm_7_link', start_state, target_pose)
            if not result is None and result.error_code.val == 1:
                break
        if result is None or result.error_code.val != 1:
            print(f'ik error: {None if result is None else result.error_code.val}')
            return None
        q_dest = {}
        for joint_name, pos in zip(result.solution.joint_state.name, result.solution.joint_state.position):
            q_dest[joint_name] = pos
        return q_dest

    def computeCartesianPath(self, side: str, js: dict[str, float],
                             waypoints: list[KdlFrame], attached_collision_objects
                             ) -> tuple[float, JointTrajectory|None]:
        result = self._moveit_client.compute_cartesian_path(attached_collision_objects, f'{side}_arm', f'{side}_arm_7_link',
                start_state=js, waypoints=[kdl_frame_to_pose(p) for p in waypoints], frame_id='world')
        if result is None or result.error_code.val != 1:
            print(f'compute_cartesian_path error: {None if result is None else result.error_code.val}')
            return 0.0, None
        return result.fraction, result.solution.joint_trajectory

    def computeTrajectory(self, side: str, q1: dict[str, float], q2: dict[str, float],
                          use_torso: bool, attached_collision_objects) -> JointTrajectory | None:
        if use_torso:
            group_name = f'{side}_arm_torso'
        else:
            group_name = f'{side}_arm'
        result = self._moveit_client.plan_joint_goal(attached_collision_objects, group_name,
                        start_state=q1, joint_goal=q2,
                        tolerance=0.002,
                        allowed_time=30,
                        num_attempts=1,
                        pipeline_id='',
                        planner_id='',
                        vel_scale=1.0,
                        acc_scale=1.0)
        if result is None or result.error_code.val != 1:
            print(f'plan_joint_goal error: {None if result is None else result.error_code.val}')
            return None
        return result.trajectory.joint_trajectory

    def computeTrajectoryToPose(self, side: str, q1: dict[str, float], target:PoseStamped,
                          use_torso: bool, attached_collision_objects,
                          pos_tol:list[float], ori_tol:float|None) -> JointTrajectory | None:
        if use_torso:
            group_name = f'{side}_arm_torso'
        else:
            group_name = f'{side}_arm'
        result = self._moveit_client.plan_to_ee_pose(
                        attached_collision_objects,
                        group=group_name,
                        ee_link=f'{side}_arm_7_link',
                        start_state=q1,
                        target=target,
                        pos_tol_x=pos_tol[0],
                        pos_tol_y=pos_tol[1],
                        pos_tol_z=pos_tol[2],
                        ori_tol=ori_tol,
                        allowed_time=5.0,
                        num_attempts=1,
                        pipeline_id='',
                        planner_id='',
                        vel_scale=1.0,
                        acc_scale=1.0,
                    )
        if result is None or result.error_code.val != 1:
            print(f'plan_to_ee_pose error: {None if result is None else result.error_code.val}')
            return None
        return result.trajectory.joint_trajectory

    def applyPlanningScene(self, planning_scene: PlanningScene) -> bool:
        return self._moveit_client.apply_planning_scene(planning_scene)

    def getObjectsAndExactPoses(self, state: State) -> list[tuple[str, KdlFrame]]:
        param_value_list = [None, None]
        value_list = getPredicateParameterValues(state, is_value_equal, 'AtPose', param_value_list)
        result = []
        for obj_id, pose_data in value_list:
            if pose_data.getType() == 'Pose':
                result.append( (obj_id.getValue(), KdlFrame.fromDict(pose_data.getValue())) )
        return result

    def getObjectExactPose(self, state: State, obj_id_str: str) -> KdlFrame:
        obj_poses = self.getObjectsAndExactPoses(state)
        for o_id, T_W_O in obj_poses:
            if o_id == obj_id_str:
                return T_W_O
        # else:
        raise Exception(f'Could not get exact pose for object {obj_id_str}')

    def getGraspedObjectsAndExactPoses(self, state: State) -> list[tuple[str, KdlFrame]]:
        result = []
        joint_names, joint_positions = self.getJointNamesAndPositions(state)
        qt = self.getConfT(state)
        obj_model_map = self.getObjectsModelsIds(state)
        # Get exact poses of grasped objects
        for obj_id_str, side_str, grasp_id_str in self.getGraspedObjects(state):
            model_id_str = obj_model_map[obj_id_str]
            obj_grasps = self._grasps_map[model_id_str]['GraspDefs']
            grasp_def = obj_grasps[grasp_id_str]
            T_G_O = self.getGraspPose(grasp_def)
            T_E_G = self.getT_E_G(side_str)
            q = self.getConfA(state, side_str)
            T_W_E = self._velma_solv.getArmFk(side_str, qt.q[0], q.q)
            #T_W_E = self.fkArmTorso(side_str, joint_names, joint_positions)
            assert not T_W_E is None
            result.append( (obj_id_str, T_W_E * T_E_G * T_G_O) )
        return result

    def getModelIdByObjectId(self, state: State, obj_id_str: str) -> str:
        models_id_map = self.getObjectsModelsIds(state)
        return models_id_map[obj_id_str]

    def getObjectsModelsIds(self, state: State) -> dict[str, str]:
        param_value_list = [None, None]
        value_list = getPredicateParameterValues(state, is_value_equal, 'IsInstance', param_value_list)
        result = {}
        for obj_id, model_id in value_list:
            result[obj_id.getValue()] = model_id.getValue()
        return result

    def getModelByModelId(self, model_id_str: str) -> dict:
        return self._models_map[model_id_str]

    def getModelByObjectId(self, state: State, obj_id_str: str) -> dict[str, Any]:
        # models_id_map = self.getObjectsModelsIds(state)
        # model_id_str = models_id_map[obj_id_str]
        return self.getModelByModelId(self.getModelIdByObjectId(state, obj_id_str))

    def getGraspedObjects(self, state: State) -> list[tuple[str, str, str]]:
        param_value_list = [None, None, None]
        value_list = getPredicateParameterValues(state, is_value_equal, 'GraspedSdGr', param_value_list)
        result = []
        for obj_id, sd, grasp_id in value_list:
            result.append( (obj_id.getValue(), sd.getValue(), grasp_id.getValue()) )
        return result

    def getGraspDef(self, state: State, obj_id_str: str, grasp_id_str: str) -> dict[str, Any]:
        model_id_str = self.getModelIdByObjectId(state, obj_id_str)
        obj_grasps = self._grasps_map[model_id_str]['GraspDefs']
        return obj_grasps[grasp_id_str]
    
    def getAllObjectGrasps(self, state: State, obj_id_str: str) -> dict:
        model_id_str = self.getModelIdByObjectId(state, obj_id_str)
        obj_grasps = self._grasps_map[model_id_str]['GraspDefs']
        return obj_grasps

    def getGraspMovements(self, side_str: str, grasp_def: dict[str, Any]) -> list[tuple[str, Any]]:
        distal_f_mult = 0.33333333
        result = []
        for mov in grasp_def['GripperMovements']:
            if 'GripperMovementT' in mov:
                result.append( ('T_G_O', KdlFrame.fromDict(mov['GripperMovementT']['Pose'])) )
            elif 'GripperMovementQ' in mov:
                js = {
                    f'{side_str}_HandFingerOneKnuckleOneJoint': float(mov['GripperMovementQ']['sp']),
                    f'{side_str}_HandFingerOneKnuckleTwoJoint': float(mov['GripperMovementQ']['f1']),
                    f'{side_str}_HandFingerOneKnuckleThreeJoint': float(mov['GripperMovementQ']['f1'])*distal_f_mult,
                    f'{side_str}_HandFingerTwoKnuckleOneJoint': float(mov['GripperMovementQ']['sp']),
                    f'{side_str}_HandFingerTwoKnuckleTwoJoint': float(mov['GripperMovementQ']['f2']),
                    f'{side_str}_HandFingerTwoKnuckleThreeJoint': float(mov['GripperMovementQ']['f2'])*distal_f_mult,
                    f'{side_str}_HandFingerThreeKnuckleTwoJoint': float(mov['GripperMovementQ']['f3']),
                    f'{side_str}_HandFingerThreeKnuckleThreeJoint': float(mov['GripperMovementQ']['f3'])*distal_f_mult
                }
                result.append( ('Q', js) )
            else:
                raise Exception(f'Wrong gripper movement type: {mov}')
        return result

    def getPregraspPose(self, grasp_def: dict[str, Any]) -> KdlFrame:
        # returns T_G_O
        for movement in grasp_def['GripperMovements']:
            if 'GripperMovementT' in movement:
                return KdlFrame.fromDict( movement['GripperMovementT']['Pose'] )
        raise Exception('No pose in grasp definition')

    def getGraspPose(self, grasp_def: dict[str, Any]) -> KdlFrame:
        # returns T_G_O
        for movement in reversed(grasp_def['GripperMovements']):
            if 'GripperMovementT' in movement:
                return KdlFrame.fromDict( movement['GripperMovementT']['Pose'] )
        raise Exception('No pose in grasp definition')

    def getPregraspConfJsMap(self, side_str: str, grasp_def: dict[str, Any]) -> dict[str, float]:
        # returns T_G_O
        for mov_type, mov_data in self.getGraspMovements(side_str, grasp_def):
            if mov_type == 'Q':
                return mov_data
        raise Exception('Could not get pregrasp config')

    def getPregraspConf(self, side_str: str, grasp_def: dict[str, Any]) -> ConfG:
        return ConfG.fromJsMap( side_str, self.getPregraspConfJsMap(side_str, grasp_def) )

    def getGraspConfJsMap(self, side_str: str, grasp_def: dict[str, Any]) -> dict[str, float]:
        # returns T_G_O
        for mov_type, mov_data in reversed(self.getGraspMovements(side_str, grasp_def)):
            if mov_type == 'Q':
                return mov_data
        raise Exception('Could not get grasp config')

    def getGraspConf(self, side_str: str, grasp_def: dict[str, Any]) -> ConfG:
        return ConfG.fromJsMap( side_str, self.getGraspConfJsMap(side_str, grasp_def) )

    def getGraspId(self, grasp_def: dict[str, Any]) -> int:
        return int(grasp_def['GraspId'])

    def getModelAttribute(self, model: dict[str, Any], attribute_name: str) -> dict | None:
        for attr in model['Attributes']:
            if attr['Name'] == attribute_name:
                return attr
        # else:
        return None

    def getObjectsIdsAndModelsIds(self, state: State) -> list[tuple[str, str]]:
        param_value_list = [None, None]
        result = []
        for obj_id, model_id in getPredicateParameterValues(state, is_value_equal, 'IsInstance', param_value_list):
            result.append( (obj_id.getValue(), model_id.getValue()) )
        return result
    
    def getMoveableObjectsIds(self, state: State) -> set[str]:
        result: set[str] = set()
        for obj_id, model_id in self.getObjectsIdsAndModelsIds(state):
            model =  self.getModelByModelId(model_id)
            attr = self.getModelAttribute(model, 'moveability')
            assert not attr is None
            is_moveable = attr['IsMoveable']
            assert isinstance(is_moveable, bool)
            if is_moveable:
                result.add(obj_id)
        return result

    def getStaticObjectsIds(self, state: State) -> set[str]:
        result: set[str] = set()
        for obj_id, model_id in self.getObjectsIdsAndModelsIds(state):
            model =  self.getModelByModelId(model_id)
            attr = self.getModelAttribute(model, 'moveability')
            assert not attr is None
            is_moveable = attr['IsMoveable']
            assert isinstance(is_moveable, bool)
            if not is_moveable:
                result.add(obj_id)
        return result

    def getFlatSurfaces(self, state: State) -> list[dict[str, Any]]:
        result = []
        for obj_id_str, T_W_O in self.getObjectsAndExactPoses(state):
            model = self.getModelByObjectId(state, obj_id_str)
            if (flat_surfaces := self.getModelAttribute(model, 'flat_surfaces')) is not None:
                for fs in flat_surfaces['FlatSurfaceContainer']:
                    T_O_F = KdlFrame.fromDict(fs['Pose'])
                    if 'Rectangle' in fs:
                        size = (float(fs['Rectangle']['x']), float(fs['Rectangle']['y']))
                    else:
                        raise Exception(f'Surface type is not supported: "{fs.keys()}"')
                    result.append( {'T_W_F': T_W_O*T_O_F, 'size': size} )
        return result

    def sampleSurfaces(self, surfaces: list[dict[str, Any]], step_size: float = 0.05,
                        margin: float = 0.05) -> list[KdlFrame]:
        # Sample in the reference frame of object
        result = []
        for surface in surfaces:
            sx, sy = surface['size']
            sx = sx - margin*2
            sy = sy - margin*2
            x_samples = max(2, int(math.ceil(sx/step_size)))
            y_samples = max(2, int(math.ceil(sy/step_size)))
            for x in np.linspace(-sx/2, sx/2, x_samples, endpoint=True):
                for y in np.linspace(-sy/2, sy/2, y_samples, endpoint=True):
                    T_W_SP = surface['T_W_F'] * KdlFrame(KdlVector(x, y, 0))
                    result.append( T_W_SP )
        return result

    def getFlatBasesCurrentPositions(self, state: State) -> list[dict[str, Any]]:
        result = []
        for obj_id_str, T_W_O in self.getObjectsAndExactPoses(state) + self.getGraspedObjectsAndExactPoses(state):
            model = self.getModelByObjectId(state, obj_id_str)
            if (flat_surfaces := self.getModelAttribute(model, 'flat_bases')) is not None:
                for fb in flat_surfaces['FlatBaseContainer']:
                    T_O_F = KdlFrame.fromDict(fb['Pose'])
                    assert 'Circle' in fb
                    result.append( {'T_W_F': T_W_O*T_O_F, 'type': 'circle',
                                                        'size': float(fb['Circle']['r'])} )
        return result

    def getFlatBases(self, state: State, obj_id_str: str) -> list[FlatBase]:
        result = []
        model = self.getModelByObjectId(state, obj_id_str)
        if (flat_surfaces := self.getModelAttribute(model, 'flat_bases')) is not None:
            for fb in flat_surfaces['FlatBaseContainer']:
                T_O_F = KdlFrame.fromDict(fb['Pose'])
                assert 'Circle' in fb
                result.append( FlatBase(T_O_F, 'circle', [float(fb['Circle']['r'])]) )
        return result

    def getObjectHole(self, state: State, obj_id_str: str) -> None|Hole:
        model = self.getModelByObjectId(state, obj_id_str)
        if (hole_attr := self.getModelAttribute(model, 'hole')) is not None:
            hole = hole_attr['Hole']
            T_O_H = KdlFrame.fromDict(hole['Pose'])
            assert 'Circle' in hole
            return Hole(T_O_H, 'circle', [float(hole['Circle']['r'])])
        # else:
        return None

    def getArmRangeMin(self) -> float:
        return 0.25
    
    def getArmRangeMax(self) -> float:
        return 1.0

    def getPossibleShoulderCenters(self, side_str: str, samples: int) -> list[KdlVector]:
        if side_str == 'left':
            return [KdlFrame(KdlRotation.RotZ(angle))*KdlVector(0, 0.27, 1.35)
                    for angle in np.linspace(0, math.radians(90), samples, endpoint=True)]
        elif side_str == 'right':
            return [KdlFrame(KdlRotation.RotZ(angle))*KdlVector(0, -0.27, 1.35)
                    for angle in np.linspace(0, math.radians(-90), samples, endpoint=True)]
        # else:
        raise Exception(f'Wrong side: "{side_str}"')

    def getEnvironmentCollisionShapes(self, state: State, ignored_obj_id_str_list: list[str],
                                      enlarge: float = 0.0) -> dict[str, list[PrimitiveShape]]:
        obj_poses = self.getObjectsAndExactPoses(state) + self.getGraspedObjectsAndExactPoses(state)
        result: dict[str, list[PrimitiveShape]] = {}
        for obj_id_str, T_W_O in obj_poses:
            if obj_id_str in ignored_obj_id_str_list:
                continue
            model = self.getModelByObjectId(state, obj_id_str)
            fcl_col_objects = self.getModelCollisionShapes(model, T_W_O, enlarge=enlarge)
            result[obj_id_str] = fcl_col_objects
        return result

    def getModelCollisionShapes(self, model: dict[str, Any], T_W_O: KdlFrame, enlarge: float = 0.0
                                            ) -> list[PrimitiveShape]:
        result = []
        for link in model['KinematicStructure']['Links']:
            for collision in link['collision']:
                T_O_C = KdlFrame.fromDict(collision['Pose']['Pose'])
                if 'Cylinder' in collision['Geometry']:
                    cylinder = collision['Geometry']['Cylinder']
                    radius = float(cylinder['radius']) + enlarge
                    length = float(cylinder['length']) + enlarge*2
                    result.append( PrimitiveShape('cylinder', [radius, length], T_W_O, T_O_C, 'world') )
                elif 'Box' in collision['Geometry']:
                    size = collision['Geometry']['Box']['Size']
                    sx = float(size['x']) + enlarge*2
                    sy = float(size['y']) + enlarge*2
                    sz = float(size['z']) + enlarge*2
                    result.append( PrimitiveShape('box', [sx, sy, sz], T_W_O, T_O_C, 'world') )
        return result

    def getReservedSpace(self, state: State) -> list[tuple[Volume, int]]:
        param_value_list = [None, None]
        result = []
        for vol, vol_id in getPredicateParameterValues(state, is_value_equal, 'KeepFree', param_value_list):
            assert isinstance(vol.getValue(), Volume)
            result.append( (vol.getValue(), vol_id.getValue()) )
        return result

    def getGripperCollisionShapes(self, side: str, T_W_G: KdlFrame,
                            q_map: dict[str, float], enlarge: float=0.0) -> list[PrimitiveShape]:

        T_G_P = self._solv_bh.getT_P_G().Inverse()
        T_W_P = T_W_G * T_G_P

        fk_map = self._solv_bh.calculateJsMapFK(side, q_map)

        col_links = [f'{side}_HandFingerOneKnuckleTwoLink', f'{side}_HandFingerOneKnuckleThreeLink',
                    f'{side}_HandFingerTwoKnuckleTwoLink', f'{side}_HandFingerTwoKnuckleThreeLink',
                    f'{side}_HandFingerThreeKnuckleTwoLink', f'{side}_HandFingerThreeKnuckleThreeLink']

        result = []
        for link_name in col_links:
            T_P_L = fk_map[link_name]
            T_W_L = T_W_P * T_P_L
            T_W_C = T_W_L * KdlFrame(KdlVector(0.03,0,0))
            sx = 0.06 + enlarge*2
            sy = 0.02 + enlarge*2
            sz = 0.02 + enlarge*2
            result.append( PrimitiveShape('box', [sx, sy, sz], T_W_C, KdlFrame(), 'world') )

        # palm link
        radius = 0.04+enlarge
        length = 0.08+enlarge*2
        result.append( PrimitiveShape('cylinder', [radius, length], T_W_P*KdlFrame(KdlVector(0,0,0.04)), KdlFrame(), 'world') )

        # gripper mount
        if side == 'left':
            T_W_C = T_W_P*KdlFrame(KdlVector(0.09,0,-0.12))
            sx = 0.02 + enlarge*2
            sy = 0.08 + enlarge*2
            sz = 0.1 + enlarge*2
            result.append( PrimitiveShape('box', [sx, sy, sz], T_W_C, KdlFrame(), 'world') )

            T_W_C = T_W_P*KdlFrame(KdlRotation.RotY(math.radians(45)), KdlVector(0.06,0,-0.035))
            sx = 0.1 + enlarge*2
            sy = 0.08 + enlarge*2
            sz = 0.02 + enlarge*2
            result.append( PrimitiveShape('box', [sx, sy, sz], T_W_C, KdlFrame(), 'world'))

        elif side == 'right':
            T_W_C = T_W_P*KdlFrame(KdlVector(-0.09,0,-0.12))
            sx = 0.02 + enlarge*2
            sy = 0.08 + enlarge*2
            sz = 0.1 + enlarge*2
            result.append( PrimitiveShape('box', [sx, sy, sz], T_W_C, KdlFrame(), 'world') )

            T_W_C = T_W_P*KdlFrame(KdlRotation.RotY(math.radians(-45)), KdlVector(-0.06,0,-0.035))
            sx = 0.1 + enlarge*2
            sy = 0.08 + enlarge*2
            sz = 0.02 + enlarge*2
            result.append( PrimitiveShape('box', [sx, sy, sz], T_W_C, KdlFrame(), 'world'))

        # arm 6 link
        T_W_C = T_W_P*KdlFrame(KdlVector(0.0,0.0,-0.12))
        radius = 0.07+enlarge
        result.append( PrimitiveShape('sphere', [radius], T_W_C, KdlFrame(), 'world'))

        return result
    
