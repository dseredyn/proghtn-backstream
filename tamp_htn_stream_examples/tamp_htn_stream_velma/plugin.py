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

import rclpy

from typing import Any
from pathlib import Path
import json
import re
import math
from ament_index_python.packages import get_package_share_directory

from .moveit_interface import PlannerClient

from .lib_spatial import StatesInterface

from tamp_htn_stream.core import TypedValue

from .data_types import ConfA, ConfT,\
    ConfG, ConfH, Volume
from .pybullet_int import PyBulletInterface
from .vwc import VelmaWristConstraint
from .velma_kinematics import KinematicsSolverVelma, KinematicsSolverBarrettHand
from .func_tools import LinearIntervalFunction

_world_state_rules = None
_node_visualization_template_html = None
_plan_visualization_template_html = None
_states_interface = None
_pybullet_interface = None
_velma_wrist_constraint = None
_velma_solver = None

def getStatesInterface() -> StatesInterface:
    global _states_interface
    if _states_interface is None:
        raise Exception('Could not get states interface: not initialized')
    return _states_interface

def getPyBulletInterface() -> PyBulletInterface:
    global _pybullet_interface
    if _pybullet_interface is None:
        raise Exception('Could not get py bullet interface: not initialized')
    return _pybullet_interface

def getVelmaWristConstraint() -> VelmaWristConstraint:
    global _velma_wrist_constraint
    if _velma_wrist_constraint is None:
        raise Exception('Could not get Velma wrist constraint: not initialized')
    return _velma_wrist_constraint

def getNodeVisualizationTemplateHtml() -> str:
    global _node_visualization_template_html
    if _node_visualization_template_html is None:
        raise Exception('Could not get visualization template for node')
    return _node_visualization_template_html

def getPlanVisualizationTemplateHtml() -> str:
    global _plan_visualization_template_html
    if _plan_visualization_template_html is None:
        raise Exception('Could not get visualization template for node')
    return _plan_visualization_template_html

def getVelmaSolver() -> VelmaSolver:
    global _velma_solver
    if _velma_solver is None:
        raise Exception('Could not get VelmaSolver')
    return _velma_solver

def _resolve_path_uri(uri, relative_path_root=None) -> str:
    if uri.startswith('package://'):
        # Package-relative path
        package_and_path = uri[10:]
        idx = package_and_path.find('/')
        package_name = package_and_path[0:idx]
        relative_path = package_and_path[idx+1:]
        share_dir = Path(get_package_share_directory(package_name))
        return str(share_dir / relative_path)
    elif uri.startswith('/'):
        # Absolute path
        return uri
    else:
        # Relative path
        if relative_path_root is None:
            raise Exception(f'Cannot process URI: {uri}; relative path is None.')
        if not relative_path_root.startswith('/'):
            raise Exception(f'Cannot process URI: {uri}; relative path is not absolutes.')
        return f'{relative_path_root}/{uri}' 


def _load_json_file(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    text = p.read_text(encoding="utf-8")
    data = json.loads(text) if text.strip() else {}
    return data

def strip_trailing_digits(s: str) -> str:
    return re.sub(r"\d+$", "", s)

def _generateUniqueName(original_name, forbidden_names):
    if not original_name in forbidden_names:
        return original_name
    for idx in range(1000):
        new_name = '{}{}'.format( strip_trailing_digits(original_name), idx )
        if not new_name in forbidden_names:
            return new_name
    # else:
    raise Exception(f'Could not find unique name for "{original_name}"')

def tryConvertToInternalRepresentation(param_type: str, param_value: dict) -> Any:
    _datatypes = [ConfA, ConfT, ConfG, ConfH, Volume]
    for dtype in _datatypes:
        if param_type == dtype.__name__:
            return dtype.fromDict(param_value)
    return param_value

def parse_world_state(json_state):
    # State is:
    # - a set of predicates with symbolic parameters
    # - map from symbolic parameters to values

    assert not _world_state_rules is None
    # Try to apply each rule to each world state item
    all_predicates = []
    values_map = {}
    for item in json_state['WorldState']:
        for rule in _world_state_rules['rules']:
            new_predicates, new_values_map = _apply_rule(item, rule, values_map.keys())
            for key, value in new_values_map.items():
                assert not key in values_map
                values_map[key] = value
            all_predicates = all_predicates + new_predicates

    for param_name in values_map: #list(values_map.keys()):
        param_data = values_map[param_name]
        # print(param_data)
        param_type = param_data['ValueType']
        param_value = tryConvertToInternalRepresentation(param_type, param_data['Value'])
        values_map[param_name] = TypedValue(param_type, param_value)

    return {'Predicates':all_predicates, 'Values':values_map}


def parse_task_network(json_tn):
    for param_name in json_tn['ParameterGroundings']:
        param_data = json_tn['ParameterGroundings'][param_name]
        param_type = param_data['ValueType']
        param_value = tryConvertToInternalRepresentation(param_type, param_data['Value'])
        json_tn['ParameterGroundings'][param_name] = TypedValue(param_type, param_value)
    return json_tn

class VelmaSolver():
    def __init__(self):
        self.vsolv = KinematicsSolverVelma()
        self.bhsolv = KinematicsSolverBarrettHand()
        self.arm_lim = self.vsolv.getArmLmitsCartImp()
        self.vwc = getVelmaWristConstraint()
        self.eval_lim_dist = LinearIntervalFunction([(-1.0, 0.0), (math.radians(10.0), 0.0), (math.radians(20.0), 0.1), (math.radians(45.0), 1.0), (10000000.0, 1.0)])
        self.eval_wrist = LinearIntervalFunction([(-1.0, 0.0), (0.2, 0.0), (1.0, 1.0), (10000000.0, 1.0)])

    def evaluateArmConf(self, side: str, q: tuple):
        # Evaluate configurations: distance from constraints:
        # joint limits, singularities, constraint at wrist
        # Constraint at wrist is hard to compute.

        # Evaluate wrist configuration
        score_wrist = self.eval_wrist.interpolate( self.vwc.getScore(side, q[5], q[6]) )
        if score_wrist < 0.0001:
            return 0.0
        # else:

        # Joint limits
        # Most of singularities are included in joint limits
        total_q_score = 1.0
        q_str = ''
        assert len(q) == 7
        for q_idx, q_val in enumerate(q):
            q_score = 0.0
            for lim_lo, lim_hi in self.arm_lim[q_idx]:
                if q_val > lim_lo and q_val < lim_hi:
                    lim_dist = min(q_val-lim_lo, lim_hi - q_val)
                    assert lim_dist > 0
                    q_score = self.eval_lim_dist.interpolate(lim_dist)
                    break
            total_q_score = min(total_q_score, q_score)
            q_str += '{:.1f}, '.format(q_val)

        # print(f'q: {q_str} {side}, scores: {score_wrist} {total_q_score}')

        # Return score: 0.0 ... 1.0
        return score_wrist * total_q_score


def configure(world_model_data, config_dir_path) -> None:
    print('Configuring tamp_htn_stream_velma')

    # from visualization_msgs.msg import Marker
    # m = Marker()
    # print(dir(m.header))
    # raise

    rclpy.init()
    moveit_client = PlannerClient()
    moveit_client.wait_for_service()

    models_abs_path = _resolve_path_uri(world_model_data['models'],
                                                                relative_path_root=config_dir_path)
    print(f'  Models path (abs): {models_abs_path}')

    models = _load_json_file(models_abs_path)
    print('Reading models:')
    # global models_map
    models_map = {}
    for model in models['ObjectModels']:
        print(f'  {model['ModelId']}')
        models_map[model['ModelId']] = model

    grasps_abs_path = _resolve_path_uri(world_model_data['grasps_dir'],
                                                                relative_path_root=config_dir_path)
    print(f'  Grasps path (abs): {grasps_abs_path}')
    grasps_map = {}
    print(f'Reading grasps:')
    for model_id in models_map:
        grasps_file_path = f'{grasps_abs_path}/{model_id}.json'
        p = Path(grasps_file_path)
        if not p.exists():
            continue
        # else:
        print(f'  {model_id}')
        grasps_map[model_id] = _load_json_file(grasps_file_path)

    global _states_interface
    _states_interface = StatesInterface(models_map, grasps_map, moveit_client)

    global _pybullet_interface
    _pybullet_interface = PyBulletInterface()

    world_state_rules_abs_path = _resolve_path_uri(world_model_data['world_state_rules'],
                                                                relative_path_root=config_dir_path)
    print(f'  World state rules path (abs): {world_state_rules_abs_path}')
    global _world_state_rules
    _world_state_rules = _load_json_file(world_state_rules_abs_path)

    # Read HTML templates for visualization
    node_visualization_template_html_abs_path = _resolve_path_uri(
            world_model_data['node_visualization_template'], relative_path_root=config_dir_path)
    with open(node_visualization_template_html_abs_path, 'r') as f:
        global _node_visualization_template_html
        _node_visualization_template_html = f.read()

    plan_visualization_template_html_abs_path = _resolve_path_uri(
            world_model_data['plan_visualization_template'], relative_path_root=config_dir_path)
    with open(plan_visualization_template_html_abs_path, 'r') as f:
        global _plan_visualization_template_html
        _plan_visualization_template_html = f.read()

    velma_wrist_constraint_image_abs_path = _resolve_path_uri(
            world_model_data['velma_wrist_constraint_image'], relative_path_root=config_dir_path)
    global _velma_wrist_constraint
    _velma_wrist_constraint = VelmaWristConstraint(velma_wrist_constraint_image_abs_path)

    global _velma_solver
    _velma_solver = VelmaSolver()
    
    print('Configured tamp_htn_stream_velma')


def _getItemValue(item, path):
    # print(f'_getItemValue({item}, {path})')
    # returns: is_found, value
    path_elements = path.split('.')
    result = item
    for path_el in path_elements:
        if path_el in result:
            result = result[path_el]
        else:
            return False, None
    return True, result


def _parsePredicate(pred_str):
    items = pred_str.split()
    result = []
    for item in items:
        item_s = item.strip()
        if item_s:
            result.append(item_s)
    return {'Name': result[0], 'Args': result[1:]}


def _apply_rule(item, rule, forbidden_param_symbols):
    # print(f'_apply_rule({item}, {rule})')
    can_apply = True
    param_name_value_map = {}
    for calc in rule['calculations']:
        if (idx := calc.find('==')) > 0:
            comp_value = calc[0:idx].strip()
            comp_source = calc[idx+2:].strip()

            item_found, item_value = _getItemValue(item, comp_source)
            if not item_found or comp_value != item_value:
                can_apply = False
                break
            # print(f'{comp_value} == {comp_source}')
        elif (idx := calc.find('=')) > 0:
            assign_to = calc[0:idx].strip()
            assign_from = calc[idx+1:].strip()

            item_found, item_value = _getItemValue(item, assign_from)
            if not item_found:
                can_apply = False
                break
            value_type = rule['parameters'][assign_to]
            #item_value = tryConvertToInternalRepresentation(value_type, item_value)
            #TypedValue(value_type, item_value)
            param_name_value_map[assign_to] = {'ValueType': value_type, 'Value': item_value}
            # print(f'{assign_to} = {assign_from}')
    if not can_apply:
        return [], {}
    # else:

    # Can apply this rule
    # TODO: types
    # types = []
    # for param_name, param_type in rule['parameters'].items():

    predicates = []
    values_map = {}
    if 'predicates' in rule:
        for pred_str in rule['predicates']:
            pred = _parsePredicate(pred_str)
            pred_arg_names = []
            for arg_name in pred['Args']:
                arg_unique_name = _generateUniqueName(arg_name, forbidden_param_symbols)
                values_map[arg_unique_name] = param_name_value_map[arg_name]
                pred_arg_names.append(arg_unique_name)
            predicates.append( {'Name':pred['Name'], 'Args':pred_arg_names} ) 
    return predicates, values_map

