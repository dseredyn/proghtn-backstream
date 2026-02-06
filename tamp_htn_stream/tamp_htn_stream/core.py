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

from dataclasses import dataclass
from types import ModuleType
from typing import Any, Dict
import re
import copy
import json
import random
import time
from .planning_domain import PlanningDomain

type State = dict[str, Any]
type TaskNetwork = dict[str, Any]

# TODO: use types matching within hierarchical structure:
# A type can be either abstract or complete. Only single inheritance is allowed (one parent).
# Every object instance is of a complete type.
# If an abstract type is required as argument to a task or a stream, then all objects of complete
# types derived from the abstract type are accepted.

class StatsCollector:
    def __init__(self):
        self._items = []
        self._uid_time = 0
        self._active_time_measurements = {}

    def add(self, item):
        self._items.append(item)

    def getItems(self):
        return self._items

    def beginTimeMeasurement(self, measurement_data) -> int:
        self._active_time_measurements[self._uid_time] = (time.time(), measurement_data)
        self._uid_time = self._uid_time + 1
        return self._uid_time - 1

    def endTimeMeasurement(self, measurement_id: int):
        assert measurement_id in self._active_time_measurements
        t_begin, measurement_data = self._active_time_measurements[measurement_id]
        measurement_time = time.time() - t_begin
        self._items.append( ['time', measurement_time, measurement_data] )
        del self._active_time_measurements[measurement_id]

_stats_collector = StatsCollector()

def clear_stats():
    global _stats_collector
    _stats_collector = StatsCollector()


def STATS() -> StatsCollector:
    global _stats_collector
    return _stats_collector


class StatsMap:
    def __init__(self):
        self._stats_map = {}
        self._verbose = False

    def toJson(self) -> dict:
        return self._stats_map

    @staticmethod
    def fromJson(d: dict) -> StatsMap:
        out = StatsMap()
        out._stats_map = d
        return out

    def increaseValue(self, path: str, delta_val: int|float):
        if self._verbose: print(f'increaseValue({path}, {delta_val})')
        assert isinstance(path, str)
        assert isinstance(delta_val, (int, float))
        names = path.split('.')
        current_dict = self._stats_map
        for name in names[:-1]:
            if not name in current_dict:
                if self._verbose: print(f'  creating new key: {name}')
                current_dict[name] = {}
            current_dict = current_dict[name]
        if not names[-1] in current_dict:
            if self._verbose: print(f'  creating new value: {names[-1]}')
            current_dict[names[-1]] = 0
        current_dict[names[-1]] += delta_val

    def setValue(self, path: str, val: int|float):
        if self._verbose: print(f'setValue({path}, {val})')
        assert isinstance(path, str)
        assert isinstance(val, (int, float))
        names = path.split('.')
        current_dict = self._stats_map
        for name in names[:-1]:
            if not name in current_dict:
                if self._verbose: print(f'  creating new key: {name}')
                current_dict[name] = {}
            current_dict = current_dict[name]
        if self._verbose: print(f'  creating new value: {names[-1]}')
        current_dict[names[-1]] = val

    def setValueObj(self, path: str, val):
        if self._verbose: print(f'setValue({path}, {val})')
        assert isinstance(path, str)
        names = path.split('.')
        current_dict = self._stats_map
        for name in names[:-1]:
            if not name in current_dict:
                if self._verbose: print(f'  creating new key: {name}')
                current_dict[name] = {}
            current_dict = current_dict[name]
        if self._verbose: print(f'  creating new value: {names[-1]}')
        current_dict[names[-1]] = val

    def getValue(self, path: str) -> int|float:
        if self._verbose: print(f'getValue({path})')
        assert isinstance(path, str)
        names = path.split('.')
        current_dict = self._stats_map
        for name in names[:-1]:
            if not name in current_dict:
                return 0
            # else:
            current_dict = current_dict[name]
        if not names[-1] in current_dict:
            return 0
        val = current_dict[names[-1]]
        assert isinstance(val, (int, float))
        return val

    def getKeysAt(self, path: str) -> list[str]:
        assert isinstance(path, str)
        names = path.split('.')
        current_dict = self._stats_map
        for name in names:
            if not name in current_dict:
                return []
            current_dict = current_dict[name]
        return list(current_dict.keys())

    def getAll(self):
        return self._stats_map


def prepareStats() -> StatsMap:
    sm = StatsMap()
    max_node_depth = 0
    for item in STATS().getItems():
        # Streams
        item_type = item[0]
        if item_type == 'stream_failure':
            stream_name, node_id, bt_nodes_ids, bt_streams_names = item[1:]
            sm.increaseValue(f'streams.{stream_name}.total_count', 1)
            sm.increaseValue(f'streams.{stream_name}.failure', 1)
            sm.increaseValue(f'streams_total.count', 1)
            sm.increaseValue(f'streams_total.failures', 1)
            if (bt_nodes_count := len(bt_nodes_ids)) > 0:
                sm.increaseValue(f'streams.{stream_name}.backtracks', 1)
                sm.increaseValue(f'streams.{stream_name}.backtracks_nodes_sum', bt_nodes_count)
                sm.increaseValue(f'streams_total.backtracks', 1)
                sm.increaseValue(f'streams_total.backtracks_nodes_sum', bt_nodes_count)
            for bt_stream_name in bt_streams_names:
                sm.increaseValue(f'streams.{bt_stream_name}.bad_param_gen', 1)
        elif item_type == 'stream_success':
            stream_name, node_id = item[1:]
            sm.increaseValue(f'streams.{stream_name}.total_count', 1)
            sm.increaseValue(f'streams.{stream_name}.success', 1)
            sm.increaseValue(f'streams_total.count', 1)
        elif item_type == 'stream_no_more':
            stream_name, node_id = item[1:]
            sm.increaseValue(f'streams.{stream_name}.total_count', 1)
            sm.increaseValue(f'streams.{stream_name}.no_more', 1)
            sm.increaseValue(f'streams_total.count', 1)
        elif item_type == 'time':
            run_time, data = item[1:]
            if data[0] == 'stream_time':
                stream_name = data[1]
                sm.increaseValue(f'streams.{stream_name}.time', run_time)
                sm.increaseValue(f'streams_total.time', run_time)
            elif data[0] == 'total_time':
                sm.setValue(f'algorithm_run.total_time', run_time)

        # Decompositions
        elif item_type == 'decomposition':
            iteration, node_id, node_depth, method_name, task_class = item[1:]
            sm.increaseValue(f'decompositions.method.{method_name}.count', 1)
            sm.increaseValue(f'decompositions.tasks.{task_class}.count', 1)
        elif item_type == 'no_more_methods':
            iteration, node_id, node_depth, task_class = item[1:]
            sm.increaseValue(f'decompositions.task.{task_class}.denied_count', 1)

        # Executions
        elif item_type == 'execution':
            iteration, node_id, node_depth, action_class = item[1:]
            sm.increaseValue(f'executions.{action_class}.count', 1)
        elif item_type == 'execution_denied':
            iteration, node_id, node_depth, action_class = item[1:]
            sm.increaseValue(f'executions.{action_class}.denied_count', 1)

        # Solutions
        elif item_type == 'solution_found':
            iteration, node_id, node_depth = item[1:]
            sm.setValue(f'solutions.{node_id}.iteration', iteration)
            sm.setValue(f'solutions.{node_id}.depth', node_depth)
        elif item_type == 'generated_plan':
            node_id, plan_length, actions = item[1:]
            sm.setValue(f'solutions.{node_id}.plan_length', plan_length)
            sm.setValueObj(f'solutions.{node_id}.actions', actions)

        # Search algorithm
        elif item_type == 'iteration':
            iteration, node_id, node_depth, subtree_max_depth = item[1:]
            sm.increaseValue(f'algorithm_run.total_iterations', 1)
            max_node_depth = max(max_node_depth, node_depth)

    sm.setValue(f'algorithm_run.max_node_depth', max_node_depth)

    # More info about streams
    for stream_name in sm.getKeysAt('streams'):
        total_count = sm.getValue(f'streams.{stream_name}.total_count')
        if total_count > 0:
            success = sm.getValue(f'streams.{stream_name}.success')
            success_rate = float(success) / total_count
            sm.setValue(f'streams.{stream_name}.succes_rate', success_rate)

            mean_time = sm.getValue(f'streams.{stream_name}.time') / total_count
            sm.setValue(f'streams.{stream_name}.mean_time', mean_time)

            # Backtrack for each stream
            backtracks = sm.getValue(f'streams.{stream_name}.backtracks')
            if backtracks > 0:
                backtracks_nodes_sum = sm.getValue(f'streams.{stream_name}.backtracks_nodes_sum')
                mean_backtracks_nodes = float(backtracks_nodes_sum) / backtracks
                sm.setValue(f'streams.{stream_name}.mean_backtracks_nodes', mean_backtracks_nodes)

    # Backtrack for all streams
    backtracks = sm.getValue(f'streams_total.backtracks')
    if backtracks > 0:
        backtracks_nodes_sum = sm.getValue(f'streams_total.backtracks_nodes_sum')
        mean_backtracks_nodes = float(backtracks_nodes_sum) / backtracks
        sm.setValue(f'streams_total.mean_backtracks_nodes', mean_backtracks_nodes)
    return sm


class TypedValue:
    def __init__(self, tp: str, value: Any) -> None:
        self._type = tp
        self._value = value

    def getType(self) -> str:
        return self._type

    def getValue(self) -> Any:
        return self._value


class GeneratorResult:
    _status: str
    _data: list[TypedValue]
    _bad_inputs: list[int]
    _debug: dict[str, Any]

    def hasValidSample(self) -> bool:
        return self._status in ('sample', 'single_sample')

    def getSample(self) -> list[TypedValue]:
        return self._data

    def hasSingleSample(self) -> bool:
        return self._status == 'single_sample'

    def isFailure(self) -> bool:
        return self._status == 'failed'
    
    def getBadInputs(self) -> list[int]:
        return self._bad_inputs

    def debugStopPlanning(self) -> GeneratorResult:
        if not hasattr(self, '_debug'):
            self._debug = {}
        self._debug['stop_planning'] = True
        return self
    
    def debugHasStopPlanning(self) -> bool:
        if hasattr(self, '_debug') and 'stop_planning' in self._debug:
            return self._debug['stop_planning']
        # else:
        return False

    @staticmethod
    def Sample(data: list) -> GeneratorResult:
        assert isinstance(data, list)
        result = GeneratorResult()
        result._status = 'sample'
        result._data = data
        return result

    @staticmethod
    def SingleSample(data: list[TypedValue]) -> GeneratorResult:
        assert isinstance(data, list)
        result = GeneratorResult()
        result._status = 'single_sample'
        result._data = data
        return result

    @staticmethod
    def Failed(bad_inputs: list[int] = []) -> GeneratorResult:
        result = GeneratorResult()
        result._status = 'failed'
        result._bad_inputs = bad_inputs
        return result

    @staticmethod
    def NoMore() -> GeneratorResult:
        result = GeneratorResult()
        result._status = 'no_more'
        return result
    

class ProblemTree:
    class Node:
        def __init__(self, state: State, task_network: TaskNetwork, streams: list):
            self.state = state
            self.task_network = task_network
            self.streams = streams

            # Additional node information
            self._closed: bool = False
            self._applied_methods: set[str] = set()
            self._node_type = ''

            self.depth: int = -1
            self.subtree_has_solution: bool = False
            self.subtree_nodes_count: int = 0
            self.subtree_max_depth: int = 0
            self.is_solution:bool = False
            self._backtrack_search_results: list = []
            self._all_subtrees_closed = False

        def isClosed(self) -> bool:
            return self._closed

        def hasClosedSubtree(self) -> bool:
            return self._all_subtrees_closed

        # def close(self, pt: ProblemTree) -> None:
        #     # WARNING: this should not be called outside PlanningTree
        #     assert not pt is None
        #     self._closed = True

        def getAppliedMethods(self) -> set[str]:
            return self._applied_methods

        def addAppliedMethod(self, method_name: str) -> None:
            assert not method_name in self._applied_methods
            self._applied_methods.add(method_name)

        def getType(self) -> str:
            return self._node_type

        def setBacktrackSearchResult(self, result: list) -> None:
            self._backtrack_search_results = result

        def getBacktrackSearchResults(self) -> list:
            return self._backtrack_search_results

    def __init__(self, state: State, task_network: TaskNetwork):
        self._nodes: dict[int, ProblemTree.Node] = {}
        self._arrows = {}
        self._arrows_inv = {}
        self._new_node_id = 0
        self._addRootNode( ProblemTree.Node(state, task_network, []) )

    def getDeepestNodeId(self) -> int:
        deepest = None
        for node_id, node in self._nodes.items():
            if deepest is None or node.depth > deepest[0]:
                deepest = (node.depth, node_id)
        assert not deepest is None
        return deepest[1]

    def setNodeType(self, node_id: int, node_type: str) -> None:
        self._nodes[node_id]._node_type = node_type

    def hasSolution(self) -> bool:
        for node_id, node in self._nodes.items():
            if node.is_solution:
                return True
        # else:
        return False

    def getAllSolutionNodesIds(self) -> list[int]:
        result = []
        for node_id, node in self._nodes.items():
            if node.is_solution:
                result.append(node_id)
        return result

    def getParentNodeId(self, node_id: int) -> int|None:
        return self._arrows_inv[node_id] if node_id in self._arrows_inv else None

    def getChildrenNodesIds(self, node_id: int) -> list[int]:
        return self._arrows[node_id]

    def getAllOpenNodesInSubtree(self, node_id):
        out = []
        if not self._nodes[node_id].isClosed():
            out.append(node_id)

        if not self._nodes[node_id]._all_subtrees_closed:
            for ch_id in self.getChildrenNodesIds(node_id):
                out += self.getAllOpenNodesInSubtree(ch_id)
        return out

    def getNodeDepth(self, node_id: int) -> int:
        return self._nodes[node_id].depth

    def getTreeDepth(self) -> int:
        return self._nodes[0].subtree_max_depth

    def getRootNodeId(self) -> int:
        return 0

    def _addRootNode(self, n: ProblemTree.Node) -> int:
        assert not self._nodes
        self._nodes[self._new_node_id] = n
        self._nodes[self._new_node_id].depth = 0
        self._arrows[self._new_node_id] = []
        self._new_node_id = self._new_node_id + 1
        return self._new_node_id - 1

    def _addArrow(self, parent_node_id: int, child_node_id: int) -> None:
        self._arrows[parent_node_id].append(child_node_id)
        assert not child_node_id in self._arrows_inv
        self._arrows_inv[child_node_id] = parent_node_id

    def addNode(self, n: ProblemTree.Node, parent_node_id: int) -> int:
        assert self._nodes
        assert parent_node_id in self._nodes
        node_id = self._new_node_id
        self._nodes[node_id] = n
        self._nodes[node_id].depth = self._nodes[parent_node_id].depth + 1
        self._arrows[node_id] = []
        self._addArrow(parent_node_id, node_id)
        self._new_node_id = self._new_node_id + 1

        # Update node information for all ancestors
        next_id = node_id
        while (next_id := self.getParentNodeId(next_id)) is not None:
            self._nodes[next_id].subtree_nodes_count += 1
            if self._nodes[next_id].subtree_max_depth < self._nodes[node_id].depth:
                self._nodes[next_id].subtree_max_depth = self._nodes[node_id].depth

        return node_id

    def getNode(self, node_id: int) -> ProblemTree.Node:
        return self._nodes[node_id]

    def getAllNodes(self) -> dict[int, ProblemTree.Node]:
        return self._nodes

    def closeNode(self, node_id: int) -> None:
        assert not self._nodes[node_id]._closed
        self._nodes[node_id]._closed = True

        # Update tree information for closed nodes and subtrees
        n_id: int|None = self.getParentNodeId(node_id)
        while not n_id is None:
            n = self.getNode(n_id)
            # Check if all subtrees are closed
            all_subtrees_closed = True
            for ch_id in self.getChildrenNodesIds(n_id):
                ch = self.getNode(ch_id)
                if not ch.isClosed() or not ch._all_subtrees_closed:
                    all_subtrees_closed = False
                    break
            if all_subtrees_closed:
                assert not n._all_subtrees_closed
                n._all_subtrees_closed = True
                # State of this node changed, so check its parent
                n_id = self.getParentNodeId(n_id)
            else:
                # Update of earlier nodes is not needed
                break

    def getAncestorNodes(self, node_id: int) -> list[int]:
        out = []
        n_id: int|None = self.getParentNodeId(node_id)
        while not n_id is None:
            out.append(n_id)
            n_id = self.getParentNodeId(n_id)
        return out

    def markAsSolution(self, node_id: int) -> None:
        self._nodes[node_id].is_solution = True
        self._nodes[node_id].subtree_has_solution = True
        next_id = node_id
        while (next_id := self.getParentNodeId(next_id)) is not None:
            self._nodes[next_id].subtree_has_solution = True

    def getOpenNodesIds(self) -> list[int]:
        result: list[int] = []
        for node_id, node in self._nodes.items():
            if not node.isClosed():
                result.append(node_id)
        return result

def getActionsSequence(pd: PlanningDomain, pt: ProblemTree, node_id: int) -> list:
    result = []
    current_id: int|None = node_id
    while not current_id is None:
        node = pt.getNode(current_id)
        if node.getType() == 'execution':
            task = getFirstTask(node.task_network)
            assert not pd.isComplexTask(task)

            param_value_list = []
            for param_name in task['Args']:
                param_value_list.append( node.task_network['ParameterGroundings'][param_name] )
            result.append( (task, param_value_list) )
        current_id = pt.getParentNodeId(current_id)
    return list(reversed(result))

def getFirstTask(tn: TaskNetwork):
    # TODO: order
    return tn['Tasks'][0]

def getTaskNetworkParameterNames(task_network):
    return set( task_network['ParameterGroundings'].keys() )

def strip_trailing_digits(s: str) -> str:
    return re.sub(r"\d+$", "", s)

def generateUniqueName(original_name, forbidden_names):
    if not original_name in forbidden_names:
        return original_name
    for idx in range(1000):
        new_name = '{}{}'.format( strip_trailing_digits(original_name), idx )
        if not new_name in forbidden_names:
            return new_name
    # else:
    raise Exception(f'Could not find unique name for "{original_name}"')

def getMappingOfMethodParameterNames(method, task, tn) -> dict[str, str]:
    # Remap names of new parameters of the inserted task network. Keep the original names
    # of parameters of the decomposed task.
    # Create a map of parameter names:
    # - for parameters of the decomposed task: method param name -> task param name
    # - for new parameter: method param name -> unique name
    param_map = {}

    # Parameters of the decomposed task
    for m_arg_name, t_arg_name in zip(method['Task']['Args'], task['Args']):
        assert m_arg_name.startswith('?')
        assert t_arg_name.startswith('?')
        param_map[ m_arg_name ] = t_arg_name
        # print(f'getMappingOfMethodParameterNames: task param: {m_arg_name} -> {t_arg_name}')

    all_tn_parameters_names = getTaskNetworkParameterNames(tn)
    print(f'all_tn_parameters_names: {all_tn_parameters_names}')
    # New parameters, generated by streams
    for method_parameter in method['Parameters']:
        new_unique_param_name = generateUniqueName(method_parameter['VarName'],
                                                                        all_tn_parameters_names)
        if not method_parameter['VarName'] in param_map:
            # print(f'getMappingOfMethodParameterNames: other param: {method_parameter['VarName']} -> {new_unique_param_name}')
            # Remap missing parameters only.
            param_map[ method_parameter['VarName'] ] = new_unique_param_name
            assert not new_unique_param_name in all_tn_parameters_names
            all_tn_parameters_names.add(new_unique_param_name)
        else:
            # print(f'getMappingOfMethodParameterNames: duplicated: {method_parameter['VarName']}')
            pass

    return param_map

def getAllTaskIds(task_network):
    result = set()
    for task in task_network['Tasks']:
        assert not task['Id'] is None
        result.add(task['Id'])
    return result

def getMethodTaskPreferredName(task):
    if task['Id'] is None:
        return task['Class']
    # else:
    return task['Id']

def getMappingOfTaskNames(method_tn, tn):
    # Create a map of parameter names:
    # - for parameters of the decomposed task: method param name -> task param name
    # - for new parameter: method param name -> unique name

    all_tn_task_names = getAllTaskIds(tn)
    new_task_names_list = []
    #name_map = {}
    for m_task in method_tn['Tasks']:
        preferred_name = getMethodTaskPreferredName(m_task)
        new_name = generateUniqueName(preferred_name, all_tn_task_names)
        all_tn_task_names.add(new_name)
        new_task_names_list.append(new_name)
        # name_map[preferred_name] = new_name
    return new_task_names_list
    # return name_map

def getPredicateParameterValues(state, func_is_value_equal, pred_name, pred_param_value_list):
    result = []
    for pred in state['Predicates']:
        if pred['Name'] == pred_name:
            match = True
            for s_pred_param_name, req_param_value in zip(pred['Args'], pred_param_value_list):
                if req_param_value is None:
                    # Any value
                    continue
                s_pred_value = state['Values'][s_pred_param_name]
                if not func_is_value_equal(s_pred_value, req_param_value):
                    match = False
                    break
            if match:
                result_param_tuple = []
                for s_pred_param_name, req_param_value in zip(pred['Args'], pred_param_value_list):
                    if req_param_value is None:
                        # Save value
                        s_pred_value = state['Values'][s_pred_param_name]
                        result_param_tuple.append(s_pred_value)
                result.append(result_param_tuple)
    return result

def isTaskNetworkEmpty(task_network):
    return len(task_network['Tasks']) == 0

def replaceTaskWithTaskNetwork(original_tn, task_to_replace, replace_tn, task_name_mapping_list,
                                                                            param_name_mapping):
    new_tn = {
        'Tasks': [],
        'ParameterGroundings': copy.deepcopy(original_tn['ParameterGroundings'])}

    # Remap names of new parameters of the inserted task network. Keep the original names
    # of parameters of the decomposed task.
    for task_idx, task in enumerate(replace_tn['Tasks']):
        mapped_task = copy.deepcopy(task)
        # m_task_name = getMethodTaskPreferredName(task)
        new_task_name = task_name_mapping_list[task_idx]

        mapped_task['Id'] = new_task_name
        for arg_idx in range(len(mapped_task['Args'])):
            if not mapped_task['Args'][arg_idx] in param_name_mapping:
                raise Exception(f'Unknown arg {mapped_task['Args'][arg_idx]} of task '
                                f'{taskToStr(task)} in method task network')
            mapped_task['Args'][arg_idx] = param_name_mapping[ mapped_task['Args'][arg_idx] ]
        new_tn['Tasks'].append(mapped_task)

    for task in original_tn['Tasks']:
        if task['Id'] == task_to_replace['Id']:
            continue
        # else:
        new_tn['Tasks'].append( copy.deepcopy(task) )

    return new_tn

def removeTask(task_network, task):
    assert len(task_network['Tasks']) > 0
    first_task = task_network['Tasks'][0]
    assert first_task['Id'] == task['Id']
    new_tn = {'Tasks': task_network['Tasks'][1:]}
    new_grounding = {}
    for tn_task in new_tn['Tasks']:
        for arg in tn_task['Args']:
            if not arg in new_grounding:
                new_grounding[arg] = task_network['ParameterGroundings'][arg]
    new_tn['ParameterGroundings'] = new_grounding
    return new_tn

def getEffectLiterals(effect):
    if effect['type'] == 'and_formula':
        pos_result = []
        neg_result = []
        for literal in effect['Literals']:
            pos, neg = getEffectLiterals(literal)
            pos_result = pos_result + pos
            neg_result = neg_result + neg
        return pos_result, neg_result
    elif effect['type'] == 'neg_atom':
        return [], [{'type': 'atom', 'Name': effect['Name'], 'Args': effect['Args']}]
    elif effect['type'] == 'atom':
        return [{'type': 'atom', 'Name': effect['Name'], 'Args': effect['Args']}], []
    # else:
    raise Exception(f'Not supported type: {effect['type']}')

def removeAtomsFromState(state: State, world_model_module: ModuleType,
                         neg_atoms, state_value_map, pm: ParamMapping) -> tuple[State, list]:
    removed_predicates = []
    result_state = {'Predicates':[]}
    for state_pred in state['Predicates']:
        keep = True
        #for atom_name, param_names in zip(atom_name_list, param_names_list):
        for atom in neg_atoms:
            atom_name = atom['Name']
            if state_pred['Name'] == atom_name:
                match = True
                # print(f'atom {atom_name}')
                for s_pred_arg_name, atom_arg_name in zip(state_pred['Args'], atom['Args']):
                # for s_pred_arg_name, atom_arg in zip(state_pred['Args'], param_names):
                    val1 = state_value_map[s_pred_arg_name]
                    val2 = pm.mapName('pd', atom_arg_name, 'val')
                    # print(f'  {s_pred_arg_name} tn -> val {val1}')
                    # print(f'  {atom_arg_name} pd -> val {val2}')
                    # pm.printAllMappings()
                    is_equal = world_model_module.is_value_equal(val1, val2)
                    # print(f'comparison ({is_equal}):')
                    # print(json.dumps(state_value_map[s_pred_arg_name], indent=2, ensure_ascii=False))
                    # print('VS')
                    # print(json.dumps(value_map[atom_arg], indent=2, ensure_ascii=False))

                    if not is_equal:
                        match = False
                        break
                if match:
                    keep = False
                    break
        if keep:
            result_state['Predicates'].append(state_pred)
        else:
            removed_predicates.append(state_pred)
    return result_state, removed_predicates

def updateState(state: State, world_model_module, effect, pm) -> State:

    result_state = state
    pos_atoms, neg_atoms = getEffectLiterals(effect)
    # Remove negative atoms
    atom_name_list = []
    mapped_arg_names_list = []
    print('updateState *********************')

    state_value_map = state['Values']

    print(f'updateState(): predicates: {len(state['Predicates'])}')
    # result_state, removed_predicates = removeAtomsFromState(result_state, world_model_module, atom_name_list,
    #                                             mapped_arg_names_list, state_value_map, value_map)

    result_state, removed_predicates = removeAtomsFromState(result_state, world_model_module,
                                                            neg_atoms, state_value_map, pm)

    verbose = False
    print(f'updateState(): after remove: {len(result_state['Predicates'])}, removed:')
    if verbose:
        for pred in removed_predicates:
            pred_arg_values = []
            for arg in pred['Args']:
                pred_arg_values.append( state_value_map[arg] )
            pred_value = {'Name': pred['Name'],
                            'Args': pred_arg_values}
            print(json.dumps(pred_value, indent=4, ensure_ascii=False))
    else:
        print('  (not printing)')

    # Keep only used parameters
    result_state['Values'] = {}
    for pred in result_state['Predicates']:
        for arg in pred['Args']:
            if not arg in result_state['Values']:
                result_state['Values'][arg] = state['Values'][arg]

    forbidden_param_names = set(result_state['Values'].keys())

    # Add positive atoms
    added_predicates = []
    for atom in pos_atoms:
        print()
        new_pred_arg_names = []
        new_pred_arg_values = []
        for arg in atom['Args']:
            mapped_arg = pm.mapName('pd', arg, 'tn')
            if mapped_arg is None:
                raise Exception(f'Could not map atom {atom['Name']} parameter {arg}, pd -> tn')
            arg_value = pm.mapName('pd', arg, 'val')
            if arg_value is None:
                raise Exception(f'Could not map atom {atom['Name']} parameter {arg}, pd -> val')
            unique_param_name = generateUniqueName(mapped_arg, forbidden_param_names)
            forbidden_param_names.add(unique_param_name)
            result_state['Values'][unique_param_name] = arg_value
            new_pred_arg_names.append( unique_param_name )
            new_pred_arg_values.append(arg_value)

            # added_param_map[unique_param_name] = arg_value

        result_state['Predicates'].append({'Name': atom['Name'], 'Args': new_pred_arg_names})
        added_predicates.append({'Name': atom['Name'], 'ArgsNames': new_pred_arg_names, 'Args': new_pred_arg_values})

    print(f'updateState(): after add: {len(result_state['Predicates'])}, added:')
    if verbose:
        for pred_value in added_predicates:
            print(json.dumps(pred_value, indent=4, ensure_ascii=False))
    else:
        print('  (not printing)')

    return result_state


def taskToStr(task, indent=0):
    args_str = ''
    for arg in task['Args']:
        if args_str:
            args_str += ', '
        args_str += arg
    return f'{' '*indent}{task['Id']}/{task['Class']}({args_str})'

def taskNetworkToStr(task_network, indent=0):
    result = ''
    for task in task_network['Tasks']:
        result += f'{taskToStr(task, indent=indent)}\n'
    result += f'{' '*indent}grounded parameters: {list(task_network['ParameterGroundings'])}'
    return result

def checkIfTaskNetworkIsGround(task_network):
    for task in task_network['Tasks']:
        for arg in task['Args']:
            if not arg in task_network['ParameterGroundings']:
                raise Exception(f'Task network is not ground (parameter "{arg}"):'
                                                        f'\n{taskNetworkToStr(task_network)}')

def getApplicableMethods(pd: PlanningDomain, world_model_module: ModuleType,
                         task: dict[str, Any], task_network: TaskNetwork,
                         state: State, ignore_methods: set[str]):
    methods = pd.getMethodsForTaskClass(task['Class'])
    result = []
    for method in methods:
        if method['Name'] in ignore_methods:
            continue
        # else:
        pm = ParamMapping()
        pm.addMappingList('tn', task['Args'], 'pd', method['Task']['Args'])
        pm.addMappingList('tn', task['Args'],
                        'val', [task_network['ParameterGroundings'][x] for x in task['Args']])

        if method['Precondition'] is None:
            formula_value = True
        else:
            formula_value, _ = deriveValueOfFormula(pd, world_model_module, state,
                                                method['Precondition'], pm, 'pd')
        print(f'  precondition for method {method['Name']}, result: {formula_value}')
        if formula_value:
            result.append(method)
    return result


def configure_planner(world_model_module, world_model_data, config_dir_path):
    world_model_module.configure(world_model_data, config_dir_path)


# def verifyDerivedPredicateArgs(d_pred, pm, ignored_params):
#     # print('verifyDerivedPredicateArgs')
#     # print(d_pred, value_map, ignored_params)
#     for arg in d_pred['Args']:
#         # arg_name = arg['VarName']
#         arg_name = pm.mapName('pd', 'tn')

#         #arg_name = param_name_map[arg['VarName']]
#         if arg_name in ignored_params:
#             continue
#         if not arg_name in value_map:
#             raise Exception(f'No value specified for derived predicate "{d_pred['Name']}" arg '
#                                 f'named "{arg_name}", value_map: {list(value_map.keys())}, '
#                                 f'ignored: {ignored_params}')
#         # else:
#         if arg['VarType'] != value_map[arg_name].getType():
#             raise Exception()


def atomValue(world_model_module, atom_name, params, state, pm, namespace, ignored_params):
    # print(f'atomValue({atom_name}, {[param['Name'] for param in params]})')
    for state_pred in state['Predicates']:
        if state_pred['Name'] == atom_name:
            match = True
            for s_pred_arg_name, atom_arg in zip(state_pred['Args'], params):
                ignore = False
                for ign_param_ns, ign_param_name in ignored_params:
                    mapped_ign_param_name = pm.mapName(namespace, atom_arg['Name'], ign_param_ns)
                    # print(f'  mapped {namespace}, {atom_arg['Name']}, to {ign_param_ns}: {mapped_ign_param_name}')
                    # pm.printAllMappings()
                    if mapped_ign_param_name == ign_param_name:
                        ignore = True
                        break
                if ignore:
                    # print(f'  ignoring parameter {atom_arg}')
                    continue
                # else:

                # Check for type:
                # print(f'  atom_arg: {atom_arg}')
                # print(f'  {s_pred_arg_name}: {state['Values'][s_pred_arg_name]}')
                if state['Values'][s_pred_arg_name] is None:
                    raise Exception(f'Atom {atom_name} arg {s_pred_arg_name} value is None. '+\
                                    f'{state_pred['Args']}, params: {params}, atom_arg: {atom_arg}')

                if state['Values'][s_pred_arg_name].getType() != atom_arg['Data'].getType():
                    # print(json.dumps(state, indent=2, ensure_ascii=False))
                    raise Exception(f'Types mismatch for atom {atom_name}, arg {atom_arg['Name']}: '
                                    f'required {atom_arg['Data'].getType()}, but there is '
                                    f'{state['Values'][s_pred_arg_name].getType()}')
                if not world_model_module.is_value_equal(state['Values'][s_pred_arg_name],
                                                                                atom_arg['Data']):
                    match = False
                    break
            if match:
                return True
    return False

def deriveValueOfFormula(pd: PlanningDomain, world_model_module: ModuleType,
                         state: State, formula, pm: ParamMapping, param_namespace: str,
                         ignored_params: set[tuple[str, str]] = set()) -> tuple[bool, list]:
    # print(f'deriveValueOfFormula(ns={param_namespace}):')
    # print(json.dumps(formula, indent=2, ensure_ascii=False))
    # print(f'  ignored_params: {ignored_params}')
    if formula['type'] == 'existential_quantification_formula':
        # new_pm = pm.getCopy()
        # new_pm.addMapping( {'pd': [], 'tn': [], 'val': []} )

        new_ignored_params: set[tuple[str, str]] = set()
        for p in formula['Param']:
            new_ignored_params.add( (param_namespace, p['VarName']) )
        new_ignored_params = new_ignored_params.union(ignored_params)
        # print(f'  new_ignored_params: {new_ignored_params}')
        # raise
        value, atoms = deriveValueOfFormula(pd, world_model_module, state,
                        formula['Formula'], pm, param_namespace, ignored_params=new_ignored_params)
        return value, atoms
    elif formula['type'] == 'and_formula':
        # print(json.dumps(formula, indent=2, ensure_ascii=False))
        atoms = []
        for literal in formula['Literals']:
            value, sub_atoms = deriveValueOfFormula(pd, world_model_module, state, literal,
                                                pm, param_namespace, ignored_params=ignored_params)
            #sub_formulas.append(subsub_formulas)
            atoms += sub_atoms
            if not value:
                return False, atoms
        return True, atoms

    elif formula['type'] == 'atom':
        if (e_pred := pd.getExtendedPredicate(formula['Name'])) is not None:

            input_values = []
            for arg in formula['Args']:
                input_values.append( pm.mapName('pd', arg, 'val') )
            value = world_model_module.calculate_extended_predicate(state, formula['Name'],
                                                                                    input_values)
            return value, [{'atom_value': value, 'atom': formula}]

        elif (d_pred := pd.getDerivedPredicate(formula['Name'])) is not None:
            new_namespace = param_namespace+'D'
            pm = pm.getCopy()
            pm.addMappingList(param_namespace, formula['Args'],
                                            new_namespace, [x['VarName'] for x in d_pred['Args']])

            value, atoms = deriveValueOfFormula(pd, world_model_module, state, d_pred['Deriviation'],
                                        pm, new_namespace, ignored_params=ignored_params)
            return value, atoms
        else:
            atom_name = formula['Name']
            atom_params = []
            for arg in formula['Args']:
                ignore_arg = False
                for ign_arg_ns, ign_arg_name in ignored_params:
                    arg_other_ns = pm.mapName(param_namespace, arg, ign_arg_ns)
                    if arg_other_ns == ign_arg_ns:
                        ignore_arg = True
                if ignore_arg:
                    continue
                param_value = pm.mapName(param_namespace, arg, 'val')
                #print(f'  Value of param {arg} in ns {param_namespace}: {param_value}')
                #pm.printAllMappings()
                atom_params.append( {'Name': arg, 'Data': param_value} )
                #atom_params.append( {'Name': arg, 'Data': value_map[arg_name]} )

            value = atomValue(world_model_module, atom_name, atom_params, state, pm, param_namespace, ignored_params)
            return value, [{'atom_value': value, 'atom': formula}]

        raise Exception('This code is not reachable')

    elif formula['type'] == 'neg_atom':
        neg_formula = {'type':'atom', 'Name':formula['Name'], 'Args':formula['Args']}
        value, atoms = deriveValueOfFormula(pd, world_model_module, state, neg_formula,
                                                pm, param_namespace, ignored_params=ignored_params)
        #return not value, [formula, sub_formulas]
        return not value, atoms
    # else:
    raise Exception(f'Formula of type {formula['type']} is not supported')


class ParamMapping:
    def __init__(self):
        self._mapping = []

    def getCopy(self):
        result = ParamMapping()
        result._mapping = copy.copy(self._mapping)
        return result

    def addMapping(self, src_ns, src_name, dst_ns, dst_name):
        self._mapping.append( (src_ns, src_name, dst_ns, dst_name) )

    def addMappingList(self, src_ns, src_name_list, dst_ns, dst_name_list):
        assert isinstance(src_name_list, list)
        assert isinstance(dst_name_list, list)
        # print(f'addMappingList: {src_name_list}, {dst_name_list}')
        if len(src_name_list) != len(dst_name_list):
            raise Exception(f'addMappingList: sizes does not match: {src_name_list}, {dst_name_list}')

        for src_name, dst_name in zip(src_name_list, dst_name_list):
            self.addMapping(src_ns, src_name, dst_ns, dst_name)

    def printAllMappings(self):
        for item in self._mapping:
            print(f'  {item}')

    def mapName(self, src_ns, src_name, dst_ns):
        if src_ns == dst_ns:
            return src_name
        # else:
        return self._recMapName(src_ns, src_name, dst_ns, [])

    def _recMapName(self, src_ns, src_name, dst_ns, visited_ns):
        if src_ns in visited_ns:
            return None
        visited_ns = visited_ns + [src_ns]
        candidates = []
        for a_ns, a_name, b_ns, b_name in self._mapping:
            if src_ns == a_ns and src_name == a_name:
                if dst_ns == b_ns:
                    return b_name
                else:
                    candidates.append( (b_ns, b_name) )
        for ns, name in candidates:
            result = self._recMapName(ns, name, dst_ns, visited_ns)
            if not result is None:
                return result

        candidates = []
        for a_ns, a_name, b_ns, b_name in self._mapping:
            if src_ns == b_ns and src_name == b_name:
                if dst_ns == a_ns:
                    return a_name
                else:
                    candidates.append( (a_ns, a_name) )
        for ns, name in candidates:
            result = self._recMapName(ns, name, dst_ns, visited_ns)
            if not result is None:
                return result

        return None


def test_ParamMapping():
    pm = ParamMapping()

    # pm.addMapping('tn', '?obj', 'pd', '?obj')
    # pm.addMapping('tn', '?obj', 'val', {'type': 'data', 'Value': 'bowl_1', 'ValueType': 'ObjectId'})
    # pm.addMapping('pd', '?obj', 'pdD', '?object')
    # pm.addMapping('pdD', '?object', 'pdDD', '?ob')
    # pm.addMapping('pdD', '?side', 'pdDD', '?sde')
    # print(f'  ?? {pm.mapName('pdDD', '?ob', 'val')}')
    # raise

    pm.addMapping('drv2', '?o',  'drv1', '?ob')
    pm.addMapping('drv1', '?ob', 'mth',  '?obj')
    pm.addMapping('drv1', '?ob', 'mth2',  '?obj2')
    pm.addMapping('mth', '?obj', 'tn',   '?objA')
    pm.addMapping('drv2', '?sd', 'drv1', '?side')

    print(f' "?ob" ?= {pm.mapName('drv2', '?o', 'drv1')}')
    print(f' "?obj" ?= {pm.mapName('drv2', '?o', 'mth')}')
    print(f' "?objA" ?= {pm.mapName('drv2', '?o', 'tn')}')

    print(f' "?obj2" ?= {pm.mapName('drv2', '?o', 'mth2')}')

    print(f' "?obj" ?= {pm.mapName('drv1', '?ob', 'mth')}')
    print(f' "?objA" ?= {pm.mapName('drv1', '?ob', 'tn')}')

    print(f' "?objA" ?= {pm.mapName('mth', '?obj', 'tn')}')

    print(f' "?obj2" ?= {pm.mapName('tn', '?objA', 'mth2')}')

    print(f' "?obj" ?= {pm.mapName('tn', '?objA', 'mth')}')
    print(f' "?ob" ?= {pm.mapName('tn', '?objA', 'drv1')}')
    print(f' "?o" ?= {pm.mapName('tn', '?objA', 'drv2')}')

    raise

# def test_DeriveValueOfFormula():
#     pred = {'type':'atom', 'Name':'Grasped', 'Args':['?obj']}
#     value_map = {'?obj': {'ValueType': 'ObjectId', 'Value': 'bowl_1'}}
#     result = deriveValueOfFormula(pd, world_model_module, init_state, pred, value_map)
#     print(f'result: {result}')

#     pred = {'type':'atom', 'Name':'GripperFree', 'Args':['?side']}
#     value_map = {'?side': {'ValueType': 'SideDesc', 'Value': 'left'}}
#     result = deriveValueOfFormula(pd, world_model_module, init_state, pred, value_map)
#     print(f'result: {result}')

#     pred = {'type':'atom', 'Name':'GripperFree', 'Args':['?side']}
#     value_map = {'?side': {'ValueType': 'SideDesc', 'Value': 'right'}}
#     result = deriveValueOfFormula(pd, world_model_module, init_state, pred, value_map)
#     print(f'result: {result}')


def findSourceNodeId(pd: PlanningDomain, pt: ProblemTree, node_id: int,
                     bad_params: list[str]) -> list:
    verbose = False
    if verbose: print(f'findSourceNodeId({node_id}, {bad_params})')

    search_info = []

    n_id: None|int = pt.getParentNodeId(node_id)
    while not n_id is None:
        node = pt.getNode(n_id)

        bad_params_to_remove: set[str] = set()
        bad_params_to_add: set[str] = set()

        if len(node.streams) > 0:
            if verbose: print(f'  Streams: {len(node.streams)}')
            # There is a stream that generated some parameters
            first_stream = node.streams[0]
            stream_name = first_stream['Name']
            if verbose: print(f'  The first stream: {stream_name}, generated: {first_stream['OutputMapping']}')
            for param_name in bad_params:
                if param_name in first_stream['OutputMapping']:
                    # The param was generated by the stream
                    if verbose: print(f'  Found the source node for param {param_name}.')

                    # Check if the parameter is relayed
                    relayed_to_list = []
                    names_map = {}
                    for pd_name, pp_name in zip(pd.getStreamInputsNames(stream_name), first_stream['InputMapping']):
                        if verbose: print(f'  input map {pd_name} -> {pp_name}')
                        names_map[pd_name] = pp_name
                    for pd_name, pp_name in zip(pd.getStreamOutputsNames(stream_name), first_stream['OutputMapping']):
                        if verbose: print(f'  output map {pd_name} -> {pp_name}')
                        names_map[pd_name] = pp_name
                    relay = pd.getStreamRelay(stream_name)
                    for out_name, in_name in relay:
                        mapped_out_name = names_map[out_name]
                        mapped_in_name = names_map[in_name]
                        if mapped_out_name == param_name:
                            relayed_to_list.append( mapped_in_name )

                    if len(relayed_to_list) == 0:
                        # The parameter originates here.
                        # Remove the parameter from search, and save the information
                        bad_params_to_remove.add(param_name)
                        search_info.append( ('generated', n_id, stream_name, param_name) )
                    else:
                        # The parameter is relayed
                        # Remove the parameter from search, and save the information
                        bad_params_to_remove.add(param_name)
                        search_info.append( ('relay', n_id, stream_name, param_name, relayed_to_list) )
                        # Add all relayed parameters
                        bad_params_to_add = bad_params_to_add.union( set(relayed_to_list) )
            for param_name in bad_params_to_remove:
                bad_params.remove(param_name)
            for param_name in bad_params_to_add:
                bad_params.append(param_name)

        n_id = pt.getParentNodeId(n_id)

    if verbose: 
        if len(search_info) == 0:
            print('  Search info is empty')
        else:
            print('  Search info:')
            for rel in search_info:
                print(f'    {rel}')
    return search_info


# Get random element within a given probability distribution
def chooseRandomElement(elememnts: list, prob_distribution: list[float]):
    assert isinstance(elememnts, list)
    assert isinstance(prob_distribution, list)
    assert len(elememnts) > 0
    assert len(elememnts) == len(prob_distribution)
    prob_distribution_sum = 0.0
    for prob in prob_distribution:
        prob_distribution_sum += prob
    rnd_value = random.uniform(0, prob_distribution_sum)

    prob_distribution_sum = 0.0
    for elem, prob in zip(elememnts, prob_distribution):
        if rnd_value >= prob_distribution_sum and rnd_value < prob_distribution_sum + prob:
            return elem
        prob_distribution_sum += prob
    return elememnts[-1]


def run_planner(pd: PlanningDomain, world_model_module: ModuleType, pt: ProblemTree,
                max_iterations: int|None, max_planning_time: int|None, use_backtracking: bool) -> tuple[list, StatsMap]:
    assert isinstance(pd, PlanningDomain)
    assert isinstance(world_model_module, ModuleType)
    assert isinstance(pt, ProblemTree)
    assert max_iterations is None or isinstance(max_iterations, int)
    assert isinstance(use_backtracking, bool)

    if max_iterations is None and max_planning_time is None:
        raise Exception('Both max_iterations is None and max_planning_time is None')
    print('run_planner()')
    print(f'Planning domain name: "{pd.getDomainName()}"')
    print(f'Backtracking: {use_backtracking}')

    print('Initial task network:')
    root_node = pt.getNode( pt.getRootNodeId() )
    print(taskNetworkToStr(root_node.task_network, 2))

    selected_node_id = pt.getRootNodeId()

    ms_total = STATS().beginTimeMeasurement(['total_time'])

    planning_start_time = time.time()
    iteration = 0
    while True:
        iteration = iteration + 1
        if not max_iterations is None and iteration > max_iterations:
            print('Reached maximum number of iterations ({max_iterations}), stopping')
            break

        planning_time = time.time() - planning_start_time
        if not max_planning_time is None and planning_time > max_planning_time:
            print('Reached maximum planning time ({max_planning_time}), stopping')
            break

        print(f'********** Iteration of algorithm: {iteration}, time: {planning_time:.1f} **********')
        node = pt.getNode(selected_node_id)
        print(f'  Selected node id: {selected_node_id}, depth: {node.depth}')
        state = node.state
        tn = node.task_network

        STATS().add(['iteration', iteration, selected_node_id, node.depth, node.subtree_max_depth])

        print_task_network = False
        if print_task_network: print('  Processing task network:')
        if print_task_network: print(taskNetworkToStr(tn, 4))
        # print(tn)
        if isTaskNetworkEmpty(tn):
            print('    Task network is empty')
            print('Solution is found')
            # TODO: solution is found
            pt.markAsSolution(selected_node_id)
            pt.closeNode(selected_node_id)
            STATS().add(['solution_found', iteration, selected_node_id, node.depth])
            break
        elif node.streams:
            print('  MODE: stream')
            # There are active streams
            print(f'  There are active streams: {[x['Name'] for x in node.streams]}')

            first_stream = node.streams[0]
            print(f'  The first stream: {first_stream['Name']}\n'
                    f'    inputs: {first_stream['InputMapping']}\n'
                    f'    outputs: {first_stream['OutputMapping']}')
            
            # Create generator instance, if needed
            if first_stream['GeneratorInstance'] is None:
                first_stream['GeneratorInstance'] = world_model_module.create_generator( first_stream['Name'], state )

            input_values = []
            # For run-time type check
            req_input_types = pd.getStreamInputsTypes(first_stream['Name'])
            for arg, req_type_list in zip(first_stream['InputMapping'], req_input_types):
                if not arg in tn['ParameterGroundings']:
                    raise Exception(f'Unknown arg {arg}')
                # Check type (run-time)
                arg_type = tn['ParameterGroundings'][arg].getType()
                if not arg_type in req_type_list:
                    raise Exception(f'Wrong input type for stream "{first_stream['Name']}": '+\
                                    f'"{arg_type}", but expected one of "{req_type_list}"')
                # Pass value
                input_values.append( tn['ParameterGroundings'][arg].getValue() )

            ms = STATS().beginTimeMeasurement(['stream_time', first_stream['Name']])
            gen_output = first_stream['GeneratorInstance'].generate(input_values)
            STATS().endTimeMeasurement(ms)

            assert isinstance(gen_output, GeneratorResult)

            if gen_output.hasValidSample():
                # A new sample was generated
                output_values = gen_output.getSample()
                print(f'  Stream "{first_stream['Name']}" generated a new sample')

                if len(output_values) != len(first_stream['OutputMapping']):
                    raise Exception(f'Stream {first_stream['Name']} generated {len(output_values)} '
                                    f'parameters, but {len(first_stream['OutputMapping'])} expected')

                new_grounding = copy.copy(tn['ParameterGroundings'])
                req_output_types = pd.getStreamOutputsTypes(first_stream['Name'])
                for arg_idx, (arg, req_type_list) in enumerate(zip(first_stream['OutputMapping'], req_output_types)):
                    if arg in new_grounding:
                        raise Exception(f'Output parameter {arg} of stream {first_stream['Name']}'
                                                                                f' is already ground')
                    # Check type (run-time)
                    out_param_value = output_values[arg_idx]
                    assert isinstance(out_param_value, TypedValue)
                    #arg_type = output_values[arg_idx].getType()
                    if not out_param_value.getType() in req_type_list:
                        raise Exception(f'Wrong output type for stream "{first_stream['Name']}": '+\
                                        f'"{out_param_value.getType()}", but expected one of "{req_type_list}"')

                    new_grounding[arg] = out_param_value

                # If the generated sample is single, then close the current node
                if gen_output.hasSingleSample():
                    print(f'  The generated sample is single, closing node {selected_node_id}')
                    pt.closeNode(selected_node_id)
                new_tn = {'Tasks': tn['Tasks'], 'ParameterGroundings': new_grounding}

                # Can copy, as instances are created at first use
                new_node_streams = copy.copy(node.streams[1:])
                # Delete all generator instances
                for new_node_stream in new_node_streams:
                    new_node_stream['GeneratorInstance'] = None
                new_node = ProblemTree.Node(state, new_tn, new_node_streams)
                selected_node_id = pt.addNode(new_node, selected_node_id)
                print(f'  Added and selected a new node: {selected_node_id}')

                STATS().add(['stream_success', first_stream['Name'], selected_node_id])

                if gen_output.debugHasStopPlanning():
                    print('Stopping: debug stop_planning')
                    break
            else:
                # No sample was generated
                print(f'  Stream "{first_stream['Name']}" could not generate a sample')

                # Close the current node
                pt.closeNode(selected_node_id)

                # In case bactracking did not work, a random node is selected
                selected_new_node = False

                if use_backtracking:
                    # Get nodes that are source of the bad input parameters of the stream
                    bt_nodes_ids: set[int] = set()
                    if gen_output.isFailure():
                        print(f'  Stream failed')
                        bad_inputs = gen_output.getBadInputs()
                        # Backtrack: find all nodes that generated any bad parameters.
                        bad_inputs_names = [first_stream['InputMapping'][param_idx] for param_idx in bad_inputs]
                        print(f'  Bad inputs: {bad_inputs_names}')
                        search_results = findSourceNodeId(pd, pt, selected_node_id, bad_inputs_names)
                        bt_nodes_ids = set( [x[1] for x in search_results if x[0] == 'generated'] )

                        # Elaborate on failure for stats
                        # Get names of all streams that created bad parameters
                        bt_streams_names = []
                        for n_id in bt_nodes_ids:
                            n = pt.getNode(n_id)
                            bt_streams_names.append( n.streams[0]['Name'] )
                        STATS().add(['stream_failure', first_stream['Name'], selected_node_id, bt_nodes_ids, bt_streams_names])

                        node.setBacktrackSearchResult(search_results)  # For debug
                    else:
                        print(f'  Stream cannot generate more samples')
                        STATS().add(['stream_no_more', first_stream['Name'], selected_node_id])


                    # Get a list of all nodes that are good for backtracking, i.e.:
                    # - an open node
                    # - a closed node with a non-closed other subtree
                    ancestor_nodes = pt.getAncestorNodes(selected_node_id)
                    candidate_bt_nodes_ids = []
                    for n_id in bt_nodes_ids:
                        n = pt.getNode(n_id)
                        if n.isClosed():
                            for ch_id in pt.getChildrenNodesIds(n_id):
                                if ch_id in ancestor_nodes:
                                    # Skip the current subtree
                                    continue
                                # else:
                                ch = pt.getNode(ch_id)
                                if not ch.hasClosedSubtree():
                                    candidate_bt_nodes_ids.append(n_id)
                        else:
                            candidate_bt_nodes_ids.append(n_id)

                    print(f'  Nodes pointed out for backtracking: {candidate_bt_nodes_ids}')
                    # Choose one node from open nodes and with increased probability of source nodes
                    # open_bt_nodes_ids = list(bt_nodes_ids.intersection(set(open_nodes_ids)))

                    if len(candidate_bt_nodes_ids) > 0:
                        # Choose from backtrack nodes only
                        prob_distribution = [1.0] * len(candidate_bt_nodes_ids)
                        selected_n_id = chooseRandomElement(candidate_bt_nodes_ids, prob_distribution)
                        print(f'  Selected node for backtrack: {selected_n_id}')
                        # If the selected node is closed, get a random open node from its other subtree
                        if pt.getNode(selected_n_id).isClosed():
                            print('  Selected node is closed')
                            subtree_open_nodes = []
                            for ch_id in pt.getChildrenNodesIds(selected_n_id):
                                if ch_id in ancestor_nodes:
                                    # Skip the current subtree
                                    continue
                                # else:
                                subtree_open_nodes += pt.getAllOpenNodesInSubtree(ch_id)
                            if len(subtree_open_nodes) > 0:
                                prob_distribution = [1.0] * len(subtree_open_nodes)
                                selected_node_id = chooseRandomElement(subtree_open_nodes, prob_distribution)
                                selected_new_node = True
                                print(f'  Selected node from the other open subtrees of {selected_n_id}: {selected_node_id}')
                            else:
                                print(f'  Subtree of {selected_n_id} has no open nodes')
                        else:
                            selected_node_id = selected_n_id
                            selected_new_node = True
                            print('  Selected node is open')

                if not selected_new_node:
                    # No backtrack is possible, get a random open node
                    open_nodes_ids = pt.getOpenNodesIds()
                    prob_distribution = [1.0] * len(open_nodes_ids)
                    for idx, open_node_id in enumerate(open_nodes_ids):
                        prob_distribution[idx] = 0.1 + 10 * float(pt.getNodeDepth(open_node_id)) /  pt.getTreeDepth()
                    selected_node_id = chooseRandomElement(open_nodes_ids, prob_distribution)
                    print(f'  No backtrack is possible, selected a random node: {selected_node_id}')

                if gen_output.debugHasStopPlanning():
                    print('Stopping: debug stop_planning')
                    break

        elif pd.isComplexTask( (task := getFirstTask(tn)) ):
            print('  MODE: decomposition')
            print(f'  First task (complex): {taskToStr(task)}')
            checkIfTaskNetworkIsGround(tn)

            # Get applicablem ethods. Discard methods already used.
            ignore_methods = node.getAppliedMethods()
            applicable_methods = getApplicableMethods(pd, world_model_module, task, tn, state,
                                                      ignore_methods)

            print(f'  Number of methods: {len(applicable_methods)}')
            # TODO: select method:
            if len(applicable_methods) > 0:
                if len(applicable_methods) == 1:
                    # This is the last method, so close the node
                    pt.closeNode(selected_node_id)

                method = applicable_methods[0]
                print(f'  Selected method: {method['Name']}')
                node.addAppliedMethod( method['Name'] )

                STATS().add(['decomposition', iteration, selected_node_id, node.depth, method['Name'], task['Class']])

                param_name_mapping = getMappingOfMethodParameterNames(method, task, tn)
                # param_name_mapping is a map: (method param) -> (decomposed task param OR unique name)

                # Sanity check: original parameter names do not change in decomposition
                for method_task_arg, task_arg in zip(method['Task']['Args'], task['Args']):
                    assert param_name_mapping[method_task_arg] == task_arg

                # print('  Parameter mapping:')
                # for p1_name, p2_name in param_name_mapping.items():
                #     print(f'    {p1_name} -> {p2_name}')
                task_name_mapping_list = getMappingOfTaskNames(method['TaskNetwork'], tn)

                # Replace the task with task network
                new_tn = replaceTaskWithTaskNetwork(tn, task, method['TaskNetwork'],
                                                        task_name_mapping_list, param_name_mapping)

                print('  Decomposed the task, new task network:')
                print(taskNetworkToStr(new_tn, 2))

                # Initialize streams
                new_str_list = []
                for idx, stream in enumerate(method['OrderedStreams']):
                    assert stream['type'] == 'stream_ref'
                    # stream['Name']
                    # stream['Inputs']
                    # stream['Outputs']
                    input_mapping = []
                    output_mapping = []
                    for arg in stream['Inputs']:
                        if not arg in param_name_mapping:
                            raise Exception(f'Stream {stream['Name']} input arg "{arg}" is unknown.')
                        input_mapping.append( param_name_mapping[arg] )
                    for arg in stream['Outputs']:
                        # Outputs should be mapped, as they are parameters of the method
                        if not arg in param_name_mapping:
                            raise Exception(f'Stream {stream['Name']} output arg "{arg}" is unknown.')
                        output_mapping.append( param_name_mapping[arg] )

                    # A stream generates data in a context of the current state. Generated output
                    # parameters are certified, and the certification is valid in the current state
                    # only. Stream domain may use certified properties of parameters or literals
                    # of the current state. The certified properties of parameters are not part
                    # of the current state - they are relevant at stream composition phase.
                    # Domain, Certified is defined in stream_def
                    generator_instance = None
                    # Generator instance will be created later
                    new_str_list.append( {
                        'Name': stream['Name'],
                        'GeneratorInstance': generator_instance,
                        'InputMapping': input_mapping,
                        'OutputMapping':output_mapping} )
                new_node = ProblemTree.Node(state, new_tn, new_str_list)
                selected_node_id = pt.addNode(new_node, selected_node_id)
                print(f'  Added and selected a new node: {selected_node_id}')
            else:
                STATS().add(['no_more_methods', iteration, selected_node_id, node.depth, task['Class']])

                # no decomposition methods available for this task
                pt.closeNode(selected_node_id)

                # Choose random node
                open_nodes_ids = pt.getOpenNodesIds()
                if len(open_nodes_ids) > 0:
                    prob_distribution = [1.0] * len(open_nodes_ids)
                    for idx, open_node_id in enumerate(open_nodes_ids):
                        prob_distribution[idx] = 0.1 + 10 * float(pt.getNodeDepth(open_node_id)) /  pt.getTreeDepth()
                    selected_node_id = chooseRandomElement(open_nodes_ids, prob_distribution)
                else:
                    STATS().add(['no_more_open_nodes', iteration, selected_node_id, node.depth, task['Class']])
                    print(f'No more open nodes, stopping')
                    break
                #print('No more methods, stopping planning')
        else:
            print('  MODE: execution')
            print(f'  First task (primitive): {taskToStr(task)}')
            checkIfTaskNetworkIsGround(tn)

            pt.setNodeType(selected_node_id, 'execution')
            pt.closeNode(selected_node_id)

            action = pd.getAction(task['Class'])
            assert not action is None

            pm = ParamMapping()
            pm.addMappingList('tn', task['Args'], 'pd', [x['VarName'] for x in action['Args']])
            pm.addMappingList('tn', task['Args'],
                                    'val', [tn['ParameterGroundings'][x] for x in task['Args']])
            # Add mapping for constants:
            const_map = world_model_module.get_constants()
            for const_name, const_val in const_map.items():
                pm.addMappingList('tn', [const_name],
                                        'val', [const_val])
                pm.addMappingList('pd', [const_name],
                                        'tn', [const_name])

            # print('Parameter mappings:')
            # pm.printAllMappings()
            # print(json.dumps(state, indent=2, ensure_ascii=False))
            if action['Precondition'] is None:
                precondition_value = True
                atoms = None
            else:
                precondition_value, atoms = deriveValueOfFormula(pd, world_model_module,
                                                        state, action['Precondition'], pm, 'pd')
            print(f'  precondition_value: {precondition_value}')
            if precondition_value:
                new_state = updateState(state, world_model_module, action['Effect'], pm)
                new_tn = removeTask(tn, task)
                new_str_list = []
                new_node = ProblemTree.Node(new_state, new_tn, new_str_list)
                selected_node_id = pt.addNode(new_node, selected_node_id)
                print(f'  Added and selected a new node: {selected_node_id}')

                STATS().add(['execution', iteration, selected_node_id, node.depth, action['Name']])

                #print(f'Executed action')

            else:
                print('state **********************************************')
                #print(json.dumps(state, indent=2, ensure_ascii=False))
                #print(state.keys())
                #'Predicates', 'Values'
                for pred in state['Predicates']:
                    print(f'  {pred['Name']}')
                    for param_name in pred['Args']:
                        val = state['Values'][param_name]
                        print(f'    {val.getValue()}')

                print('atoms **********************************************')
                # print(json.dumps(atoms, indent=2, ensure_ascii=False))
                assert not atoms is None
                atoms_values = []
                for atom in atoms:
                    # print (atom)
                    mapped_args = []
                    for arg in atom['atom']['Args']:
                        mapped_args.append( pm.mapName('pd', arg, 'val') )
                        # mapped_args.append( value_map[param_name_map[arg]] )
                    mapped_atom = {'Name': atom['atom']['Name'], 'Args': mapped_args}
                    atoms_values.append({'atom_value': atom['atom_value'], 'atom':mapped_atom})
                #print(json.dumps(atoms_values, indent=2, ensure_ascii=False))
                for atom_val in atoms_values:
                    print(f'  {atom_val['atom']['Name']} = {atom_val['atom_value']}')
                    for atom_param_value in atom_val['atom']['Args']:
                        if atom_param_value is None:
                            print(f'    None')
                        else:
                            print(f'    {atom_param_value.getValue()}')

                STATS().add(['execution_denied', iteration, selected_node_id, node.depth, action['Name']])

                # print('value_map **********************************************')
                # print(json.dumps(value_map, indent=2, ensure_ascii=False))
                raise Exception(f'Primitive task precondition is {precondition_value}')

    STATS().endTimeMeasurement(ms_total)

    plans = []
    if pt.hasSolution():
        for solution_node_id in pt.getAllSolutionNodesIds():
            plan = getActionsSequence(pd, pt, solution_node_id)
            plans.append(plan)
            print(f'Generated plan of length {len(plan)}')
            for task, param_value_list in plan:
                print(f'  ({task['Class']} {task['Args']})')

            actions = [task['Class'] for task, _ in plan]
            STATS().add(['generated_plan', solution_node_id, len(plan), actions])

    stats = prepareStats()
    print(json.dumps(stats.toJson(), indent=2, ensure_ascii=False))

    return plans, stats
