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

class PlanningDomain:
    def __init__(self, planning_domain):
        self._planning_domain = planning_domain

        self._extended_pred_map = {}
        for e_pred in self._planning_domain['ExtendedPredicates']:
            # Check for doubles:
            assert not e_pred['Name'] in self._extended_pred_map
            self._extended_pred_map[e_pred['Name']] = e_pred

        self._derived_pred_map = {}
        for d_pred in self._planning_domain['DerivedPredicates']:
            # Check for doubles:
            assert not d_pred['Name'] in self._derived_pred_map
            self._derived_pred_map[ d_pred['Name'] ] = d_pred

        self._actions_map = {}
        for action in self._planning_domain['Actions']:
            # Check for doubles:
            assert not action['Name'] in self._actions_map
            self._actions_map[ action['Name'] ] = action

        # Create a map: type name -> matching types (i.e. the type and all its descendants)
        self._matching_types_map: dict[str, list[str]] = {}
        for tp, tp_parent in self._planning_domain['Types']:
            if not tp in self._matching_types_map:
                self._matching_types_map[tp] = [tp]
            if not tp_parent in self._matching_types_map:
                self._matching_types_map[tp_parent] = [tp_parent]
            if not tp in self._matching_types_map[tp_parent]:
                self._matching_types_map[tp_parent].append( tp )

        # Create a map of complex tasks and check:
        # - if each task is declared once
        # - if each parameter name is unique
        # - if each parameter type is known
        self._complex_tasks_map: dict[str, Any] = {}
        for task in self._planning_domain['Tasks']:
            assert task['type'] == 'task_def'
            task_class = task['Name']
            # Each task is declared once
            if task_class in self._complex_tasks_map:
                raise Exception(f'Duplicate definiction of complex task "{task_class}"')
            self._complex_tasks_map[task_class] = task
            arg_names: set[str] = set()
            for arg in task['Args']:
                arg_name = arg['VarName']
                # Parameter name is unique
                if arg_name in arg_names:
                    raise Exception(f'Duplicated parameter name "{arg_name}" in complex task "{task_class}"')
                # else:
                arg_names.add(arg_name)
                type_name = arg['VarType']
                if not type_name in self._matching_types_map:
                    raise Exception(f'Unknown type "{type_name}" of arg "{arg_name}" '+\
                                    f'of complex task "{task_class}"')

        # Create a map of primitive tasks and check:
        # - if each task is declared once
        # - if each parameter name is unique
        # - if each parameter type is known
        self._primitive_tasks_map: dict[str, Any] = {}
        for task in self._planning_domain['Actions']:
            assert task['type'] == 'action'
            task_class = task['Name']
            # Each task is declared once
            if task_class in self._primitive_tasks_map:
                raise Exception(f'Duplicate definiction of action "{task_class}"')
            self._primitive_tasks_map[task_class] = task
            arg_names: set[str] = set()
            for arg in task['Args']:
                arg_name = arg['VarName']
                # Parameter name is unique
                if arg_name in arg_names:
                    raise Exception(f'Duplicated parameter name "{arg_name}" in action "{task_class}"')
                # else:
                arg_names.add(arg_name)
                type_name = arg['VarType']
                if not type_name in self._matching_types_map:
                    raise Exception(f'Unknown type "{type_name}" of arg "{arg_name}" '+\
                                    f'of action "{task_class}"')

            # TODO: check precondition and effect
#   "Actions": [
#     {
#       "type": "action",
#       "Name": "pMvHead",
#       "Args": [
#         {
#           "type": "typed_variable",
#           "VarName": "?q1",
#           "VarType": "HeadConf"
#         },
#         {
#           "type": "typed_variable",
#           "VarName": "?q2",
#           "VarType": "HeadConf"
#         }
#       ],
#       "Precondition": {
#         "type": "atom",
#         "Name": "AtConfH",
#         "Args": [
#           "?q1"
#         ]
#       },

        # Create a map of streams and check:
        # - if each streams is declared once
        # - if each parameter name is unique
        # - if each parameter type is known
        # - check formulas for domain and certified
        self._streams_map: dict[str, Any] = {}
        for stream_def in self._planning_domain['Streams']:
            assert stream_def['type'] == 'stream_def'
            stream_name = stream_def['Name']
            if stream_name in self._streams_map:
                raise Exception(f'Stream "{stream_name}" is declared twice')
            self._streams_map[stream_name] = stream_def

            # Check inputs
            input_names: set[str] = set()
            for param in stream_def['Inputs']:
                param_name = param['VarName']
                if param_name in input_names:
                    raise Exception(f'Stream "{stream_name}" input parameter '+\
                                    f'"{param_name}" is declared twice')
                param_type = param['VarType']
                if not param_type in self._matching_types_map:
                    raise Exception(f'Stream "{stream_name}" input parameter '+\
                                    f'"{param_name}" is of unknown type "{param_type}')
            # Check outputs
            output_names: set[str] = set()
            for param in stream_def['Outputs']:
                param_name = param['VarName']
                if param_name in output_names:
                    raise Exception(f'Stream "{stream_name}" output parameter '+\
                                    f'"{param_name}" is declared twice')
                if param_name in input_names:
                    raise Exception(f'Stream "{stream_name}" output parameter '+\
                                    f'"{param_name}" is declared as input')
                param_type = param['VarType']
                if not param_type in self._matching_types_map:
                    raise Exception(f'Stream "{stream_name}" output parameter '+\
                                    f'"{param_name}" is of unknown type "{param_type}')

            # TODO: check "domain"
            stream_def['Domain']
            # TODO: check "certified"
            stream_def['Certified']

        self._checkMethods()
        # TODO: verify integrity of planning domain:
        # self._everyTaskHasMethods()
        # self._everyMethodIsForTask()
        # self._everyTaskNetworkIsTotallyOrdered()
        # self._everyTaskOrActionIsKnown()
        # self._everyTypeIsKnown()
        # self._taskParametersMatch()
        # self._actionParametersMatch()
        # Check if stream_ref outputs conflict with method parameters (overwrite)
        # Check if data types match in method: method parameters - tasks,
        #   method parameters - streams

        # Rules (?)
        # task ?t match method ?m <=> t.Class == m.task
        # for each action: for each arg: type of arg is known

    def getStreamRelay(self, name: str) -> list[tuple[str, str]]:
        result = []
        for rel_item in self._streams_map[name]['Relay']:
            result.append( (rel_item['FromOut'], rel_item['ToIn']) )
        return result

    def getStreamInputsNames(self, name: str) -> list[str]:
        return [x['VarName'] for x in self._streams_map[name]['Inputs']]

    def getStreamOutputsNames(self, name: str) -> list[str]:
        return [x['VarName'] for x in self._streams_map[name]['Outputs']]

    def _hasComplexTask(self, task_class: str):
        return task_class in self._complex_tasks_map

    def _getTaskArgTypes(self, task_class: str) -> list[str]:
        if task_class in self._complex_tasks_map:
            task = self._complex_tasks_map[task_class]
            return [arg['VarType'] for arg in task['Args']]
        elif task_class in self._primitive_tasks_map:
            task = self._primitive_tasks_map[task_class]
            return [arg['VarType'] for arg in task['Args']]
        else:
            raise Exception(f'Unknown task or action class "{task_class}"')

    def _checkComplexTask(self, err_prefix: str, task_class: str, task_arg_names: list[str],
                          parameters_types_map: dict[str, str], allow_derived_types: bool) -> None:
        if not self._hasComplexTask(task_class):
            raise Exception(f'{err_prefix}: unknown task "{task_class}"')
        # else:

        self._checkTask(err_prefix, task_class, task_arg_names, parameters_types_map,
                        allow_derived_types)

    def _checkTask(self, err_prefix: str, task_class: str, task_arg_names: list[str],
                   parameters_types_map: dict[str, str], allow_derived_types: bool) -> None:
        # Check if number of task parameters match
        task_arg_types = self._getTaskArgTypes(task_class)
        if len(task_arg_names) != len(task_arg_types):
            raise Exception(f'{err_prefix}: wrong number of parameters for '+\
                            f'task "{task_class}": {len(task_arg_names)}, '+\
                            f'expected {len(task_arg_types)}')
        for arg_name, task_def_arg_type in zip(task_arg_names, task_arg_types):
            # Check if parameter is declared
            if not arg_name in parameters_types_map:
                raise Exception(f'{err_prefix}: parameter "{arg_name}" is not declared')
            
            # Check if type of the parameter is compliant to task definition
            arg_type = parameters_types_map[arg_name]
            if allow_derived_types:
                if not self.typeMatch(arg_type, task_def_arg_type):
                    raise Exception(f'{err_prefix}: arg "{arg_name}" '+\
                                f'type "{arg_type}" of task "{task_class}" '+\
                                f'does not match required type (or its descendants) "{task_def_arg_type}"')
            else:
                if arg_type != task_def_arg_type:
                    raise Exception(f'{err_prefix}: arg "{arg_name}" '+\
                                    f'type "{arg_type}" of task "{task_class}" '+\
                                    f'does not match required type "{task_def_arg_type}"')

    def _getStreamInputTypes(self, stream_name: str) -> list[str]:
        assert stream_name in self._streams_map
        return [param['VarType'] for param in self._streams_map[stream_name]['Inputs']]

    def _getStreamOutputTypes(self, stream_name: str) -> list[str]:
        assert stream_name in self._streams_map
        return [param['VarType'] for param in self._streams_map[stream_name]['Outputs']]

    def _checkMethods(self) -> None:
        method_names: set[str] = set()
        for method in self._planning_domain['Methods']:
            assert method['type'] == 'method'

            # Method name is unique
            method_name = method['Name']
            if method_name in method_names:
                raise Exception(f'Duplicated method name: "{method_name}"')
            # else:
            method_names.add( method_name )

            # Detect unused parameters of method
            # A parameter is used, if:
            # - it is a parameter of any task of the method task network
            # - it is an input of any stream
            used_parameters: set[str] = set()

            # Check if:
            # - all types are known
            # - all parameter names are unique
            method_parameters_types_map: dict[str, str] = {}
            method_parameters = method['Parameters']

            for param in method_parameters:
                param_name = param['VarName']
                param_type = param['VarType']
                # Parameter name is unique
                if param_name in method_parameters_types_map:
                    raise Exception(f'Duplicated parameter: "{param_name}" in method "{method_name}"')
                # else:
                # Parameter type is known
                if not param_type in self._matching_types_map:
                    raise Exception(f'Method "{method_name}" parameter "{param_name}" type '+\
                                    f'"{param_type}" is unknown')
                # Save parameter name and its type
                method_parameters_types_map[param_name] = param_type

            # Check if:
            # - the task is known
            # - all parameters are known
            # - number of task parameters match
            # - types of parameters are the same as declared in method definition
            task_class = method['Task']['Class']
            task_arg_names = method['Task']['Args']
            self._checkComplexTask(f'Method "{method_name}", decomposed task', task_class,
                                   task_arg_names, method_parameters_types_map, False)

            # TODO: check precondition
            # Check if:
            # - all atoms are known
            # - all parameters are known
            # - all parameters are of types matching parameters of atoms
            method['Precondition']

            subtask_ids: set[str] = set()
            for subtask in method['TaskNetwork']['Tasks']:
                assert subtask['type'] == 'task_ref'
                # Subtask id is None or is unique
                subtask_id = subtask['Id']
                if not subtask_id is None:
                    if subtask_id in subtask_ids:
                        raise Exception(f'Duplicated subtask id: "{subtask_id}"')
                    # else:
                    subtask_ids.add( subtask_id )

                # Save parameters of the task as used parameters
                for arg_name in subtask['Args']:
                    used_parameters.add( arg_name )

                # Check each subtask
                subtask_class = subtask['Class']
                subtask_arg_names = subtask['Args']
                self._checkTask(f'Method "{method_name}", subtask "{subtask_id}/{subtask_class}"',
                                    subtask_class, subtask_arg_names, method_parameters_types_map, True)

            # Check if:
            # - all streams are known
            # - all parameters are known
            # - all parameters are of types matching parameters of atoms
            # - all inputs are ground
            # - the final task network is ground
            # At first, only parameters of decomposed task are ground
            ground_parameters: set[str] = set( task_arg_names )
            ordered_streams = method['OrderedStreams']
            for stream_ref in ordered_streams:
                assert stream_ref['type'] == 'stream_ref'
                stream_name = stream_ref['Name']

                # Check if a given stream exists
                if not stream_name in self._streams_map:
                    raise Exception(f'Method "{method_name}" uses unknown stream "{stream_name}"')

                # Check if number of inputs match to stream definition
                def_stream_inputs_types = self._getStreamInputTypes(stream_name)
                stream_inputs = stream_ref['Inputs']
                if len(stream_inputs) != len(def_stream_inputs_types):
                    raise Exception(f'Method "{method_name}" stream "{stream_name}": '+\
                                        f'wrong number of inputs {len(stream_inputs)}, '+\
                                        f'expected {len(def_stream_inputs_types)}')

                # Check each input
                for input_name, req_input_type in zip(stream_inputs, def_stream_inputs_types):
                    # Check if parameters are known
                    if not input_name in method_parameters_types_map:
                        raise Exception(f'Method "{method_name}" stream "{stream_name}" '+\
                                        f'input "{input_name}" is unknown')
                    # Check if parameter type is compliant with stream definition
                    param_type = method_parameters_types_map[input_name]
                    if not self.typeMatch(param_type, req_input_type):
                        raise Exception(f'Method "{method_name}" stream "{stream_name}" '+\
                                        f'input "{input_name}" has wrong type: {param_type}, '+\
                                        f'expected: {req_input_type}')

                    # Save input of the stream as used parameter
                    used_parameters.add( input_name )

                    # TODO
                    # Check if input parameter is ground
                    if not input_name in ground_parameters:
                        raise Exception(f'Method "{method_name}" stream "{stream_name}" '+\
                                        f'input "{input_name}" is not ground')

                # Check if number of outputs match to stream definition
                def_stream_outputs_types = self._getStreamOutputTypes(stream_name)
                stream_outputs = stream_ref['Outputs']
                if len(stream_outputs) != len(def_stream_outputs_types):
                    raise Exception(f'Method "{method_name}" stream "{stream_name}": '+\
                                        f'wrong number of inputs {len(stream_outputs)}, '+\
                                        f'expected {len(def_stream_outputs_types)}')

                for output_name in stream_outputs:
                    # Check if parameters are known
                    if not output_name in method_parameters_types_map:
                        raise Exception(f'Method "{method_name}" stream "{stream_name}" '+\
                                        f'output "{output_name}" is unknown')
                    # Check if parameter type is compliant with stream definition
                    param_type = method_parameters_types_map[output_name]
                    # TODO
                    # Check if output parameter is not ground (double assign is forbidden)
                    if output_name in ground_parameters:
                        raise Exception(f'Method "{method_name}" stream "{stream_name}" '+\
                                        f'output "{output_name}" is already ground')
                    ground_parameters.add( output_name )

            # Finally, print warnings about unused parameters
            for param in method_parameters:
                param_name = param['VarName']
                if not param_name in used_parameters:
                    print(f'WARNING: method {method_name} has an unused parameter: {param_name}')


    def getDomainName(self):
        return self._planning_domain['DomainName']

    def isComplexTask(self, task):
        task_class = task['Class']
        for task in self._planning_domain['Tasks']:
            assert task['type'] == 'task_def'
            if task['Name'] == task_class:
                return True
        for task in self._planning_domain['Actions']:
            assert task['type'] == 'action'
            if task['Name'] == task_class:
                return False
        # else:
        raise Exception(f'Unknown task class: {task_class}')

    def getMethodsForTaskClass(self, task_class: str):
        result = []
        for method in self._planning_domain['Methods']:
            if method['Task']['Class'] == task_class:
                result.append(method)
        return result

    def getDerivedPredicate(self, name):
        return self._derived_pred_map[name] if name in self._derived_pred_map else None

    def getExtendedPredicate(self, name):
        return self._extended_pred_map[name] if name in self._extended_pred_map else None

    def getAction(self, class_name):
        return self._actions_map[class_name] if class_name in self._actions_map else None

    def getStreamInputsTypes(self, stream_name: str) -> list[str]:
        result = []
        for stream_def in self._planning_domain['Streams']:
            if stream_def['Name'] == stream_name:
                for inp in stream_def['Inputs']:
                    result.append( self._matching_types_map[inp['VarType']] )
        return result

    def getStreamOutputsTypes(self, stream_name: str) -> list[str]:
        result = []
        for stream_def in self._planning_domain['Streams']:
            if stream_def['Name'] == stream_name:
                for inp in stream_def['Outputs']:
                    result.append( self._matching_types_map[inp['VarType']] )
        return result

    def typeMatch(self, type_name: str, req_type_name: str) -> bool:
        return type_name in self._matching_types_map[req_type_name]

    def getComplexTasksClasses(self) -> list[str]:
        out = []
        for task_def in self._planning_domain['Tasks']:
            out.append( task_def['Name'] )
        return out
    
    def getComplexTaskDef(self, task_class: str) -> dict[str, Any]:
        return self._complex_tasks_map[task_class]

