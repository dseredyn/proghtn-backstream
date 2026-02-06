# -*- coding: utf-8 -*-

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


from .ebnf import VisitorAbstract
import copy

class VisitorHSDDL(VisitorAbstract):
    def visit_any_ext_name(self, args):
        # print('visit_any_ext_name({})'.format(args[0]))
        return args[0]

    def visit_ext_name(self, args):
        # print('visit_ext_name({})'.format(args[1]))
        return args[1]

    def visit_any_name(self, args):
        # print('visit_any_name({})'.format(args[0]))
        return args[0]

    def visit_name(self, args):
        # print('visit_name({})'.format(args[1]))
        return args[1]

    def visit_domain_name(self, args):
        # print('visit_domain_name({})'.format(args[4]))
        return args[4]

    def visit_parent_type(self, args):
        # print('visit_parent_type({})'.format(args[2]))
        return args[2]

    def visit_var_name(self, args):
        # print('visit_var_name({})'.format(args[2]))
        return '?'+args[2]

    def visit_method_task(self, args):
        # print('visit_method_task({}, {})'.format(args[4], args[5]))
        # TODO: task_name, args
        # print(args[5])
        return {'Class': args[4], 'Args': args[5]}

    def visit_var_const(self, args):
        #print('visit_var_const({})'.format(args[1][0][0]))
        return args[1][0][0]

    def visit_type_def(self, args):
        # print('{}, {}'.format(args[1], args[3]))
        # There can be many types
        # There can be 0 or 1 parent type
        types_list = []
        if len(args[3]) == 0:
            parent_type = None
        else:
            parent_type = args[3][0]
        for type_name in args[1]:
            types_list.append( (type_name, parent_type) )
        return types_list

    def visit_types(self, args):
        types_list = []
        for sub_list in args[4]:
            types_list = types_list + sub_list
        # print('visit_types()')
        # print(types_list)
        return types_list

    def visit_constants(self, args):
        constant_list = []
        for sub_list in args[4]:
            constant_list = constant_list + sub_list
        # print('visit_constants()')
        # print(constant_list)
        return constant_list

    def visit_typed_var(self, args):
        result = []
        for var_name in args[0]:
            result.append(
                {
                    'type': 'typed_variable',
                    'VarName': var_name,
                    'VarType': args[3]} )
        # print('visit_typed_var()')
        # print(result)
        return result

    def visit_predicate(self, args):
        pred_args = []
        for arg_list in args[3]:
            for a in arg_list:
                pred_args.append(a)
        result = {'type':'predicate', 'Name':args[2], 'Args':pred_args}
        # print('visit_predicate({})'.format(result))
        return result

    def visit_com_pred(self, args):
        # print('visit_predicates({})'.format(args[4]))
        return args[4]

    def visit_ext_pred(self, args):
        # print('visit_predicates({})'.format(args[4]))
        return args[4]

    def visit_parameters(self, args):
        result = []
        for arg_list in args[4]:
            for a in arg_list:
                result.append(a)
        # print('visit_parameters({})'.format(result))
        return result

    def visit_task_def(self, args):
        result = {'type':'task_def', 'Name':args[4], 'Args':args[5]}
        # print('visit_task({})'.format(result))
        return result

    def visit_action(self, args):
        if len(args[6]) == 0:
            precondition = None
        else:
            assert len(args[6]) == 1
            precondition = args[6][0]
        if len(args[7]) == 0:
            effect = None
        else:
            assert len(args[7]) == 1
            effect = args[7][0]
        result = {'type':'action', 'Name':args[4], 'Args':args[5], 'Precondition':precondition,
                                                                                'Effect':effect}
        # print('visit_action({})'.format(result))
        return result

    def visit_atom(self, args):
        result = {'type':'atom', 'Name':args[3], 'Args':args[4]}
        #print('visit_atom({})'.format(result))
        return result

    def visit_neg_atom(self, args):
        result = {'type':'neg_atom', 'Name':args[5]['Name'], 'Args':args[5]['Args']}
        # print('visit_neg_atom({})'.format(result))
        return result

    def visit_literal(self, args):
        result = args[0][0][0]
        # print('visit_literal({})'.format(result))
        return result

    def visit_and_formula(self, args):
        return {
            'type': 'and_formula',
            'Literals': args[4]}

    def visit_formula(self, args):
        assert len(args) == 1
        assert len(args[0]) == 1
        if isinstance(args[0][0], list):
            assert not args[0][0]
            result = None
        else:
            result = args[0][0]
        # print('visit_formula( {} )'.format(result))
        return result

    def visit_empty_block(self, args):
        return []

    def visit_precondition(self, args):
        # print('visit_precondition( {} )'.format(args[3]))
        return args[3]

    def visit_effect(self, args):
        return args[3]

    def visit_inputs_ref(self, args):
        # print('visit_inputs_ref( {} )'.format(args[4]))
        return args[4]

    def visit_outputs_ref(self, args):
        # print('visit_outputs_ref( {} )'.format(args[4]))
        return args[4]

    def visit_stream_ref(self, args):
        if len(args[3]) == 0:
            inputs_ref = []
        else:
            assert len(args[3]) == 1
            inputs_ref = args[3][0]

        if len(args[4]) == 0:
            outputs_ref = []
        else:
            assert len(args[4]) == 1
            outputs_ref = args[4][0]
        result = {'type':'stream_ref', 'Name':args[2], 'Inputs':inputs_ref, 'Outputs':outputs_ref}
        # print('visit_stream( {} )'.format(result))
        return result

    def visit_and_streams(self, args):
        # print('visit_and_streams( {} )'.format(args[4]))
        return args[4]

    def visit_ordered_streams(self, args):
        # print('visit_ordered_streams( {} )'.format(args[2]))
        return args[2]

    def visit_task_ref(self, args):
        return {
            'type':'task_ref',
            'Id':None,
            'Class':args[2],
            'Args':args[3]}

    def visit_named_task(self, args):
        return {
            'type':'task_ref',
            'Id': args[3],
            'Class':args[4]['Class'],
            'Args':args[4]['Args']}

    def visit_opt_named_task(self, args):
        assert len(args[0]) == 1
        # print('visit_opt_named_task( {} )'.format(args[0][0]))
        return args[0][0]

    def visit_and_tasks(self, args):
        # assert len(args[4]) == 1
        # print('visit_and_tasks( {} )'.format(args[4]))
        return args[4]

    def visit_ordered_subtasks(self, args):
        assert len(args[2]) == 1
        assert len(args[2][0]) == 1
        # print('visit_ordered_subtasks( {} )'.format(args[2][0][0]))
        return args[2][0][0]

    def visit_task_network(self, args):
        return {'Tasks': args[0]}

    def visit_method(self, args):
        if len(args[7]) == 0:
            precondition = None
        else:
            assert len(args[7]) == 1
            precondition = args[7][0]
        if len(args[9]) == 0:
            ordered_streams = []
        else:
            assert len(args[9]) == 1
            ordered_streams = args[9][0]
        result = {'type':'method', 'Name':args[4], 'Parameters':args[5], 'Task':args[6],
                                        'Precondition':precondition, 'TaskNetwork':args[8],
                                                            'OrderedStreams': ordered_streams}
        # print('visit_method( {} )'.format(result))
        return result

    def visit_stream_def(self, args):
        if len(args[5]) == 0:
            inputs = []
        else:
            assert len(args[5]) == 1
            inputs = args[5][0]

        if len(args[6]) == 0:
            domain = None
        else:
            assert len(args[6]) == 1
            domain = args[6][0]

        if len(args[7]) == 0:
            outputs = []
        else:
            assert len(args[7]) == 1
            outputs = args[7][0]

        if len(args[8]) == 0:
            certified = None
        else:
            assert len(args[8]) == 1
            certified = args[8][0]

        if len(args[9]) == 0:
            relay = []
        else:
            assert len(args[9]) == 1
            relay = args[9][0]

        result = {'type':'stream_def', 'Name':args[4], 'Inputs':inputs, 'Domain':domain,
                    'Outputs':outputs, 'Certified':certified, 'Relay': relay}
        # print('visit_stream_def( {} )'.format(result))
        return result

    def visit_inputs_def(self, args):
        result = []
        for v in args[4]:
            result = result + v
        # print('visit_inputs_def( {} )'.format(result))
        return result

    def visit_outputs_def(self, args):
        result = []
        for v in args[4]:
            result = result + v
        # print('visit_outputs_def( {} )'.format(result))
        return result

    def visit_str_domain(self, args):
        return args[3]

    def visit_str_certified(self, args):
        return args[3]

    def visit_relay_def(self, args):
        return args[4]
    
    def visit_relay_item(self, args):
        return {
            'type': 'relay_item',
            'FromOut': args[1],
            'ToIn': args[5]
        }

    def visit_tmas(self, args):
        assert len(args) == 1
        assert len(args[0]) == 1
        return args[0][0]

    def visit_derived_predicate(self, args):
        # print('visit_derived_predicate')
        return {
            'type': 'derived_predicate',
            'Name': args[4],
            'Args': args[5],
            'Deriviation': args[6],
            }

    def visit_derivation(self, args):
        # print('visit_derivation')
        assert len(args[3]) == 1
        assert len(args[3][0]) == 1
        return args[3][0][0]

    def visit_existential_quant_formula(self, args):
        assert len(args[6]) == 1
        return {
            'type': 'existential_quantification_formula',
            'Param': args[6][0],
            'Formula': args[9]
        }

    def visit_planning_domain(self, args):
        if len(args[6]) == 0:
            types = []
        else:
            assert len(args[6]) == 1
            types = args[6][0]

        if len(args[7]) == 0:
            constants = []
        else:
            assert len(args[7]) == 1
            constants = args[7][0]

        if len(args[8]) == 0:
            com_pred = []
        else:
            assert len(args[8]) == 1
            com_pred = args[8][0]

        if len(args[9]) == 0:
            ext_pred = []
        else:
            assert len(args[9]) == 1
            ext_pred = args[9][0]

        derived_pred = args[10]

        task_list = []
        action_list = []
        method_list = []
        stream_list = []
        for tmas in args[11]:
            if tmas['type'] == 'task_def':
                task_list.append(tmas)
            elif tmas['type'] == 'action':
                action_list.append(tmas)
            elif tmas['type'] == 'method':
                method_list.append(tmas)
            elif tmas['type'] == 'stream_def':
                stream_list.append(tmas)
            else:
                raise Exception('Unsupported type: "{}"'.format(tmas['type']))

        result = {'DomainName': args[4], 'Types': types, 'Constants': constants,
                    'CommonPredicates': com_pred, 'ExtendedPredicates': ext_pred, 'DerivedPredicates': derived_pred,
                    'Tasks': task_list,
                    'Actions': action_list, 'Methods': method_list, 'Streams': stream_list}
        return result
