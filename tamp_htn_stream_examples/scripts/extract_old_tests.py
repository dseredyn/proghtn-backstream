#!/usr/bin/env python3

# TODO: add author and license

# Decription: this script extracts input data (world state, task network) from the older version
# of planner suite, of test cases presented in the article

# Usage:
# ros2 run tamp_htn_stream_examples extract_old_tests.py


from __future__ import annotations

import argparse
import sys

import random
import math
import time
import copy
import json

import PyKDL

import xml.etree.ElementTree as ET

class DataXmlParser:
    def __init__(self, root_output_mode):
        self._root_output_mode = root_output_mode

    def parseDocument(self, path):
        # self._path = path
        tree = ET.parse(path)
        root = tree.getroot()
        data = self._parse(root, self._root_output_mode)
        for attrib_name, attrib_value in root.attrib.items():
            data[attrib_name] = attrib_value
        return data

    def _parse(self, elem, output_mode):
        result = {}
        for ch in elem:
            if hasattr(self, f'parse_{ch.tag}'):
                parse_method = getattr(self, f'parse_{ch.tag}')
                key_value = parse_method(ch)
                if not key_value is None:
                    if output_mode == 'list':
                        key, value = key_value
                        if not key in result:
                            result[key] = []
                        result[key].append(value)
                    elif output_mode == 'dict':
                        key, value = key_value
                        result[key] = value
                    else:
                        raise Exception('Unknown output_mode: "{}"'.format(output_mode))
            else:
                raise Exception('Unknown tag: "{}"'.format(ch.tag))
        return result

    def parse_GeomPose(self, elem):
        return 'Pose', {
            'p': {
                'x': elem.attrib['tx'],
                'y': elem.attrib['ty'],
                'z': elem.attrib['tz'],
            },
            'M': {
                'x': elem.attrib['qx'],
                'y': elem.attrib['qy'],
                'z': elem.attrib['qz'],
                'w': elem.attrib['qw'],
            }
        }

    def parse_ObjectId(self, elem):
        return 'ObjectId', elem.text

    def parse_NoValue(self, elem):
        return 'NoValue', None

    def parse_SideDesc(self, elem):
        return 'Side', elem.text

    def parse_collisionContainer(self, elem):
        return 'CollisionContainer', self._parse(elem, 'list')['collision']

    def parse_collision(self, elem):
        data = self._parse(elem, 'dict')
        data['Name'] = elem.attrib['name']
        return 'collision', data

    def parse_pose(self, elem):
        items = elem.text.split()
        qx, qy, qz, qw = PyKDL.Rotation.RPY(float(items[3]), float(items[4]), float(items[5])).GetQuaternion()
        data = {
            'Pose': {
                'p': {
                    'x': items[0],
                    'y': items[1],
                    'z': items[2],
                },
                'M': {
                    'x': qx,
                    'y': qy,
                    'z': qz,
                    'w': qw
                }
                # 'RPY': {
                #     'r': items[3],
                #     'p': items[4],
                #     'y': items[5]
                # }
            }
        }
        if 'frame' in elem.attrib:
            data['Frame'] = elem.attrib['frame']
        return 'Pose', data

    def parse_geometry(self, elem):
        return 'Geometry', self._parse(elem, 'dict')

    def parse_box(self, elem):
        return 'Box', self._parse(elem, 'dict')

    def parse_size(self, elem):
        items = elem.text.split()
        return 'Size', {'x':items[0], 'y':items[1], 'z':items[2]}


class WorldModelXmlParser(DataXmlParser):
    def __init__(self):
        super().__init__('dict')

    def parseDocument(self, path):
        self.model_ids = set()
        result = super().parseDocument(path)
        obj_model_map = {}
        for obj_inst in result['ObjectInstances']:
            if 'ObjectModelId' in obj_inst['ObjectInstance']:
                obj_model_map[obj_inst['ObjectInstance']['ObjectId']] = obj_inst['ObjectInstance']['ObjectModelId']

        for item in result['WorldState']:
            if 'PhysicalObjectState' in item:
                obj_id_str = item['PhysicalObjectState']['ObjectId']
                if obj_id_str in obj_model_map:
                    item['PhysicalObjectState']['ObjectModelId'] = obj_model_map[obj_id_str]
        return result

    def parse_FakePerceptionKnowledge(self, elem):
        data = self._parse(elem, 'dict')
        if data:
            return 'WorldStateSource', self._parse(elem, 'dict')
        else:
            return None

    def parse_GazeboWorldPath(self, elem):
        return 'GazeboWorld', self._parse(elem, 'dict')['RosPath']

    def parse_RosPath(self, elem):
        assert elem.attrib['package'] == "roko_nodes"
        assert elem.attrib['path'].startswith('data/gazebo/worlds')
        print(f'Using file: package://{elem.attrib['package']}/{elem.attrib['path']}')
        return 'RosPath', {'package': elem.attrib['package'], 'path': elem.attrib['path']}

    def parse_FakePerceptionModel(self, elem):
        return None

    def parse_ObjectModelId(self, elem):
        self.model_ids.add(elem.text)
        return 'ObjectModelId', elem.text

    def parse_ObjectInstance(self, elem):
        return 'ObjectInstance', self._parse(elem, 'dict')

    def parse_ObjectInstances(self, elem):
        data = self._parse(elem, 'list')['ObjectInstance']
        object_instances = []
        for obj_inst in data:
            object_instances.append( {'ObjectInstance': obj_inst} )
        return 'ObjectInstances', object_instances

    def parse_WorldState(self, elem):
        return 'WorldState', self._parse(elem, 'dict')['DataStorage']['DataValue']

    def parse_DataStorage(self, elem):
        return 'DataStorage', self._parse(elem, 'list')

    def parse_Comment(self, elem):
        lines = []
        for line in elem.text.split('\n'):
            line_s = line.strip()
            if line_s:
                lines.append(line_s)
        return 'Comment', lines

    def parse_DataValue(self, elem):
        if elem.attrib['name'].find('taken_over') >= 0:
            # Ignore "taken_over" information
            return None
        # else:

        result = self._parse(elem, 'dict')
        if 'GripperState' in result:
            if elem.attrib['name'] == 'gripper.right.state':
                result['GripperState']['Side'] = 'right'
            elif elem.attrib['name'] == 'gripper.left.state':
                result['GripperState']['Side'] = 'left'
            else:
                raise Exception()
        return 'DataValue', result

    def parse_PhysicalObjectState(self, elem):
        result = copy.deepcopy(self._parse(elem, 'dict'))
        if 'PoseDescription' in result and 'GraspedPose' in result['PoseDescription']:
            del result['PoseDescription']
        return 'PhysicalObjectState', result

    def parse_PoseState(self, elem):
        return 'PoseDescription', self._parse(elem, 'dict')['PoseRegion']

    def parse_PoseRegion(self, elem):
        return 'PoseRegion', self._parse(elem, 'dict')

    def parse_PlaceDesc(self, elem):
        return 'PlaceDesc', self._parse(elem, 'dict')

    def parse_PlacePreposition(self, elem):
        return 'PlacePreposition', elem.text

    def parse_RelativeTo(self, elem):
        return 'RelativeTo', self._parse(elem, 'dict')

    def parse_GraspedPose(self, elem):
        return 'GraspedPose', {'Side': elem.attrib['side']}

    def parse_InternalState(self, elem):
        return None

    def parse_GripperState(self, elem):
        data = self._parse(elem, 'dict')
        data['State'] = elem.attrib['state']
        return 'GripperState', data

    def parse_GraspDef(self, elem):
        return 'GraspDef', {'GraspId': elem.attrib['id']}

    def parse_BodyPose(self, elem):
        #result = {'torso_angle': elem.attrib['torso_angle']}
        data = self._parse(elem, 'dict')
        data['TorsoConfig'] = elem.attrib['torso_angle']
        return 'BodyPose', data

    def parse_RightArm(self, elem):
        result = self._parse(elem, 'dict')
        result['Side'] = 'right'
        return 'RightArm', result

    def parse_LeftArm(self, elem):
        result = self._parse(elem, 'dict')
        result['Side'] = 'left'
        return 'LeftArm', result

    def parse_RightHand(self, elem):
        result = self._parse(elem, 'dict')
        result['Side'] = 'right'
        return 'RightHand', result

    def parse_LeftHand(self, elem):
        result = self._parse(elem, 'dict')
        result['Side'] = 'left'
        return 'LeftHand', result

    def parse_ArmConfig(self, elem):
        return 'ArmConfig', {'q0': elem.attrib['q0'], 'q1': elem.attrib['q1'],
            'q2': elem.attrib['q2'], 'q3': elem.attrib['q3'], 'q4': elem.attrib['q4'],
            'q5': elem.attrib['q5'], 'q6': elem.attrib['q6']}

    def parse_HandConfig(self, elem):
        return 'HandConfig', {'f0a': elem.attrib['f0a'], 'f0b': elem.attrib['f0b'],
            'f1a': elem.attrib['f1a'], 'f1b': elem.attrib['f1b'], 'f2a': elem.attrib['f2a'],
            'f2b': elem.attrib['f2b'], 'sp': elem.attrib['sp']}

    def parse_HeadConfig(self, elem):
        return 'HeadConfig', {'hp': elem.attrib['hp'], 'ht': elem.attrib['ht']}


class TaskNetworkXmlParser(DataXmlParser):
    def __init__(self):
        super().__init__('list')

    def parseDocument(self, path):
        result = super().parseDocument(path)
        result['type'] = 'TaskNetwork'
        arg_value_map = {}
        for task in result['Tasks']:
            new_args = []
            for arg in task['Args']:
                new_args.append(arg['Name'])
                arg_value_map[arg['Name']] = arg['Value']
            task['Args'] = new_args
        result['ParameterGroundings'] = {}
        for name, value in arg_value_map.items():
            obj_type = list(value.keys())[0]
            obj_value = value[obj_type]
            result['ParameterGroundings'][name] = {
                    'type': 'data',
                    'Value': obj_value,
                    'ValueType':obj_type}
        return result

    def parse_GroundTask(self, elem):
        data = self._parse(elem, 'list')
        data['Class'] = elem.attrib['name']
        data['Id'] = elem.attrib['id']
        data['type'] = 'task_ref'
        return 'Tasks', data

    def parse_arg(self, elem):
        return 'Args', {
            'type':'Arg',
            'Name': f'?{elem.attrib['name']}',
            'Value': self._parse(elem, 'dict')}

    def parse_ObjectIdContainer(self, elem):
        data = self._parse(elem, 'list')
        if not data:
            return 'ObjectIdContainer', []
        # else:
        return 'ObjectIdContainer', data


class ModelsXmlParser(DataXmlParser):
    def __init__(self):
        super().__init__('list')

    def parse_ObjectModel(self, elem):
        data = self._parse(elem, 'list')
        data['KinematicStructure'] = data['KinematicStructure'][0]
        data['ModelId'] = elem.attrib['id']
        data['ParentClasses'] = data['ParentClasses'][0]
        moveable_models_map = {
            'jar': True,
            'jar_markers': True,
            'cabinet_no_door': False,
            'cabinet_no_door_foam': False,
            'bowl_high': True,
            'bowl_high_markers': True,
            'table': False,
            'table_a': False,
            'table_b': False,
            'jar_lid': True,
            'cabinet_door': False,
            'chessboard': True,
            'box1': True,
            'box2': True,
            'box3': True,
        }
        is_moveable = moveable_models_map[data['ModelId']]
        if not 'Attributes' in data:
            data['Attributes'] = []
        data['Attributes'].append( {'Name': 'moveability', 'IsMoveable': is_moveable} )
            
        return 'ObjectModels', data

    def parse_ParentClasses(self, elem):
        return 'ParentClasses', self._parse(elem, 'list')['ParentClass']

    def parse_ParentClass(self, elem):
        return 'ParentClass', elem.text

    def parse_Attribute(self, elem):
        data = self._parse(elem, 'dict')
        data['Name'] = elem.attrib['name']
        return 'Attributes', data

    def parse_Hole(self, elem):
        return 'Hole', self._parse(elem, 'dict')

    def parse_Circle(self, elem):
        return 'Circle', {'r': elem.attrib['r']}

    def parse_FlatBaseContainer(self, elem):
        return 'FlatBaseContainer', self._parse(elem, 'list')['FlatBase']

    def parse_FlatBase(self, elem):
        data = self._parse(elem, 'dict')
        if 'Cylinder' in data:
            data['Circle'] = {'r': data['Cylinder']['radius']}
            del data['Cylinder']
        return 'FlatBase', data

    def parse_cylinder(self, elem):
        return 'Cylinder', self._parse(elem, 'dict')

    def parse_radius(self, elem):
        return 'radius', elem.text

    def parse_length(self, elem):
        return 'length', elem.text

    def parse_KinematicStructure(self, elem):
        return 'KinematicStructure', self._parse(elem, 'list')

    def parse_link(self, elem):
        data = self._parse(elem, 'list')
        data['Name'] = elem.attrib['name']
        return 'Links', data

    def parse_JointConnection(self, elem):
        return 'JointConnections', {
            'Name':elem.attrib['attr_name'],
            'JointName':elem.attrib['joint_name'],
            'Mult':elem.attrib['mult'],
            'Offset':elem.attrib['offset'],
        }

    def parse_joint(self, elem):
        data = self._parse(elem, 'dict')
        data['Name'] = elem.attrib['name']
        data['Type'] = elem.attrib['type']
        return 'Joints', data

    def parse_parent(self, elem):
        return 'Parent', elem.text

    def parse_child(self, elem):
        return 'Child', elem.text

    def parse_axis(self, elem):
        return 'Axis', self._parse(elem, 'dict')

    def parse_xyz(self, elem):
        items = elem.text.split()
        return 'JointAxis', {
            'x':items[0],
            'y':items[1],
            'z':items[2],
        }

    def parse_limit(self, elem):
        return 'Limit', self._parse(elem, 'dict')

    def parse_lower(self, elem):
        return 'Lower', elem.text

    def parse_upper(self, elem):
        return 'Upper', elem.text

    def parse_use_parent_model_frame(self, elem):
        return 'UseParentModelFrame', elem.text

    def parse_ArMarker(self, elem):
        return None

    def parse_FlatSurfaceContainer(self, elem):
        return 'FlatSurfaceContainer', self._parse(elem, 'list')['FlatSurface']

    def parse_FlatSurface(self, elem):
        return 'FlatSurface', self._parse(elem, 'dict')

    def parse_Rectangle(self, elem):
        return 'Rectangle', {'x':elem.attrib['x'], 'y':elem.attrib['y']}

    def parse_GeomPoint(self, elem):
        return 'Point', {'x':elem.attrib['x'], 'y':elem.attrib['y'], 'z':elem.attrib['z']}

class GraspsXmlParser(DataXmlParser):
    def __init__(self):
        super().__init__('list')

    def parseDocument(self, path):
        result = super().parseDocument(path)
        grasp_def_map = {}
        for grasp_def in result['GraspDefs']:
            grasp_def_map[ grasp_def['GraspId'] ] = grasp_def
        result['GraspDefs'] = grasp_def_map
        return result
# GraspsContainer model_id="bowl_high">
#   <GraspDef id="0">
# <SimilarGrasps id1="3" id2="4"/>
    # def parse_GraspsContainer(self, elem):
    #     elem.attrib['model_id']
    #     data = self._parse(elem, 'list')
    #     return 'GraspsContainer', data

    #     # data['KinematicStructure'] = data['KinematicStructure'][0]
    #     # data['ModelId'] = elem.attrib['id']
    #     # data['ParentClasses'] = data['ParentClasses'][0]
    #     # return 'ObjectModels', data

    def parse_GraspDef(self, elem):
        data = self._parse(elem, 'list')
        data['GraspId'] = elem.attrib['id']
        return 'GraspDefs', data

    def parse_SimilarGrasps(self, elem):
        return 'SimilarGrasps', {
            'id1': elem.attrib['id1'],
            'id2': elem.attrib['id2'],
        }

    def parse_GripperMovement(self, elem):
        return 'GripperMovements', self._parse(elem, 'dict')

    def parse_GripperMovementT(self, elem):
        return 'GripperMovementT', self._parse(elem, 'dict')

    def parse_GripperMovementQ(self, elem):
        return 'GripperMovementQ', {
            'f1': elem.attrib['f1'],
            'f2': elem.attrib['f2'],
            'f3': elem.attrib['f3'],
            'sp': elem.attrib['sp'],
        }

def show_wrist_constraint():
    import matplotlib.pyplot as plt

    wcc_r = [0.65, -2.90, 1.79, -2.91, 1.78, -1.43, 0.77, -1.39, 0.36, -1.00, -0.15, -0.26, 0.35, 0.41, 0.8, 0.94, 1.8, 1.01, 1.81 ,2.88, -0.4, 2.89, -0.81, 2.27, -1.82, 2.29, -1.83, -1.66, -0.84, -1.73, -0.42, -2.09]
    wcc_l = [-0.65, 2.90, -1.79, 2.91, -1.78, 1.43, -0.77, 1.39, -0.36, 1.00, 0.15, 0.26, -0.35, -0.41, -0.8, -0.94, -1.8, -1.01, -1.81, -2.88, 0.4, -2.89, 0.81, -2.27, 1.82, -2.29, 1.83, 1.66, 0.84, 1.73, 0.42, 2.09]
    wcc = wcc_l
    wcc = wcc + wcc[0:2]
    fig, ax = plt.subplots()
    #plt.axis('equal')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([-3, 3]) # type: ignore
    ax.set_ylim([-3, 3]) # type: ignore
    ax.plot([x for idx, x in enumerate(wcc) if idx%2 == 0], [x for idx, x in enumerate(wcc) if idx%2 == 1])
    
    # This is for right arm
    # circles = [
    #     (0.85, 1.93, 0.9),
    #     (0.16, 1.8, 1.0),
    #     (-0.41, 1.29, 1.0),
    #     (-0.65, 1.05, 1.1),
    #     (-0.87, 0.45, 0.9),
    #     (-0.98, -0.16, 0.8),
    #     (-0.95, -0.75, 0.85),
    #     (-0.42, -1.14, 0.65),
    #     (-0.16, -1.38, 0.6),
    #     (0.15, -1.66, 0.6),
    #     (0.42, -1.91, 0.6),
    #     (1.0, -2.13, 0.7),
    # ]
    # for cx, cy, cr in circles:
    #     circle = plt.Circle((cx, cy), cr, color='r')
    #     ax.add_patch(circle)
    # ax.plot([x[0] for x in circles], [x[1] for x in circles], 'bo')
    plt.show()

    return 0

def main(argv=None) -> int:
    # return show_wrist_constraint()

    src_ws_path = '/home/dseredyn/ws_velma_2021_04_24/ws_velma_os/src'
    world_models_path = f'{src_ws_path}/roko/roko_planning/data/tests'
    task_networks_path = f'{src_ws_path}/roko/roko_nodes/data/task_networks'
    models_path = f'{src_ws_path}/roko/roko_nodes/data/xml/models.xml'
    grasps_path = f'{src_ws_path}/roko/roko_nodes/data/xml/grasps'

    dst_ws_path = '/home/dseredyn/ws_tamp/src'
    world_states_output_path = f'{dst_ws_path}/tamp_htn_stream_examples/examples/wut_velma/world_states'
    task_networks_output_path = f'{dst_ws_path}/tamp_htn_stream_examples/examples/wut_velma/task_networks'
    models_output_path = f'{dst_ws_path}/tamp_htn_stream_examples/examples/wut_velma/models/models.json'
    grasps_output_path = f'{dst_ws_path}/tamp_htn_stream_examples/examples/wut_velma/grasps'

    tests_list = [
        'kb_world_model_013.xml place_bowl.xml a',
        'simple_jar_grasped.xml pour.xml b',
        'kb_world_model_011.xml takeover_jar_left.xml c',
        'simple_jar_bowl_grasped.xml pour.xml d',
        'kb_world_model_013.xml takeover_jar_right.xml e',
        'kb_world_model_013.xml takeover_jar_left.xml f',
        'kb_world_model_013.xml pour.xml g',
        'exp_2022_11_21_env2_bowl_occl_box2_c.xml takeover_bowl_right.xml h',
        'exp_2022_11_21_env2_bowl_occl_box2_c.xml takeover_jar_left.xml i',
        'exp_2022_11_21_env2_bowl_occl_box2_c.xml takeover_jar_right.xml j',
        'kb_world_model_011.xml pour.xml k',
        'exp_2022_11_21_env2_bowl_occl_box2_c.xml takeover_bowl_left.xml l',
        'complex_01_jar_opened_localized_fixed.xml takeover_jar_left.xml m',
        'exp_2022_11_21_env2_bowl_occl_box2_b.xml pour.xml n',
        'exp_2022_11_21_env2_bowl_occl_box2_c.xml pour.xml o',
        'world_05.xml pour.xml p',
        'world_06.xml pour.xml q',
        'world_08.xml pour.xml r',
        'world_09b.xml pour.xml s',
        'world_10.xml pour.xml t',
        'world_12.xml clear_table.xml u',]

    # Uwaga: stan w pliku complex_01_jar_opened_localized.xml był wczytywany ze świata Gazebo;
    # zamieniono na plik complex_01_jar_opened_localized_fixed.xml z kompletnym stanem na bazie
    # oryginalnego.

    model_ids = set()
    for test_str in tests_list:
        items = test_str.split()
        assert len(items) == 3
        world_model = items[0]
        task_network = items[1]
        test_case = items[2]
        world_model_path = '{}/{}'.format(world_models_path, world_model)
        task_network_path = '{}/{}'.format(task_networks_path, task_network)

        # World states
        print(f'Parsing {world_model}')
        parser = WorldModelXmlParser()
        data = parser.parseDocument(world_model_path)
        model_ids = model_ids.union(parser.model_ids)
        json_string = json.dumps(data, indent=2)

        # Write to output file
        output_file_path = f'{world_states_output_path}/{world_model[0:-4]}.json'
        print(f'Writing to {output_file_path}')
        with open(output_file_path, 'w') as f:
            f.write(json_string)

        # Task networks
        print(f'Parsing {task_network}')
        parser = TaskNetworkXmlParser()
        data = parser.parseDocument(task_network_path)
        json_string = json.dumps(data, indent=2)

        # Write to output file
        output_file_path = f'{task_networks_output_path}/{task_network[0:-4]}.json'
        print(f'Writing to {output_file_path}')
        with open(output_file_path, 'w') as f:
            f.write(json_string)

    print(f'Model ids: {list(model_ids)}')

    # Models
    print(f'Parsing models')
    parser = ModelsXmlParser()
    data = parser.parseDocument(models_path)
    json_string = json.dumps(data, indent=2)
    # Write to output file
    print(f'Writing to {models_output_path}')
    with open(models_output_path, 'w') as f:
        f.write(json_string)

    # Grasps
    print('Parsing grasps')
    grasps_models = ['bowl_high', 'jar', 'box1', 'box2', 'box3']
    for model_id in grasps_models:
        grasps_input_file_name = f'{grasps_path}/{model_id}.xml'
        parser = GraspsXmlParser()
        data = parser.parseDocument(grasps_input_file_name)
        json_string = json.dumps(data, indent=2)
        # Write to output file
        grasps_output_file_name = f'{grasps_output_path}/{model_id}.json'
        with open(grasps_output_file_name, 'w') as f:
            f.write(json_string)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
