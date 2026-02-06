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
import math
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose, Vector3, Point
from std_msgs.msg import ColorRGBA

from tamp_htn_stream.core import TypedValue
from .data_types import KdlVector, KdlRotation, KdlFrame, PrimitiveShape,\
    ConfA, ConfT

def is_value_equal(data1: TypedValue, data2: TypedValue):
    if not isinstance(data1, TypedValue):
        print(data1)
        raise Exception()
    if not isinstance(data2, TypedValue):
        print(data2)
        raise Exception()
    # print(f'data1: {data1}')
    # print(f'data2: {data2}')
    if data1.getType() != data2.getType():
        return False
    # else:
    tp = data1.getType()
    v1 = data1.getValue()
    v2 = data2.getValue()
    if tp in ['ObjectId', 'ObjectModelId', 'Side', 'GraspId', 'ConfA', 'ConfT', 'ConfG', 'ConfH', 'Volume']:
        result = (v1 == v2)
        # print(f'is_value_equal({tp}, [{v1}], [{v2}]): {result}')
        return result
    elif tp == 'Pose':
        f1 = KdlFrame.fromDict(v1)
        f2 = KdlFrame.fromDict(v2)
        diff = f1.diff(f2, 1.0)
        return diff.vel.Norm() < 0.0001 and diff.rot.Norm() < 0.0001
    # else:
    raise Exception(f'Not supported data type: "{tp}"')

def jsonToJsStr(d: Any, indent: None|int = 0) -> str:
    if indent is None:
        ind_str = ''
        indent_next = None
    else:
        ind_str = '  '*indent
        indent_next = indent + 1
    if isinstance(d, dict):
        js_str = ''
        for key, value in d.items():
            if js_str:
                js_str += ',\n'
            js_str += f'{ind_str}{key}: {jsonToJsStr(value, indent_next)}'
        return f'{ind_str}{{\n{js_str}}}'
    elif isinstance(d, list):
        js_str = ''
        for value in d:
            if js_str:
                js_str += ',\n'
            js_str += f'{ind_str}{jsonToJsStr(value, indent_next)}'
        return f'{ind_str}[\n{js_str}]'
    elif isinstance(d, bool):
        return 'true' if d else 'false'
    elif isinstance(d, int):
        return f'{d}'
    elif isinstance(d, float):
        return f'{d:.4f}'
    elif isinstance(d, str):
        return f'"{d}"'
    else:
        raise Exception(f'Not supported data type: "{d}"')

# def kdlFrameFromDict(d: dict[str, Any]) -> KdlFrame:
#     if 'Quaternion' in d:
#         M = KdlRotation.Quaternion(float(d['Quaternion']['x']), float(d['Quaternion']['y']),
#                                         float(d['Quaternion']['z']), float(d['Quaternion']['w']))
#     elif 'RPY' in d:
#         M = KdlRotation.RPY(float(d['RPY']['r']), float(d['RPY']['p']), float(d['RPY']['y']))
#     else:
#         raise Exception(d)

#     return KdlFrame(M,
#         KdlVector(
#         x=float(d['Vector']['x']),
#         y=float(d['Vector']['y']),
#         z=float(d['Vector']['z']))
#         )

# def kdlFrameToDict(f: KdlFrame) -> dict[str, Any]:
#     qx, qy, qz, qw = f.M.GetQuaternion()
#     return {'Vector': {'x': f.p.x(), 'y': f.p.y(), 'z': f.p.z()},
#                         'Quaternion': {'x': qx, 'y': qy, 'z': qz, 'w': qw} }

def cVector3(x: float, y: float, z: float) -> Vector3:
    result = Vector3()
    result.x = x
    result.y = y
    result.z = z
    return result

def cPoint(x: float, y: float, z: float) -> Point:
    result = Point()
    result.x = x
    result.y = y
    result.z = z
    return result

def cColorRGBA(r: float, g: float, b: float, a: float) -> ColorRGBA:
    result = ColorRGBA()
    result.r = r
    result.g = g
    result.b = b
    result.a = a
    return result

def cMarker(ns:str, idx:int, pose:KdlFrame, shape_type: str, size: list[float],
            color: list[float]) -> Marker:
    if shape_type == 'box':
        return cMarkerBox(ns, idx, pose, size, color)
    elif shape_type == 'cylinder':
        return cMarkerCylinder(ns, idx, pose, size[0], size[1], color)
    elif shape_type == 'sphere':
        return cMarkerSphere(ns, idx, pose, size[0], color)
    # else:
    raise Exception(f'Unknown shape type: "{shape_type}"')

def cMarkerSphere(ns:str, idx:int, pose:KdlFrame, radius:float, color: list[float]) -> Marker:
    assert isinstance(pose, KdlFrame)
    color_a = color[3] if len(color) == 4 else 1.0
    marker = Marker()
    marker.header.frame_id = 'world'
    marker.ns = ns
    marker.id = idx
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose = pose.toRosPose()
    marker.scale = cVector3(radius*2, radius*2, radius*2)
    marker.color = cColorRGBA(color[0], color[1], color[2], color_a)
    return marker

def cMarkerSphereList(ns:str, idx:int, points: list[KdlVector], radius:float, color: list[float]) -> Marker:
    color_a = color[3] if len(color) == 4 else 1.0
    marker = Marker()
    marker.header.frame_id = 'world'
    marker.ns = ns
    marker.id = idx
    marker.type = Marker.SPHERE_LIST
    marker.action = Marker.ADD
    marker.pose = KdlFrame().toRosPose()
    marker.scale = cVector3(radius*2, radius*2, radius*2)
    marker.color = cColorRGBA(color[0], color[1], color[2], color_a)
    marker.points = [cPoint(p.x(), p.y(), p.z()) for p in points]
    return marker

def cMarkerCylinder(ns:str, idx:int, pose:KdlFrame, radius: float, length: float,
                    color: list[float], frame_id: str = 'world') -> Marker:
    assert isinstance(pose, KdlFrame)
    color_a = color[3] if len(color) == 4 else 1.0
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.ns = ns
    marker.id = idx
    marker.type = Marker.CYLINDER
    marker.action = Marker.ADD
    marker.pose = pose.toRosPose()
    marker.scale = cVector3(radius*2, radius*2, length)
    marker.color = cColorRGBA(color[0], color[1], color[2], color_a)
    return marker

def cMarkerArrow(ns:str, idx:int, p1: KdlVector, p2: KdlVector, width: float, color: list[float],
                 frame_id: str = 'world') -> Marker:
    color_a = color[3] if len(color) == 4 else 1.0
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.ns = ns
    marker.id = idx
    marker.type = Marker.ARROW
    marker.action = Marker.ADD
    marker.pose = KdlFrame().toRosPose()
    marker.scale = cVector3(width, width*2, width*3)
    marker.color = cColorRGBA(color[0], color[1], color[2], color_a)
    marker.points = [cPoint(p1.x(), p1.y(), p1.z()), cPoint(p2.x(), p2.y(), p2.z())]
    return marker

def cMarkerBox(ns:str, idx:int, pose:KdlFrame, size: list[float],
                    color: list[float]) -> Marker:
    assert isinstance(pose, KdlFrame)
    color_a = color[3] if len(color) == 4 else 1.0
    marker = Marker()
    marker.header.frame_id = 'world'
    marker.ns = ns
    marker.id = idx
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.pose = pose.toRosPose()
    marker.scale = cVector3(size[0], size[1], size[2])
    marker.color = cColorRGBA(color[0], color[1], color[2], color_a)
    return marker

def cMarkerCubeList(ns: str, idx: int, points: list[KdlVector], size:list[float],
                    color: list[float]) -> Marker:
    color_a = color[3] if len(color) == 4 else 1.0
    marker = Marker()
    marker.header.frame_id = 'world'
    marker.ns = ns
    marker.id = idx
    marker.type = Marker.CUBE_LIST
    marker.action = Marker.ADD
    marker.pose = KdlFrame().toRosPose()
    marker.scale = cVector3(size[0], size[1], size[2])
    marker.color = cColorRGBA(color[0], color[1], color[2], color_a)
    marker.points = [cPoint(p.x(), p.y(), p.z()) for p in points]
    return marker

def cMarkerFrame(ns: str, base_idx: int, T: KdlFrame, size: float,
                 frame_id: str = 'world') -> list[Marker]:
    length = size
    radius = 0.1 * size
    return [
        cMarkerCylinder(ns, base_idx+0,
            T*KdlFrame(KdlRotation.RotY(math.radians(90))) * KdlFrame(KdlVector(0, 0, length/2)),
            radius, length, [1.0, 0.0, 0.0], frame_id=frame_id),
        cMarkerCylinder(ns, base_idx+1,
            T*KdlFrame(KdlRotation.RotX(math.radians(-90))) * KdlFrame(KdlVector(0, 0, length/2)),
            radius, length, [0.0, 1.0, 0.0], frame_id=frame_id),
        cMarkerCylinder(ns, base_idx+2, T*KdlFrame(KdlVector(0, 0, length/2)),
            radius, length, [0.0, 0.0, 1.0], frame_id=frame_id)
    ]

def rosPoseToJson(f: KdlFrame):
    qx, qy, qz, qw = f.M.GetQuaternion()
    return {'position': {'x': f.p.x(), 'y': f.p.y(), 'z': f.p.z()},
                        'orientation': {'x': qx, 'y': qy, 'z': qz, 'w': qw} }

def rosMsgToDict(obj):
    if isinstance(obj, dict):
        ignored_keys = ['_check_fields']
        return dict((key.lstrip("_"), rosMsgToDict(val)) for key, val in obj.items() if not key in ignored_keys)
    elif hasattr(obj, "_ast"):
        return rosMsgToDict(obj._ast())
    elif hasattr(obj, "__iter__") and not isinstance(obj, str):
        return [rosMsgToDict(v) for v in obj]
    elif hasattr(obj, '__dict__'):
        return rosMsgToDict(vars(obj))
    elif hasattr(obj, '__slots__'):
        return rosMsgToDict(dict((name, getattr(obj, name)) for name in getattr(obj, '__slots__')))
    return obj

def completeColorAlpha(color: list[float]) -> list[float]:
    # Optional alpha
    if len(color) == 4:
        return color
    else:
        return [color[0], color[1], color[2], 1.0]

def primitiveShapeToRvizMarker(ns: str, base_id: int, shape: PrimitiveShape,
                                color: list[float]):
    frame_id = 'world'
    color4 = completeColorAlpha(color)
    # marker.header.stamp = stamp
    if shape.tp == 'cylinder':
        radius = shape.size[0]
        length = shape.size[1]
        return cMarkerCylinder(ns, base_id, shape.getShapePose(frame_id), radius, length, color4)
    elif shape.tp == 'box':
        return cMarkerBox(ns, base_id, shape.getShapePose(frame_id), shape.size, color4)
    elif shape.tp == 'sphere':
        radius = shape.size[0]
        return cMarkerSphere(ns, base_id, shape.getShapePose(frame_id), radius, color4)
    else:
        raise Exception(f'Not supported shape: {shape.tp}')
  
def modelToRvizMarkers(ns: str, base_id: int, model: dict[str, Any],
                       T_B_O: KdlFrame, color: list[float]) -> tuple[list[Marker], int]:
    frame_id = 'world'
    markers: list[Marker] = []
    color4 = completeColorAlpha(color)
    marker_id = base_id
    for link in model['KinematicStructure']['Links']:
        for col in link['collision']:
            T_O_C = KdlFrame.fromDict( col['Pose']['Pose'] )
            T_B_C = T_B_O * T_O_C
            # marker.header.stamp = stamp
            marker_id = marker_id + 1
            if 'Cylinder' in (geom := col['Geometry']):
                radius = float(geom['Cylinder']['radius'])
                length = float(geom['Cylinder']['length'])
                marker = cMarkerCylinder(ns, marker_id, T_B_C, radius, length, color4)
            elif 'Box' in geom:
                sx = float(geom['Box']['Size']['x'])
                sy = float(geom['Box']['Size']['y'])
                sz = float(geom['Box']['Size']['z'])
                marker = cMarkerBox(ns, marker_id, T_B_C, [sx, sy, sz], color4)
            else:
                raise Exception(f'Not supported shape: {geom}')

            markers.append(marker)
            marker_id = marker_id + 1

    return markers, marker_id


###################################################################################################
# Configurations ##################################################################################
###################################################################################################

def joinJsMap(conf_list: list[dict[str, float]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for conf in conf_list:
        for name, val in conf.items():
            if name in result:
                raise Exception(f'Already present: {name}')
            result[name] = val
    return result

# def confToJsMap(conf_list: list[dict[str, Any]]) -> dict[str, float]:
#     result = {}
#     for conf in conf_list:
#         # if isinstance(conf, ConfA):

#         # if not 'ValueType' in conf:
#         #     raise Exception(f'The data has no "ValueType" property: {conf}')
#         if isinstance(conf, (str, float, int)): #'ConfT'
#             result['torso_0_joint'] = float(conf)
#         elif 'ArmConfig' in conf:
#             side_str = conf['Side']
#             for idx in range(7):
#                 result[f'{side_str}_arm_{idx}_joint'] = float(conf['ArmConfig'][f'q{idx}'])
#         elif 'HandConfig' in conf:
#             side_str = conf['Side']
#             result[f'{side_str}_HandFingerOneKnuckleOneJoint'] = float(conf['HandConfig']['sp'])
#             result[f'{side_str}_HandFingerOneKnuckleTwoJoint'] = float(conf['HandConfig']['f0a'])
#             result[f'{side_str}_HandFingerOneKnuckleThreeJoint'] = float(conf['HandConfig']['f0b'])
#             result[f'{side_str}_HandFingerTwoKnuckleOneJoint'] = float(conf['HandConfig']['sp'])
#             result[f'{side_str}_HandFingerTwoKnuckleTwoJoint'] = float(conf['HandConfig']['f1a'])
#             result[f'{side_str}_HandFingerTwoKnuckleThreeJoint'] = float(conf['HandConfig']['f1b'])
#             result[f'{side_str}_HandFingerThreeKnuckleTwoJoint'] = float(conf['HandConfig']['f2a'])
#             result[f'{side_str}_HandFingerThreeKnuckleThreeJoint'] = float(conf['HandConfig']['f2b'])
#         elif 'hp' in conf:
#             result['head_pan_joint'] = float(conf['hp'])
#             result['head_tilt_joint'] = float(conf['ht'])
#         else:
#             raise Exception(f'Worng conf type: {conf['ValueType']}')
#     return result

# def jsMapToConfA(side_str: str, js: dict[str, float]) -> dict[str, Any]:
#     return {
#             'type': 'data',
#             'ValueType': 'ConfA',
#             'Value': {
#               "ArmConfig": {
#                 "q0": js[f'{side_str}_arm_0_joint'],
#                 "q1": js[f'{side_str}_arm_1_joint'],
#                 "q2": js[f'{side_str}_arm_2_joint'],
#                 "q3": js[f'{side_str}_arm_3_joint'],
#                 "q4": js[f'{side_str}_arm_4_joint'],
#                 "q5": js[f'{side_str}_arm_5_joint'],
#                 "q6": js[f'{side_str}_arm_6_joint']
#               },
#               "Side": side_str
#               }
#             }

# def jsMapToConfT(js: dict[str, float]) -> dict[str, Any]:
#     return {
#             'type': 'data',
#             'ValueType': 'ConfT',
#             'Value': js['torso_0_joint']
#             }

# def jsMapToConfG(side_str: str, js: dict[str, float]) -> dict[str, Any]:
#     return {
#         'type': 'data',
#         'ValueType': 'ConfG',
#         'Value': {
#             'HandConfig': {
#                 'sp': js[f'{side_str}_HandFingerOneKnuckleOneJoint'],
#                 'f0a': js[f'{side_str}_HandFingerOneKnuckleTwoJoint'],
#                 'f0b': js[f'{side_str}_HandFingerOneKnuckleThreeJoint'],
#                 'f1a': js[f'{side_str}_HandFingerTwoKnuckleTwoJoint'],
#                 'f1b': js[f'{side_str}_HandFingerTwoKnuckleThreeJoint'],
#                 'f2a': js[f'{side_str}_HandFingerThreeKnuckleTwoJoint'],
#                 'f2b': js[f'{side_str}_HandFingerThreeKnuckleThreeJoint']
#             },
#             'Side': side_str
#         }
#     }

# def jsMapToConfH(js: dict[str, float]) -> dict[str, Any]:
#     return {
#         'type': 'data',
#         'ValueType': 'ConfH',
#         'Value': {
#             'hp': js['head_pan_joint'],
#             'ht': js['head_tilt_joint'],
#         }
#     }

# TODO: universal conversions - only destination type is provided
# def toJsMap(data: Any) -> dict[str, float]:
#     if isinstance(data, list):
#         confToJsMap[data]
#     else:
#         toJsMap([data])

# def toListConfA(conf: dict[str, Any]) -> list[float]:
#     return [float(conf['ArmConfig'][x]) for x in ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6']]

def armJointNameList(side: str) -> list[str]:
    return [
        f'{side}_arm_0_joint',
        f'{side}_arm_1_joint',
        f'{side}_arm_2_joint',
        f'{side}_arm_3_joint',
        f'{side}_arm_4_joint',
        f'{side}_arm_5_joint',
        f'{side}_arm_6_joint'
    ]
