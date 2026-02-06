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

from typing import Any, overload
import PyKDL
import numpy as np
import math
import copy

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.msg import PlanningScene, CollisionObject, AttachedCollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose, Vector3, Point
from builtin_interfaces.msg import Duration

# PyKDL wrapper
class KdlVector:
    def __init__(self, x: None | int | float = None,
                 y: None | int | float = None,
                 z: None | int | float = None):
        if x is None:
            assert y is None and z is None
            self._v = PyKDL.Vector()
        elif isinstance(x, (int, float)):
            assert isinstance(y, (int, float)) and isinstance(z, (int, float))
            self._v = PyKDL.Vector(float(x), float(y), float(z))
        else:
            raise Exception()

    def Norm(self) -> float:
        return self._v.Norm()

    def Normalize(self) -> float:
        return self._v.Normalize()

    def ReverseSign(self) -> None:
        self._v.ReverseSign()

    def Zero(self) -> None:
        self._v.Zero()

    def x(self) -> float:
        return self._v.x()

    def y(self) -> float:
        return self._v.y()

    def z(self) -> float:
        return self._v.z()
    
    def dot(self, v2: KdlVector) -> float:
        assert isinstance(v2, KdlVector)
        return PyKDL.dot(self._v, v2._v)

    def __add__(self, v: KdlVector) -> KdlVector:
        assert isinstance(v, KdlVector)
        return KdlVector.fromKDL(self._v + v.toKDL())

    def __sub__(self, v: KdlVector) -> KdlVector:
        return KdlVector.fromKDL(self._v - v.toKDL())

    #| KdlRotation | KdlTwist | KdlWranch | Kdl
    def __mul__(self, v: float | KdlVector) -> KdlVector:
        if isinstance(v, float):
            return KdlVector.fromKDL(self._v * v)
        elif isinstance(v, KdlVector):
            return KdlVector.fromKDL(self._v * v._v)
        raise Exception()

    def __truediv__(self, v: float) -> KdlVector:
        return KdlVector.fromKDL(self._v / v)

    def __floordiv__(self, v: int) -> KdlVector:
        return KdlVector.fromKDL(self._v / float(v))

    def __neg__(self) -> KdlVector:
        return KdlVector.fromKDL(-self._v)

    def __pos__(self) -> KdlVector:
        return self

    def toKDL(self) -> PyKDL.Vector:
        return self._v

    # Static functions:
    @staticmethod
    def fromKDL(v: PyKDL.Vector) -> KdlVector:
        assert isinstance(v, PyKDL.Vector)
        result = KdlVector()
        result._v = v
        return result

    @staticmethod
    def fromDict(d: dict[str, Any]) -> KdlVector:
        return KdlVector( float(d['x']), float(d['y']), float(d['z']) )

    def toDict(self) -> dict[str, Any]:
        return {'x': self._v.x(), 'y': self._v.y(), 'z': self._v.z()}

    def __str__(self) -> str:
        return f'[{self.x():.2f}, {self.y():.2f}, {self.z():.2f}]'

# class Kdl:
#     @staticmethod
#     def dot(v1: KdlVector, v2: KdlVector) -> float:
#         return PyKDL.dot(v1._v, v2._v)

#     def operator*(v1, v2) -> Vector

#         Returns the cross producs of two vectors
#         Parameters:	

#             v1 (Vector) – the first vector
#             v2 (Vector) – the second vector

class KdlRotation:
    def __init__(self, Xx: None | float | KdlVector = None, Yx: None | float | KdlVector = None,
                 Zx: None | float | KdlVector = None, Xy: None | float | KdlVector = None,
                 Yy: None | float | KdlVector = None, Zy: None | float | KdlVector = None,
                 Xz: None | float | KdlVector = None, Yz: None | float | KdlVector = None,
                 Zz: None | float | KdlVector = None):
        if Xx is None:
            assert Yx is None and Zx is None and\
                    Xy is None and Yy is None and Zy is None and\
                    Xz is None and Yz is None and Zz is None
            self._r = PyKDL.Rotation()
        elif isinstance(Xx, float):
            assert isinstance(Yx, float) and isinstance(Zx, float) and\
                    isinstance(Xy, float) and isinstance(Yy, float) and isinstance(Zy, float) and\
                    isinstance(Xz, float) and isinstance(Yz, float) and isinstance(Zz, float)
            self._r = PyKDL.Rotation(Xx, Yx, Zx, Xy, Yy, Zy, Xz, Yz, Zz)
        elif isinstance(Xx, KdlVector):
            assert isinstance(Yx, KdlVector) and isinstance(Zx, KdlVector)
            self._r = PyKDL.Rotation(Xx.toKDL(), Yx.toKDL(), Zx.toKDL())
        else:
            raise Exception()

    def DoRotX(self, angle: float) -> None:
        self._r.DoRotX(angle)

    def DoRotY(self, angle: float) -> None:
        self._r.DoRotY(angle)

    def DoRotZ(self, angle: float) -> None:
        self._r.DoRotZ(angle)

    def GetEulerZYX(self) -> tuple[float, float, float]:
        return self._r.GetEulerZYX()

    def GetEulerZYZ(self) -> tuple[float, float, float]:
        return self._r.GetEulerZYZ()

    def GetQuaternion(self) -> tuple[float, float, float, float]:
        return self._r.GetQuaternion()

    def GetRPY(self) -> tuple[float, float, float]:
        return self._r.GetRPY()

    def GetRot(self) -> float:
        return self._r.GetRot()

    def GetRotAngle(self) -> tuple[float, KdlVector]:
        angle, axis = self._r.GetRotAngle()
        return angle, KdlVector.fromKDL(axis)

    def Inverse(self) -> KdlRotation:
        return KdlRotation.fromKDL(self._r.Inverse())

    # def Rot2(self) -> None:
    #     return

    # def SetInverse(self) -> None:
    #     return

    def UnitX(self) -> KdlVector:
        return KdlVector.fromKDL( self._r.UnitX())

    def UnitY(self) -> KdlVector:
        return KdlVector.fromKDL( self._r.UnitY())

    def UnitZ(self) -> KdlVector:
        return KdlVector.fromKDL( self._r.UnitZ())

    #| KdlRotation | KdlTwist | KdlWranch | Kdl
    def __mul__(self, v: KdlVector) -> KdlVector:
        return KdlVector.fromKDL(self._r * v.toKDL())

    # def operator*(self, Twist) -> Twist:
    #     return

    #     Changes the refenrece frame of a Twist

    # def operator*(self, Wrench) -> Wrench:
    #     return

    #     Changes the refenrece frame of a Wrench

    def toKDL(self) -> PyKDL.Rotation:
        return self._r

    # Static functions:
    @staticmethod
    def fromKDL(r: PyKDL.Rotation):
        assert isinstance(r, PyKDL.Rotation)
        result = KdlRotation()
        result._r = r
        return result

    @staticmethod
    def Identity() -> KdlRotation:
        return KdlRotation()

    @staticmethod
    def Quaternion(qx: float, qy: float, qz: float, qw: float) -> KdlRotation:
        return KdlRotation.fromKDL( PyKDL.Rotation.Quaternion(qx, qy, qz, qw) )
    
    @staticmethod
    def Rot(axis: KdlVector, angle: float) -> KdlRotation:
        return KdlRotation.fromKDL( PyKDL.Rotation.Rot(axis.toKDL(), angle) )

    @staticmethod
    def RotX(angle: float)-> KdlRotation:
        return KdlRotation.fromKDL( PyKDL.Rotation.RotX(angle) )

    @staticmethod
    def RotY(angle: float) -> KdlRotation:
        return KdlRotation.fromKDL( PyKDL.Rotation.RotY(angle) )

    @staticmethod
    def RotZ(angle: float) -> KdlRotation:
        return KdlRotation.fromKDL( PyKDL.Rotation.RotZ(angle) )

    @staticmethod
    def EulerZYX(z: float, y: float, x: float) -> KdlRotation:
        return KdlRotation.fromKDL( PyKDL.Rotation.EulerZYX(z, y, x) )

    @staticmethod
    def EulerZYZ(z1: float, y: float, z2: float) -> KdlRotation:
        return KdlRotation.fromKDL( PyKDL.Rotation.EulerZYZ(z1, y, z2) )

    @staticmethod
    def RPY(r: float, p: float, y: float) -> KdlRotation:
        return KdlRotation.fromKDL( PyKDL.Rotation.RPY(r, p, y) )

    @staticmethod
    def fromDict(d: dict[str, Any]) -> KdlRotation:
        return KdlRotation.Quaternion( float(d['x']), float(d['y']), float(d['z']), float(d['w']) )

    def toDict(self) -> dict[str, Any]:
        qx, qy, qz, qw = self._r.GetQuaternion()
        return {'x': qx, 'y': qy, 'z': qz, 'w': qw}

    def __str__(self) -> str:
        qx, qy, qz, qw = self.GetQuaternion()
        return f'[{qx:.2f}, {qy:.2f}, {qz:.2f}, {qw:.2f}]'

# @overload
# def func(a: str) -> str:
#     return a

# @overload
# def func(a: float) -> float:
#     return a

# def func(a: float | str) -> float | str:
#     return 0.0

class KdlFrame:
    def __init__(self, a: None | KdlRotation | KdlVector = None, b: None | KdlVector = None):
        if a is None:
            assert b is None
            self._f = PyKDL.Frame()
        elif isinstance(a, KdlRotation):
            if b is None:
                self._f = PyKDL.Frame(a.toKDL())
            else:
                assert isinstance(b, KdlVector)
                self._f = PyKDL.Frame(a.toKDL(), b.toKDL())
        elif isinstance(a, KdlVector):
            assert b is None
            self._f = PyKDL.Frame(a.toKDL())
        else:
            raise Exception()

    @property
    def p(self) -> KdlVector:
        return KdlVector.fromKDL(self._f.p)

    @p.setter
    def p(self, p: KdlVector):
        self._f.p = p.toKDL()

    @property
    def M(self) -> KdlRotation:
        return KdlRotation.fromKDL(self._f.M)

    @M.setter
    def M(self, M: KdlRotation):
        self._f.M = M.toKDL()


    # Integrate(twist, frequency) → None

    #     This frame is integrated into an updated frame with sample frequence, using first order integration
    #     Parameters:	

    #         twist (Twist) – this twist is represented with respect to the current frame
    #         frequency – the sample frequency to update this frame

    def toKDL(self) -> PyKDL.Frame:
        return self._f

    def Inverse(self) -> KdlFrame:
        return KdlFrame.fromKDL( self._f.Inverse() )

    @overload
    def __mul__(self, x: KdlVector) -> KdlVector:
        pass

    @overload
    def __mul__(self, x: KdlFrame) -> KdlFrame:
        pass

    def __mul__(self, x: KdlVector | KdlFrame) -> KdlVector | KdlFrame:
        if isinstance(x, KdlVector):
            return KdlVector.fromKDL(self._f * x.toKDL())
        elif isinstance(x, KdlFrame):
            return KdlFrame.fromKDL(self._f * x.toKDL())
        else:
            raise Exception()

    #     Changes both the reference frame and the reference point of a Vector. Use this operator when the vector represents a point

    # operator*(Twist) -> Twist

    #     Changes bothe the refenrece frame and the referece point of a Twist

    # operator*(Wrench) -> Wrench

    #     Changes both the refenrece frame and the reference point of a Wrench

    # Static functions:

    @staticmethod
    def fromKDL(f: PyKDL.Frame) -> KdlFrame:
        assert isinstance(f, PyKDL.Frame)
        result = KdlFrame()
        result._f = f
        return result
    
    @staticmethod
    def Identity() -> KdlFrame:
        return KdlFrame()

    # PyKDL.HD(a, alpha, d, theta) → Frame

    #     Constructs a transformationmatrix T_link(i-1)_link(i) with the Denavit-Hartenberg convention as described in the original publictation: Denavit, J. and Hartenberg, R. S., A kinematic notation for lower-pair mechanisms based on matrices, ASME Journal of Applied Mechanics, 23:215-221, 1955.

    # PyKDL.DH_Craig1989(a, alpha, d, theta) → Frame

    #     Constructs a transformationmatrix T_link(i-1)_link(i) with the Denavit-Hartenberg convention as described in the Craigs book: Craig, J. J.,Introduction to Robotics: Mechanics and Control, Addison-Wesley, isbn:0-201-10326-5, 1986.

    def addDelta(self, t: KdlTwist, d: float) -> KdlFrame:
        return KdlFrame.fromKDL( PyKDL.addDelta(self._f, t.toKdl(), d) )

    # PyKDL.AddDelta(f, t, d) → Frame

    #     Constructs a frame that is obtained by: starting from frame f, apply twist t, during time d
    #     Parameters:	

    #         f (Frame) – the frame to start the integration from
    #         t (Twist) – the twist to apply, represented in the same reference frame as f, and with reference point at the origin of f
    #         d (double) – the duration to apply twist t

    def diff(self, f2: KdlFrame, d: float) -> KdlTwist:
        return KdlTwist.fromKDL( PyKDL.diff(self._f, f2._f, d) )

    @staticmethod
    def fromDict(d: dict[str, Any]) -> KdlFrame:
        return KdlFrame( KdlRotation.fromDict(d['M']), KdlVector.fromDict(d['p']) )

    def toDict(self) -> dict[str, Any]:
        return {'p': KdlVector.fromKDL(self._f.p).toDict(),
                'M': KdlRotation.fromKDL(self._f.M).toDict()}

    def __str__(self) -> str:
        return f'{{M: {self.M}, p: {self.p}}}'

    def toRosPose(self) -> Pose:
        result = Pose()
        result.position.x = self.p.x()
        result.position.y = self.p.y()
        result.position.z = self.p.z()
        qx, qy, qz, qw = self.M.GetQuaternion()
        result.orientation.x = qx
        result.orientation.y = qy
        result.orientation.z = qz
        result.orientation.w = qw
        return result


class KdlTwist:

    # RefPoint() → None

    # ReverseSign() → None

    # Zero() → None

    def __init__(self, rot: None | KdlVector = None, vel: None | KdlVector = None):
        if rot is None:
            assert vel is None
            self._t = PyKDL.Twist()
        elif isinstance(rot, KdlVector):
            assert isinstance(vel, KdlVector)
            self._t.rot = rot.toKDL()
            self._t.vel = vel.toKDL()
        else:
            raise Exception()

    # @property
    # def a(self):
    #     return self.__a

    # ## the attribute name and the method name must be same which is used to set the value for the attribute
    # @a.setter
    # def a(self, var):
    #     if var > 0 and var % 2 == 0:
    #         self.__a = var
    #     else:
    #         self.__a = 2

    @property
    def rot(self) -> KdlVector:
        return KdlVector.fromKDL(self._t.rot)

    @rot.setter
    def rot(self, rot: KdlVector):
        self._t.rot = rot.toKDL()

    @property
    def vel(self) -> KdlVector:
        return KdlVector.fromKDL(self._t.vel)

    @vel.setter
    def vel(self, vel: KdlVector):
        self._t.vel = vel.toKDL()

    @staticmethod
    def fromKDL(t: PyKDL.Twist) -> KdlTwist:
        result = KdlTwist()
        result._t = t
        return result
    
    def toKdl(self) -> PyKDL.Twist:
        return self._t

    def __str__(self) -> str:
        return f'{{v: {self.vel}, r: {self.rot}}}'

class PrimitiveShape:
    tp: str
    size: list[float]
    _frame_id: str
    T: KdlFrame

    # For cylinder: size is [radius, length]
    def __init__(self, tp: str, size: list[float], T: KdlFrame, Tint: KdlFrame, frame_id: str):
        self.tp = tp
        self.size = size
        self.setObjectPose(T, frame_id)
        self._Tint = Tint

    def getShapePose(self, frame_id: str) -> KdlFrame:
        if frame_id != self._frame_id:
            raise Exception(f'Tried to get pose in "{frame_id}" frame, but the pose is in "{self._frame_id}"')
        return self._T * self._Tint
    
    def setObjectPose(self, T:KdlFrame, frame_id: str):
        self._T = T
        self._frame_id = frame_id

    def toDict(self) -> dict[str, Any]:
        return {
            'type': 'PrimitiveShape',
            'shape_type': self.tp,
            'size': self.size,
            'T': self._T.toDict(),
            'Tint': self._Tint.toDict(),
            'frame_id' : self._frame_id
        }
    
    @staticmethod
    def fromDict(d: dict[str, Any]) -> PrimitiveShape:
        assert d['type'] == 'PrimitiveShape'
        return PrimitiveShape(d['shape_type'], d['size'],
                              KdlFrame.fromDict(d['T']), KdlFrame.fromDict(d['Tint']),
                              d['frame_id'])


class PoseWithFreeDOF:
    def __init__(self):
        self._transforms = []

    def addStaticTransform(self, T: KdlFrame) -> None:
        assert isinstance(T, KdlFrame)
        self._transforms.append( ('static', T) )

    def addRotation(self, dof_name: str, axis: KdlVector) -> None:
        self._transforms.append( ('rotation', dof_name, axis) )

    def calculate(self, name_pos_map: dict[str, float]) -> KdlFrame:
        result = KdlFrame()
        for transform in self._transforms:
            if transform[0] == 'static':
                result = result * transform[1]
            elif transform[0] == 'rotation':
                dof_name = transform[1]
                axis = transform[2]
                assert dof_name in name_pos_map
                frame = KdlFrame(KdlRotation.Rot(axis, name_pos_map[dof_name]))
                result = result * frame
        return result

    def getAxis(self, axis_name: str, name_pos_map: dict[str, float]) -> tuple[KdlFrame, KdlVector]:
        # Returns a tuple: (origin_frame_of_axis, axis)
        # name_pos_map must contain positions for all previous axes to axis_name
        result = KdlFrame()
        for transform in self._transforms:
            if transform[0] == 'static':
                result = result * transform[1]
            elif transform[0] == 'rotation':
                dof_name = transform[1]
                axis = transform[2]
                if dof_name == axis_name:
                    return result, axis
                assert dof_name in name_pos_map
                frame = KdlFrame(KdlRotation.Rot(axis, name_pos_map[dof_name]))
                result = result * frame
        raise Exception(f'Could not find axis named "{axis_name}"')


def cAddCollisionObject(obj_id: str, obj_type: str, obj_size: list[float],
                        T_W_O: KdlFrame) -> CollisionObject:
    assert isinstance(obj_size, list)
    result = CollisionObject()
    result.header.frame_id = 'world'
    result.pose = T_W_O.toRosPose()
    result.id = obj_id
    prim = SolidPrimitive()
    if obj_type == 'box':
        prim.type = SolidPrimitive.BOX
        assert len(obj_size) == 3
        # total size
        prim.dimensions = obj_size
    elif obj_type == 'sphere':
        prim.type = SolidPrimitive.SPHERE
        assert len(obj_size) == 1
        # radius
        prim.dimensions = obj_size
    elif obj_type == 'cylinder':
        prim.type = SolidPrimitive.CYLINDER
        assert len(obj_size) == 2
        # radius, total length
        radius, length = obj_size
        # In CollisionObject cylinder dimensions are opposite!
        prim.dimensions = [length, radius]
        # prim.dimensions[SolidPrimitive.CYLINDER_RADIUS] = obj_size[0]
        # prim.dimensions[SolidPrimitive.CYLINDER_HEIGHT] = obj_size[1]

        # obj_size[SolidPrimitive.CYLINDER_RADIUS],
        #                    obj_size[SolidPrimitive.CYLINDER_HEIGHT]]
    else:
        raise Exception(f'Unknown shape type: {obj_type}')
    #SolidPrimitive.CONE

    result.primitives = [prim]
    result.primitive_poses = [KdlFrame().toRosPose()]
    result.operation = CollisionObject.ADD
    return result


def cRemoveCollisionObject(obj_id: str) -> CollisionObject:
    result = CollisionObject()
    # result.header.frame_id = 'world'
    result.id = obj_id
    result.operation = CollisionObject.REMOVE
    return result


class PlanningSceneCreator:
    def __init__(self) -> None:
        self._planning_scene_objects: dict[str, CollisionObject] = {}
        self._grasped_objects: dict[str, AttachedCollisionObject] = {}

    def hasObject(self, obj_id: str) -> bool:
        return obj_id in self._planning_scene_objects

    def addObject(self, obj_id: str, prim: PrimitiveShape) -> None:
                  #T_W_O: KdlFrame) -> None:

        if self.hasObject(obj_id):
            raise Exception(f'Object {obj_id} is already added')

        self._planning_scene_objects[obj_id] = cAddCollisionObject(obj_id,
                prim.tp, prim.size, prim.getShapePose('world'))

    def addGraspedObject(self, side: str, prim: PrimitiveShape) -> None:
                         #T_E_O: KdlFrame) -> None:
        if side in self._grasped_objects:
            raise Exception(f'Object is already added as grasped by gripper {side}')

        col_links = [f'{side}_HandFingerOneKnuckleOneLink', f'{side}_HandFingerOneKnuckleTwoLink', f'{side}_HandFingerOneKnuckleThreeLink',
                    f'{side}_HandFingerTwoKnuckleOneLink', f'{side}_HandFingerTwoKnuckleTwoLink', f'{side}_HandFingerTwoKnuckleThreeLink',
                    f'{side}_HandFingerThreeKnuckleTwoLink', f'{side}_HandFingerThreeKnuckleThreeLink',
                    f'{side}_HandPalmLink']

        ee_link_name = f'{side}_arm_7_link'
        att_obj = AttachedCollisionObject()
        att_obj.link_name = ee_link_name
        att_obj.object = cAddCollisionObject(side, prim.tp, prim.size, prim.getShapePose(ee_link_name))
        att_obj.object.header.frame_id = ee_link_name
        att_obj.touch_links = col_links
        self._grasped_objects[side] = att_obj

    def getGraspedObjects(self) -> list[AttachedCollisionObject]:
        return [att_obj for side, att_obj in self._grasped_objects.items()]

    def buildPlanningScene(self) -> PlanningScene:
        result = PlanningScene()
        # result.name
        # result.robot_state.attached_collision_objects = [col_obj for obj_id, col_obj in self._grasped_objects.items()]
        # result.robot_model_name
        # result.fixed_frame_transforms
        # result.allowed_collision_matrix
        # result.link_padding
        # all link scales
        # result.link_scale
        # result.object_colors
        result.world.collision_objects = [col_obj for obj_id, col_obj in self._planning_scene_objects.items()]
        result.is_diff = True
        return result

    def cleanupPlanningScene(self) -> PlanningScene:
        result = PlanningScene()
        # result.robot_state.attached_collision_objects = []
        # for side_id in self._grasped_objects:
        #     att_obj = AttachedCollisionObject()
        #     ee_link_name = f'{side_id}_arm_7_link'
        #     att_obj.link_name = ee_link_name
        #     att_obj.object = cRemoveCollisionObject(side_id)
        #     att_obj.object.header.frame_id = ee_link_name
        #     #att_obj.touch_links = col_links
        #     result.robot_state.attached_collision_objects.append( att_obj )
        #     result.robot_state.is_diff = True
        
        result.world.collision_objects = [cRemoveCollisionObject(obj_id) for obj_id in self._planning_scene_objects]
        result.is_diff = True
        self._planning_scene_objects: dict[str, CollisionObject] = {}
        return result

def reverseTrajectory(traj: JointTrajectory) -> JointTrajectory:
    result = JointTrajectory()
    result.header = traj.header
    result.joint_names = traj.joint_names
    trajectory_time = traj.points[-1].time_from_start # type: ignore
    for pt in reversed(traj.points): # type: ignore
        pt_new = JointTrajectoryPoint()
        pt_new.positions = pt.positions
        pt_new.time_from_start = sub_duration_msgs(trajectory_time, pt.time_from_start)
        result.points.append(pt_new) # type: ignore
    return result

def floatToDuration(d: float) -> Duration:
    result = Duration()
    result.sec = int(math.floor(d))
    result.nanosec = int((d - result.sec) * 1_000_000_000)
    return result

def add_duration_msgs(a: Duration, b: Duration) -> Duration:
    out = Duration()
    out.sec = a.sec + b.sec
    out.nanosec = a.nanosec + b.nanosec
    if out.nanosec > 1_000_000_000:
        out.nanosec -= 1_000_000_000
        out.sec += 1
    # total_ns = (a.sec * 1_000_000_000 + a.nanosec) + (b.sec * 1_000_000_000 + b.nanosec)
    # out = Duration()
    # out.sec = int(total_ns // 1_000_000_000)
    # out.nanosec = int(total_ns % 1_000_000_000)
    return out

def sub_duration_msgs(a: Duration, b: Duration) -> Duration:
    out = Duration()
    out.sec = a.sec - b.sec
    out.nanosec = a.nanosec - b.nanosec
    if out.nanosec < 0:
        out.nanosec += 1_000_000_000
        out.sec -= 1
    return out
    # total_ns = (a.sec * 1_000_000_000 + a.nanosec) - (b.sec * 1_000_000_000 + b.nanosec)
    # out.sec = int(total_ns // 1_000_000_000)
    # out.nanosec = int(total_ns % 1_000_000_000)
    # return out

def interpolateToJointTrajectory(q1: dict[str, float], q2: dict[str, float]) -> JointTrajectory:
    assert len(q1) == len(q2)
    traj = JointTrajectory()
    traj.joint_names = [joint_name for joint_name in q1]
    # Interpolate between q1 and q2
    for f in np.linspace(0, 1, 10, endpoint=True):
        pt = JointTrajectoryPoint()
        pt.time_from_start = floatToDuration(f)
        for joint_name in traj.joint_names:
            pt.positions.append( q1[joint_name]*(1.0-f) + q2[joint_name]*f )
        traj.points.append(pt) # type: ignore
    return traj


class GraspTraj:
    class Movement:
        def __init__(self, mov_type: str, traj: JointTrajectory,
                     q1: dict[str, float], q2: dict[str, float]) -> None:
            assert mov_type in ['arm', 'fingers']
            self.mov_type = mov_type
            self.traj = traj
            self.q1 = q1
            self.q2 = q2

    def __init__(self, side: str):
        self._side = side
        self._movements: list[GraspTraj.Movement] = []

    def addArmMovement(self, traj: JointTrajectory, q1: dict[str, float], q2: dict[str, float]) -> None:
        self._movements.append( GraspTraj.Movement('arm', traj, q1, q2) )

    def addFingersMovement(self, q1: dict[str, float], q2: dict[str, float]) -> None:
        traj = interpolateToJointTrajectory(q1, q2)
        self._movements.append( GraspTraj.Movement('fingers', traj, q1, q2) )

    def getReversed(self) -> GraspTraj:
        traj_rev = GraspTraj(self._side)
        # mov_rev = list(reversed(self._movements))
        # for idx in range(len(mov_rev)-1):
        #     mov1 = mov_rev[idx]
        #     mov2 = mov_rev[idx+1]
        #     if mov1.mov_type == 'arm' and mov2.mov_type == 'fingers':
        #         mov_rev[idx] = mov2
        #         mov_rev[idx+1] = mov1

        # for mov in mov_rev: #reversed(self._movements):
        for mov in reversed(self._movements):
            if mov.mov_type == 'arm':
                traj_rev.addArmMovement(reverseTrajectory(mov.traj), mov.q2, mov.q1)
            elif mov.mov_type == 'fingers':
                traj_rev.addFingersMovement(mov.q2, mov.q1)
        return traj_rev

    def getArmTrajectories(self) -> list[JointTrajectory]:
        return [mov.traj for mov in self._movements if mov.mov_type == 'arm']

    def getMovements(self):
        return self._movements

    def getInitialFingersConf(self) -> dict[str, float]:
        for mov in self._movements:
            if mov.mov_type == 'fingers':
                return mov.q1
        raise Exception()

class ConfA:
    def __init__(self, side: str, q: list[float]):
        self.side = side
        self.q = q

    @staticmethod
    def fromDict(d: dict) -> ConfA:
        return ConfA(d['Side'], [float(d['ArmConfig'][x]) for x in ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6']])

    @staticmethod
    def fromJsMap(side: str, js_map: dict[str, float]) -> ConfA:
        return ConfA(side, [js_map[f'{side}_arm_{idx}_joint'] for idx in range(7)])

    def toJsMap(self) -> dict[str, float]:
        return {f'{self.side}_arm_{idx}_joint': self.q[idx] for idx in range(7)}

    def getJointNames(self) -> list[str]:
        return [f'{self.side}_arm_{idx}_joint' for idx in range(7)]

    def __str__(self):
        out = ''
        for q in self.q:
            if out:
                out += ', '
            out += f'{q:.1f}'
        return out

    def __eq__(self, o):
        global _epsilon
        if not isinstance(o, ConfA):
            return False
        for q1, q2 in zip(self.q, o.q):
            if abs(q1-q2) > _epsilon:
                return False
        return True
    

class ConfT:
    def __init__(self, q: float):
        self.q = [q]

    @staticmethod
    def fromDict(d) -> ConfT:
        return ConfT(float(d))

    @staticmethod
    def fromJsMap(js_map: dict[str, float]) -> ConfT:
        return ConfT(js_map['torso_0_joint'])

    def toJsMap(self) -> dict[str, float]:
        return {'torso_0_joint': self.q[0]}

    def getJointNames(self) -> list[str]:
        return ['torso_0_joint']

    def __str__(self):
        return f'{self.q[0]:.2f}'

    def __eq__(self, o):
        global _epsilon
        if not isinstance(o, ConfT):
            return False
        for q1, q2 in zip(self.q, o.q):
            if abs(q1-q2) > _epsilon:
                return False
        return True
    

class ConfG:
    _joint_names = [
            '_HandFingerOneKnuckleOneJoint',
            '_HandFingerOneKnuckleTwoJoint',
            '_HandFingerOneKnuckleThreeJoint',
            '_HandFingerTwoKnuckleOneJoint',
            '_HandFingerTwoKnuckleTwoJoint',
            '_HandFingerTwoKnuckleThreeJoint',
            '_HandFingerThreeKnuckleTwoJoint',
            '_HandFingerThreeKnuckleThreeJoint']

    def __init__(self, side: str, q: list[float]):
        self.side = side
        self.q = q

    @staticmethod
    def fromDict(d: dict) -> ConfG:
        return ConfG(d['Side'], [float(d['HandConfig'][x]) for x in ['sp', 'f0a', 'f0b', 'sp', 'f1a', 'f1b', 'f2a', 'f2b']])

    @staticmethod
    def fromJsMap(side: str, js_map: dict[str, float]) -> ConfG:
        js_map = copy.copy(js_map)
        if not f'{side}_HandFingerTwoKnuckleOneJoint' in js_map:
            # spread joint
            js_map[f'{side}_HandFingerTwoKnuckleOneJoint'] = js_map[f'{side}_HandFingerOneKnuckleOneJoint']
        return ConfG(side, [js_map[f'{side}{name}'] for idx, name in enumerate(ConfG._joint_names)])

    def toJsMap(self) -> dict[str, float]:
        return {f'{self.side}{name}': self.q[idx] for idx, name in enumerate(self._joint_names)}

    def getJointNames(self) -> list[str]:
        return [f'{self.side}{name}' for name in self._joint_names]

    def __str__(self):
        out = ''
        for q in self.q:
            if out:
                out += ', '
            out += f'{q:.1f}'
        return out

    def __eq__(self, o):
        global _epsilon
        if not isinstance(o, ConfG):
            return False
        # Do not check distal joints
        for idx in [0, 1, 4, 6]:
            q1 = self.q[idx]
            q2 = o.q[idx]
            if abs(q1-q2) > _epsilon:
                return False
        return True
    

class ConfH:
    def __init__(self, hp: float, ht: float):
        self.q = [hp, ht]

    @staticmethod
    def fromDict(d) -> ConfH:
        return ConfH(float(d['hp']), float(d['ht']))

    @staticmethod
    def fromJsMap(side: str, js_map: dict[str, float]) -> ConfH:
        return ConfH(js_map['head_pan_joint'], js_map['head_tilt_joint'])

    def toJsMap(self) -> dict[str, float]:
        return {'head_pan_joint': self.q[0], 'head_tilt_joint': self.q[1]}

    def getJointNames(self) -> list[str]:
        return ['head_pan_joint', 'head_tilt_joint']

    def __str__(self):
        return f'{self.q[0]}, {self.q[1]}'
    
    def __eq__(self, o):
        global _epsilon
        if not isinstance(o, ConfH):
            return False
        for q1, q2 in zip(self.q, o.q):
            if abs(q1-q2) > _epsilon:
                return False
        return True

_epsilon = 0.000001

class Placement:
    def __init__(self, T_W_F: PoseWithFreeDOF, T_F_O_list: list[KdlFrame]):
        self.T_W_F = T_W_F
        self.T_F_O_list = T_F_O_list

def buildFakeTrajectory(side: str, q1_map: dict[str, float],
                        q2_map: dict[str, float]) -> JointTrajectory:
    result = JointTrajectory()
    qa1 = ConfA.fromJsMap(side, q1_map)
    qt1 = ConfT.fromJsMap(q1_map)
    qa2 = ConfA.fromJsMap(side, q2_map)
    qt2 = ConfT.fromJsMap(q2_map)
    result.joint_names = qt1.getJointNames() + qa1.getJointNames()
    p1 = JointTrajectoryPoint()
    p2 = JointTrajectoryPoint()
    p1.positions = qt1.q + qa1.q
    p1.time_from_start = Duration()
    p2.positions = qt2.q + qa2.q
    p2.time_from_start = Duration(sec=2.0)
    result.points = [p1, p2]
    return result

class Volume:
    def __init__(self, col_shapes: list[PrimitiveShape], ign_obj_list: list[str]):
        self.col_shapes = col_shapes
        self.ign_obj_list = ign_obj_list
        # import json
        # print(json.dumps(self.toDict(), indent=2, ensure_ascii=False))
        # raise

    def __eq(self, o):
        if len(self.ign_obj_list) != len(o.ign_obj_list):
            return False
        for x in self.ign_obj_list:
            if not x in o.ign_obj_list:
                return False
            
        if len(self.col_shapes) != len(o.col_shapes):
            return False
        # The same order is assumed
        for col1, col2 in zip(self.col_shapes, o.col_shapes):
            if col1 != col2:
                return False
        return True

    @staticmethod
    def fromDict(d) -> Volume:
        col_shapes = [PrimitiveShape.fromDict(shape) for shape in d['col_shapes']]
        ign_obj_list = d['ign_obj_list']
        return Volume(col_shapes, ign_obj_list)

    def toDict(self) -> dict[str, Any]:
        out = {
            'col_shapes': [shape.toDict() for shape in self.col_shapes],
            'ign_obj_list': self.ign_obj_list,
        }
        return out

    def __str__(self):
        return f'shapes: {len(self.col_shapes)}'
    

class FlatFeature:
    def __init__(self, T_O_F, tp, size: list[float]):
        self.T_O_F = T_O_F
        if tp == 'circle':
            assert len(size) == 1
        elif tp == 'rectangle':
            assert len(size) == 2
        else:
            raise Exception(f'')
        self.tp = tp
        self.size = size

class FlatBase(FlatFeature):
    pass

class Hole(FlatFeature):
    pass

