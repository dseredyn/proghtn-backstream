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

from tamp_htn_stream.core import State

from .data_types import PrimitiveShape
from .plugin import getStatesInterface, getPyBulletInterface
from .data_types import KdlFrame, KdlVector, Volume
from .velma_kinematics import KinematicsSolverVelma
from .generators import getCollidingObjects


####################################################################################################
# Extended predicates ##############################################################################
####################################################################################################


# (CanPlaceAt ?obj - ObjectId ?p - GeomPose)
def CanPlaceAt(state: State, inputs: list) -> bool:
    return True

# (OtherArmTucked ?sd - Side)
def OtherArmTucked(state: State, inputs: list) -> bool:
    o_side = inputs[0].getValue()
    if o_side == 'left':
        side = 'right'
    else:
        side = 'left'

    if side == 'right':
        T_T0_E = KdlFrame(KdlVector(0, -0.35, 0.55))
    else:
        T_T0_E = KdlFrame(KdlVector(0, 0.35, 0.55))
    pos_tol = [0.15, 0.15, 0.15]

    sti = getStatesInterface()
    q = sti.getConfA(state, side)
    qt = sti.getConfT(state)

    velma_solv = KinematicsSolverVelma()
    T_W_E = velma_solv.getArmFk(side, qt.q[0], q.q)
    T_W_T0 = velma_solv.getTorsoFk(qt.q[0])
    T_T0_Ec = T_W_T0.Inverse() * T_W_E

    diff = (T_T0_E.p - T_T0_Ec.p)
    result = abs(diff.x()) < pos_tol[0] and abs(diff.y()) < pos_tol[1] and abs(diff.z()) < pos_tol[2]
    print(f'OtherArmTucked: diff: {diff}, result: {result}')
    return result


def TrajBegin(state: State, inputs: list) -> bool:
    return True

def TrajEnd(state: State, inputs: list) -> bool:
    return True

def GraspTrajBegin(state: State, inputs: list) -> bool:
    return True

def GraspTrajEnd(state: State, inputs: list) -> bool:
    return True

def ObjSideGood(state: State, inputs: list) -> bool:
    sd, obj = inputs
    # TODO
    return True

def ObjSideBad(state: State, inputs: list) -> bool:
    sd, obj = inputs
    # TODO
    return True

def MovObjClear(state: State, inputs: list) -> bool:
    assert len(inputs) == 1
    assert inputs[0].getType() == 'Volume'
    vol = inputs[0].getValue()
    assert isinstance(vol, Volume)

    # Filter out colliding poses of the gripper for given placements
    pb = getPyBulletInterface()
    sti = getStatesInterface()
    GROUP_ENV = 1 << 0
    GROUP_VOL = 1 << 1

    for col_idx, col in enumerate(vol.col_shapes):
        assert isinstance(col, PrimitiveShape)
        pb.addObject(f'vol/vol/{col_idx}', col.tp, col.size,
                                        col.getShapePose('world'), GROUP_VOL, GROUP_ENV)

    # Prepare environment model with all moveable objects except volume-ignored object.
    # Ignore all grasped objects, static objects and volume-ignored objects.
    ign_obj_id_list = [obj_id_str for obj_id_str, _, _ in sti.getGraspedObjects(state)]
    ign_obj_id_list += sti.getStaticObjectsIds(state)
    ign_obj_id_list += vol.ign_obj_list
    env_shapes_dict = sti.getEnvironmentCollisionShapes(state, ign_obj_id_list)

    for env_obj_id, shape_list in env_shapes_dict.items():
        for idx, shape in enumerate(shape_list):
            assert isinstance(shape, PrimitiveShape)
            pb.addObject(f'env/{env_obj_id}/{idx}', shape.tp, shape.size, shape.getShapePose('world'), GROUP_ENV, GROUP_VOL)

    contacts = pb.getContacts()

    pb.clearAll()
    return len(contacts) == 0

def Picked(state: State, inputs: list) -> bool:
    assert len(inputs) == 1
    assert inputs[0].getType() == 'ObjectId'
    obj_id = inputs[0].getValue()
    sti = getStatesInterface()
    obj_model = sti.getModelByObjectId(state, obj_id)
    grasped_obj_list = sti.getGraspedObjectsAndExactPoses(state)
    T_W_O = None
    for gr_obj_id, gr_T_W_O in grasped_obj_list:
        if gr_obj_id == obj_id:
            T_W_O = gr_T_W_O
            break
    if T_W_O is None:
        raise Exception(f'Predicate "Picked" failed: object "{obj_id}" is not grasped.')
    # else:

    vol = Volume(sti.getModelCollisionShapes(obj_model, T_W_O, 0.03), [obj_id])
    collisions = getCollidingObjects(state, [vol])[0]
    return len(collisions) == 0

extended_predicates = [CanPlaceAt, OtherArmTucked, TrajBegin,
                       TrajEnd, GraspTrajBegin, GraspTrajEnd,
                       ObjSideGood, ObjSideBad, MovObjClear,
                       Picked]
def calculate_extended_predicate(state: State, name: str, inputs: list) -> bool:
    for ext_pred_func in extended_predicates:
        if name == ext_pred_func.__name__:
            return ext_pred_func(state, inputs)
    raise Exception(f'Unknown extended predicate: {name}')
