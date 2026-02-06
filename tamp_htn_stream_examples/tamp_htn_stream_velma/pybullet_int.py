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

from .data_types import KdlFrame

import pybullet as p
import pybullet_data

# Use case #1:
# - add some objects from environment with mask (e.g. tables, cabinets, some moveable objects)
# - add some additional objects (e.g. possible placements)

# Use case #2:
# - add some objects from environment with mask (e.g. tables, cabinets, some moveable objects)
# - add some other objects from environment with mask (e.g. an object)
# - ray cast

class PyBulletInterface:
    def __init__(self):
        self._cid = p.connect(p.DIRECT)  # albo p.GUI żeby zobaczyć
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)
        self._obj_id_to_bullet_id_map: dict[str, int] = {}
        self._bullet_id_to_obj_id_map: dict[int, str] = {}

    def assertEmpty(self):
        assert len(self._obj_id_to_bullet_id_map) == 0
        assert len(self._bullet_id_to_obj_id_map) == 0

    def getObjectIdPyBulletId(self, bullet_id: int) -> str:
        return self._bullet_id_to_obj_id_map[bullet_id]

    def getBulletIdByObjectId(self, obj_id: str) -> int:
        return self._obj_id_to_bullet_id_map[obj_id]

    def hasObject(self, obj_id: str) -> bool:
        return obj_id in self._obj_id_to_bullet_id_map

    def addObject(self, obj_id: str, obj_type: str, obj_size: list[float],
                  T_W_O: KdlFrame, group: int|None, mask: int|None) -> None:
        # Groups and masks:
        # GROUP_ROBOT    = 1 << 0
        # GROUP_OBSTACLE = 1 << 1
        if self.hasObject(obj_id):
            raise Exception(f'Object {obj_id} is already in the scene')
        if obj_type == 'box':
            col = p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[obj_size[0]/2, obj_size[1]/2, obj_size[2]/2]  # (x,y,z)/2
            )
        elif obj_type == 'cylinder':
            col = p.createCollisionShape(
                shapeType=p.GEOM_CYLINDER,
                radius=obj_size[0],
                height=obj_size[1]
                )
        elif obj_type == 'sphere':
            col = p.createCollisionShape(
                shapeType=p.GEOM_SPHERE,
                radius=obj_size[0]
                )
        else:
            raise Exception(f'Not supported: {obj_type}')

        qx, qy, qz, qw = T_W_O.M.GetQuaternion()
        bullet_id = p.createMultiBody(
            baseMass=0,  # 0 => static object
            baseCollisionShapeIndex=col,
            basePosition=[T_W_O.p.x(), T_W_O.p.y(), T_W_O.p.z()],
            baseOrientation=[qx, qy, qz, qw]
        )
        if not group is None and not mask is None:
            self._setGroupMaskForBody(bullet_id, group, mask)

        self._obj_id_to_bullet_id_map[obj_id] = bullet_id
        assert not bullet_id in self._bullet_id_to_obj_id_map
        self._bullet_id_to_obj_id_map[bullet_id] = obj_id

    def removeObject(self, obj_id: str) -> None:
        if not self.hasObject(obj_id):
            raise Exception(f'Object {obj_id} is not in the scene')
        bullet_id = self._obj_id_to_bullet_id_map[obj_id]
        p.removeBody(bullet_id)
        del self._obj_id_to_bullet_id_map[obj_id]
        del self._bullet_id_to_obj_id_map[bullet_id]

    def clearAll(self) -> None:
        for bullet_id in self._bullet_id_to_obj_id_map:
            p.removeBody(bullet_id)
        self._obj_id_to_bullet_id_map = {}
        self._bullet_id_to_obj_id_map = {}

    def _setGroupMaskForBody(self, body_id: int, group: int, mask: int) -> None:
        # ważne: ustaw dla bazy (-1) i dla wszystkich linków
        for link in [-1] + list(range(p.getNumJoints(body_id))):
            p.setCollisionFilterGroupMask(body_id, link, group, mask)

    def getContacts(self) -> list[tuple[str, str]]:
        # Wymuś detekcję kolizji bez symulacji krokowej
        p.performCollisionDetection()

        contacts = p.getContactPoints()
        print(f"contacts count = {len(contacts)}")

        result = []
        # Każdy kontakt to krotka (14 pól)
        # [5] positionOnAInWS, [6] positionOnBInWS, [7] contactNormalOnBInWS, [8] contactDistance
        for i, c in enumerate(contacts):
            bullet_id1 = c[1]
            bullet_id2 = c[2]
            if not bullet_id1 in self._bullet_id_to_obj_id_map or not bullet_id2 in self._bullet_id_to_obj_id_map:
                print(f'wrong id returned from pybullet: {bullet_id1}.{c[3]} {bullet_id2}.{c[4]}')
            #     b_id_list = sorted(list(self._bullet_id_to_obj_id_map.keys()))
            #     print( b_id_list )
            #     print( [self._bullet_id_to_obj_id_map[b_id] for b_id in b_id_list] )
                continue
            if bullet_id1 < bullet_id2:
                result.append( (self._bullet_id_to_obj_id_map[bullet_id1],
                                self._bullet_id_to_obj_id_map[bullet_id2]) )
            else:
                result.append( (self._bullet_id_to_obj_id_map[bullet_id2],
                                self._bullet_id_to_obj_id_map[bullet_id1]) )
            # posA = c[5]
            # posB = c[6]
            # nB = c[7]
            # dist = c[8]  # < 0 oznacza penetrację
            # normal_force = c[9]
            # print(f"[{i}] dist={dist:.6f}, normal_force={normal_force:.3f}")
            # print(f"    posA={posA}")
            # print(f"    posB={posB}")
            # print(f"    nB  ={nB}")
        return result

    def rayTestBatch(self, rays_from, rays_to):
        max_batch = p.MAX_RAY_INTERSECTION_BATCH_SIZE
        left_rays_from = rays_from
        left_rays_to = rays_to
        results = []
        while True:
            if len(left_rays_from) > max_batch:
                current_batch_from = left_rays_from[:max_batch]
                current_batch_to = left_rays_to[:max_batch]
                left_rays_from = left_rays_from[max_batch:]
                left_rays_to = left_rays_to[max_batch:]
            elif len(left_rays_from) > 0:
                current_batch_from = left_rays_from
                current_batch_to = left_rays_to
                left_rays_from = []
                left_rays_to = []
            else:
                break

            current_results = p.rayTestBatch(current_batch_from, current_batch_to)
            results = results + list(current_results)
        return results

            # # Remap results
            # for i, hit in enumerate(current_results):
            #     hit_obj, hit_link, hit_frac, hit_pos, hit_n = hit
            #     if hit_obj == -1:
            #         #print(f"Ray {i}: no hit")
            #         pass
            #     else:
            #         print(f"Ray {i}: hit obj={hit_obj}, frac={hit_frac:.3f}, pos={hit_pos}, n={hit_n}")



        

# def getEnvironmentCollisionShapesFcl(state: dict[str, Any], ignored_obj_id_str_list: list[str]
#                                      ) -> list[FclCollisionObject]:
#     sti = getStatesInterface()
#     obj_poses = sti.getObjectsAndExactPoses(state) + sti.getGraspedObjectsAndExactPoses(state)
#     result = []
#     for obj_id_str, T_W_O in obj_poses:
#         model = sti.getModelByObjectId(state, obj_id_str)
#         fcl_col_objects = getModelCollisionShapesFcl(model, T_W_O)
#         result = result + fcl_col_objects
#     return result

# def collideEnvironmentToMultipleObjects(evn_shapes: list[FclCollisionObject],
#                                         collision_objects: list[dict[str, Any]]) -> BatchCollisionResult:

#     # Create map from geometry IDs to objects
#     env_geom_id_to_idx = { id(geom) : idx for idx, (geom, obj) in enumerate(evn_shapes) }
#     env_shape_list = [obj for geom, obj in evn_shapes]

#     dyn_index = {}
#     dyn_shape_list = []
#     for col_obj in collision_objects:
#         # for shape_idx, shape in enumerate(col_obj['shapes']):
#         for geom, obj in col_obj['shapes']:
#             dyn_shape_list.append(obj)
#             dyn_index[id(geom)] = col_obj['id'] #f'{col_obj['id']}.{shape_idx}'

#     env_mgr = fcl.DynamicAABBTreeCollisionManager()
#     dyn_mgr = fcl.DynamicAABBTreeCollisionManager()

#     env_mgr.clear()
#     env_mgr.registerObjects(env_shape_list)
#     env_mgr.setup()

#     dyn_mgr.clear()
#     dyn_mgr.registerObjects(dyn_shape_list)
#     dyn_mgr.setup()

#     pairs: list[tuple[int, int]] = []

#     return_pairs = True

#     # callback for collection of pairs
#     # def _cb(o1, o2, cdata):
#     #     # cdata to CollisionData
#     #     if fcl.collide(o1, o2, cdata.request, cdata.result) > 0:
#     #         #print(f'col: {o1} {o2}')
#     #         if return_pairs:
#     #             # UWAGA: o1/o2 mogą przyjść w dowolnej kolejności
#     #             if o1 in dyn_index and o2 in env_index:
#     #                 print(f'pair: {dyn_index[o1]}, {env_index[o2]}')
#     #                 pairs.append((dyn_index[o1], env_index[o2]))
#     #             elif o2 in dyn_index and o1 in env_index:
#     #                 print(f'pair: {dyn_index[o2]}, {env_index[o1]}')
#     #                 pairs.append((dyn_index[o2], env_index[o1]))
#     #         # jeśli nie zbieramy par, możemy uciąć wcześniej
#     #         return True
#     #     return False

#     request = fcl.CollisionRequest(
#         enable_contact=True,
#         num_max_contacts=len(env_shape_list) * len(dyn_shape_list),
#     )
    
#     cdata = fcl.CollisionData(request, fcl.CollisionResult())

#     # broadphase: manager vs manager
#     # dyn_mgr.collide(env_mgr, cdata, _cb)
#     dyn_mgr.collide(env_mgr, cdata, fcl.defaultCollisionCallback)

#     objs_in_collision = []
#     for contact in cdata.result.contacts:
#         # Extract collision geometries that are in contact
#         coll_geom_0 = contact.o1
#         coll_geom_1 = contact.o2

#         # Get their names
#         if id(coll_geom_0) in env_geom_id_to_idx and id(coll_geom_1) in dyn_index:
#             objs_in_collision.append( (env_geom_id_to_idx[id(coll_geom_0)], dyn_index[id(coll_geom_1)]) ) 
#         elif id(coll_geom_1) in env_geom_id_to_idx and id(coll_geom_0) in dyn_index:
#             objs_in_collision.append( (env_geom_id_to_idx[id(coll_geom_1)], dyn_index[id(coll_geom_0)]) )

#     # print(f'pairs: {len(objs_in_collision)}')
#     any_col = cdata.result.is_collision
#     return BatchCollisionResult(any_collision=any_col, pairs=objs_in_collision)

# @dataclass
# class BatchCollisionResult:
#     any_collision: bool
#     pairs: list[tuple[int, int]]  # (idxA, idxB) indices in groups A and B

