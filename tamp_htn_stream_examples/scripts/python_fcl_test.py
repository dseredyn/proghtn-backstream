#!/usr/bin/env python3

from __future__ import annotations
import numpy as np
import fcl
import sys
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional


# ----------------------------
# Helpers: transform builders
# ----------------------------
def fcl_tf_from_xyz_rpy(xyz: np.ndarray, rpy: np.ndarray) -> fcl.Transform:
    """
    xyz: (3,)
    rpy: (3,) roll,pitch,yaw [rad]
    """
    rx, ry, rz = rpy
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    # Rz * Ry * Rx
    R = np.array([
        [cz*cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx],
        [sz*cy, sz*sy*sx + cz*cx, sz*sy*cx - cz*sx],
        [-sy,   cy*sx,            cy*cx],
    ], dtype=float)

    t = np.array(xyz, dtype=float)
    return fcl.Transform(R, t)


def fcl_tf_identity(x=0.0, y=0.0, z=0.0) -> fcl.Transform:
    return fcl.Transform(np.eye(3), np.array([x, y, z], dtype=float))


def collide_dynamic_vs_static(self, return_pairs: bool = False) -> BatchCollisionResult:
    """
    Batch collision test: (dynamic group) vs (static group).
    Zwraca:
      - any_collision
      - (opcjonalnie) listę par (i_dyn, i_stat)
    """

    pairs: List[Tuple[int, int]] = []

    # map: object -> index (dla par)
    dyn_index = {obj: i for i, obj in enumerate(self.dynamic_objects)}
    sta_index = {obj: i for i, obj in enumerate(self.static_objects)}

    # callback zbierający pary
    def _cb(o1, o2, cdata):
        # cdata to CollisionData
        if fcl.collide(o1, o2, cdata.request, cdata.result) > 0:
            if return_pairs:
                # UWAGA: o1/o2 mogą przyjść w dowolnej kolejności
                if o1 in dyn_index and o2 in sta_index:
                    pairs.append((dyn_index[o1], sta_index[o2]))
                elif o2 in dyn_index and o1 in sta_index:
                    pairs.append((dyn_index[o2], sta_index[o1]))
            # jeśli nie zbieramy par, możemy uciąć wcześniej
            return True
        return False

    cdata = fcl.CollisionData(self.request, fcl.CollisionResult())

    # broadphase: manager vs manager
    self.dynamic_mgr.collide(self.static_mgr, cdata, _cb)  # :contentReference[oaicite:1]{index=1}

    any_col = cdata.result.is_collision
    return BatchCollisionResult(any_collision=any_col, pairs=pairs)

# ----------------------------
# Batch collision wrapper
# ----------------------------
@dataclass
class BatchCollisionResult:
    any_collision: bool
    pairs: List[Tuple[int, int]]  # (idxA, idxB) indices in groups A and B


class FCLBatchChecker:
    """
    Batch collision/distance between:
      - group A (dynamic) and group B (static)  OR  A vs A self-collision
    using DynamicAABBTreeCollisionManager.

    Typical usage:
      checker = FCLBatchChecker(static_objects=obstacles)
      checker.set_dynamic_objects(robot_links)
      checker.update_dynamic_transforms(list_of_transforms)
      res = checker.collide_dynamic_vs_static(return_pairs=True)
    """

    def __init__(
        self,
        static_objects: Optional[List[fcl.CollisionObject]] = None,
        enable_contact: bool = False,
        num_max_contacts: int = 1,
    ):
        self.static_mgr = fcl.DynamicAABBTreeCollisionManager()
        self.dynamic_mgr = fcl.DynamicAABBTreeCollisionManager()

        self.static_objects: List[fcl.CollisionObject] = []
        self.dynamic_objects: List[fcl.CollisionObject] = []

        self.request = fcl.CollisionRequest(
            enable_contact=bool(enable_contact),
            num_max_contacts=int(num_max_contacts),
        )

        if static_objects:
            self.set_static_objects(static_objects)

    # ---- manager setup ----
    def set_static_objects(self, objects: List[fcl.CollisionObject]) -> None:
        self.static_objects = objects
        self.static_mgr.clear()
        self.static_mgr.registerObjects(self.static_objects)
        self.static_mgr.setup()

    def set_dynamic_objects(self, objects: List[fcl.CollisionObject]) -> None:
        self.dynamic_objects = objects
        self.dynamic_mgr.clear()
        self.dynamic_mgr.registerObjects(self.dynamic_objects)
        self.dynamic_mgr.setup()

    def update_dynamic_transforms(self, transforms: List[fcl.Transform]) -> None:
        """
        Szybki update pozycji dla wszystkich obiektów dynamicznych.
        """
        if len(transforms) != len(self.dynamic_objects):
            raise ValueError(
                f"Expected {len(self.dynamic_objects)} transforms, got {len(transforms)}"
            )

        for obj, tf in zip(self.dynamic_objects, transforms):
            obj.setTransform(tf)
            # obj.computeAABB()

        # po zmianach warto odświeżyć broadphase
        self.dynamic_mgr.update()

    def update_static_transforms(self, transforms: List[fcl.Transform]) -> None:
        """
        Jeśli przeszkody też są ruchome.
        """
        if len(transforms) != len(self.static_objects):
            raise ValueError(
                f"Expected {len(self.static_objects)} transforms, got {len(transforms)}"
            )

        for obj, tf in zip(self.static_objects, transforms):
            obj.setTransform(tf)
            # obj.computeAABB()

        self.static_mgr.update()

    # ---- batch collide ----
    def collide_dynamic_vs_static(self, return_pairs: bool = False) -> BatchCollisionResult:
        """
        Batch collision test: (dynamic group) vs (static group).
        Zwraca:
          - any_collision
          - (opcjonalnie) listę par (i_dyn, i_stat)
        """

        pairs: List[Tuple[int, int]] = []

        # map: object -> index (dla par)
        dyn_index = {obj: i for i, obj in enumerate(self.dynamic_objects)}
        sta_index = {obj: i for i, obj in enumerate(self.static_objects)}

        # callback zbierający pary
        def _cb(o1, o2, cdata):
            # cdata to CollisionData
            if fcl.collide(o1, o2, cdata.request, cdata.result) > 0:
                if return_pairs:
                    # UWAGA: o1/o2 mogą przyjść w dowolnej kolejności
                    if o1 in dyn_index and o2 in sta_index:
                        pairs.append((dyn_index[o1], sta_index[o2]))
                    elif o2 in dyn_index and o1 in sta_index:
                        pairs.append((dyn_index[o2], sta_index[o1]))
                # jeśli nie zbieramy par, możemy uciąć wcześniej
                return True
            return False

        cdata = fcl.CollisionData(self.request, fcl.CollisionResult())

        # broadphase: manager vs manager
        self.dynamic_mgr.collide(self.static_mgr, cdata, _cb)  # :contentReference[oaicite:1]{index=1}

        any_col = cdata.result.is_collision
        return BatchCollisionResult(any_collision=any_col, pairs=pairs)

    def collide_dynamic_self(self, return_pairs: bool = False) -> BatchCollisionResult:
        """
        Batch self-collision inside dynamic group (N^2, ale broadphase optymalizuje).
        Zwraca:
          - any_collision
          - (opcjonalnie) listę par (i, j) gdzie i<j
        """

        pairs: List[Tuple[int, int]] = []
        dyn_index = {obj: i for i, obj in enumerate(self.dynamic_objects)}

        def _cb(o1, o2, cdata):
            if fcl.collide(o1, o2, cdata.request, cdata.result) > 0:
                if return_pairs:
                    i, j = dyn_index[o1], dyn_index[o2]
                    if i != j:
                        pairs.append((min(i, j), max(i, j)))
                return True
            return False

        cdata = fcl.CollisionData(self.request, fcl.CollisionResult())
        self.dynamic_mgr.collide(cdata, _cb)  # self-collide :contentReference[oaicite:2]{index=2}
        return BatchCollisionResult(bool(cdata.result.is_collision), pairs)

    # ---- batch distance ----
    def distance_dynamic_vs_static(self) -> float:
        """
        Minimalny dystans między dynamiczną grupą i statyczną grupą.
        Zwraca jeden skalar (min distance).
        """
        dreq = fcl.DistanceRequest(enable_nearest_points=False)
        dres = fcl.DistanceResult()

        # callback dystansowy: minimalizujemy globalnie
        def _dcb(o1, o2, ddata):
            d = fcl.distance(o1, o2, ddata.request, ddata.result)
            if d < ddata.result.min_distance:
                ddata.result.min_distance = d
            # False => nie ucinaj, bo chcemy prawdziwe minimum
            return False

        ddata = fcl.DistanceData(dreq, dres)
        #self.dynamic_mgr.distance(self.static_mgr, ddata, _dcb)
        self.dynamic_mgr.distance(self.static_mgr, ddata, fcl.defaultDistanceCallback)
        
        return float(ddata.result.min_distance)


# ----------------------------
# Example: build objects
# ----------------------------
def make_box(size_xyz: Tuple[float, float, float], tf: fcl.Transform) -> fcl.CollisionObject:
    geom = fcl.Box(*size_xyz)
    return fcl.CollisionObject(geom, tf)

def make_cylinder(radius: float, length: float, tf: fcl.Transform) -> fcl.CollisionObject:
    geom = fcl.Cylinder(radius, length)
    return fcl.CollisionObject(geom, tf)


def main():
    # statyczne przeszkody
    obstacles = [
        make_box((1.0, 1.0, 1.0), fcl_tf_identity(0.0, 0.0, 0.0)),
        make_cylinder(0.2, 1.0, fcl_tf_identity(2.0, 0.0, 0.0)),
    ]

    checker = FCLBatchChecker(static_objects=obstacles)

    # obiekty dynamiczne (np. linki robota, paczki na taśmie)
    dyn = [
        make_box((0.5, 0.2, 0.2), fcl_tf_identity(0.5, 0.0, 0.0)),
        make_cylinder(0.1, 0.5, fcl_tf_identity(3.0, 0.0, 0.0)),
    ]
    checker.set_dynamic_objects(dyn)

    # update pozycji (np. w pętli)
    checker.update_dynamic_transforms([
        fcl_tf_identity(0.4, 0.0, 0.0),
        fcl_tf_identity(2.1, 0.0, 0.0),
    ])

    # batch collide + pary
    res = checker.collide_dynamic_vs_static(return_pairs=True)
    print(res.any_collision, res.pairs)

    # minimalny dystans między grupami
    dmin = checker.distance_dynamic_vs_static()
    print("min distance:", dmin)

if __name__ == "__main__":
    raise SystemExit(main())
