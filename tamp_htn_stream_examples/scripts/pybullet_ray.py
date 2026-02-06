#!/usr/bin/env python3

import pybullet as p
import pybullet_data


def main():
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")

    # kilka obiektów
    box_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.3])
    ids = []
    for x in [1.5, 2.5, 3.5]:
        ids.append(p.createMultiBody(0, box_col, basePosition=[x, 0, 0.3]))

    print(f'ids: {ids}')
    
    rays_from = [
        [0, 0, 0.3],
        [0, 0.2, 0.3],
        [0, -0.2, 0.3],
    ]
    rays_to = [
        [5, 0, 0.3],
        [5, 0.2, 0.3],
        [5, -0.2, 0.3],
    ]

    results = p.rayTestBatch(rays_from, rays_to)

    for i, hit in enumerate(results):
        hit_obj, hit_link, hit_frac, hit_pos, hit_n = hit
        if hit_obj == -1:
            print(f"Ray {i}: no hit")
        else:
            print(f"Ray {i}: hit obj={hit_obj}, frac={hit_frac:.3f}, pos={hit_pos}, n={hit_n}")

    p.disconnect(cid)


if __name__ == "__main__":
    main()
