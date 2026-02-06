#!/usr/bin/env python3

import pybullet as p
import pybullet_data


def main():
    cid = p.connect(p.DIRECT)  # albo p.GUI żeby zobaczyć
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)

    # --- Tworzymy kształty kolizyjne ---
    box_col = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[0.5, 0.3, 0.2]  # (x,y,z)/2
    )

    cyl_col = p.createCollisionShape(
        shapeType=p.GEOM_CYLINDER,
        radius=0.25,
        height=1.0
    )

    # --- Ciała (MultiBody) ---
    # box jako statyczna przeszkoda
    box_id = p.createMultiBody(
        baseMass=0,  # 0 => statyczny
        baseCollisionShapeIndex=box_col,
        basePosition=[0, 0, 0.2]
    )

    # cylinder jako dynamiczny (masa > 0)
    cyl_id = p.createMultiBody(
        baseMass=1.0,
        baseCollisionShapeIndex=cyl_col,
        basePosition=[0.3, 0.0, 0.2]  # celowo blisko/nałożony na box
    )
# for bid in p.getBodyUniqueIdList():
#     p.removeBody(bid)
    print(f'box_id: {box_id}')
    print(f'cyl_id: {cyl_id}')

    # Wymuś detekcję kolizji bez symulacji krokowej
    p.performCollisionDetection()

    # --- Kontakty ---
    contacts = p.getContactPoints(bodyA=box_id, bodyB=cyl_id)
    print(f"contacts count = {len(contacts)}")

    # Każdy kontakt to krotka (14 pól)
    # [5] positionOnAInWS, [6] positionOnBInWS, [7] contactNormalOnBInWS, [8] contactDistance
    for i, c in enumerate(contacts[:5]):
        posA = c[5]
        posB = c[6]
        nB = c[7]
        dist = c[8]  # < 0 oznacza penetrację
        normal_force = c[9]
        print(f"[{i}] dist={dist:.6f}, normal_force={normal_force:.3f}")
        print(f"    posA={posA}")
        print(f"    posB={posB}")
        print(f"    nB  ={nB}")
        print(c)

    # --- Alternatywnie: "closest points" (działa też dla nie-kolidujących) ---
    closest = p.getClosestPoints(bodyA=box_id, bodyB=cyl_id, distance=0.05)
    print(f"closestPoints (<=0.05m): {len(closest)}")

    p.disconnect(cid)


if __name__ == "__main__":
    main()
