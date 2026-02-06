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


import math
from .data_types import KdlVector, KdlRotation, KdlFrame

def wrapAngle(angle):
    while angle > math.pi:
        angle = angle - 2.0*math.pi
    while angle < -math.pi:
        angle = angle + 2.0*math.pi
    return angle

class KinematicsSolverLWR4:
    """!
    Kinematics solver (FK, IK) for Kuka LWR 4+
    """

    def __init__(self):
        """!
        Initialization.
        """
        self.__A = 0.078
        self.__B = 0.39
        self.__C = 0.2
        self.__D = 0.3105
        self.__arm_len_a = 2*self.__C
        self.__arm_len_b = self.__B
        self.__debug = {}

        self.m_lim_lo = [-2.96, -2.09, -2.96, -2.095, -2.96, -2.09, -2.96]
        self.m_lim_up = [2.96, 2.09, 2.96, 2.095, 2.96, 2.09, 2.96]

    def getLimits(self):
        return zip(self.m_lim_lo, self.m_lim_up)

    def getDebug(self, key):
        """!
        Get some debug information after solving inverse kinematics.

        @param key string: one of values: 'T_A0_A6d', 'pt_shoulder', 'elbow_pt'.

        @return Value for the given key or None, if the key does not exist.
        """

        if key in self.__debug:
            return self.__debug[key]
        else:
            return None

    def __calcQ3(self, dist):
        a = self.__arm_len_a
        b = self.__arm_len_b
        acos_val = (a**2 + b**2 - dist**2) / (2*a*b)
        if abs(acos_val) <= 1.0:
            return math.pi - math.acos( acos_val )
        else:
            #print('ERROR: Could not compute arccos of {} = ({}**2 + {}**2 - {}**2) / (2*{}*{})'.format(acos_val, a, b, dist, a, b))
            pass
        #    raise Exception('Could not compute arccos of {} = ({}**2 + {}**2 - {}**2) / (2*{}*{})'.format(acos_val, a, b, dist, a, b))
        return None

    def __heronArea(self, a, b, c):
        s = (a+b+c)/2.0
        val = s*(s-a)*(s-b)*(s-c)
        if val < 0:
            raise Exception('Square root of negative number: {} = {}*({}-{})*({}-{})*({}-{})'.format(val, s, s, a, s, b, s, c))
        return math.sqrt(val)

    def __calculateIkPart1(self, T_A0_A6d: KdlFrame, elbow_circle_angle, flip_shoulder, flip_elbow):

        pt_shoulder = KdlVector(0, 0, self.__D)
        dir_vec = T_A0_A6d.p - pt_shoulder
        dist = dir_vec.Norm()
        q3 = self.__calcQ3(dist)

        if q3 is None:
            return None, None, None, None

        area = self.__heronArea(dist, self.__arm_len_a, self.__arm_len_b)
        height = area / (0.5*dist)
        dist_a = math.sqrt( self.__arm_len_a**2 - height**2 )

        # The elbow circle angle is calculated wrt the axis of the first joint of the arm.
        # This is the parameter for IK that deals with redundancy problem.

        shoulder_vec = pt_shoulder - KdlVector()
        shoulder_vec_norm = shoulder_vec.Norm()
        nz = dir_vec * 1.0
        nx = shoulder_vec * 1.0
        ny = nz*nx
        nx = ny*nz
        nx.Normalize()
        ny.Normalize()
        nz.Normalize()
        shoulder_frame = KdlFrame( KdlRotation(nx, ny, nz), pt_shoulder )
        shoulder_frame_rot = shoulder_frame * KdlFrame( KdlRotation.RotZ(elbow_circle_angle), KdlVector() )
        elbow_pt = shoulder_frame_rot * KdlVector(height, 0, dist_a)
        elbow_vec = elbow_pt - pt_shoulder
        q0 = math.atan2(-elbow_pt.y(), -elbow_pt.x())

        q1 = -math.atan2(math.sqrt(elbow_vec.x()**2 + elbow_vec.y()**2), elbow_vec.z())

        # Calculate q2:
        # Project dir_vec to frame of right_arm_2_link, and compute atan2(-dir_vec.y(), -dir_vec.x())

        nx = shoulder_vec * 1.0
        nz = elbow_vec * 1.0
        ny = nz*nx
        nx = ny*nz
        nx.Normalize()
        ny.Normalize()
        nz.Normalize()
        T_A0_A2 = KdlFrame( KdlRotation(nx, ny, nz), pt_shoulder )
        dir_vec_A0 = dir_vec
        dir_vec_A2 = T_A0_A2.Inverse().M * dir_vec_A0
        q2 = math.atan2(-dir_vec_A2.y(), -dir_vec_A2.x())
        # print(f'{nx.toDict()}; {ny.toDict()}; {nz.toDict()}')
        # print(f'__calculateIkPart1: {dir_vec.x()}, {dir_vec.y()}; {dir_vec_A2.x()}, {dir_vec_A2.y()}, {q2}')

        assert abs(shoulder_vec_norm - shoulder_vec.Norm()) < 0.0000001

        # There are two alternative angles for q3
        if flip_elbow:
            q0 = math.pi + q0
            q1 = -q1
            q3 = -q3

        if flip_shoulder:
            q0 = math.pi + q0
            q1 = -q1
            q2 = math.pi + q2

        # These points are saved for debugging
        self.__debug['pt_shoulder'] = pt_shoulder
        self.__debug['elbow_pt'] = elbow_pt
        #self.__debug['ee_pt'] = ee_pt

        q0 = wrapAngle(q0)
        q1 = wrapAngle(q1)
        q2 = wrapAngle(q2)
        q3 = wrapAngle(q3)
        if q0 < self.m_lim_lo[0] or q0 > self.m_lim_up[0] or\
                q1 < self.m_lim_lo[1] or q1 > self.m_lim_up[1] or\
                q2 < self.m_lim_lo[2] or q2 > self.m_lim_up[2] or\
                q3 < self.m_lim_lo[3] or q3 > self.m_lim_up[3]:
            return None, None, None, None

        # else
        return q0, q1, q2, q3

    def __calculateIkPart2(self, T_A0_A7d, q0, q1, q2, q3, flip_ee):

        #print T_A0_A7d
        s1 = math.sin(q0)
        c1 = math.cos(q0)
        s2 = math.sin(q1)
        c2 = math.cos(q1)
        s3 = math.sin(q2)
        c3 = math.cos(q2)
        s4 = math.sin(q3)
        c4 = math.cos(q3)
        nx = T_A0_A7d.M.UnitX()
        ny = T_A0_A7d.M.UnitY()
        nz = T_A0_A7d.M.UnitZ()
        r11 = nx.x()
        r21 = nx.y()
        r31 = nx.z()
        r12 = ny.x()
        r22 = ny.y()
        r32 = ny.z()
        r13 = nz.x()
        r23 = nz.y()
        r33 = nz.z()
        px = T_A0_A7d.p.x()
        py = T_A0_A7d.p.y()
        pz = T_A0_A7d.p.z()


        m02 = r13*((-s1*s3 + c1*c2*c3)*c4 + s2*s4*c1) + r23*((s1*c2*c3 + s3*c1)*c4 + s1*s2*s4) + r33*(-s2*c3*c4 + s4*c2)
        m12 = r13*(-s1*c3 - s3*c1*c2) + r23*(-s1*s3*c2 + c1*c3) + r33*s2*s3
        m20 = r11*(-(-s1*s3 + c1*c2*c3)*s4 + s2*c1*c4) + r21*(-(s1*c2*c3 + s3*c1)*s4 + s1*s2*c4) + r31*(s2*s4*c3 + c2*c4)
        m21 = r12*(-(-s1*s3 + c1*c2*c3)*s4 + s2*c1*c4) + r22*(-(s1*c2*c3 + s3*c1)*s4 + s1*s2*c4) + r32*(s2*s4*c3 + c2*c4)
        m22 = r13*(-(-s1*s3 + c1*c2*c3)*s4 + s2*c1*c4) + r23*(-(s1*c2*c3 + s3*c1)*s4 + s1*s2*c4) + r33*(s2*s4*c3 + c2*c4)

        #tg7 = s7/c7 = -m[2,1]/m[2,0]
        q6 = math.atan2(m21, -m20)

        #tg5 = s5/c5 = m[1,2]/m[0,2]
        q4 = math.atan2(m12, m02)

        #tg6 = s6/c6 = m[2,1]/(s7*m[2,2])
        q5 = math.atan2(m21, math.sin(q6)*m22)
        if m21 < 0:
            # This is tricky
            q5 = q5 + math.pi

        if flip_ee:
            q4 = q4 + math.pi
            q5 = -q5
            q6 = q6 + math.pi

        q4 = wrapAngle(q4)
        q5 = wrapAngle(q5)
        q6 = wrapAngle(q6)

        if q4 < self.m_lim_lo[4] or q4 > self.m_lim_up[4] or\
                q5 < self.m_lim_lo[5] or q5 > self.m_lim_up[5] or\
                q6 < self.m_lim_lo[6] or q6 > self.m_lim_up[6]:
            return None, None, None
        # else
        return q4, q5, q6

    def calculateIk(self, T_A0_A7d, elbow_circle_angle, flip_shoulder, flip_elbow, flip_ee):
        """!
        Calculate inverse kinematics (IK) for Kuka LWR 4+.

        @param T_A0_A7d KdlFrame: pose of the end-effector (the last link) wrt. the base of arm
        (the first link).
        @param elbow_circle_angle float: IK parameter.
        @param flip_shoulder bool: IK parameter.
        @param flip_elbow bool: IK parameter.
        @param flip_ee bool: IK parameter.

        @return 7-tuple: arm configuration or a tuple of Nones, if the solution does not exist.
        """

        self.__debug = {}
        T_A6_A7 = KdlFrame(KdlRotation(), KdlVector(0, 0, self.__A))
        T_A7_A6 = T_A6_A7.Inverse()
        T_A0_A6d = T_A0_A7d * T_A7_A6
        self.__debug['T_A0_A6d'] = T_A0_A6d
        q0, q1, q2, q3 = self.__calculateIkPart1(T_A0_A6d, elbow_circle_angle, flip_shoulder, flip_elbow)
        if q0 is None:
            return None, None, None, None, None, None, None
        # else
        q4, q5, q6 = self.__calculateIkPart2(T_A0_A7d, q0, q1, q2, q3, flip_ee)
        if q4 is None:
            return None, None, None, None, None, None, None
        # else
        return q0, q1, q2, q3, q4, q5, q6

    def calculateIkSet(self, T_A0_A7d, elbow_circle_angles):
        """!
        Calculate inverse kinematics (IK) for Kuka LWR 4+.

        @param T_A0_A7d KdlFrame: pose of the end-effector (the last link) wrt. the base of arm
        (the first link).
        @param elbow_circle_angles list of float: list of values for IK parameter: elbow_circle_angle (@see calculateIk).

        @return list of 7-tuples: arm configurations.
        """
        flips = [(False, False, False), (False, False, True), (False, True, False),
            (False, True, True), (True, False, False), (True, False, True), (True, True, False),
            (True, True, True)]
        solutions = []
        for elbow_circle_angle in elbow_circle_angles:
            for flip_shoulder, flip_elbow, flip_ee in flips:
                q = self.calculateIk(T_A0_A7d, elbow_circle_angle, flip_shoulder, flip_elbow,
                                                                                        flip_ee)
                #print('calculateIkSet: {}, {}, {}, {}'.format(elbow_circle_angle, flip_shoulder,
                #                                                            flip_elbow, flip_ee))
                #print('  {}'.format(q))
                if not q[0] is None:
                    solutions.append( q )
        return solutions

    def calculateFk(self, q):
        """!
        Calculate forward kinematics (FK) for Kuka LWR 4+.

        @param q 7-tuple: arm configuration.

        @return KdlFrame: pose of the last link wrt. the first link of the arm.
        """
        s1 = math.sin(q[0])
        c1 = math.cos(q[0])
        s2 = math.sin(q[1])
        c2 = math.cos(q[1])
        s3 = math.sin(q[2])
        c3 = math.cos(q[2])
        s4 = math.sin(q[3])
        c4 = math.cos(q[3])
        s5 = math.sin(q[4])
        c5 = math.cos(q[4])
        s6 = math.sin(q[5])
        c6 = math.cos(q[5])
        s7 = math.sin(q[6])
        c7 = math.cos(q[6])

        A = self.__A
        B = self.__B
        C = self.__C
        D = self.__D

        #m00 = -(((s4*s6 + c4*c5*c6)*c7 - s5*s7*c4)*s3 + (s5*c6*c7 + s7*c5)*c3)*s1 + (((s4*s6 + c4*c5*c6)*c7 - s5*s7*c4)*c2*c3 + ((s4*c5*c6 - s6*c4)*c7 - s4*s5*s7)*s2 - (s5*c6*c7 + s7*c5)*s3*c2)*c1
        #m01 = -((-(s4*s6 + c4*c5*c6)*s7 - s5*c4*c7)*s3 + (-s5*s7*c6 + c5*c7)*c3)*s1 + ((-(s4*s6 + c4*c5*c6)*s7 - s5*c4*c7)*c2*c3 + (-(s4*c5*c6 - s6*c4)*s7 - s4*s5*c7)*s2 - (-s5*s7*c6 + c5*c7)*s3*c2)*c1
        #m02 = -((-s4*c6 + s6*c4*c5)*s3 + s5*s6*c3)*s1 + ((-s4*c6 + s6*c4*c5)*c2*c3 + (s4*s6*c5 + c4*c6)*s2 - s3*s5*s6*c2)*c1
        #m03 = -((-A*s4*c6 - B*s4 + A*s6*c4*c5)*s3 + A*s5*s6*c3)*s1 + ((-A*s4*c6 - B*s4 + A*s6*c4*c5)*c2*c3 + (A*s4*s6*c5 + A*c4*c6 + B*c4 + C)*s2 + C*s2 - A*s3*s5*s6*c2)*c1
        #m10 = (((s4*s6 + c4*c5*c6)*c7 - s5*s7*c4)*s3 + (s5*c6*c7 + s7*c5)*c3)*c1 + (((s4*s6 + c4*c5*c6)*c7 - s5*s7*c4)*c2*c3 + ((s4*c5*c6 - s6*c4)*c7 - s4*s5*s7)*s2 - (s5*c6*c7 + s7*c5)*s3*c2)*s1
        #m11 = ((-(s4*s6 + c4*c5*c6)*s7 - s5*c4*c7)*s3 + (-s5*s7*c6 + c5*c7)*c3)*c1 + ((-(s4*s6 + c4*c5*c6)*s7 - s5*c4*c7)*c2*c3 + (-(s4*c5*c6 - s6*c4)*s7 - s4*s5*c7)*s2 - (-s5*s7*c6 + c5*c7)*s3*c2)*s1
        #m12 = ((-s4*c6 + s6*c4*c5)*s3 + s5*s6*c3)*c1 + ((-s4*c6 + s6*c4*c5)*c2*c3 + (s4*s6*c5 + c4*c6)*s2 - s3*s5*s6*c2)*s1
        #m13 = ((-A*s4*c6 - B*s4 + A*s6*c4*c5)*s3 + A*s5*s6*c3)*c1 + ((-A*s4*c6 - B*s4 + A*s6*c4*c5)*c2*c3 + (A*s4*s6*c5 + A*c4*c6 + B*c4 + C)*s2 + C*s2 - A*s3*s5*s6*c2)*s1
        #m20 = -((s4*s6 + c4*c5*c6)*c7 - s5*s7*c4)*s2*c3 + ((s4*c5*c6 - s6*c4)*c7 - s4*s5*s7)*c2 + (s5*c6*c7 + s7*c5)*s2*s3
        #m21 = -(-(s4*s6 + c4*c5*c6)*s7 - s5*c4*c7)*s2*c3 + (-(s4*c5*c6 - s6*c4)*s7 - s4*s5*c7)*c2 + (-s5*s7*c6 + c5*c7)*s2*s3
        #m22 = -(-s4*c6 + s6*c4*c5)*s2*c3 + (s4*s6*c5 + c4*c6)*c2 + s2*s3*s5*s6
        #m23 = -(-A*s4*c6 - B*s4 + A*s6*c4*c5)*s2*c3 + (A*s4*s6*c5 + A*c4*c6 + B*c4 + C)*c2 + A*s2*s3*s5*s6 + C*c2 + D

        c4c5c6 = c4*c5*c6
        c4c5s6 = s6*c4*c5
        s4s6 = s4*s6
        s5c6s7 = s5*s7*c6
        c4s5s7 = s5*s7*c4
        s4c6 = s4*c6
        s5c6c7 = s5*c6*c7
        c4s5c7 = s5*c4*c7
        s4c5c6 = s4*c5*c6
        s4s5c7 = s4*s5*c7
        c5s7 = s7*c5
        c4s6 = s6*c4
        c3s5s6 = s5*s6*c3
        s3s5s6 = s3*s5*s6
        s2c3 = s2*c3
        c5c7 = c5*c7
        c2s3 = s3*c2
        c4c6 = c4*c6
        s4c5s6 = s4s6*c5
        e1 = (s4s6 + c4c5c6)*c7
        e2 = (s4s6 + c4c5c6)*s7
        e3 = (s4c5c6 - c4s6)*s7
        e4 = (s4c5c6 - c4s6)*c7
        e6 = (-A*s4c6 - B*s4 + A*c4c5s6)
        e8 = (A*s4c5s6 + A*c4c6 + B*c4 + C)
        e9 = e6*c2*c3 + e8*s2 + C*s2 - A*s3s5s6*c2
        m00 = -((e1 - c4s5s7)*s3 + (s5c6c7 + c5s7)*c3)*s1 + ((e1 - c4s5s7)*c2*c3 + (e4 - s4*s5*s7)*s2 - (s5c6c7 + c5s7)*c2s3)*c1
        m01 = -((-e2 - c4s5c7)*s3 + (-s5c6s7 + c5c7)*c3)*s1 + ((-e2 - c4s5c7)*c2*c3 + (-e3 - s4s5c7)*s2 - (-s5c6s7 + c5c7)*c2s3)*c1
        m02 = -((-s4c6 + c4c5s6)*s3 + c3s5s6)*s1 + ((-s4c6 + c4c5s6)*c2*c3 + (s4c5s6 + c4c6)*s2 - s3s5s6*c2)*c1
        m03 = -(e6*s3 + A*c3s5s6)*s1 + e9*c1
        m10 = ((e1 - c4s5s7)*s3 + (s5c6c7 + c5s7)*c3)*c1 + ((e1 - c4s5s7)*c2*c3 + (e4 - s4*s5*s7)*s2 - (s5c6c7 + c5s7)*c2s3)*s1
        m11 = ((-e2 - c4s5c7)*s3 + (-s5c6s7 + c5c7)*c3)*c1 + ((-e2 - c4s5c7)*c2*c3 + (-e3 - s4s5c7)*s2 - (-s5c6s7 + c5c7)*c2s3)*s1
        m12 = ((-s4c6 + c4c5s6)*s3 + c3s5s6)*c1 + ((-s4c6 + c4c5s6)*c2*c3 + (s4c5s6 + c4c6)*s2 - s3s5s6*c2)*s1
        m13 = (e6*s3 + A*c3s5s6)*c1 + e9*s1
        m20 = -(e1 - c4s5s7)*s2c3 + (e4 - s4*s5*s7)*c2 + (s5c6c7 + c5s7)*s2*s3
        m21 = -(-e2 - c4s5c7)*s2c3 + (-e3 - s4s5c7)*c2 + (-s5c6s7 + c5c7)*s2*s3
        m22 = -(-s4c6 + c4c5s6)*s2c3 + (s4c5s6 + c4c6)*c2 + s2*s3s5s6
        m23 = -e6*s2c3 + e8*c2 + A*s2*s3s5s6 + C*c2 + D

        nx = KdlVector(m00, m10, m20)
        ny = KdlVector(m01, m11, m21)
        nz = KdlVector(m02, m12, m22)
        return KdlFrame( KdlRotation(nx, ny, nz), KdlVector(m03, m13, m23))


class KinematicsSolverVelma:
    """!
    Kinematics solver (FK, IK) for WUT Velma robot
    """

    __T_Er_Gr = KdlFrame( KdlRotation.RPY(0, math.pi/2, 0), KdlVector(0.235, 0, -0.078) )
    __T_El_Gl = KdlFrame( KdlRotation.RPY(0, -math.pi/2, 0), KdlVector(-0.235, 0, -0.078) )
    __T_Er_Pr = KdlFrame( KdlRotation.RPY(0, math.pi/2, 0), KdlVector(0.115, 0, -0.078) )
    __T_El_Pl = KdlFrame( KdlRotation.RPY(0, -math.pi/2, 0), KdlVector(-0.115, 0, -0.078) )
    __T_Gr_Er = __T_Er_Gr.Inverse()
    __T_Gl_El = __T_El_Gl.Inverse()
    __T_Pr_Er = __T_Er_Pr.Inverse()
    __T_Pl_El = __T_El_Pl.Inverse()

    def __init__(self):
        """!
        Initialize.
        """

        # self.__robot = URDF.from_parameter_server()
        # self.__tree = kdl_tree_from_urdf_model(self.__robot)

        # self.__chain_ar_base = self.__tree.getChain('torso_base', 'calib_right_arm_base_link')
        # self.__chain_al_base = self.__tree.getChain('torso_base', 'calib_left_arm_base_link')
        # self.__chain_torso_link0 = self.__tree.getChain('torso_base', 'torso_link0')
        # self.__chain_ar_elbow = self.__tree.getChain('torso_base', 'right_arm_4_link')
        # self.__chain_al_elbow = self.__tree.getChain('torso_base', 'left_arm_4_link')

        # self.__fk_kdl_ar_base = PyKDL.ChainFkSolverPos_recursive(self.__chain_ar_base)
        # self.__fk_kdl_al_base = PyKDL.ChainFkSolverPos_recursive(self.__chain_al_base)
        # self.__fk_kdl_torso_link0 = PyKDL.ChainFkSolverPos_recursive(self.__chain_torso_link0)
        # self.__fk_kdl_ar_elbow = PyKDL.ChainFkSolverPos_recursive(self.__chain_ar_elbow)
        # self.__fk_kdl_al_elbow = PyKDL.ChainFkSolverPos_recursive(self.__chain_al_elbow)

        self.__ik_solver_lwr = KinematicsSolverLWR4()

    def getArmShoulderPosition(self, side_str, torso_angle):
        """!
        Calculate forward kinematics for shoulder.

        @param side_str string: either 'left' or 'right'.
        @param torso_angle float: angle of torso joint.

        @return KdlFrame: pose of the shoulder given by side_str.
        """

        if side_str == 'left':
            return self.getLeftArmShoulderPosition(torso_angle)
        elif side_str == 'right':
            return self.getRightArmShoulderPosition(torso_angle)
        else:
            raise Exception()

    def getLeftArmShoulderPosition(self, torso_angle):
        """!
        Calculate forward kinematics for left shoulder.

        @param torso_angle float: angle of torso joint.

        @return KdlFrame: pose of the left shoulder.
        """
        T_B_A = self.getLeftArmBaseFk(torso_angle)
        return T_B_A * KdlVector(0, 0, 0.2005+0.11)

    def getRightArmShoulderPosition(self, torso_angle):
        """!
        Calculate forward kinematics for right shoulder.

        @param torso_angle float: angle of torso joint.

        @return KdlFrame: pose of the right shoulder.
        """
        T_B_A = self.getRightArmBaseFk(torso_angle)
        return T_B_A * KdlVector(0, 0, 0.2005+0.11)

    def getArmBaseFk(self, side_str, torso_angle):
        if side_str == 'left':
            return self.getLeftArmBaseFk(torso_angle)
        elif side_str == 'right':
            return self.getRightArmBaseFk(torso_angle)
        else:
            raise Exception('Wrong side: "{}"'.format(side_str))

    def getLeftArmBaseFk(self, torso_angle):
        """!
        Calculate forward kinematics for left arm base.

        @param torso_angle float: angle of torso joint.

        @return KdlFrame: pose of the left arm base.
        """

        T_B_T0 = self.getTorsoFk(torso_angle)

        # T_T0_Abl = KdlFrame( KdlRotation.Quaternion(-0.34302, 0.36396, 0.59424, 0.62988),
        #                      KdlVector(-0.00067672, -0.0012734, 1.2033) )
        T_T0_Abl = KdlFrame( KdlRotation.Quaternion(-0.35367, 0.35363, 0.61266, 0.61197),
                             KdlVector(-0.0006, -0.0013113, 1.1733) )



        return T_B_T0 * T_T0_Abl

        # T_B_AB = KdlFrame()
        # q_kdl = PyKDL.JntArray(1)
        # q_kdl[0] = torso_angle
        # self.__fk_kdl_al_base.JntToCart(q_kdl, T_B_AB)
        # return T_B_AB

        # # FK for left ARM base
        # s0 = math.sin(torso_angle)
        # c0 = math.cos(torso_angle)
        # m00 = -0.5*s0
        # m01 = -c0
        # m02 = -0.86602540378614*s0
        # m03 = -0.000188676*s0
        # m10 = 0.5*c0
        # m11 = -s0
        # m12 = 0.86602540378614*c0
        # m13 = 0.000188676*c0
        # m20 = -0.866025403786140
        # m21 = 0.0
        # m22 = 0.5
        # m23 = 1.20335
        # nx = KdlVector(m00, m10, m20)
        # ny = KdlVector(m01, m11, m21)
        # nz = KdlVector(m02, m12, m22)
        # return KdlFrame( KdlRotation(nx, ny, nz), KdlVector(m03, m13, m23) )

    def getRightArmBaseFk(self, torso_angle):
        """!
        Calculate forward kinematics for right arm base.

        @param torso_angle float: angle of torso joint.

        @return KdlFrame: pose of the right arm base.
        """

        T_B_T0 = self.getTorsoFk(torso_angle)

        # T_T0_Abr = KdlFrame( KdlRotation.Quaternion(0.34279, -0.36416, 0.5937, 0.63039),
        #                      KdlVector(-0.000452, -0.00086339, 1.2033) )
        T_T0_Abr = KdlFrame( KdlRotation.Quaternion(0.35344, -0.35383, 0.61214, 0.6125),
                             KdlVector(-0.0004, -0.00088868, 1.1733) )

        return T_B_T0 * T_T0_Abr

        # T_B_AB = KdlFrame()
        # q_kdl = PyKDL.JntArray(1)
        # q_kdl[0] = torso_angle
        # self.__fk_kdl_ar_base.JntToCart(q_kdl, T_B_AB)
        # return T_B_AB

        # # FK for right arm base
        # s0 = math.sin(torso_angle)
        # c0 = math.cos(torso_angle)
        # m00 = -0.5*s0
        # m01 = -c0
        # m02 = 0.86602540378614*s0
        # m03 = 0.000188676*s0
        # m10 = 0.5*c0
        # m11 = -s0
        # m12 = -0.86602540378614*c0
        # m13 = -0.000188676*c0
        # m20 = 0.866025403786140
        # m21 = 0.0
        # m22 = 0.5
        # m23 = 1.20335
        # nx = KdlVector(m00, m10, m20)
        # ny = KdlVector(m01, m11, m21)
        # nz = KdlVector(m02, m12, m22)
        # return KdlFrame( KdlRotation(nx, ny, nz), KdlVector(m03, m13, m23) )

    # def getArmElbowFk(self, side_str, torso_angle, q):
    #     """!
    #     Calculate forward kinematics for arm elbow.

    #     @param torso_angle float: angle of torso joint.
    #     @param q 7-tuple: arm configuration.

    #     @return KdlFrame: pose of the arm elbow.
    #     """

    #     T_B_EL = KdlFrame()
    #     q_kdl = PyKDL.JntArray(5)
    #     q_kdl[0] = torso_angle
    #     q_kdl[1] = q[0]
    #     q_kdl[2] = q[1]
    #     q_kdl[3] = q[2]
    #     q_kdl[4] = q[3]
    #     if side_str == 'right':
    #         self.__fk_kdl_ar_elbow.JntToCart(q_kdl, T_B_EL)
    #     elif side_str == 'left':
    #         self.__fk_kdl_al_elbow.JntToCart(q_kdl, T_B_EL)
    #     else:
    #         raise Exception('Wrong side: "{}". Expected "left" or "right".'.format(side_str))
    #     return T_B_EL

    def getArmFkJsMap(self, side: str, js_map: dict[str, float]) -> KdlFrame:
        torso_angle = js_map['torso_0_joint']
        q: list[float] = []
        for idx in range(7):
            q.append( js_map[f'{side}_arm_{idx}_joint'] )
        return self.getArmFk(side, torso_angle, q)

    def getArmFk(self, side: str, torso_angle: float, q: list[float]) -> KdlFrame:
        if side == 'left':
            return self.getLeftArmFk(torso_angle, q)
        elif side == 'right':
            return self.getRightArmFk(torso_angle, q)
        else:
            raise Exception('Wrong side: "{}"'.format(side))

    def getLeftArmFk(self, torso_angle, q):
        """!
        Calculate forward kinematics for left arm end-effector (the last link).

        @param torso_angle float: angle of torso joint.
        @param q 7-tuple: arm configuration.

        @return KdlFrame: pose of the left arm end-effector (the last link).
        """

        T_B_AB = self.getLeftArmBaseFk(torso_angle)
        T_AB_E = self.__ik_solver_lwr.calculateFk(q)
        return T_B_AB * T_AB_E

    def getRightArmFk(self, torso_angle, q):
        """!
        Calculate forward kinematics for right arm end-effector (the last link).

        @param torso_angle float: angle of torso joint.
        @param q 7-tuple: arm configuration.

        @return KdlFrame: pose of the right arm end-effector (the last link).
        """
        T_B_AB = self.getRightArmBaseFk(torso_angle)
        T_AB_E = self.__ik_solver_lwr.calculateFk(q)
        return T_B_AB * T_AB_E

    @staticmethod
    def getLeft_T_E_G():
        """!
        Get constant transformation from the left end-effector (the last link, frame 'E') to the left grasp frame 'G' in the center of the left gripper.

        @return KdlFrame: pose of frame 'G' wrt. frame 'E'.
        """
        return KinematicsSolverVelma.__T_El_Gl

    @staticmethod
    def getRight_T_E_G():
        """!
        Get constant transformation from the right end-effector (the last link, frame 'E') to the right grasp frame 'G' in the center of the right gripper.

        @return KdlFrame: pose of frame 'G' wrt. frame 'E'.
        """
        return KinematicsSolverVelma.__T_Er_Gr

    @staticmethod
    def getT_E_G(side_str):
        """!
        Get constant transformation from the the end-effector (the last link, frame 'E') to the grasp frame ('G', in the center of the gripper).

        @param side_str str: either 'left' or 'right'.

        @return KdlFrame: pose of frame 'E' wrt. frame 'G'.
        """
        if side_str == 'left':
            return KinematicsSolverVelma.getLeft_T_E_G()
        elif side_str == 'right':
            return KinematicsSolverVelma.getRight_T_E_G()
        # else:
        raise Exception('Wrong side_str: "{}"'.format(side_str))

    @staticmethod
    def getLeft_T_E_P():
        """!
        Get constant transformation from the left end-effector (the last link, frame 'E') to the left palm link (frame 'P').

        @return KdlFrame: pose of frame 'P' wrt. frame 'E'.
        """
        return KinematicsSolverVelma.__T_El_Pl

    @staticmethod
    def getRight_T_E_P():
        """!
        Get constant transformation from the right end-effector (the last link, frame 'E') to the right palm link (frame 'P').

        @return KdlFrame: pose of frame 'P' wrt. frame 'E'.
        """
        return KinematicsSolverVelma.__T_Er_Pr

    @staticmethod
    def getT_E_P(side_str):
        """!
        Get constant transformation from the end-effector (the last link, frame 'E') to the left palm link (frame 'P').

        @param side_str str: either 'left' or 'right'.

        @return KdlFrame: pose of frame 'P' wrt. frame 'E'.
        """
        if side_str == 'left':
            return KinematicsSolverVelma.__T_El_Pl
        elif side_str == 'right':
            return KinematicsSolverVelma.__T_Er_Pr
        # else:
        raise Exception('Wrong value for side_str: "{}", expected "left" or "right"'.format(
                                                                                        side_str))

    @staticmethod
    def getLeft_T_G_E():
        """!
        Inverse of velma_kinematics.velma_ik_geom.KinematicsSolverVelma.getLeft_T_E_G.

        @return KdlFrame: pose of frame 'E' wrt. frame 'G'.
        """
        return KinematicsSolverVelma.__T_Gl_El

    @staticmethod
    def getRight_T_G_E():
        """!
        Inverse of velma_kinematics.velma_ik_geom.KinematicsSolverVelma.getRight_T_E_G.

        @return KdlFrame: pose of frame 'E' wrt. frame 'G'.
        """
        return KinematicsSolverVelma.__T_Gr_Er

    @staticmethod
    def getT_G_E(side_str):
        """!
        Get constant transformation from the grasp frame 'G' in the center of the gripper to the end-effector (the last link, frame 'E').

        @param side_str str: either 'left' or 'right'.

        @return KdlFrame: pose of frame 'E' wrt. frame 'G'.
        """
        if side_str == 'left':
            return KinematicsSolverVelma.getLeft_T_G_E()
        elif side_str == 'right':
            return KinematicsSolverVelma.getRight_T_G_E()
        # else:
        raise Exception('Wrong side_str: "{}"'.format(side_str))

    @staticmethod
    def getLeft_T_P_E():
        """!
        Inverse of velma_kinematics.velma_ik_geom.KinematicsSolverVelma.getLeft_T_E_P.

        @return KdlFrame: pose of frame 'E' wrt. frame 'P'.
        """
        return KinematicsSolverVelma.__T_Pl_El

    @staticmethod
    def getRight_T_P_E():
        """!
        Inverse of velma_kinematics.velma_ik_geom.KinematicsSolverVelma.getRight_T_E_P.

        @return KdlFrame: pose of frame 'E' wrt. frame 'P'.
        """
        return KinematicsSolverVelma.__T_Pr_Er

    def calculateIkRightArm(self, T_B_W, torso_angle, elbow_circle_angle, flip_shoulder,
                                                                            flip_elbow, flip_ee):
        """!
        Calculate inverse kinematics (IK) for WUT Velma robot for right arm.
        @see calculateIkArm
        """
        return self.calculateIkArm('right', T_B_W, torso_angle, elbow_circle_angle, flip_shoulder,
                                                                            flip_elbow, flip_ee)

    def calculateIkLeftArm(self, T_B_W, torso_angle, elbow_circle_angle, flip_shoulder,
                                                                            flip_elbow, flip_ee):
        """!
        Calculate inverse kinematics (IK) for WUT Velma robot for left arm.
        @see calculateIkArm
        """
        return self.calculateIkArm('left', T_B_W, torso_angle, elbow_circle_angle, flip_shoulder,
                                                                            flip_elbow, flip_ee)

    def calculateIkArm(self, side_str, T_B_W, torso_angle, elbow_circle_angle, flip_shoulder,
                                                                            flip_elbow, flip_ee):
        """!
        Calculate inverse kinematics (IK) for WUT Velma robot arm.

        @param side_str string: either 'left' or 'right'.
        @param T_B_W KdlFrame: pose of the end-effector (the last link) wrt. the robot base.
        @param torso_angle float: angle of the torso joint.
        @param elbow_circle_angle float: IK parameter.
        @param flip_shoulder bool: IK parameter.
        @param flip_elbow bool: IK parameter.
        @param flip_ee bool: IK parameter.

        @return 7-tuple: arm configuration or a tuple of Nones, if the solution does not exist.
        """
        assert isinstance(T_B_W, KdlFrame)
        assert isinstance(torso_angle, float)
        assert isinstance(elbow_circle_angle, float)

        T_A0_A7d = self.getArmBaseFk(side_str, torso_angle).Inverse() * T_B_W
        return self.__ik_solver_lwr.calculateIk(T_A0_A7d, elbow_circle_angle, flip_shoulder,
                                                                                flip_elbow, flip_ee)

    def calculateIkSet(self, side_str, T_B_W, torso_angle, elbow_circle_angles):
        """!
        @see KinematicsSolverLWR4.calculateIkSet
        """
        assert side_str in ('left', 'right')
        assert isinstance(T_B_W, KdlFrame)
        assert isinstance(torso_angle, float)

        T_A0_A7d = self.getArmBaseFk(side_str, torso_angle).Inverse() * T_B_W
        return self.__ik_solver_lwr.calculateIkSet(T_A0_A7d, elbow_circle_angles)

    def getArmLimits(self):
        return self.__ik_solver_lwr.getLimits()

    def getTorsoLimits(self):
        return (-1.57, 1.57)

    def getTorsoFk(self, torso_angle) -> KdlFrame:
        """!
        Calculate forward kinematics for torso_link0.

        @param torso_angle float: angle of torso joint.

        @return KdlFrame: pose of torso_link0.
        """

        T_B_T0 = KdlFrame(KdlRotation.RotZ(torso_angle), KdlVector(0, 0, 0.03))
        return T_B_T0

        # T_B_T = KdlFrame()
        # q_kdl = PyKDL.JntArray(1)
        # q_kdl[0] = torso_angle
        # self.__fk_kdl_torso_link0.JntToCart(q_kdl, T_B_T)
        # return T_B_T

    def getArmLmitsCartImp(self) -> list[list[tuple[float, float]]]:
        return [
            [(-2.96706, 2.96706)],
            [(-2.0944, -0.2), (0.2, 2.0944)],
            [(-2.96706, 2.96706)],
            [(-2.0944, -0.2), (0.2, 2.0944)],
            [(-2.96706, 2.96706)],
            [(-2.0944, -0.2), (0.2, 2.0944)],
            [(-2.96706, 2.96706)]
        ]


class KinematicsSolverBarrettHand:
    """!
    Kinematics solver (FK) for BarrettHand gripper
    """

    def __init__(self):
        self.__T_P_G = KdlFrame( KdlVector(0, 0, 0.12) )

        # joint_name, parent_link, child_link, joint_rpy, joint_xyz, joint_axis
        kin_parameters = [
            ('_HandFingerThreeKnuckleTwoJoint',
                '_HandPalmLink', '_HandFingerThreeKnuckleTwoLink',
                (-1.57079632679, 0, -1.57079632679), (0, -0.05, 0.0754), (0, 0, -1)),
            ('_HandFingerThreeKnuckleThreeJoint',
                '_HandFingerThreeKnuckleTwoLink', '_HandFingerThreeKnuckleThreeLink',
                (0, 0, -0.690452252), (0.07, 0, 0), (0, 0, -1)),
            ('_HandFingerOneKnuckleOneJoint',
                '_HandPalmLink', '_HandFingerOneKnuckleOneLink',
                (0, 0, 1.57079632679), (0.02497, 0, 0.0587966), (0, 0, -1)),
            ('_HandFingerOneKnuckleTwoJoint',
                '_HandFingerOneKnuckleOneLink', '_HandFingerOneKnuckleTwoLink',
                (-1.57079632679, 0, 0), (0.05, 0, 0.0166034), (0, 0, -1)),
            ('_HandFingerOneKnuckleThreeJoint',
                '_HandFingerOneKnuckleTwoLink', '_HandFingerOneKnuckleThreeLink',
                (0, 0, -0.690452252), (0.07, 0, 0), (0, 0, -1)),
            ('_HandFingerTwoKnuckleOneJoint',
                '_HandPalmLink', '_HandFingerTwoKnuckleOneLink',
                (0, 0, 1.57079632679), (-0.02497, 0, 0.0587966), (0, 0, 1)),
            ('_HandFingerTwoKnuckleTwoJoint',
                '_HandFingerTwoKnuckleOneLink', '_HandFingerTwoKnuckleTwoLink',
                (-1.57079632679, 0, 0), (0.05, 0, 0.0166034), (0, 0, -1)),
            ('_HandFingerTwoKnuckleThreeJoint',
                '_HandFingerTwoKnuckleTwoLink', '_HandFingerTwoKnuckleThreeLink',
                (0, 0, -0.690452252), (0.07, 0, 0), (0, 0, -1)),
        ]
        self._kin_parameters = {}
        for side in ['left', 'right']:
            self._kin_parameters[side] = []
            for joint_name, parent_link_name, link_name, rpy, xyz, axis in kin_parameters:
                self._kin_parameters[side].append( (f'{side}{joint_name}',
                        f'{side}{parent_link_name}', f'{side}{link_name}', rpy, xyz, axis) )

    def getT_P_G(self):
        return self.__T_P_G


    def calculateFK(self, side: str, q: list[float]) -> dict[str, KdlFrame]:
        if len(q) == 4:
            f0a = q[0]
            f0b = q[0] * 0.3333333
            f1a = q[1]
            f1b = q[1] * 0.3333333
            f2a = q[2]
            f2b = q[2] * 0.3333333
            sp = q[3]
        elif len(q) == 7:
            f0a = q[0]
            f0b = q[1]
            f1a = q[2]
            f1b = q[3]
            f2a = q[4]
            f2b = q[5]
            sp = q[6]
        else:
            raise Exception('KinematicsSolverBarrettHand.calculateFK({}): '\
                    'Wrong number of joint angles'.format(q))

        q_map = {
            f'{side}_HandFingerOneKnuckleTwoJoint':f0a,
            f'{side}_HandFingerOneKnuckleThreeJoint':f0b,
            f'{side}_HandFingerTwoKnuckleTwoJoint':f1a,
            f'{side}_HandFingerTwoKnuckleThreeJoint':f1b,
            f'{side}_HandFingerThreeKnuckleTwoJoint':f2a,
            f'{side}_HandFingerThreeKnuckleThreeJoint':f2b,
            f'{side}_HandFingerOneKnuckleOneJoint':sp,
            f'{side}_HandFingerTwoKnuckleOneJoint':sp,
        }

        return self.calculateJsMapFK(side, q_map)

    def calculateJsMapFK(self, side: str, q_map: dict[str, float]) -> dict[str, KdlFrame]:
        # Calculate positions of distal joints, if they are missing:
        distal_suffixes = {
            f'{side}_HandFingerOneKnuckleThreeJoint': (f'{side}_HandFingerOneKnuckleTwoJoint', 0.3333333),
            f'{side}_HandFingerTwoKnuckleThreeJoint': (f'{side}_HandFingerTwoKnuckleTwoJoint', 0.3333333),
            f'{side}_HandFingerThreeKnuckleThreeJoint': (f'{side}_HandFingerThreeKnuckleTwoJoint', 0.3333333)}
        for distal_suffix in distal_suffixes:
            if not distal_suffix in q_map:
                proximal_suffix, mult = distal_suffixes[distal_suffix]
                q_map[distal_suffix] = q_map[proximal_suffix] * mult

        fk = {f'{side}_HandPalmLink': KdlFrame()}

        for joint_param in self._kin_parameters[side]:
            joint_name, parent_link, child_link, joint_rpy, joint_xyz, joint_axis = joint_param
            q = q_map[joint_name]
            axis = KdlVector(joint_axis[0], joint_axis[1], joint_axis[2])

            fk[child_link] = fk[parent_link] *\
                    KdlFrame(KdlRotation.RPY(joint_rpy[0], joint_rpy[1], joint_rpy[2]),
                                    KdlVector(joint_xyz[0], joint_xyz[1], joint_xyz[2])) *\
                                                        KdlFrame(KdlRotation.Rot(axis, q))

        result = {}
        for link_name, pose in fk.items():
            result[link_name] = pose
        return result

