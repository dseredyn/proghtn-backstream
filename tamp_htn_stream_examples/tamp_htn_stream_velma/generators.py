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
from .data_types import KdlVector, KdlRotation, KdlFrame, PoseWithFreeDOF,\
    PlanningSceneCreator, GraspTraj, reverseTrajectory, ConfA, ConfT, ConfH,\
    ConfG, PrimitiveShape, Placement, buildFakeTrajectory, Volume, FlatBase, Hole
import math
import time
import copy
import numpy as np
import random

from visualization_msgs.msg import Marker
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from tamp_htn_stream.core import State, GeneratorResult, TypedValue
from .moveit_interface import get_trajectory_first_point, get_trajectory_last_point,\
     cPoseStamped, cHeader

from .plugin import getStatesInterface, getPyBulletInterface, getVelmaWristConstraint,\
    getVelmaSolver

from .conversions import cMarkerSphereList, cMarkerCylinder, cMarkerBox, cMarkerCubeList,\
    cMarkerArrow, cMarker, joinJsMap

from .velma_kinematics import KinematicsSolverBarrettHand

from .func_tools import LinearIntervalFunction

####################################################################################################
# Utility functions ################################################################################
####################################################################################################

#_simplified_trajectory_generation = True
_simplified_trajectory_generation = False

_sampling_type = 'random_pdf'

def set_generator_sampling_type(sampling_type: str):
    assert sampling_type in ['random_pdf', 'best', 'random']
    global _sampling_type
    _sampling_type = sampling_type


# Get a random element within a given probability distribution
def chooseRandomPdfElementIdx(prob_distribution: list[float]) -> int:
    assert isinstance(prob_distribution, list)

    if len(prob_distribution) == 0:
        raise Exception('Cannot sample from 0 samples')

    prob_distribution_sum = 0.0
    for prob in prob_distribution:
        prob_distribution_sum += prob
    rnd_value = random.uniform(0, prob_distribution_sum)

    prob_distribution_sum = 0.0
    for idx, prob in enumerate(prob_distribution):
        if rnd_value >= prob_distribution_sum and rnd_value < prob_distribution_sum + prob:
            return idx
        prob_distribution_sum += prob
    return len(prob_distribution) - 1

def test_chooseRandomPdfElementIdx():
    pdf = [0.3, 0.1, 0.05, 0.5, 0.03, 0.02]
    samples_count = {}
    total_samples = 100000
    for idx in range(total_samples):
        elem = chooseRandomPdfElementIdx(pdf)
        if not elem in samples_count:
            samples_count[elem] = 0
        samples_count[elem] += 1

    for idx in range(len(pdf)):
        if idx in samples_count:
            count = samples_count[idx]
        else:
            count = 0
        print(f'{idx}, {pdf[idx]:.2f} -> {count/total_samples:.2f}')


# Get an element with the best score
def chooseBestElementIdx(score_list: list[float]) -> int:
    if len(score_list) == 0:
        raise Exception('Cannot sample from 0 samples')
    max_score = None
    best_idx = None
    for idx, score in enumerate(score_list):
        if max_score is None or score > max_score:
            best_idx = idx
            max_score = score
    assert not best_idx is None
    return best_idx


# Get a random element
def chooseRandomElementIdx(samples_list: list) -> int:
    if len(samples_list) == 0:
        raise Exception('Cannot sample from 0 samples')
    return random.randint(0, len(samples_list)-1)


def getGripperVisualization(ns: str, m_id: int, side: str, T_W_G: KdlFrame,
                            q_map: dict[str, float]) -> list[Marker]:
    
    sti = getStatesInterface()
    ns = 'gripper'
    color = [0.0, 0.0, 1.0]
    markers = []
    marker_id = m_id
    for idx, col in enumerate(sti.getGripperCollisionShapes(side, T_W_G, q_map)):
        markers.append( cMarker(ns, marker_id, col.getShapePose('world'), col.tp, col.size, color) )
        marker_id = marker_id + 1

    return markers

def angleBetweenVectors(v1: KdlVector, v2: KdlVector) -> float:
    v1 = copy.copy(v1)
    v2 = copy.copy(v2)
    v1.Normalize()
    v2.Normalize()
    dot_product = min(1.0, max(-1.0, v1.dot(v2) ))
    return math.acos(dot_product)


####################################################################################################
# Generators #######################################################################################
####################################################################################################

class GenCurrentGrasp:
    # (:stream GenCurrentGrasp
    #   :inputs (?obj - ObjectId)
    #   :outputs (?sd - Side ?gr - GraspId)
    #   :certified (and (GraspedSdGr ?obj ?sd ?gr))
    # )
    def __init__(self, state: State):
        self._state = state
        self._generated = False

    def generate(self, inputs: list) -> GeneratorResult:
        assert len(inputs) == 1

        # Single-shot
        if self._generated:
            return GeneratorResult.NoMore()

        self._generated = True
        obj_id_str = inputs[0]
        sti = getStatesInterface()

        for o_id_str, sd_str, gr_str in sti.getGraspedObjects(self._state):
            if o_id_str == obj_id_str:
                return GeneratorResult.SingleSample([
                    TypedValue('Side', sd_str),
                    TypedValue('GraspId', gr_str)])
        # else:
        return GeneratorResult.Failed()


# Find a put-down pose for a grasped object.
# Sample every surface that:
# - is horizontal and top-oriented
# - is of size that is reasonable for put-down
# - is in range of the arm at side ?sd
# - for a cylindrical volume of an enlarged shape:
#   - allows a collision-less put-down of an object
#   - does not collide with reserved space


def generatePlacements(points: list[KdlFrame], flat_bases: list[FlatBase]) -> list[Placement]:
    # Place each flat base at each point, so that the base z axis is oriented upwards and it is
    # a free DOF.
    result: list[Placement] = []
    for pt in points:
        T_W_F = PoseWithFreeDOF()
        assert isinstance(pt, KdlFrame)
        T_W_F.addStaticTransform( pt )
        T_W_F.addRotation('vert_rot', KdlVector(0, 0, 1))
        result.append( Placement(T_W_F, [base.T_O_F.Inverse() for base in flat_bases]) )

    return result

# For each placement, check collision with environment
def removeCollidingPlacements(state: State, obj_id_str: str,
                                placements: list[Placement]) -> list[Placement]:
    sti = getStatesInterface()

    # Prepare collision shapes for each object placement and enlarge each collision
    enl = 0.03
    lift = enl + 0.01
    T_W_Wlift = KdlFrame(KdlVector(0, 0, lift))
    # Create a list of collisions to check
    model = sti.getModelByObjectId(state, obj_id_str)
    collision_objects = {}
    for idx, placement in enumerate(placements):
        # Assume, that the object is regular, so its orientation is not important
        # TODO
        T_W_Olift = T_W_Wlift * placement.T_W_F.calculate({'vert_rot': 0.0}) * placement.T_F_O_list[0]
        # Make a collision object: 1) lift each shape by "enl", 2) enlarge each shape by "enl"
        if 'Joints' in model['KinematicStructure'] and model['KinematicStructure']['Joints']:
            raise Exception(f'Objects with internal DOF are not supported; obj_id: "{obj_id_str}"')
        col_shapes = sti.getModelCollisionShapes(model, T_W_Olift, enlarge=enl)
        collision_objects[idx] = col_shapes

    # Prepare environment model with all objects except grasped objects
    # Ignore all grasped objects
    ign_obj_id_list = [gr_obj_id_str for gr_obj_id_str, _, _ in sti.getGraspedObjects(state)]
    env_shapes_dict = sti.getEnvironmentCollisionShapes(state, ign_obj_id_list)

    GROUP_ENV    = 1 << 0
    GROUP_PLACEMENTS = 1 << 1
    GROUP_RES = 1 << 2
    pb = getPyBulletInterface()
    pb.assertEmpty()
    # Add environment
    for env_obj_id, shape_list in env_shapes_dict.items():
        for idx, shape in enumerate(shape_list):
            pb.addObject(f'env/{env_obj_id}/{idx}', shape.tp, shape.size, shape.getShapePose('world'), GROUP_ENV, GROUP_PLACEMENTS)
    # Add placements
    for pl_obj_id, shape_list in collision_objects.items():
        for idx, shape in enumerate(shape_list):
            pb.addObject(f'pla/{pl_obj_id}/{idx}', shape.tp, shape.size, shape.getShapePose('world'), GROUP_PLACEMENTS, GROUP_ENV | GROUP_RES)

    # Add reserved space
    res_space = sti.getReservedSpace(state)
    for vol, vol_id in res_space:
        #vol['ignore_obj']
        for idx, shape in enumerate(vol.col_shapes):
            pb.addObject(f'res/{vol_id}/{idx}', shape.tp, shape.size, shape.getShapePose('world'), GROUP_RES, GROUP_PLACEMENTS)

    contacts = pb.getContacts()
    colliding_placements = set()
    for name1, name2 in contacts:
        items1 = name1.split('/')
        items2 = name2.split('/')
        if items1[0] == 'pla':
            colliding_placements.add( int(items1[1]) )
        if items2[0] == 'pla':
            colliding_placements.add( int(items2[1]) )

    pb.clearAll()

    result: list[Placement] = []
    for idx, placement in enumerate(placements):
        if not idx in colliding_placements:
            result.append( placement )
    return result

class GenPlaceSimple:
#   (:stream GenPlaceSimple
#     :inputs (?obj - ObjectId)
#     :outputs (?p - Placement ?sd)
#   )
    def __init__(self, state: State) -> None:
        self._state = state
        self._visualization = {}
        self._placements = None
        self._eval_other_sample = LinearIntervalFunction([(-1.0, 0.0), (0.0, 0.0), (0.2, 1.0), (10000000.0, 0.0)])

    def _initialize(self, inputs: list):
        obj_id_str = inputs[0]

        sti = getStatesInterface()

        self._prepareShoulderDistEvaluator()

        if sti.hasReservedSpace(self._state):
            print(f'GenPlace, there is reserved space')
        else:
            print(f'GenPlace, there is no reserved space')

        flat_surfaces = sti.getFlatSurfaces(self._state)
        samples = sti.sampleSurfaces(flat_surfaces, step_size=0.025, margin=0.05)
        self._visualization['samples'] = samples

        # Only for visualization
        self._visualization['all_bases'] = sti.getFlatBasesCurrentPositions(self._state)

        close_samples = self._removeUnreachablePlaces(samples)
        self._visualization['close_samples'] = close_samples

        flat_bases = sti.getFlatBases(self._state, obj_id_str)
        placements = generatePlacements(close_samples, flat_bases)
        print(f'Total placements (not collision-checked): {len(placements)}')

        self._placements = removeCollidingPlacements(self._state, obj_id_str, placements)
        self._visualization['placements'] = self._placements

    def _removeUnreachablePlaces(self, samples: list[KdlFrame]) -> list[KdlFrame]:
        result = []
        for pt in samples:
            ev_left, ev_right = self._evalShoulderDist(pt.p)
            if ev_left > 0.01 and ev_right > 0.01:
                result.append(pt)
        return result

    def _prepareShoulderDistEvaluator(self):
        # Prepare evaluation functions
        sti = getStatesInterface()
        arm_range_min = sti.getArmRangeMin()
        arm_range_max = sti.getArmRangeMax() + 0.2     # Add some range - gripper is long
        arm_range_med1 = arm_range_min*0.75 + arm_range_max*0.25
        arm_range_med2 = arm_range_min*0.25 + arm_range_max*0.75
        self._eval_arm_dist = LinearIntervalFunction(
            [(-1.0, 0.0), (arm_range_min, 0.0), (arm_range_med1, 1.0),
             (arm_range_med2, 1.0), (arm_range_max, 0.0), (10000000.0, 0.0)])

        self._shoulder_centers_left = sti.getPossibleShoulderCenters('left', 3)
        self._shoulder_centers_right = sti.getPossibleShoulderCenters('right', 3)

    def _evalShoulderDist(self, pt:KdlVector) -> tuple[float, float]:
        result_left = 0.0
        for pt_shoulder in self._shoulder_centers_left:
            dist = (pt_shoulder-pt).Norm()
            result_left = max(result_left, self._eval_arm_dist.interpolate(dist))

        result_right = 0.0
        for pt_shoulder in self._shoulder_centers_right:
            dist = (pt_shoulder-pt).Norm()
            result_right = max(result_right, self._eval_arm_dist.interpolate(dist))
        return result_left, result_right

    def generate(self, inputs: list) -> GeneratorResult:
        assert len(inputs) == 1

        if not self._placements:
            self._initialize(inputs)
            assert not self._placements is None
            self._returned_placements_pts = []

        # Select the best placement
        evaluated_placements: list[tuple[float, Placement, float, float]] = []
        for placement in self._placements:
            # Add some height
            pt_placement = placement.T_W_F.calculate({'vert_rot': 0.0}).p
            ev_left, ev_right = self._evalShoulderDist(pt_placement + KdlVector(0,0, 0.1))

            # Penalty for samples close to previous
            prev_score = 1.0
            for pt_returned in self._returned_placements_pts:
                dist = (pt_returned-pt_placement).Norm()
                prev_score = min(prev_score, self._eval_other_sample.interpolate(dist))
            
            range_score = max(ev_left, ev_right)
            score = range_score * prev_score
            evaluated_placements.append( (score, placement, ev_left, ev_right) )
        if len(evaluated_placements) == 0:
            return GeneratorResult.Failed()

        placement_samples_scores = [x[0] for x in evaluated_placements]

        if _sampling_type == 'random_pdf':
            chooseElement_func = chooseRandomPdfElementIdx
        elif _sampling_type == 'best':
            chooseElement_func = chooseBestElementIdx
        elif _sampling_type == 'random':
            chooseElement_func = chooseRandomElementIdx
        else:
            raise Exception(f'Unknown sampling type: {_sampling_type}')

        sample_idx = chooseElement_func(placement_samples_scores)
        selected_score, selected_placement, ev_left, ev_right = evaluated_placements[sample_idx]

        side_samples_scores = [ev_left, ev_right]
        selected_side_idx = chooseElement_func(side_samples_scores)
        selected_side = ['left', 'right'][selected_side_idx]

        # evaluated_placements = sorted(evaluated_placements, key=lambda x: x[1], reverse=True)
        # selected_placement, selected_score = evaluated_placements[0]
        placement_pt = selected_placement.T_W_F.calculate({'vert_rot': 0.0}).p
        print(f'GenPlaceSimple: selected placement: {placement_pt}, score: {selected_score}')
        if selected_score < 0.01:
            if self._returned_placements_pts:
                return GeneratorResult.NoMore()
            else:
                return GeneratorResult.Failed()
        # else:
        self._returned_placements_pts.append( placement_pt )

        # This pose has one DOF
        return GeneratorResult.Sample([
            TypedValue('Placement', selected_placement),
            TypedValue('Side', selected_side)
            ])

    def can_visualize(self) -> bool:
        return bool(self._visualization)

    def get_visualization(self) -> tuple[str, list[Marker]]:
        markers = []

        markers.append( cMarkerSphereList( 'GenPlaceSimple.samples', 0,
                                               self._visualization['samples'], 0.02, [0, 0, 1] ) )

        for idx, fb in enumerate(self._visualization['all_bases']):
            assert fb['type'] == 'circle'
            markers.append( cMarkerCylinder('GenPlaceSimple.bases', idx, fb['T_W_F'],
                fb['size'], 0.02, [0, 0, 1]) )

        markers.append( cMarkerSphereList( 'GenPlaceSimple.close_samples', 0,
                                               self._visualization['close_samples'], 0.02, [0, 1, 1] ) )

        placement_points: list[KdlVector] = []
        if 'placements' in self._visualization:
            for pl in self._visualization['placements']:
                placement_points.append( pl.T_W_F.calculate({'vert_rot': 0.0}).p )

            markers.append( cMarkerCubeList( 'GenPlaceSimple.placements', 0,
                placement_points,
                [0.01, 0.01, 0.05], [1, 1, 1] ) )

        generator_data = ''
        return generator_data, markers
    

class GenPlace:
    # (:stream GenPlace
    #   :inputs (?obj - ObjectId ?sd - Side ?sd2 - SidePref)
    #   :outputs (?p - PoseWithFreeDOF)
    # )
    def __init__(self, state: State) -> None:
        self._state = state
        self._visualization = {}
        self._placements = None
        self._eval_other_sample = LinearIntervalFunction([(-1.0, 0.0), (0.01, 0.0), (0.2, 1.0), (10000000.0, 0.0)])
        self._generated_sample_index = 0

    def _initialize(self, inputs: list):
        obj_id_str, side_str, side_pref = inputs

        sti = getStatesInterface()

        # Prepare evaluation functions
        arm_range_min = sti.getArmRangeMin()
        arm_range_max = sti.getArmRangeMax() + 0.2     # Add some range - gripper is long
        arm_range_med1 = arm_range_min*0.75 + arm_range_max*0.25
        arm_range_med2 = arm_range_min*0.25 + arm_range_max*0.75
        self._eval_arm_dist = LinearIntervalFunction(
            [(-1.0, 0.0), (arm_range_min, 0.0), (arm_range_med1, 1.0),
             (arm_range_med2, 1.0), (arm_range_max, 0.0), (10000000.0, 0.0)])

        self._shoulder_centers = sti.getPossibleShoulderCenters(side_str, 3)
        if side_pref != 'any' and side_pref != side_str:
            self._shoulder_centers_pref = sti.getPossibleShoulderCenters(side_pref, 3)
        else:
            self._shoulder_centers_pref = None


        if sti.hasReservedSpace(self._state):
            print(f'GenPlace, there is reserved space')
        else:
            print(f'GenPlace, there is no reserved space')

        flat_surfaces = sti.getFlatSurfaces(self._state)
        samples = sti.sampleSurfaces(flat_surfaces, step_size=0.025, margin=0.05)
        #self._visualization['samples'] = samples

        # Only for visualization
        self._visualization['all_bases'] = sti.getFlatBasesCurrentPositions(self._state)

        close_samples, far_samples = self._removeUnreachablePlaces(samples)
        self._visualization['close_samples'] = close_samples
        self._visualization['far_samples'] = far_samples

        flat_bases = sti.getFlatBases(self._state, obj_id_str)
        placements = generatePlacements(close_samples, flat_bases)
        print(f'Total placements (not collision-checked): {len(placements)}')

        placements = removeCollidingPlacements(self._state, obj_id_str, placements)
        self._visualization['placements'] = placements
        return placements

        # print(f'Collision free placements: {len(self._placements)}')
        # raise

    def _removeUnreachablePlaces(self, samples: list[KdlFrame]
                                 ) -> tuple[list[KdlFrame], list[KdlFrame]]:
        result_good = []
        result_bad = []
        for pt in samples:
            if self._evalShoulderDist(pt.p) > 0.01:
                result_good.append(pt)
            else:
                result_bad.append(pt)
        return result_good, result_bad

    def _evalShoulderDist(self, pt:KdlVector) -> float:
        result = 0.0
        for pt_shoulder in self._shoulder_centers:
            dist = (pt_shoulder-pt).Norm()
            result = max(result, self._eval_arm_dist.interpolate(dist))

        if not self._shoulder_centers_pref is None:
            result_pref = 0.0
            for pt_shoulder in self._shoulder_centers_pref:
                dist = (pt_shoulder-pt).Norm()
                result_pref = max(result, self._eval_arm_dist.interpolate(dist))
            result = min(result, result_pref)
        return result

    def generate(self, inputs: list) -> GeneratorResult:
        assert len(inputs) == 3

        if self._placements is None:
            self._dbg = 4
            self._placements = self._initialize(inputs)
            self._generated_placements_pts = []
            if len(self._placements) == 0:
                return GeneratorResult.Failed()

        # Select the best placement
        evaluated_placements: list[tuple[Placement, float]] = []
        for placement in self._placements:
            pt_placement = placement.T_W_F.calculate({'vert_rot': 0.0}).p
            # Add some height
            range_score = self._evalShoulderDist(pt_placement + KdlVector(0,0, 0.1))

            # Penalty for samples close to previous
            prev_score = 1.0
            for pt_returned, pt_score in self._generated_placements_pts:
                dist = (pt_returned-pt_placement).Norm()
                prev_score = min(prev_score, self._eval_other_sample.interpolate(dist))
            
            score = range_score * prev_score
            evaluated_placements.append( (placement, score) )
        if len(evaluated_placements) == 0:
            return GeneratorResult.Failed()

        if _sampling_type == 'random_pdf':
            chooseElement_func = chooseRandomPdfElementIdx
        elif _sampling_type == 'best':
            chooseElement_func = chooseBestElementIdx
        elif _sampling_type == 'random':
            chooseElement_func = chooseRandomElementIdx
        else:
            raise Exception(f'Unknown sampling type: {_sampling_type}')
    
        # Visualize generator's all evaluations
        if not 'evaluated_placements' in self._visualization:
            self._visualization['evaluated_placements'] = []
            self._visualization['generated_placements_pts'] = []
        self._visualization['evaluated_placements'].append( copy.copy(evaluated_placements) )

        # This is used for visualization only / debugging: select samples with y < 0
        # Do not use this in normal cases.
        # while True:
        #     sample_idx = chooseElement_func([y for x, y in evaluated_placements])
        #     selected_score = evaluated_placements[sample_idx][1]
        #     selected_placement = evaluated_placements[sample_idx][0]
        #     if selected_placement.T_W_F.calculate({'vert_rot': 0.0}).p.y() < 0:
        #         break

        sample_idx = chooseElement_func([y for x, y in evaluated_placements])
        selected_score = evaluated_placements[sample_idx][1]
        selected_placement = evaluated_placements[sample_idx][0]
        
        placement_pt = selected_placement.T_W_F.calculate({'vert_rot': 0.0}).p
        print(f'GenPlace: selected placement: {placement_pt}, score: {selected_score}')
        if selected_score < 0.01:
            if self._generated_placements_pts:
                return GeneratorResult.NoMore()
            else:
                return GeneratorResult.Failed()
        # else:
        self._generated_placements_pts.append( (placement_pt, selected_score) )
        self._visualization['generated_placements_pts'].append( copy.copy(self._generated_placements_pts) )

        self._generated_sample_index = self._generated_sample_index + 1

        # This is used for visualization only / debugging: generate a number of samples
        # Do not use this in normal cases.
        # if self._generated_sample_index < 5:
        #     return self.generate(inputs)

        # This pose has one DOF
        return GeneratorResult.Sample([TypedValue('Placement', selected_placement)])

        # This is used for visualization only / debugging: stop planning after one generation.
        # Do not use this in normal cases.
        # return GeneratorResult.Sample([TypedValue('Placement', selected_placement)]).debugStopPlanning()

    def can_visualize(self) -> bool:
        return bool(self._visualization)

    def get_visualization(self) -> tuple[str, list[Marker]]:
        markers = []

        sample_radius = 0.025
        far_samples_points = [T.p for T in self._visualization['far_samples']]
        markers.append( cMarkerSphereList( 'GenPlace.far_samples', 0,
                                            far_samples_points,
                                            sample_radius, [1, 0, 0] ) )

        for idx, fb in enumerate(self._visualization['all_bases']):
            assert fb['type'] == 'circle'
            markers.append( cMarkerCylinder('GenPlace.bases', idx, fb['T_W_F'],
                fb['size'], sample_radius, [0, 0, 1]) )

        close_samples_points = [T.p for T in self._visualization['close_samples']]
        markers.append( cMarkerSphereList( 'GenPlace.close_samples', 0,
                                            close_samples_points,
                                            sample_radius, [0, 1, 0] ) )

        if 'evaluated_placements' in self._visualization:
            for gen_idx, vis_ev_placements in enumerate(self._visualization['evaluated_placements']):
                # placement_points: list[KdlVector] = []
                for idx, (pl, score) in enumerate(vis_ev_placements):
                    sample_height = 0.1 * score
                    #T_W_Pl = KdlFrame(pl.T_W_F.calculate({'vert_rot': 0.0}).p + KdlVector(0, 0, sample_radius+sample_height/2))
                    T_W_Pl = KdlFrame(KdlVector(0, 0, sample_radius+sample_height/2)) * pl.T_W_F.calculate({'vert_rot': 0.0})
                    markers.append( cMarkerBox( f'GenPlace.ev_placements_{gen_idx}', idx,
                        T_W_Pl, [0.025, 0.025, sample_height], [0, 0, 1] ) )

        if 'generated_placements_pts' in self._visualization:
            for gen_idx, vis_generated_placements_pts in enumerate(self._visualization['generated_placements_pts']):
                # placement_points: list[KdlVector] = []
                for idx, (pl, pl_score) in enumerate(vis_generated_placements_pts):
                    sample_height = 0.1 * pl_score
                    markers.append( cMarkerArrow( f'GenPlace.gen_placements_{gen_idx}', idx,
                        pl + KdlVector(0, 0, 0.3+sample_radius+sample_height), pl + KdlVector(0, 0, sample_radius+sample_height),
                        0.02, [0, 1, 0] ) )

                    # placement_points.append(
                    #     pl + KdlVector(0, 0, sample_radius+sample_height/2) )

                # markers.append( cMarkerCubeList( f'GenPlace.gen_placements_{gen_idx}', 0,
                #     placement_points,
                #     [0.025, 0.025, sample_height], [1, 1, 1] ) )

        generator_data = ''
        return generator_data, markers


def getCollidingGripperPoses(state: State, side: str,
            samples: list[GraspCollisionSpace],
            ignored_obj: list[str] = []) -> tuple[set[int], dict[int, set[str]]]:
    # Filter out colliding poses of the gripper for given placements
    pb = getPyBulletInterface()
    pb.assertEmpty()
    sti = getStatesInterface()
    T_E_G = sti.getT_E_G(side)
    GROUP_ENV    = 1 << 0
    GROUP_GRIPPER = 1 << 1
    for sample_idx, sample in enumerate(samples):
        T_W_Gg = sample.T_W_O * sample.T_O_Gg
        T_W_Gp = sample.T_W_O * sample.T_O_Gp
        col_shapes = sti.getGripperCollisionShapes(side, T_W_Gg, sample.qg.toJsMap())
        col_shapes += sti.getGripperCollisionShapes(side, T_W_Gg, sample.qp.toJsMap())
        col_shapes += sti.getGripperCollisionShapes(side, T_W_Gp, sample.qp.toJsMap())
        for col_idx, col in enumerate(col_shapes):
            pb.addObject(f'gripper/{sample_idx}/{col_idx}', col.tp, col.size,
                                            col.getShapePose('world'), GROUP_GRIPPER, GROUP_ENV)

    # Prepare environment model with all objects except grasped objects
    # Ignore all grasped objects
    ign_obj_id_list = [obj_id_str for obj_id_str, _, _ in sti.getGraspedObjects(state)] + ignored_obj
    env_shapes_dict = sti.getEnvironmentCollisionShapes(state, ign_obj_id_list)

    # Collisions with static objects are not acceptable
    static_obj_ids = sti.getStaticObjectsIds(state)

    for env_obj_id, shape_list in env_shapes_dict.items():
        for idx, shape in enumerate(shape_list):
            pb.addObject(f'env/{env_obj_id}/{idx}', shape.tp, shape.size, shape.getShapePose('world'), GROUP_ENV, GROUP_GRIPPER)

    contacts = pb.getContacts()
    pb.clearAll()

    hard_colliding_samples: set[int] = set()
    soft_colliding_samples: dict[int, set[str]] = {}
    for name1, name2 in contacts:
        items1 = name1.split('/')
        items2 = name2.split('/')
        if items1[0] == 'gripper':
            sample_id = int(items1[1])
            col_obj_id = items2[1]
        elif items2[0] == 'gripper':
            sample_id = int(items2[1])
            col_obj_id = items1[1]
        else:
            raise Exception('Unreachable code')

        if col_obj_id in static_obj_ids:
            hard_colliding_samples.add( sample_id )
        else:
            if not sample_id in soft_colliding_samples:
                soft_colliding_samples[sample_id] = set()
            soft_colliding_samples[sample_id].add(col_obj_id)

    return hard_colliding_samples, soft_colliding_samples

class GraspCollisionSpace:
    def __init__(self, grasp_id: str, T_W_O: KdlFrame, T_G_E: KdlFrame, T_O_Gp: KdlFrame, T_O_Gg: KdlFrame,
                 qp: ConfG, qg: ConfG):
        self.grasp_id = grasp_id
        self.T_W_O = T_W_O
        self.T_G_E = T_G_E
        self.T_O_Gp = T_O_Gp
        self.T_O_Gg = T_O_Gg
        self.qp = qp
        self.qg = qg

    def getT_W_Eg(self) -> KdlFrame:
        return self.T_W_O * self.T_O_Gg * self.T_G_E


class GenUngraspPose:
    # (:stream GenUngraspPose
    #   :inputs (?obj - ObjectId ?sd - Side ?gr - GraspId  ?ap - Approaches ?p - Placement)
    #   :outputs (?p_obj ?p_ee - Pose)
    # )
    def __init__(self, state: State):
        self._state = state
        self._initialized = False
        self._visualization = {}
        # self._vsolv = KinematicsSolverVelma()
        self._vsolv = getVelmaSolver()
        self._elbow_angles = list(np.linspace(math.radians(-180), math.radians(180), 9, endpoint=False))
        self._eval_approach_diff = LinearIntervalFunction(
            [(-1.0, 1.0), (0.0, 1.0), (math.radians(15), 1.0), (math.radians(45), 0.1),
             (math.radians(90), 0.0), (math.radians(181), 0.0)])

    def _initialize(self, inputs: list):
        obj_id_str, side, grasp_id_str, ap, pose_put_down = inputs
        assert isinstance(pose_put_down, Placement)

        sti = getStatesInterface()
        grasp_def = sti.getGraspDef(self._state, obj_id_str, grasp_id_str)
        q_grasp = sti.getGraspConf(side, grasp_def)
        q_pregrasp = sti.getPregraspConf(side, grasp_def)
        T_Gg_O = sti.getGraspPose(grasp_def)
        T_Gp_O = sti.getPregraspPose(grasp_def)
        T_E_G = sti.getT_E_G(side)

        T_O_Gg = T_Gg_O.Inverse()
        T_O_Gp = T_Gp_O.Inverse()
        T_G_E = T_E_G.Inverse()

        solv = KinematicsSolverBarrettHand()
        self._T_E_P = T_E_G * solv.getT_P_G().Inverse()

        # TODO: add collision detection for access from free space
        # TODO: add collision detection for reserved space
        # assert not sti.hasReservedSpace(self._state)

        # Generate samples: multiple bases * rotation of the grasped object to be placed
        samples: list[GraspCollisionSpace] = []
        for T_F_O in pose_put_down.T_F_O_list:
            for angle in np.linspace(-math.pi, math.pi, 24, endpoint=False):
                T_W_Fd = pose_put_down.T_W_F.calculate({'vert_rot': angle})
                T_W_Od = T_W_Fd * T_F_O
                samples.append( GraspCollisionSpace(grasp_id_str, T_W_Od, T_G_E, T_O_Gp, T_O_Gg, q_pregrasp, q_grasp) )

        hard_col_samples, soft_col_samples = getCollidingGripperPoses(self._state, side, samples)
        soft_col_samples = set([samp_id for samp_id in soft_col_samples])
        col_samples = hard_col_samples.union(soft_col_samples)
        # col_samples = hard_col_samples

        self._samples = [s for s_idx, s in enumerate(samples) if not s_idx in col_samples]

        # self._samples_col = []
        # for s_idx, _ in enumerate(samples):
        #     if s_idx in hard_col_samples:
        #         continue
        #     if s_idx in soft_col_samples:
        #         self._samples_col.append(soft_col_samples[s_idx])
        #     else:
        #         self._samples_col.append( [] )

        # assert len(self._samples_col) == len(self._samples)

        arm_range_min = sti.getArmRangeMin()
        arm_range_max = sti.getArmRangeMax() + 0.2
        arm_range_med1 = arm_range_min*0.75 + arm_range_max*0.25
        arm_range_med2 = arm_range_min*0.25 + arm_range_max*0.75
        self._eval_arm_dist = LinearIntervalFunction(
            [(-1.0, 0.0), (arm_range_min, 0.0), (arm_range_med1, 1.0),
             (arm_range_med2, 1.0), (arm_range_max, 0.0), (10000000.0, 0.0)])

        self._shouleder_centers = sti.getPossibleShoulderCenters(side, 3)

    def _evalEePose(self, side: str, T_W_E: KdlFrame) -> float:
        torso_samples = 5
        max_solutions = len(self._elbow_angles) * torso_samples
        solutions_count = 0
        for torso_angle in np.linspace(math.radians(-85), math.radians(85), torso_samples, endpoint=True):
            solutions = self._vsolv.vsolv.calculateIkSet(side, T_W_E, torso_angle, self._elbow_angles)
            solutions_count += len(solutions)
        return solutions_count / max_solutions

    def _evalApproach(self, ap: Approaches, approach_dir: KdlVector) -> float:
        result_score = 0.0
        for ap_score, ap_dir in ap.ap:
            angle = angleBetweenVectors(ap_dir, approach_dir)
            single_score = self._eval_approach_diff.interpolate(angle)
            result_score = max(result_score, single_score)
        return result_score

    def generate(self, inputs: list) -> GeneratorResult:
        assert len(inputs) == 5
        obj_id_str, side_str, grasp_id_str, ap, pose_put_down = inputs

        if not self._initialized:
            self._initialize(inputs)
            self._initialized = True
            assert not self._samples is None
            if len(self._samples) == 0:
                # No samples generated - failure
                # TODO: determine reason of failure
                return GeneratorResult.Failed([4])

        if len(self._samples) == 0:
            # No more samples
            return GeneratorResult.NoMore()

        evaluated_samples = []
        for idx, sample in enumerate(self._samples):
            #idx, (T_W_Od, T_W_Ed)
            T_W_Ed = sample.getT_W_Eg()
            # score_range = self._evalShoulderDist(T_W_Ed.p)
            score_range = self._evalEePose(side_str, T_W_Ed)
            T_W_P = T_W_Ed * self._T_E_P
            approach_dir = sample.T_W_O.p - T_W_P.p
            score_approach = self._evalApproach(ap, approach_dir)
            score = score_range * score_approach
            evaluated_samples.append( (score, sample) )

        if len(evaluated_samples) == 0:
            return GeneratorResult.Failed()

        if _sampling_type == 'random_pdf':
            chooseElement_func = chooseRandomPdfElementIdx
        elif _sampling_type == 'best':
            chooseElement_func = chooseBestElementIdx
        elif _sampling_type == 'random':
            chooseElement_func = chooseRandomElementIdx
        else:
            raise Exception(f'Unknown sampling type: {_sampling_type}')

        sample_idx = chooseElement_func([x for x, y in evaluated_samples])
        idx = sample_idx
        score = evaluated_samples[idx][0]

        # best_sample = (None, None)
        # for idx, (T_W_Od, T_W_Ed) in enumerate(self._samples):
        #     score_range = self._evalShoulderDist(T_W_Ed.p)
        #     T_W_P = T_W_Ed * self._T_E_P
        #     approach_dir = T_W_Od.p - T_W_P.p
        #     score_approach = self._evalApproach(ap, approach_dir)
        #     score = score_range * score_approach
        #     if best_sample[0] is None or best_sample[0] < score:
        #         best_sample = (score, idx)
        # score, idx = best_sample
        assert not score is None
        assert not idx is None
        if score < 0.01:
            # No more samples
            return GeneratorResult.NoMore()
        # else:
        sample = self._samples[idx]
        #T_W_Od, T_W_Ed
        T_W_Ed = sample.getT_W_Eg()

        sti = getStatesInterface()
        grasp_def = sti.getGraspDef(self._state, obj_id_str, grasp_id_str)
        q_map = sti.getGraspConfJsMap(side_str, grasp_def)

        T_E_G = sti.getT_E_G(side_str)
        T_W_G = T_W_Ed * T_E_G
        self._visualization['grasp'] = (side_str, T_W_G, q_map)
        del self._samples[idx]

        return GeneratorResult.Sample([TypedValue('Pose', sample.T_W_O.toDict()), TypedValue('Pose', T_W_Ed.toDict())])

    def can_visualize(self) -> bool:
        return 'grasp' in self._visualization

    def get_visualization(self) -> tuple[str, list[Marker]]:

        side, T_W_G, q_map = self._visualization['grasp']

        markers = getGripperVisualization('gripper', 0, side, T_W_G, q_map)

        generator_data = ''
        return generator_data, markers


class GenGraspSd:
#   (:stream GenGraspSd
#     :inputs (?obj - ObjectId ?sd - Side ?ap - Approaches)
#     :outputs (?gr - GraspId ?p_ee - Pose)
#   )
    def __init__(self, state: State, no_moveable_collisions: bool = False):
        self._state = state
        self._no_moveable_collisions = no_moveable_collisions
        self._elbow_angles = list(np.linspace(math.radians(-180), math.radians(180), 9, endpoint=False))
        self._eval_approach_diff = LinearIntervalFunction(
            [(-1.0, 1.0), (0.0, 1.0), (math.radians(15), 1.0), (math.radians(45), 0.1),
             (math.radians(90), 0.0), (math.radians(181), 0.0)])
        self._visualization = {}
        self._initialized = False

    def _initialize(self, inputs: list) -> list[GraspCollisionSpace]:
        obj_id_str, side, ap = inputs

        sti = getStatesInterface()
        T_E_G = sti.getT_E_G(side)
        T_G_E = T_E_G.Inverse()

        self._T_W_O = sti.getObjectExactPose(self._state, obj_id_str)
        grasps = sti.getAllObjectGrasps(self._state, obj_id_str)
        gr_samples: list[GraspCollisionSpace] = []
        for gr_id, grasp_def in grasps.items():
            q_grasp = sti.getGraspConf(side, grasp_def)
            q_pregrasp = sti.getPregraspConf(side, grasp_def)
            T_Gp_O = sti.getPregraspPose(grasp_def)
            T_Gg_O = sti.getGraspPose(grasp_def)
            T_O_Gp = T_Gp_O.Inverse()
            T_O_Gg = T_Gg_O.Inverse()

            gr_samples.append( GraspCollisionSpace(gr_id, self._T_W_O, T_G_E, T_O_Gp, T_O_Gg, q_pregrasp, q_grasp) )

        solv = KinematicsSolverBarrettHand()
        self._T_E_P = T_E_G * solv.getT_P_G().Inverse()

        # TODO: add collision detection for access from free space
        # TODO: add collision detection for reserved space
        # assert not sti.hasReservedSpace(self._state)

        ignored_obj = [obj_id_str]
        hard_col_samples, soft_col_samples = getCollidingGripperPoses(self._state, side, gr_samples, ignored_obj)
        if self._no_moveable_collisions:
            col_samples = hard_col_samples.union( set( [samp_id for samp_id in soft_col_samples] ) )
        else:
            col_samples = hard_col_samples
        samples = [s for s_idx, s in enumerate(gr_samples) if not s_idx in col_samples]
        print(f'  samples: {len(samples)}')

        self._samples_col = []
        for s_idx, _ in enumerate(gr_samples):
            if s_idx in hard_col_samples:
                continue
            if s_idx in soft_col_samples:
                self._samples_col.append(soft_col_samples[s_idx])
            else:
                self._samples_col.append( [] )

        print(f'  self._samples_col: {len(self._samples_col)}')

        arm_range_min = sti.getArmRangeMin()
        arm_range_max = sti.getArmRangeMax()
        arm_range_med1 = arm_range_min*0.75 + arm_range_max*0.25
        arm_range_med2 = arm_range_min*0.25 + arm_range_max*0.75
        self._eval_arm_dist = LinearIntervalFunction(
            [(-1.0, 0.0), (arm_range_min, 0.0), (arm_range_med1, 1.0),
             (arm_range_med2, 1.0), (arm_range_max, 0.0), (10000000.0, 0.0)])

        self._shouleder_centers = sti.getPossibleShoulderCenters(side, 3)

        return samples

    def _evalShoulderDist(self, pt:KdlVector) -> float:
        result = 0.0
        for pt_shoulder in self._shouleder_centers:
            dist = (pt_shoulder-pt).Norm()
            result = max(result, self._eval_arm_dist.interpolate(dist))
        return result

    # def _evalEePose(self, side: str, T_W_E: KdlFrame) -> float:
    #     torso_samples = 5
    #     max_solutions = len(self._elbow_angles) * torso_samples
    #     solutions_count = 0
    #     for torso_angle in np.linspace(math.radians(-80), math.radians(80), torso_samples, endpoint=True):
    #         solutions = self._vsolv.calculateIkSet(side, T_W_E, torso_angle, self._elbow_angles)
    #         solutions_count += len(solutions)
    #     return solutions_count / max_solutions

    def _evalApproach(self, ap: Approaches, approach_dir: KdlVector) -> float:
        result_score = 0.0
        for ap_score, ap_dir in ap.ap:
            angle = angleBetweenVectors(ap_dir, approach_dir)
            single_score = self._eval_approach_diff.interpolate(angle)
            score = ap_score * single_score
            result_score = max(result_score, score)
        return result_score

    def generate(self, inputs: list) -> GeneratorResult:
        assert len(inputs) == 3
        obj_id_str, side, ap = inputs

        if not self._initialized:
            self._initialized = True

            samples = self._initialize(inputs)

            self._ev_samples: list[tuple[float, GraspCollisionSpace]] = []
            #best_sample = (None, None)
            for idx, sample in enumerate(samples):
                T_W_E = sample.T_W_O * sample.T_O_Gg * sample.T_G_E
                score_range = self._evalShoulderDist(T_W_E.p)
                # score_range = self._evalEePose(side, T_W_E)
                T_W_P = T_W_E * self._T_E_P
                approach_dir = self._T_W_O.p - T_W_P.p
                score_approach = self._evalApproach(ap, approach_dir)
                # print(f'  appr: {score_approach}, range: {score_range}')
                score = score_range * score_approach
                # if best_sample[0] is None or score > best_sample[0]:
                #     best_sample = (score, idx)
                if score > 0.01:
                    self._ev_samples.append( (score, sample) )

            if len(self._ev_samples) == 0:
                # No samples - failure
                print('  GenGraspSd: no samples generated')
                # Reason of failure: maybe the object is wrong?
                return GeneratorResult.Failed([0])

        if len(self._ev_samples) == 0:
            # No more samples
            print('  GenGraspSd: no more samples')
            # Reason of failure: maybe the object is wrong?
            return GeneratorResult.Failed([0])


        if _sampling_type == 'random_pdf':
            chooseElement_func = chooseRandomPdfElementIdx
        elif _sampling_type == 'best':
            chooseElement_func = chooseBestElementIdx
        elif _sampling_type == 'random':
            chooseElement_func = chooseRandomElementIdx
        else:
            raise Exception(f'Unknown sampling type: {_sampling_type}')

        sample_idx = chooseElement_func([score for score, sample in self._ev_samples])

        # score, idx = best_sample
        # if score is None or score < 0.01:
        #     print(f'  All samples have bad scores')
        #     # No more samples
        #     return GeneratorResult.NoMore()
        # # else:
        # assert not idx is None
        #gr_id, T_W_E, q_grasp
        score, sample = self._ev_samples[sample_idx]
        T_W_E = sample.getT_W_Eg()

        print(f'  GenGraspSd: generated sample (grasp_id: {sample.grasp_id}) with score {score}')

        sti = getStatesInterface()

        T_E_G = sti.getT_E_G(side)
        T_W_G = T_W_E * T_E_G
        self._visualization['grasp'] = (side, T_W_G, sample.qg.toJsMap())
        del self._ev_samples[sample_idx]

        return GeneratorResult.Sample([TypedValue('GraspId', str(sample.grasp_id)),
                                       TypedValue('Pose', T_W_E.toDict())])

    def can_visualize(self) -> bool:
        return 'grasp' in self._visualization

    def get_visualization(self) -> tuple[str, list[Marker]]:

        side, T_W_G, q_map = self._visualization['grasp']

        markers = getGripperVisualization('gripper', 0, side, T_W_G, q_map)

        generator_data = ''
        return generator_data, markers


class GenGraspSdCl(GenGraspSd):
    def __init__(self, state: State):
        super().__init__(state, no_moveable_collisions=True)


def getModelCenter(model: dict[str, Any]) -> KdlVector:
    central_pt = KdlVector()
    count = 0
    for link in model['KinematicStructure']['Links']:
        for col in link['collision']:
            T_O_C = KdlFrame.fromDict( col['Pose']['Pose'] )
            central_pt = central_pt + T_O_C.p
            count += 1
    return central_pt / count

type SingleRay = tuple[KdlFrame, float]

def getApproachView(T_W_O: KdlFrame, model: dict[str, Any],
                        v_dir: KdlVector):
    z_back = 0.6

    # Start at the central point
    pt_c = T_W_O * getModelCenter(model)

    z_axis = v_dir
    if abs(z_axis.z()) < 0.8:
        y_axis = KdlVector(0, 0, 1)
    else:
        y_axis = KdlVector(0, 1, 0)

    x_axis = y_axis * z_axis
    y_axis = z_axis * x_axis

    x_axis.Normalize()
    y_axis.Normalize()
    z_axis.Normalize()

    T_W_V = KdlFrame(KdlRotation(x_axis, y_axis, z_axis), pt_c - v_dir * z_back)
    return T_W_V


class RayHits:
    def __init__(self, solid_id: str|None,
                 solid_hit: KdlVector|None, transp_ids: set[str],
                 p1: KdlVector, p2: KdlVector):
        self.solid_id = solid_id
        self.solid_hit = solid_hit
        self.transp_ids = transp_ids
        self.p1 = p1
        self.p2 = p2

    def getRaySolidHitOrEnd(self) -> KdlVector:
        if self.solid_hit is None:
            return self.p2
        return self.solid_hit
    
    def hitSolid(self) -> bool:
        return not self.solid_id is None

    def hitTransparent(self) -> bool:
        return len(self.transp_ids) > 0


def collideRays(rays: list[tuple[KdlVector, KdlVector]],
                    solid_objects_ids: set[str]) -> list[RayHits]:

    pb = getPyBulletInterface()

    rays_from = [[ray[0].x(), ray[0].y(), ray[0].z()] for ray in rays]
    rays_to = [[ray[1].x(), ray[1].y(), ray[1].z()] for ray in rays]
    solid_objects_int_ids = set()
    for obj_id in solid_objects_ids:
        bullet_id = pb.getBulletIdByObjectId(obj_id)
        assert bullet_id != -1
        solid_objects_int_ids.add( bullet_id )
    b_result = collideRaysRec(pb, rays_from, rays_to, solid_objects_int_ids)
    
    result: list[RayHits] = []
    for ray_idx, sub in enumerate(b_result):
        assert sub.finished
        if sub.solid_id is None:
            solid_id = None
            solid_hit = None
        else:
            solid_id = pb.getObjectIdPyBulletId(sub.solid_id)
            assert not solid_id is None
            assert not sub.solid_hit is None
            solid_hit = KdlVector(sub.solid_hit[0], sub.solid_hit[1], sub.solid_hit[2])

        transp_ids = set()
        for b_transp_id in sub.transp_ids:
            transp_id = pb.getObjectIdPyBulletId(b_transp_id)
            transp_ids.add( transp_id )
        
        result.append( RayHits(solid_id, solid_hit, transp_ids, rays[ray_idx][0], rays[ray_idx][1]) )
    return result

class RayHitsPyBullet:
    def __init__(self, finished: bool, solid_id: int|None,
                 solid_hit: list[float]|None, transp_ids: set[int]):
        self.finished = finished
        self.solid_id = solid_id
        self.solid_hit = solid_hit
        self.transp_ids = transp_ids

def collideRaysRec(pb, rays_from: list[list[float]], rays_to: list[list[float]],
                    solid_objects_ids: set[int]
                    ) -> list[RayHitsPyBullet]:
    # Collide rays multiple time, until each ray ends at solid object or
    # at its end point.
    # It is assumed that pybullet scene is already set.
    # Returns a list that for each ray has the following information (a tuple):
    # (is_finished, solid_id_hit, solid_hit_pos, list_of_transparent_id_hit)

    results = pb.rayTestBatch(rays_from, rays_to)

    result: list[RayHitsPyBullet] = []
    next_it_rays_from = []
    next_it_rays_to = []
    next_it_map = []
    for i, hit in enumerate(results):
        hit_obj, hit_link, hit_frac, hit_pos, hit_n = hit
        if hit_obj == -1:
            # No hit, finish ray
            result.append( RayHitsPyBullet(True, None, None, set()) )
        else:
            if hit_obj in solid_objects_ids:
                # Solid hit, finish ray
                result.append( RayHitsPyBullet(True, hit_obj, hit_pos, set()) )
            else:
                # Transparent hit, continue
                p1 = KdlVector(hit_pos[0], hit_pos[1], hit_pos[2])
                p2 = KdlVector(rays_to[i][0], rays_to[i][1], rays_to[i][2])
                ray_dir = p2 - p1
                ray_dir.Normalize()
                new_p1 = p1 + ray_dir*0.01

                if (p2-new_p1).dot(ray_dir) > 0:
                    result.append( RayHitsPyBullet(False, None, None, set([hit_obj])) )
                    next_it_rays_from.append( [new_p1.x(), new_p1.y(), new_p1.z()] )
                    next_it_rays_to.append( rays_to[i] )
                    next_it_map.append( i )
                else:
                    result.append( RayHitsPyBullet(True, None, None, set([hit_obj])) )


    # print(f'next_it_rays: {len(next_it_rays_from)}')
    # Iterate through transparent objects
    if next_it_rays_from:
        sub_result = collideRaysRec(pb, next_it_rays_from, next_it_rays_to, solid_objects_ids)
    else:
        sub_result = []

    for sub_idx, sub in enumerate(sub_result):
        ray_idx = next_it_map[sub_idx]
        transp_ids = result[ray_idx].transp_ids
        assert not result[ray_idx].finished
        result[ray_idx] = RayHitsPyBullet(sub.finished, sub.solid_id,
                                          sub.solid_hit, sub.transp_ids.union(transp_ids))
    return result


def collideRaysAtApproach(state: State, obj_id_str: str, T_W_O: KdlFrame,
                          directions: list[KdlVector], penetrate_moveable_objects: bool = True
                          ) -> dict[int, ApproachHits]:
    sti = getStatesInterface()

    # Prepare environment model with all objects except grasped objects and the main object
    # Ignore all grasped objects
    ign_obj_id_list = [gr_obj_id_str for gr_obj_id_str, _, _ in sti.getGraspedObjects(state)] + [obj_id_str]
    env_shapes_dict = sti.getEnvironmentCollisionShapes(state, ign_obj_id_list)

    static_objects_ids = sti.getStaticObjectsIds(state)

    enlarge_obj = 0.02

# TODO: use T_W_O_rot
# Important note:
# Possible approach directions are connected to object properties, e.g. possible grasps.
# It is not necessary to generate approaches from sided that are not reachable,
# e.g. for a pan.
# Approach directions are:
# - for a grasped object, a grasp defines approach direction wrt. the object. All possible
#   approach directions in the world frame depend on possible placements of the object.
# - for an object to be grasped, all approach directions depend on possible grasps and
#   the current placement of the object
# Initial directions of approach can be filterd out to match all possibilities - in next generators.
# This supports the use of GenApproach as a separate generator, as it can be used for various tasks.

    T_W_Wlift = KdlFrame(KdlVector(0, 0, enlarge_obj))
    T_W_Olift = T_W_Wlift * T_W_O

    obj_model = sti.getModelByObjectId(state, obj_id_str)
    obj_shapes_dict = {'obj': sti.getModelCollisionShapes(obj_model, T_W_Olift, enlarge=enlarge_obj)}

    # Prepare rays
    samples_one_dim = 20
    window_size = 0.4
    ray_length = 1.0

    # rays_from = []
    # rays_to = []
    rays = []
    ray_idx = 0
    ray_idx_to_dir_map = {}
    for dir_idx, v_dir in enumerate(directions):
        # Get view from direction at center of the main object
        T_W_V = getApproachView(T_W_O, obj_model, v_dir)
        # Generate a matrix of rays for the view
        for dx in np.linspace(-window_size/2, window_size/2, samples_one_dim, endpoint=True):
            for dy in np.linspace(-window_size/2, window_size/2, samples_one_dim, endpoint=True):
                p1 = T_W_V * KdlVector(dx, dy, 0)
                p2 = T_W_V * KdlVector(dx, dy, ray_length)
                # print(f'ray: {p1} -> {p2}')
                rays.append( (p1, p2) )
                # rays_from.append( [p1.x(), p1.y(), p1.z()] )
                # rays_to.append( [p2.x(), p2.y(), p2.z()] )
                ray_idx_to_dir_map[ray_idx] = dir_idx
                ray_idx = ray_idx + 1

    print(f'number of rays: {len(rays)}')
    GROUP_ENV    = 1 << 0
    GROUP_OTHER = 1 << 1
    pb = getPyBulletInterface()
    pb.assertEmpty()

    # Check ray collisions with the object and without environment
    # Add the main object, enlarged
    solid_obj_ids = set()
    for pl_obj_id, shape_list in obj_shapes_dict.items():
        for idx, shape in enumerate(shape_list):
            obj_id = f'obj:{pl_obj_id}/{idx}'
            pb.addObject(obj_id, shape.tp, shape.size, shape.getShapePose('world'), None, None)
            solid_obj_ids.add(obj_id)

    # print(f'solid_obj_ids: {solid_obj_ids}')
    result = collideRays(rays, solid_obj_ids)

    # Keep only the rays that hit the object
    hit_rays = []
    hit_ray_idx = 0
    hit_ray_idx_to_dir_map = {}
    for ray_idx, hit in enumerate(result):
        if hit.hitSolid():
            p1 = rays[ray_idx][0]
            p2 = hit.solid_hit
            hit_rays.append( (p1, p2) )
            dir_idx = ray_idx_to_dir_map[ray_idx]
            hit_ray_idx_to_dir_map[hit_ray_idx] = dir_idx
            hit_ray_idx = hit_ray_idx + 1

    ray_idx_to_dir_map = None
    rays = None

    # Check ray collisions with the object and with environment
    pb.clearAll()

    # Add environment without the main object
    solid_obj_ids = set()
    for env_obj_id, shape_list in env_shapes_dict.items():
        assert env_obj_id != obj_id_str

        # print(f'creating env model: {env_obj_id}')
        for idx, shape in enumerate(shape_list):
            obj_id = f'env:{env_obj_id}/{idx}'
            pb.addObject(obj_id, shape.tp, shape.size, shape.getShapePose('world'), None, None)
            if env_obj_id in static_objects_ids:
                solid_obj_ids.add(obj_id)

    print(f'rays hit: {len(hit_rays)}')

    result = collideRays(hit_rays, solid_obj_ids)

    dir_rays_hit_map = {}
    for i, hit in enumerate(result):
        # For each direction we want to know:
        # - what rays hit solid objects
        # - what rays hit transparent objects
        # - what transparent objects were hit

        dir_idx = hit_ray_idx_to_dir_map[i]
        if not dir_idx in dir_rays_hit_map:
            dir_rays_hit_map[dir_idx] = ApproachHits()
        dir_rays_hit_map[dir_idx].addHit( hit )

    pb.clearAll()
    return dir_rays_hit_map

class ApproachHits:
    def __init__(self):
        self.hits: list[RayHits] = []
        self.transp_ids: set[str] = set()

    def addHit(self, hit: RayHits):
        self.hits.append(hit)
        self.transp_ids = self.transp_ids.union( hit.transp_ids )

    def getRaysCountSolidHit(self) -> int:
        count = 0
        for hit in self.hits:
            if hit.hitSolid():
                count += 1
        return count
    
    def getRaysCountTransparentHit(self) -> int:
        count = 0
        for hit in self.hits:
            if len(hit.transp_ids) > 0:
                count += 1
        return count

    def getRaysCount(self) -> int:
        return len(self.hits)


class Approaches:
    ap: list[tuple[float, KdlVector]]
    def __init__(self, pt: KdlVector, ap: list[tuple[float, KdlVector]]):
        self.pt = pt
        self.ap = ap


class GenApproaches:
    # (:stream GenApproached
    #   :inputs (?obj - ObjectId ?p_obj - SomePose)
    #   :outputs (?ap - Approaches)
    # )
    def __init__(self, state: State):
        self._state = state
        self._dir_rays_hit_map = None
        # self._eval_approach_hits = LinearIntervalFunction([(-1.0, 0.0), (0.0, 0.0), (0.7, 0.0), (1, 1.0), (2, 1.0)])
        self._eval_approach_solid_hits = LinearIntervalFunction([(-1.0, 0.0), (0.0, 0.0), (0.7, 0.0), (1, 1.0), (2, 1.0)])
        self._eval_approach_transp_hits = LinearIntervalFunction([(-1.0, 0.1), (0.0, 0.1), (1, 1.0), (2, 1.0)])

    def _initialize(self, inputs):
        assert len(inputs) == 2
        obj_id_str, p_obj = inputs

        if isinstance(p_obj, PoseWithFreeDOF):
            T_W_O = p_obj.calculate({'vert_rot': 0.0})
        elif isinstance(p_obj, Placement):
            T_W_O = p_obj.T_W_F.calculate({'vert_rot': 0.0}) * p_obj.T_F_O_list[0]
        else:
            T_W_O = KdlFrame.fromDict(p_obj)
        self._T_W_O = T_W_O
        self._directions = getUniformDirections(math.radians(20.0))
        self._dir_rays_hit_map = collideRaysAtApproach(self._state, obj_id_str, T_W_O, self._directions)

        # The score is non-linear
        score_threshold = 0.01

        # Summarize results
        self._scored_dirs = []
        self._dir_score_map = {}
        for dir_idx, rays_info in self._dir_rays_hit_map.items():
            ratio_solid = 1.0 - rays_info.getRaysCountSolidHit() / float(rays_info.getRaysCount())
            ratio_transp = 1.0 - rays_info.getRaysCountTransparentHit() / float(rays_info.getRaysCount())

            score_solid = self._eval_approach_solid_hits.interpolate(ratio_solid)
            score_transp = self._eval_approach_transp_hits.interpolate(ratio_transp)
            score = score_solid * score_transp
            if score > score_threshold:
                self._scored_dirs.append( (score, dir_idx) )
                self._dir_score_map[dir_idx] = score

        self._scored_dirs = list(sorted(self._scored_dirs, key=lambda x: x[0], reverse=True))

    def generate(self, inputs: list) -> GeneratorResult:
        assert len(inputs) == 2

        if self._dir_rays_hit_map is None:
            self._initialize(inputs)
            if not self._scored_dirs:
                print('GenApproaches - no directions')

                # No samples - failure
                # TODO: reason of failure
                return GeneratorResult.Failed([0, 1])
                #.debugStopPlanning()
        else:
            return GeneratorResult.NoMore()

        # Single-shot
        result = Approaches( self._T_W_O.p, [(score, self._directions[dir_idx]) for score, dir_idx in self._scored_dirs] )

        return GeneratorResult.SingleSample([TypedValue('Approaches', result)])

    def can_visualize(self) -> bool:
        return not self._dir_rays_hit_map is None

    def _getCentralRay(self, rays_info: ApproachHits,#list[tuple[str, KdlVector, KdlVector]]
                       ) -> tuple[float, RayHits] | None:
        center = KdlVector()
        # for obj_id, pos, hit_pos in rays_info:
        for hit in rays_info.hits:
            pos = hit.p1
            center = center + pos
        center = center / rays_info.getRaysCount()
        # center = center / len(rays_info)
        result = None
        # for ray in rays_info:
        for hit in rays_info.hits:
            pos = hit.p1
            # obj_id, pos, hit_pos = ray
            dist = (pos - center).Norm()
            if result is None or result[0] > dist:
                result = (dist, hit)
        return result

    def get_visualization(self) -> tuple[str, list[Marker]]:
        assert not self._dir_rays_hit_map is None
        markers = []

        for dir_idx, v_dir in enumerate(self._directions):
            color = [0.0, 1.0, 1.0]
            markers.append( cMarkerArrow(f'GenApproaches.all_directions', dir_idx,
                                          self._T_W_O.p - v_dir*0.5, self._T_W_O.p - v_dir*0.05,
                                          0.002, color) )

        #show_all = True
        show_rays_dirs = [KdlVector(1, 0, -0.2), KdlVector(1, 0, 0.5), KdlVector(0.4, 1, -0.5)]
        show_rays_dir_idx = []
        for show_rays_dir in show_rays_dirs:
            best_match = None
            # Get the closest direction
            # print(f'GenApproaches.visualize: showing rays for dir {show_rays_dir}')
            for dir_idx, v_dir in enumerate(self._directions):
                angle = angleBetweenVectors(v_dir, show_rays_dir)
                # print(f'GenApproaches.visualize: angle for dir {dir_idx}, {v_dir}: {angle}')

                if best_match is None or angle < best_match[0]:
                    best_match = (angle, dir_idx)
            assert not best_match is None
            show_rays_dir_idx.append( best_match[1] )

        for dir_idx, rays_info in self._dir_rays_hit_map.items():
            if dir_idx in self._dir_score_map and (score := self._dir_score_map[dir_idx]) > 0.001:
                # Show central ray
                width = score * 0.03
                central_ray = self._getCentralRay(rays_info)
                assert not central_ray is None

                pos = central_ray[1].p1
                hit_pos = central_ray[1].getRaySolidHitOrEnd()
                markers.append( cMarkerArrow(f'GenApproaches.c_rays_{dir_idx}', 0, pos, hit_pos, width, [0.0, 0.0, 1.0]) )

            # Show all rays
            if dir_idx in show_rays_dir_idx:
                marker_idx = 0
                ray_width = 0.005
                for hit in rays_info.hits:
                    p1 = hit.p1
                    p2 = hit.getRaySolidHitOrEnd()

                    if hit.hitSolid():
                        color = [1.0, 0.0, 0.0]
                    elif hit.hitTransparent():
                        color = [1.0, 0.5, 0.0]
                    else:
                        color = [0.0, 1.0, 0.0]
                    markers.append( cMarkerArrow(f'GenApproaches.rays_{dir_idx}', marker_idx,
                                                 p1, p2, ray_width, color) )
                    marker_idx = marker_idx + 1

        generator_data = ''
        return generator_data, markers


class GenGraspHandConfig:
    # (:stream GenGraspHandConfig
    #   :inputs (?obj - ObjectId ?gr - GraspId ?sd - Side)
    #   :outputs (?qh - HandConf)
    #   :certified (GraspConf ?gr ?sd ?qh)
    # )
    def __init__(self, state: State):
        self._state = state
        self._generated = False

    def generate(self, inputs: list) -> GeneratorResult:
        assert len(inputs) == 3
        # Single-shot
        if self._generated:
            return GeneratorResult.NoMore()
        # else:
        obj_id_str, grasp_id_str, side_str = inputs
        sti = getStatesInterface()
        grasp_def = sti.getGraspDef(self._state, obj_id_str, grasp_id_str)
        self._generated = True
        return GeneratorResult.SingleSample([
            TypedValue('ConfG', sti.getGraspConf(side_str, grasp_def))])


class GenPregraspHandConfig:
    # (:stream GenPregraspHandConfig
    #   :inputs (?obj - ObjectId ?gr - GraspId ?sd - Side)
    #   :outputs (?qh - HandConf)
    # )
    def __init__(self, state: State):
        self._state = state
        self._generated = False

    def generate(self, inputs: list) -> GeneratorResult:
        assert len(inputs) == 3
        # Single-shot
        if self._generated:
            return GeneratorResult.NoMore()
        # else:
        obj_id_str, grasp_id_str, side_str = inputs
        sti = getStatesInterface()

        grasp_def = sti.getGraspDef(self._state, obj_id_str, grasp_id_str)
        self._generated = True
        return GeneratorResult.SingleSample([
            TypedValue('ConfG', sti.getPregraspConf(side_str, grasp_def))])

class GenIk:
#   (:stream GenIk
#     :inputs (?p - Pose ?sd - Side ?obj - ObjectId
#               ?qh - ConfG)
#     :outputs (?q - ConfA ?qt - ConfT)
#     :relay (?q -> ?p)
#   )
    def __init__(self, state: State):
        self._state = state
        self._candidate_solutions = None
        self._velma_solv = getVelmaSolver()

    def _initialize(self, inputs: list):
        p, side, obj, qh = inputs

        self._T_B_E = KdlFrame.fromDict(p)
        # print(f'GenIk for pose: {T_B_E}')

        # Uses KinematicsSolverVelma to start up the moveit IK solver
        # Evaluation of configurations - distance from constraints:
        # joint limits, singularities, at wrist

        # Generate a number of options: torso_angle, elbow rotation, discrete configurations.
        # Evaluate each option.
        # print('GenIk: calculating set of IK solutions')

        candidate_solutions = []
        for torso_angle in np.linspace(math.radians(-85), math.radians(85), 9, endpoint=True):
            elbow_rot_list = list(np.linspace(math.radians(-180), math.radians(180), 13, endpoint=False))
            solutions = self._velma_solv.vsolv.calculateIkSet(side, self._T_B_E, torso_angle, elbow_rot_list)
            #print(f'torso_angle {torso_angle}, solutions: {len(solutions)}')
            for q in solutions:
                q_map = {
                    'torso_0_joint': torso_angle,
                    f'{side}_arm_0_joint': q[0],
                    f'{side}_arm_1_joint': q[1],
                    f'{side}_arm_2_joint': q[2],
                    f'{side}_arm_3_joint': q[3],
                    f'{side}_arm_4_joint': q[4],
                    f'{side}_arm_5_joint': q[5],
                    f'{side}_arm_6_joint': q[6]
                }

                # Sanity check (sti.fkArmTorso is slow):
                # T_W_Efk = sti.fkArmTorso(side, pos_map=q_map)
                # assert not T_W_Efk is None
                # diff = T_W_Efk.diff(T_B_E, 1.0)
                # if diff.rot.Norm() > 0.01 or diff.vel.Norm() > 0.01:
                #     #print('ik error')
                #     # print(f'fk: {T_W_Efk.toDict()}')
                #     # print(f'dest: {T_B_E.toDict()}')
                #     raise Exception()

                score = self._velma_solv.evaluateArmConf(side, q)
                if score > 0.01:
                    # Save solution
                    candidate_solutions.append( (score, torso_angle, q) )
        # print('GenIk: calculating set of IK solutions: done')
        print(f'GenIK candidate solutions: {len(candidate_solutions)} for object {obj}')
        if len(candidate_solutions) == 0:
            # In this case, the problem is in the ee pose.
            return None
        
        self._candidate_solutions = sorted(candidate_solutions, key=lambda x: x[0], reverse=True)

    def generate(self, inputs: list) -> GeneratorResult:
        assert len(inputs) == 4
        p, side, obj, qg = inputs

        if self._candidate_solutions is None:
            self._initialize(inputs)
            if not self._candidate_solutions:
                # Failed: probalby the pose is out of reach
                # TODO: reason

                return GeneratorResult.Failed([0])
        # else:

        # Pick one candidate and verify using MoveIt
        sti = getStatesInterface()

        psc = PlanningSceneCreator()

        # Prepare environment model with all objects except grasped objects
        # Ignore all grasped objects
        # TODO: attach grasped objects in the scene!
        ign_obj_id_list = [obj_id_str for obj_id_str, _, _ in sti.getGraspedObjects(self._state)] + [obj]
        env_shapes_dict = sti.getEnvironmentCollisionShapes(self._state, ign_obj_id_list)
        for env_obj_id, shape_list in env_shapes_dict.items():
            for idx, shape in enumerate(shape_list):
                psc.addObject(f'env/{env_obj_id}/{idx}', shape)

        assert sti.applyPlanningScene( psc.buildPlanningScene() )

        js = None
        qt_dest = None
        for idx in range(3):
            if not self._candidate_solutions:
                assert sti.applyPlanningScene( psc.cleanupPlanningScene() )
                return GeneratorResult.NoMore()

            score, qt_dest, q = self._candidate_solutions.pop(0)
            # print(f'Trying IK candidate no {idx}, score: {score}, {q}')
            print(f'Trying IK candidate no {idx}, score: {score}')
            q_map = {
                f'{side}_arm_0_joint': q[0],
                f'{side}_arm_1_joint': q[1],
                f'{side}_arm_2_joint': q[2],
                f'{side}_arm_3_joint': q[3],
                f'{side}_arm_4_joint': q[4],
                f'{side}_arm_5_joint': q[5],
                f'{side}_arm_6_joint': q[6]
            }
            qg_map = qg.toJsMap()
            q_map = addGripperToJsMap(side, q_map, qg_map)

            js = sti.ikArmTorso(side, qt_dest, self._T_B_E, start_pose=q_map)
            # TODO: check if the result IK is much different than initial IK
            if not js is None:
                # print(f'moveit IK: {js}')
                assert sti.applyPlanningScene( psc.cleanupPlanningScene() )
                return GeneratorResult.Sample([
                    TypedValue('ConfA', ConfA.fromJsMap(side, js)),
                    TypedValue('ConfT', ConfT(qt_dest))])
            #raise
        assert sti.applyPlanningScene( psc.cleanupPlanningScene() )

        # The best 10 candidates failed, there is a problem.
        # This time, the problem is in collisions.
        # TODO: reason
        return GeneratorResult.Failed([0])


def addTorsoToTraj(traj: JointTrajectory, qt: ConfT):
    result = copy.copy(traj)
    if not 'torso_0_joint' in result.joint_names:
        result.joint_names.append( 'torso_0_joint' ) # type: ignore
        for pt_idx in range(len(traj.points)):
            traj.points[pt_idx].positions.append(qt.q[0]) # type: ignore
    return result

def addGripperToJsMap(side: str, js_map: dict[str, float], js_map_add: dict[str, float]):
    gripper_joint_names = [
        f'{side}_HandFingerOneKnuckleOneJoint',
        f'{side}_HandFingerOneKnuckleTwoJoint',
        f'{side}_HandFingerOneKnuckleThreeJoint',
        f'{side}_HandFingerThreeKnuckleTwoJoint',
        f'{side}_HandFingerThreeKnuckleThreeJoint',
        f'{side}_HandFingerTwoKnuckleOneJoint',
        f'{side}_HandFingerTwoKnuckleTwoJoint',
        f'{side}_HandFingerTwoKnuckleThreeJoint']

    result = copy.copy(js_map)

    for joint_name in gripper_joint_names:
        if not joint_name in result:
            result[joint_name] = js_map_add[joint_name]
    return result

def addGripperToTraj(side: str, traj: JointTrajectory, js_map: dict[str, float]):
    gripper_joint_names = [
        f'{side}_HandFingerOneKnuckleOneJoint',
        f'{side}_HandFingerOneKnuckleTwoJoint',
        f'{side}_HandFingerOneKnuckleThreeJoint',
        f'{side}_HandFingerThreeKnuckleTwoJoint',
        f'{side}_HandFingerThreeKnuckleThreeJoint',
        f'{side}_HandFingerTwoKnuckleOneJoint',
        f'{side}_HandFingerTwoKnuckleTwoJoint',
        f'{side}_HandFingerTwoKnuckleThreeJoint']

    result = copy.copy(traj)

    for joint_name in gripper_joint_names:
        if not joint_name in result.joint_names:
            result.joint_names.append( joint_name ) # type: ignore
            for pt_idx in range(len(traj.points)):
                traj.points[pt_idx].positions.append( js_map[joint_name] ) # type: ignore
    return result


class GenUngraspTraj:
#   (:stream GenUngraspTraj
#     :inputs ( ?obj - ObjectId
#               ?p - Pose
#               ?gr - GraspId
#               ?sd - Side
#               ?q1 - ConfA
#               ?qt - ConfT)
#     :outputs (?tr - GraspTraj ?q2 - ConfA)
#     :certified (and (GraspTrajBegin ?tr ?sd ?q1 ?qt) (GraspTrajEnd ?tr ?sd ?q2 ?qt))
#   )
    def __init__(self, state: State):
        self._state = state
        self._grasp_traj = None
        self._velma_solv = getVelmaSolver()

    def _initialize(self, inputs):
        obj_id_str, p, grasp_id_str, side_str, q1, qt = inputs

        # (q1, qt) is a configuration when the robot holds the object
        # placed at the destination pose (put down pose)

        T_W_O = KdlFrame.fromDict(p)
        sti = getStatesInterface()

        # Calculate pose of the possibly grasped object
        T_W_E = self._velma_solv.vsolv.getArmFk(side_str, qt.q[0], q1.q)

        grasp_def = sti.getGraspDef(self._state, obj_id_str, grasp_id_str)

        T_G_Og = sti.getGraspPose(grasp_def)
        qg_g = sti.getGraspConfJsMap(side_str, grasp_def)

        T_G_Op = sti.getPregraspPose(grasp_def)
        qg_p = sti.getPregraspConfJsMap(side_str, grasp_def)

        T_E_G = sti.getT_E_G(side_str)
        T_G_E = T_E_G.Inverse()

        movements = sti.getGraspMovements(side_str, grasp_def)
        # movements:
        # 1. to ee pose
        # 2. to q_gr or ee_pose
        # ...
        # n-1. to ee_pose
        # n. to q_gr (at grasp, i.e. q1)

        # TODO: there is a bug somewhere here - the ungrasp goes sideways
        # Some Cartesian paths cannot be followed due to kinematic constraints
        # TODO: choose more comfortable poses for the arm - use python based IK
        # for WUT Velma (KinematicsSolverVelma) to speedup choice of a good configuration.

        q_current = joinJsMap([q1.toJsMap(), qt.toJsMap()])
        #qg_current = sti.getPregraspConfJsMap(side_str, grasp_def)
        #p_ee_current = sti.fkArmTorso(side_str, pos_map=q_current)

        T_W_Ed = T_W_O * T_G_Op.Inverse() * T_G_E
        attached_collision_objects = []
        self._fraction, traj = sti.computeCartesianPath(side_str, q_current, [T_W_Ed], attached_collision_objects)

        # sti.applyPlanningScene( psc.cleanupPlanningScene() )

        if traj is None or self._fraction < 0.9:
            self._grasp_traj = None
            return
        
        # The first configuration q1 (at grasp) is known
        grasp_traj = GraspTraj(side_str)
        grasp_traj.addFingersMovement(qg_g, qg_p)
        q_next = get_trajectory_last_point(traj)
        grasp_traj.addArmMovement(traj, q_current, q_next)

        self._grasp_traj = grasp_traj
        self._last_q = q_next


    def generate(self, inputs: list) -> GeneratorResult:
        assert len(inputs) == 6
        obj, p, gr, side_str, q1, qt = inputs

        # Single-shot
        if self._grasp_traj is None:
            self._initialize(inputs)

            if self._grasp_traj is None:
                # The problem may be with grasp, object position or configuration
                return GeneratorResult.Failed([1, 2, 4])

            return GeneratorResult.SingleSample([TypedValue('GraspTraj', self._grasp_traj),
                                                 TypedValue('ConfA', ConfA.fromJsMap(side_str, self._last_q))])

        return GeneratorResult.NoMore()



class GenGraspTrajRev:
#   (:stream GenGraspTrajRev
#     :inputs (?tr - GraspTraj
#             ?sd - Side
#             ?q1 ?q2 - ConfA
#             ?qt - ConfT)
#     :domain (and (GraspTrajBegin ?tr ?sd ?q1 ?qt) (GraspTrajEnd ?tr ?sd ?q2 ?qt))
#     :outputs (?tr_rev - GraspTraj)
#     :certified (and (GraspTrajBegin ?tr_rev ?sd ?q2 ?qt) (GraspTrajEnd ?tr ?sd ?q1 ?qt))
#   )
    def __init__(self, state: State):
        self._state = state
        self._traj_rev = None

    def _initialize(self, inputs):
        traj, side_str, q1, q2, qt = inputs
        assert isinstance(traj, GraspTraj)
        self._traj_rev = traj.getReversed()

    def generate(self, inputs: list) -> GeneratorResult:
        assert len(inputs) == 5
        tr, side_str, q1, q2, qt = inputs

        # Single-shot
        if self._traj_rev is None:
            self._initialize(inputs)

            return GeneratorResult.SingleSample([
                TypedValue('GraspTraj', self._traj_rev)])
        return GeneratorResult.NoMore()


def getUniformDirections(angle: float) -> list[KdlVector]:
    result = []
    alpha_angles_count = int(math.pi/angle)+1;
    if alpha_angles_count < 2:
        raise Exception('angle is too big')

    for alpha in np.linspace(0, math.pi, alpha_angles_count+1, endpoint=True):
        z = math.cos(alpha)
        r = math.sin(alpha)
        l = 2.0*math.pi*r
        beta_angles_count = int(l/angle)+1;
        for beta in np.linspace(0, 2.0*math.pi, beta_angles_count, endpoint=False):
            x = r * math.cos(beta)
            y = r * math.sin(beta)
            vec = KdlVector(x, y, z)
            assert abs(vec.Normalize()-1.0) < 0.001
            result.append( vec )
    return result


class GenPickTraj:
    # (:stream GenPickTraj
    #   :inputs (?obj - ObjectId ?gr - GraspId ?sd - Side ?ap - Approaches
    #             ?q1 - ConfA ?qt - ConfT)
    #   :outputs (?tr - JointTraj ?q2 - ConfA ?qt2 - ConfT)
    #   :certified (and (TrajBegin ?tr ?sd ?q1 ?qt) (TrajEnd ?tr ?sd ?q2 ?qt))
    # )
    def __init__(self, state: State):
        self._state = state
        self._traj = None
        self._velma_solv = getVelmaSolver()

    def _initialize(self, inputs: list):
        obj_id_str, grasp_id_str, side_str, q1, qt = inputs

        sti = getStatesInterface()

        # Calculate pose of the possibly grasped object
        js_start = joinJsMap([q1.toJsMap(), qt.toJsMap()])
        T_W_E = self._velma_solv.vsolv.getArmFk(side_str, qt.q[0], q1.q)
        assert not T_W_E is None
        T_G_O = sti.getGraspPose(sti.getGraspDef(self._state, obj_id_str, grasp_id_str))
        T_E_G = sti.getT_E_G(side_str)
        T_W_O = T_W_E * T_E_G * T_G_O

        lift_dist = 0.02
        T_W_Od = KdlFrame(KdlVector(0, 0, lift_dist)) * T_W_O
        T_W_Ed = T_W_Od * T_G_O.Inverse() * T_E_G.Inverse()

        # TODO: add planning scene
        # This may fail
        attached_collision_objects = []
        self._fraction, traj = sti.computeCartesianPath(side_str, js_start, [T_W_Ed], [])
        if traj is None:
            self._traj = None
        else:
            if self._fraction < 0.9:
                self._traj = None
            else:
                self._traj = addTorsoToTraj(traj, qt)
                self._js_end = get_trajectory_last_point(self._traj)


    def generate(self, inputs: list) -> GeneratorResult:
        assert len(inputs) == 5
        obj, gr, side_str, q1, qt = inputs

        # Single-shot
        if self._traj is None:
            self._initialize(inputs)

            if self._traj is None:
                # Generation failed, probably the configuration is bad
                return GeneratorResult.Failed([1, 4])

            return GeneratorResult.SingleSample([
                TypedValue('JointTraj', self._traj),
                TypedValue('ConfA', ConfA.fromJsMap(side_str, self._js_end))])

        return GeneratorResult.NoMore()


class GenTrajRev:
    # (:stream GenTrajRev
    #   :inputs ( ?tr - JointTraj
    #             ?sd - Side
    #             ?q1 - ConfA
    #             ?qt1 - ConfT
    #             ?q2 - ConfA
    #             ?qt2 - ConfT)
    #   :domain (and (TrajBegin ?tr ?sd ?q1 ?qt1) (TrajEnd ?tr ?sd ?q2 ?qt2))
    #   :outputs (?tr_rev - JointTraj)
    #   :certified (and (TrajBegin ?tr_rev ?sd ?q2 ?qt2) (TrajEnd ?tr_rev ?sd ?q1 ?qt1))
    # )
    def __init__(self, state: State):
        self._state = state
        self._generated = False

    def generate(self, inputs: list) -> GeneratorResult:
        assert len(inputs) == 6
        traj, side, q1, qt1, q2, qt2 = inputs

        # Single-shot
        if self._generated:
            return GeneratorResult.NoMore()
        
        self._generated = True
        traj_rev = reverseTrajectory(traj)
        return GeneratorResult.SingleSample([
            TypedValue('JointTraj', traj_rev)])


class GenCurrentConfAT:
    # (:stream GenCurrentConfAT
    #   :inputs (?sd - Side)
    #   :outputs (?q - ConfA ?qt - ConfT)
    #   :certified (AtConfA ?q AtConfT ?qt)
    # )
    def __init__(self, state: State):
        self._state = state
        self._generated = False

    def generate(self, inputs: list) -> GeneratorResult:
        assert len(inputs) == 1
        side_str = inputs[0]

        # Single-shot
        if self._generated:
            return GeneratorResult.NoMore()
        # else:
        self._generated = True
        sti = getStatesInterface()
        return GeneratorResult.SingleSample([
            TypedValue('ConfA', sti.getConfA(self._state, side_str)),
            TypedValue('ConfT', sti.getConfT(self._state))])


class GenCurrentConfT:
    # (:stream GenCurrentConfT
    #   :outputs (?qt - ConfT)
    #   :certified (AtConfT ?qt)
    # )
    def __init__(self, state: State):
        self._state = state
        self._generated = False

    def generate(self, inputs: list) -> GeneratorResult:
        assert len(inputs) == 0
        # Single-shot
        if self._generated:
            return GeneratorResult.NoMore()
        # else:
        self._generated = True
        sti = getStatesInterface()
        return GeneratorResult.SingleSample([
            TypedValue('ConfT', sti.getConfT(self._state))])


class GenCurrentConfH:
    # (:stream GenCurrentConfH
    #   :outputs (?q - ConfH)
    #   :certified (AtConfH ?q)
    # )
    def __init__(self, state: State):
        self._state = state
        self._generated = False

    def generate(self, inputs: list) -> GeneratorResult:
        assert len(inputs) == 0
        # Single-shot
        if self._generated:
            return GeneratorResult.NoMore()
        # else:
        sti = getStatesInterface()
        
        self._generated = True
        for pred in self._state['Predicates']:
            if pred['Name'] == 'AtConfH':
                return GeneratorResult.SingleSample([
                    TypedValue('ConfH', sti.getConfH(self._state))])
        raise Exception()


class GenCurrentConfG:
#   (:stream GenCurrentConfG
#     :inputs (?sd - Side)
#     :outputs (?q - ConfG)
#     :certified (AtConfG ?q)
#   )
    def __init__(self, state: State):
        self._state = state
        self._generated = False

    def generate(self, inputs: list) -> GeneratorResult:
        assert len(inputs) == 1
        side = inputs[0]
        # Single-shot
        if self._generated:
            return GeneratorResult.NoMore()
        # else:
        self._generated = True
        sti = getStatesInterface()
        return GeneratorResult.SingleSample([
            TypedValue('ConfG', sti.getConfG(self._state, side))])

class GenLookAt:
    # (:stream GenLookAt
    #   :inputs (?pt - SomePoint ?qt - ConfT)
    #   :outputs (?qn - ConfH)
    #   # TODO: domain, certified
    # )
    def __init__(self, state: State):
        self._state = state
        self._generated = False

    def _simplifiedHeadIK(self, torso_angle: ConfT, pt: KdlVector) -> tuple[float, float]:
        # yaw is a sum of torso angle and head pan angle
        yaw = math.atan2( pt.y(), pt.x() ) - torso_angle.q[0]

        # pitch is head tilt angle
        xy_dist = math.sqrt( pt.x()**2 + pt.y()**2 )
        head_z = 1.75
        pitch = -math.atan2( pt.z() - head_z, xy_dist )

        yaw = max(-1.5, min(1.5, yaw))
        pitch = max(-1.0, min(1.3, pitch))
        return (yaw, pitch)

    def generate(self, inputs: list) -> GeneratorResult:
        assert len(inputs) == 2

        # Single-shot
        if self._generated:
            return GeneratorResult.NoMore()

        p, qt = inputs
        if isinstance(p, PoseWithFreeDOF):
            T_W_O = p.calculate({'vert_rot': 0.0})
        elif isinstance(p, Placement):
            T_W_O = p.T_W_F.calculate({'vert_rot': 0.0})
        else:
            T_W_O = KdlFrame.fromDict(p)

        hp, ht = self._simplifiedHeadIK(qt, T_W_O.p)

        self._generated = True
        return GeneratorResult.SingleSample([
            TypedValue('ConfH', ConfH(hp, ht))])


class GenTraj:
#   (:stream GenTraj
#     :inputs ( ?sd - Side
#               ?q1 - ConfA
#               ?qt1 - ConfT
#               ?q2 - ConfA
#               ?qt2 - ConfT
#               ?qg - ConfG)
#     :outputs (?tr - JointTraj)
#     :certified (and (TrajBegin ?tr ?sd ?q1 ?qt1) (TrajEnd ?tr ?sd ?q2 ?qt2))
#   )
    def __init__(self, state: State):
        self._state = state
        self._velma_solv = getVelmaSolver()
        self._visualization = {}

    def generate(self, inputs: list) -> GeneratorResult:
        assert len(inputs) == 6
        side, q1, qt1, q2, qt2, qg = inputs

        q1_map = joinJsMap([q1.toJsMap(), qt1.toJsMap(), qg.toJsMap()])
        q2_map = joinJsMap([q2.toJsMap(), qt2.toJsMap(), qg.toJsMap()])
        
        sti = getStatesInterface()
        psc = PlanningSceneCreator()

        # Prepare environment model with all objects except grasped objects
        # Ignore all grasped objects
        # TODO: attach grasped objects in the scene!
        ign_obj_id_list = [obj_id_str for obj_id_str, _, _ in sti.getGraspedObjects(self._state)]
        #ign_obj_id_list = []

        self._visualization['grasped'] = []
        # TODO: get grasped objects from generator inputs, instead of current state
        for gr_obj_id, gr_side, gr_grasp_id in sti.getGraspedObjects(self._state):
            model = sti.getModelByObjectId(self._state, gr_obj_id)
            col_objects = sti.getModelCollisionShapes(model, KdlFrame(), enlarge=0.0)
            # TODO: add support for compound objects
            assert len(col_objects) == 1
            shape = col_objects[0]
            grasp_def = sti.getGraspDef(self._state, gr_obj_id, gr_grasp_id)
            T_G_O = sti.getGraspPose(grasp_def)
            T_E_G = sti.getT_E_G(gr_side)
            T_E_O = T_E_G * T_G_O
            # psc.addGraspedObject(gr_side, shape.tp, shape.size, T_E_O, f'{side}_arm_7_link')
            ee_link_name = f'{gr_side}_arm_7_link'
            shape.setObjectPose(T_E_O, ee_link_name)
            psc.addGraspedObject(gr_side, shape)
            self._visualization['grasped'].append( (q1, qt1, q2, qt2, gr_side, shape) )


        env_shapes_dict = sti.getEnvironmentCollisionShapes(self._state, ign_obj_id_list)
        for env_obj_id, shape_list in env_shapes_dict.items():
            for idx, shape in enumerate(shape_list):
                # psc.addObject(f'env/{env_obj_id}/{idx}', shape.tp, shape.size, shape.T)
                psc.addObject(f'env/{env_obj_id}/{idx}', shape)

        if _simplified_trajectory_generation:
            traj = buildFakeTrajectory(side, q1_map, q2_map)
        else:
            assert sti.applyPlanningScene( psc.buildPlanningScene() )
            attached_collision_objects = psc.getGraspedObjects()
            traj = sti.computeTrajectory(side, q1_map, q2_map, True, attached_collision_objects)
            assert sti.applyPlanningScene( psc.cleanupPlanningScene() )

        if traj is None:
            # TODO: reason
            # raise
            return GeneratorResult.Failed([3])
            #.debugStopPlanning()
        # else:

        # This is only for visualization:
        # traj = addGripperToTraj(side, traj, q1_map)

        return GeneratorResult.Sample([
            TypedValue('JointTraj', traj)])

    def can_visualize(self) -> bool:
        return bool(self._visualization)

    def get_visualization(self) -> tuple[str, list[Marker]]:
        markers = []
        marker_id = 0
        ns = 'volume'
        color = [1.0, 1.0, 0.0]

        for x in self._visualization['grasped']:
            q1, qt1, q2, qt2, gr_side, shape = x
            assert isinstance(shape, PrimitiveShape)
            T_W_E = self._velma_solv.vsolv.getArmFk(gr_side, qt1.q[0], q1.q)
            T_W_O = T_W_E * shape.getShapePose(f'{gr_side}_arm_7_link')
            markers.append( cMarker(ns, marker_id, T_W_O, shape.tp, shape.size, color) )
            marker_id = marker_id + 1

            T_W_E = self._velma_solv.vsolv.getArmFk(gr_side, qt2.q[0], q2.q)
            T_W_O = T_W_E * shape.getShapePose(f'{gr_side}_arm_7_link')
            markers.append( cMarker(ns, marker_id, T_W_O, shape.tp, shape.size, color) )
            marker_id = marker_id + 1

        generator_data = ''
        return generator_data, markers
        

class GenObjectCurrentPose:
#   (:stream GenObjectCurrentPose
#     :inputs (?obj - ObjectId)
#     :outputs (?p_obj - Pose)
#     :certified (and (AtPose ?obj ?p_obj))
#   )
    def __init__(self, state: State):
        self._state = state
        self._generated = False

    def generate(self, inputs: list) -> GeneratorResult:
        assert len(inputs) == 1

        # Single-shot
        if self._generated:
            return GeneratorResult.NoMore()
        # else:
        obj_id_str = inputs[0]
        sti = getStatesInterface()
        T_W_O = sti.getObjectExactPose(self._state, obj_id_str)
        self._generated = True
        return GeneratorResult.SingleSample([
            TypedValue('Pose', T_W_O.toDict())])


class GenFk:
#   (:stream GenFk
#     :inputs (?q - ConfA ?qt - ConfT ?sd - Side)
#     :outputs (?p_ee - Pose)
#   )
    def __init__(self, state: State):
        self._state = state
        self._generated = False
        self._velma_solv = getVelmaSolver()

    def generate(self, inputs: list) -> GeneratorResult:
        assert len(inputs) == 3
        q, qt, side = inputs

        # Single-shot
        if self._generated:
            return GeneratorResult.NoMore()
        # else:
        self._generated = True

        sti = getStatesInterface()

        # TODO: use internal FK
        # js_start = joinJsMap([q.toJsMap(), qt.toJsMap()])
        T_W_E = self._velma_solv.vsolv.getArmFk(side, qt.q[0], q.q)
        # T_W_E = sti.fkArmTorso(side, pos_map=js_start)
        assert not T_W_E is None

        return GeneratorResult.SingleSample([
            TypedValue('Pose', T_W_E.toDict())])

class GenTuckArmTraj:
#   (:stream GenTuckArmTraj
#     :inputs ( ?sd - Side
#               ?q1 - ConfA
#               ?qt - ConfT
#               ?qg - ConfG)
#     :outputs (?tr - JointTraj
#               ?q2 - ConfA)
#     :certified (and (TrajBegin ?tr ?sd ?q1 ?qt) (TrajEnd ?tr ?sd ?q2 ?qt))
#   )
    def __init__(self, state: State):
        self._state = state
        self._generated = False

    def generate(self, inputs: list) -> GeneratorResult:
        assert len(inputs) == 4
        side, q1, qt, qg = inputs

        # Single-shot
        if self._generated:
            return GeneratorResult.NoMore()
        # else:
        self._generated = True

        sti = getStatesInterface()

        q1_map = joinJsMap([q1.toJsMap(), qt.toJsMap(), qg.toJsMap()])

        if side == 'right':
            T_T0_E = KdlFrame(KdlVector(0, -0.35, 0.55))
        else:
            T_T0_E = KdlFrame(KdlVector(0, 0.35, 0.55))
        pos_tol = [0.15, 0.15, 0.15]

        target = cPoseStamped(cHeader(None, 'torso_link0'), T_T0_E.toRosPose())
        
        sti = getStatesInterface()
        psc = PlanningSceneCreator()

        # Prepare environment model with all objects except grasped objects
        # Ignore all grasped objects
        # TODO: attach grasped objects in the scene!
        ign_obj_id_list = [obj_id_str for obj_id_str, _, _ in sti.getGraspedObjects(self._state)]

        #self._visualization['grasped'] = []
        # TODO: get grasped objects from generator inputs, instead of current state
        for gr_obj_id, gr_side, gr_grasp_id in sti.getGraspedObjects(self._state):
            model = sti.getModelByObjectId(self._state, gr_obj_id)
            col_objects = sti.getModelCollisionShapes(model, KdlFrame(), enlarge=0.0)
            # TODO: add support for compound objects
            assert len(col_objects) == 1
            shape = col_objects[0]
            grasp_def = sti.getGraspDef(self._state, gr_obj_id, gr_grasp_id)
            T_G_O = sti.getGraspPose(grasp_def)
            T_E_G = sti.getT_E_G(gr_side)
            T_E_O = T_E_G * T_G_O
            ee_link_name = f'{gr_side}_arm_7_link'
            shape.setObjectPose(T_E_O, ee_link_name)
            psc.addGraspedObject(gr_side, shape)
            #self._visualization['grasped'].append( (q1, qt1, q2, qt2, gr_side, shape) )

        env_shapes_dict = sti.getEnvironmentCollisionShapes(self._state, ign_obj_id_list)
        for env_obj_id, shape_list in env_shapes_dict.items():
            for idx, shape in enumerate(shape_list):
                psc.addObject(f'env/{env_obj_id}/{idx}', shape)

        # if _simplified_trajectory_generation:
        #     traj = buildFakeTrajectory(side, q1_map, q2_map)
        # else:
        assert sti.applyPlanningScene( psc.buildPlanningScene() )
        attached_collision_objects = psc.getGraspedObjects()
        traj = sti.computeTrajectoryToPose(side, q1_map, target, False,
                        attached_collision_objects, pos_tol, None )
        assert sti.applyPlanningScene( psc.cleanupPlanningScene() )

        if traj is None:
            # This is very unproblable, but possible
            return GeneratorResult.Failed()

        q_end = get_trajectory_last_point(traj)

        return GeneratorResult.SingleSample([
            TypedValue('JointTraj', traj),
            TypedValue('ConfA', ConfA.fromJsMap(side, q_end))])


class GenOtherSide:
#   (:stream GenOtherSide
#     :inputs (?sd - Side)
#     :outputs (?sd2 - Side)
#     :certified (OtherSide ?sd ?sd2)
#   )
    def __init__(self, state: State):
        self._state = state
        self._generated = False

    def generate(self, inputs: list) -> GeneratorResult:
        assert len(inputs) == 1
        side = inputs[0]

        # Single-shot
        if self._generated:
            return GeneratorResult.NoMore()
        # else:
        self._generated = True
        if side == 'left':
            other_side_str = 'right'
        elif side == 'right':
            other_side_str = 'left'
        else:
            raise Exception(f'Wrong side "{side}"')

        return GeneratorResult.SingleSample([
            TypedValue('Side', other_side_str)])


class GenCurrentGraspAtSide:
#   (:stream GenCurrentGraspAtSide
#     :inputs (?sd - Side)
#     :outputs (?obj - ObjectId ?gr - GraspId)
#     :certified (and (GraspedSdGr ?obj ?sd ?gr))
#   )
    def __init__(self, state: State):
        self._state = state
        self._generated = False

    def generate(self, inputs: list) -> GeneratorResult:
        assert len(inputs) == 1
        side = inputs[0]

        if self._generated:
            return GeneratorResult.NoMore()
        # else:
        sti = getStatesInterface()

        for o_id_str, sd_str, gr_str in sti.getGraspedObjects(self._state):
            if sd_str == side:
                return GeneratorResult.SingleSample([
                    TypedValue('ObjectId', o_id_str),
                    TypedValue('GraspId', gr_str)])
        # else:
        # TODO: reason - problem is in planning domain
        return GeneratorResult.Failed()


# This is a global variable used by GenResourceId generator
next_resource_id = 0

class GenResourceId:
#   (:stream GenResourceId
#     :outputs (?res_id - ResourceId)
#   )
    def __init__(self, state: State):
        self._state = state
        self._generated = False

    def generate(self, inputs: list) -> GeneratorResult:
        assert len(inputs) == 0
        # Single-shot
        if self._generated:
            return GeneratorResult.NoMore()
        # else:
        self._generated = True

        global next_resource_id
        resource_id = next_resource_id
        next_resource_id = next_resource_id + 1

        return GeneratorResult.SingleSample([
            TypedValue('ResourceId', resource_id)])


class GenGraspReservedSpace:
#   (:stream GenGraspReservedSpace
#     :inputs (?obj - ObjectId ?sd - Side ?gr - GraspId ?ee_gr - Pose)
#     :outputs (?gr_vol - Volume)
#   )
    def __init__(self, state: State):
        self._state = state
        self._generated = False
        self._visualization = {}

    def generate(self, inputs: list) -> GeneratorResult:
        assert len(inputs) == 4

        # Single-shot
        if self._generated:
            return GeneratorResult.NoMore()
        # else:
        self._generated = True

        obj_id, side, grasp_id, p_ee = inputs
        T_W_E = KdlFrame.fromDict(p_ee)

        sti = getStatesInterface()
        obj_model = sti.getModelByObjectId(self._state, obj_id)
        grasp_def = sti.getGraspDef(self._state, obj_id, grasp_id)
        q_grasp = sti.getGraspConfJsMap(side, grasp_def)
        T_G_O = sti.getGraspPose(grasp_def)
        T_E_G = sti.getT_E_G(side)
        T_W_G = T_W_E * T_E_G
        T_W_O = T_W_G * T_G_O

        solv = KinematicsSolverBarrettHand()
        T_E_P = T_E_G * solv.getT_P_G().Inverse()

        T_W_P = T_W_E * T_E_P
        approach_dir = T_W_O.p - T_W_P.p

        # There are multiple possible representations, e.g.:
        #   1. This volume consists of a number of rays that:
        #     - are along pulling motion pull_dir
        #     - hit enlarged gripper or enlarged object
        #     - start in the free space near the robot
        #     - end at plane perpendicular to pull_dir that is centered at object centre
        #   2. It is build of enlarged shapes of the object and gripper repeated along pull_dir

        enlarge = 0.02
        pull_dist_max = 0.3
        pull_dir = -approach_dir
        pull_dir.Normalize()
        samples_count = int(pull_dist_max / enlarge)
        assert samples_count > 3

        #sti.getEnvironmentCollisionShapes(self._state, ign_objects, enlarge)
        collisions = []
        for pull_dist in np.linspace(0, pull_dist_max, samples_count, endpoint=True):
            # Displacement
            T_W_Wds = KdlFrame( pull_dir * pull_dist )
            obj_col = sti.getModelCollisionShapes(obj_model, T_W_Wds * T_W_O, enlarge)
            gr_col = sti.getGripperCollisionShapes(side, T_W_Wds * T_W_G, q_grasp, enlarge)
            collisions = collisions + obj_col + gr_col

        self._visualization['volume'] = collisions
        return GeneratorResult.SingleSample([
            TypedValue('Volume', Volume(collisions, [obj_id]))
            ])
    
    def can_visualize(self) -> bool:
        return bool(self._visualization)

    def get_visualization(self) -> tuple[str, list[Marker]]:
        markers = []
        marker_id = 0
        ns = 'volume'
        color = [1.0, 1.0, 0.0]
        for idx, col in enumerate(self._visualization['volume']):
            markers.append( cMarker(ns, marker_id, col.getShapePose('world'), col.tp, col.size, color) )
            marker_id = marker_id + 1
        
        generator_data = ''
        return generator_data, markers

# For each volume, check collisions with environment.
# Returns a list of the same length as the list of volumes.
# Each element in the output list is a set of colliding objects with
# the corresponding volume in the input list.
def getCollidingObjects(state: State, vol_list: list[Volume]) -> list[set[str]]:
    assert len(vol_list) > 0

    sti = getStatesInterface()

    # Ignored objects must be identical for all volumes
    ign_obj_list: None|list[str] = None
    out_col_list: list[set[str]] = []
    for vol in vol_list:
        out_col_list.append(set())
        if ign_obj_list is None:
            ign_obj_list = vol.ign_obj_list
        else:
            assert len(ign_obj_list) == len(vol.ign_obj_list)
            for ign_obj in ign_obj_list:
                assert ign_obj in vol.ign_obj_list
    assert not ign_obj_list is None

    # Prepare environment model with all objects except grasped objects and ignored objects
    # Ignore all grasped objects
    ign_obj_id_list = [gr_obj_id_str for gr_obj_id_str, _, _ in sti.getGraspedObjects(state)] + ign_obj_list
    env_shapes_dict = sti.getEnvironmentCollisionShapes(state, ign_obj_id_list)

    GROUP_ENV = 1 << 0
    GROUP_VOL = 1 << 1
    pb = getPyBulletInterface()
    pb.assertEmpty()

    # Add environment
    for env_obj_id, shape_list in env_shapes_dict.items():
        for idx, shape in enumerate(shape_list):
            pb.addObject(f'env/{env_obj_id}/{idx}', shape.tp, shape.size,
                         shape.getShapePose('world'), GROUP_ENV, GROUP_VOL)

    for vol_idx, vol in enumerate(vol_list):
        for shape_idx, shape in enumerate(vol.col_shapes):
            pb.addObject(f'vol/{vol_idx}/{shape_idx}', shape.tp, shape.size,
                         shape.getShapePose('world'), GROUP_VOL, GROUP_ENV)

    contacts = pb.getContacts()
    pb.clearAll()

    for name1, name2 in contacts:
        items1 = name1.split('/')
        items2 = name2.split('/')
        if items1[0] == 'vol':
            vol_idx = int(items1[1])
            env_id = items2[1]
        elif items2[0] == 'vol':
            vol_idx = int(items2[1])
            env_id = items1[1]
        else:
            raise Exception(f'Unexpected output: {items1}, {items2}')

        out_col_list[vol_idx].add(env_id)

    return out_col_list


class GenClear:
#   (:stream GenClear
#     :inputs (?vol - Volume)
#     :outputs (?obj - ObjectId)
#   )
    def __init__(self, state: State):
        self._state = state
        self._generated = False

    def _prepareShoulderDistEvaluator(self):
        # Prepare evaluation functions
        sti = getStatesInterface()
        arm_range_min = sti.getArmRangeMin()
        arm_range_max = sti.getArmRangeMax() + 0.2     # Add some range - gripper is long
        arm_range_med1 = arm_range_min*0.75 + arm_range_max*0.25
        arm_range_med2 = arm_range_min*0.25 + arm_range_max*0.75
        self._eval_arm_dist = LinearIntervalFunction(
            [(-1.0, 0.0), (arm_range_min, 0.0), (arm_range_med1, 1.0),
             (arm_range_med2, 1.0), (arm_range_max, 0.0), (10000000.0, 0.0)])

        self._shoulder_centers_left = sti.getPossibleShoulderCenters('left', 3)
        self._shoulder_centers_right = sti.getPossibleShoulderCenters('right', 3)

    def _evalShoulderDist(self, pt:KdlVector) -> tuple[float, float]:
        result_left = 0.0
        for pt_shoulder in self._shoulder_centers_left:
            dist = (pt_shoulder-pt).Norm()
            result_left = max(result_left, self._eval_arm_dist.interpolate(dist))

        result_right = 0.0
        for pt_shoulder in self._shoulder_centers_right:
            dist = (pt_shoulder-pt).Norm()
            result_right = max(result_right, self._eval_arm_dist.interpolate(dist))
        return result_left, result_right

    def generate(self, inputs: list) -> GeneratorResult:
        assert len(inputs) == 1
        # Single-shot
        if not self._generated:
            # based on: removeCollidingPlacements
            vol = inputs[0]
            assert isinstance(vol, Volume)

            sti = getStatesInterface()
            self._prepareShoulderDistEvaluator()
            static_objects_ids = sti.getStaticObjectsIds(self._state)
            col_objs = getCollidingObjects(self._state, [vol])[0]
            print(f'  Colliding objects: {col_objs}')
            # closest_obj = None
            self._ev_obj_list = []
            for col_obj in col_objs:
                if col_obj in static_objects_ids:
                    # Nothing to do with a static object
                    continue
                # else
                T_W_O = sti.getObjectExactPose(self._state, col_obj)
                score_range_left, score_range_right = self._evalShoulderDist(T_W_O.p)
                score_range = max(score_range_left, score_range_right)
                #dist = math.sqrt( T_W_O.p.x()**2 + T_W_O.p.y()**2 )
                if score_range < 0.01:
                    # There is a problem: the colliding object cannot be removed
                    return GeneratorResult.Failed([0])
                else:
                    self._ev_obj_list.append( (score_range, col_obj) )

            self._generated = True
        # else:

        # print(f'GenClear: volume size: {len(vol.col_shapes)}, ignore: {vol.ign_obj_list}')

        if len(self._ev_obj_list) == 0:
            return GeneratorResult.NoMore()

        if _sampling_type == 'random_pdf':
            chooseElement_func = chooseRandomPdfElementIdx
        elif _sampling_type == 'best':
            chooseElement_func = chooseBestElementIdx
        elif _sampling_type == 'random':
            chooseElement_func = chooseRandomElementIdx
        else:
            raise Exception(f'Unknown sampling type: {_sampling_type}')

        
        sample_idx = chooseElement_func([score for score, col_obj in self._ev_obj_list])
        score, closest_obj_id = self._ev_obj_list[sample_idx]
        del self._ev_obj_list[sample_idx]

        # else:
#        dist, closest_obj_id = self._ev_obj_list.pop(0)
        print(f'  Selected object: {closest_obj_id}, score: {score}')

        return GeneratorResult.Sample([
            TypedValue('ObjectId', closest_obj_id)])


class GenManipSide:
#   (:stream GenManipSide
#     :inputs (?p_obj - SomePose)
#     :outputs (?sd - Side)
#   )
    def __init__(self, state: State):
        self._state = state
        self._preferred_sides = None

    def generate(self, inputs: list) -> GeneratorResult:
        assert len(inputs) == 1

        if self._preferred_sides is None:
            p = inputs[0]
            if isinstance(p, PoseWithFreeDOF):
                T_W_O = p.calculate({'vert_rot': 0.0})
            elif isinstance(p, Placement):
                T_W_O = p.T_W_F.calculate({'vert_rot': 0.0})
            else:
                T_W_O = KdlFrame.fromDict(p)

            if T_W_O.p.y() > 0:
                self._preferred_sides = ['left', 'right']
            else:
                self._preferred_sides = ['right', 'left']

        if len(self._preferred_sides) == 0:
            return GeneratorResult.NoMore()
        # else:

        side = self._preferred_sides.pop(0)

        return GeneratorResult.SingleSample([
            TypedValue('Side', side)])


class GenClosedG:
#   (:stream GenClosedG
#     :inputs (?sd - Side)
#     :outputs (?qg - ConfG)
#   )
    def __init__(self, state: State):
        self._state = state
        self._generated = False

    def generate(self, inputs: list) -> GeneratorResult:
        assert len(inputs) == 1
        # Single-shot
        if self._generated:
            return GeneratorResult.NoMore()
        # else:
        self._generated = True
        side = inputs[0]

        prox = math.radians(110.0)
        dist = prox * 0.333333
        js = {
        f'{side}_HandFingerOneKnuckleOneJoint': 0.0,
        f'{side}_HandFingerOneKnuckleTwoJoint': prox,
        f'{side}_HandFingerOneKnuckleThreeJoint': dist,
        f'{side}_HandFingerTwoKnuckleTwoJoint': prox,
        f'{side}_HandFingerTwoKnuckleThreeJoint': dist,
        f'{side}_HandFingerThreeKnuckleTwoJoint': prox,
        f'{side}_HandFingerThreeKnuckleThreeJoint': dist
        }
        return GeneratorResult.SingleSample([
            TypedValue('ConfG', ConfG.fromJsMap(side, js))])


class GenAnySidePref:
#   (:stream GenAnySidePref
#     :outputs (?sd - SidePref)
#   )
    def __init__(self, state: State):
        self._state = state
        self._generated = False

    def generate(self, inputs: list) -> GeneratorResult:
        assert len(inputs) == 0
        # Single-shot
        if self._generated:
            return GeneratorResult.NoMore()
        # else:
        self._generated = True
        return GeneratorResult.SingleSample([
            TypedValue('SidePref', 'any')])


def getPouringVolume(obj_id: str, pt: KdlVector) -> Volume:
    col_shapes = []

    # The 'pour' shape: a vertical cylinder above the objects
    radius = 0.25
    height = 0.6
    lift = 0.02
    col_shapes.append( PrimitiveShape('cylinder', [radius, height],
                                        KdlFrame(pt + KdlVector(0, 0, height/2 + lift)),
                                        KdlFrame(), 'world'))
    
    # The 'bridge' shape: a horizontal cylinder that joins the robot
    # and the pour shape.
    bridge_z = 1.3
    # Calculate the frame of bridge
    pt1 = KdlVector(0, 0, bridge_z)
    pt2 = KdlVector(pt.x(), pt.y(), bridge_z)
    pt_c = (pt1+pt2)/2

    axis_z = pt2-pt1
    bridge_length = axis_z.Normalize()
    axis_x = KdlVector(0, 0, 1)
    axis_y = axis_z * axis_x
    T_W_BR = KdlFrame(KdlRotation(axis_x, axis_y, axis_z), pt_c)
    col_shapes.append( PrimitiveShape('cylinder', [radius, bridge_length],
                                        T_W_BR, KdlFrame(), 'world'))

    return Volume(col_shapes, [obj_id])


class GenPourSpace:
#   (:stream GenPourSpace
#     :inputs (?obj - ObjectId ?p - SomePose)
#     :outputs (?vol - Volume)
#   )
    def __init__(self, state: State):
        self._state = state
        self._generated = False

    def _initialize(self, inputs: list) -> None|Volume:
        obj, p = inputs
        if isinstance(p, PoseWithFreeDOF):
            T_W_O = p.calculate({'vert_rot': 0.0})
        elif isinstance(p, Placement):
            T_W_O = p.T_W_F.calculate({'vert_rot': 0.0})
        else:
            T_W_O = KdlFrame.fromDict(p)

        pt_pour = T_W_O.p
        vol = getPouringVolume(obj, pt_pour)

        colliding_objects = getCollidingObjects(self._state, [vol])[0]
        if len(colliding_objects) > 0:
            return None
        #else:
        return vol

    def generate(self, inputs: list) -> GeneratorResult:
        assert len(inputs) == 2

        # Single-shot

        # Check if pouring to object ?obj is possible at pose ?p and generate
        # volume of space to be reserved for this task.
        if self._generated:
            # Already generated
            return GeneratorResult.NoMore()
        # else:
        self._generated = True

        vol = self._initialize(inputs)
        if vol is None:
            return GeneratorResult.Failed([1])
        # else:

        return GeneratorResult.SingleSample([
            TypedValue('Volume', vol)])


class GenPourPlace:
#   (:stream GenPourPlace
#     :inputs (?dst - ObjectId)
#     :domain (and (HasHole ?src) (HasHole ?dst))
#     :outputs (?p_obj - Placement ?vol - Volume)
#   )
    def __init__(self, state: State):
        self._state = state
        self._visualization = {}
        self._samples: None|list[tuple[Placement, Volume]] = None
        self._eval_other_sample = LinearIntervalFunction([(-1.0, 0.0), (0.0, 0.0), (0.2, 1.0), (10000000.0, 0.0)])

    def _initialize(self, inputs: list) -> list[tuple[Placement, Volume]]:
        obj = inputs[0]

        sti = getStatesInterface()
        flat_bases = sti.getFlatBases(self._state, obj)
        if len(flat_bases) != 1:
            raise Exception(f'GenPourPlace supports destination objects with exactly one base.')
        obj_flat_base = flat_bases[0]

        # Prepare evaluation functions
        arm_range_min = sti.getArmRangeMin()
        arm_range_max = sti.getArmRangeMax() + 0.2     # Add some range - gripper is long
        arm_range_med1 = arm_range_min*0.75 + arm_range_max*0.25
        arm_range_med2 = arm_range_min*0.25 + arm_range_max*0.75
        self._eval_arm_dist = LinearIntervalFunction(
            [(-1.0, 0.0), (arm_range_min, 0.0), (arm_range_med1, 1.0),
             (arm_range_med2, 1.0), (arm_range_max, 0.0), (10000000.0, 0.0)])

        self._shoulder_centers_left = sti.getPossibleShoulderCenters('left', 3)
        self._shoulder_centers_right = sti.getPossibleShoulderCenters('right', 3)


        flat_surfaces = sti.getFlatSurfaces(self._state)
        samples = sti.sampleSurfaces(flat_surfaces, step_size=0.025, margin=0.05)
        self._visualization['samples'] = samples

        close_samples = self._removeUnreachablePlaces(samples)
        self._visualization['close_samples'] = close_samples

        # Use collision detection to eliminate colliding placements
        pour_vol_list: list[Volume] = []
        for pt_pour in close_samples:
            pour_vol_list.append( getPouringVolume(obj, pt_pour.p) )

        collisions = getCollidingObjects(self._state, pour_vol_list)

        assert len(close_samples) == len(pour_vol_list)
        assert len(pour_vol_list) == len(collisions)

        T_O_F = obj_flat_base.T_O_F
        assert isinstance(T_O_F, KdlFrame)
        T_F_O = T_O_F.Inverse()

        out: list[tuple[Placement, Volume]] = []
        for pt_pour, pour_vol, env_col in zip(close_samples, pour_vol_list, collisions):
            if len(env_col) > 0:
                continue
            # else:
            T_W_F = PoseWithFreeDOF()
            T_W_F.addStaticTransform( pt_pour )
            T_W_F.addRotation('vert_rot', KdlVector(0, 0, 1))
            placement = Placement(T_W_F, [T_F_O])
            out.append( (placement, pour_vol) )

        return out
    

    def _removeUnreachablePlaces(self, samples: list[KdlFrame]) -> list[KdlFrame]:
        result = []
        for pt in samples:
            ev_left, ev_right = self._evalShoulderDist(pt.p)
            if ev_left > 0.01 and ev_right > 0.01:
                result.append(pt)
        return result

    def _evalShoulderDist(self, pt:KdlVector) -> tuple[float, float]:
        result_left = 0.0
        for pt_shoulder in self._shoulder_centers_left:
            dist = (pt_shoulder-pt).Norm()
            result_left = max(result_left, self._eval_arm_dist.interpolate(dist))

        result_right = 0.0
        for pt_shoulder in self._shoulder_centers_right:
            dist = (pt_shoulder-pt).Norm()
            result_right = max(result_right, self._eval_arm_dist.interpolate(dist))
        return result_left, result_right
    
    def generate(self, inputs: list) -> GeneratorResult:
        assert len(inputs) == 1

        # Check if pouring to object ?obj is possible at pose ?p and generate
        # volume of space to be reserved for this task.
        if self._samples is None:
            self._samples = self._initialize(inputs)
            self._returned_placements_pts = []
            if len(self._samples) == 0:
                return GeneratorResult.Failed()
            # else:

        if len(self._samples) == 0:
            return GeneratorResult.NoMore()

        # Select the best placement
        evaluated_placements: list[tuple[Placement, Volume, float]] = []
        for placement, vol in self._samples:
            # Add some height
            pt_placement = placement.T_W_F.calculate({'vert_rot': 0.0}).p
            ev_left, ev_right = self._evalShoulderDist(pt_placement + KdlVector(0,0, 0.1))
            range_score = max(ev_left, ev_right)

            # Penalty for samples close to previous
            prev_score = 1.0
            for pt_returned in self._returned_placements_pts:
                dist = (pt_returned-pt_placement).Norm()
                prev_score = min(prev_score, self._eval_other_sample.interpolate(dist))
            
            score = range_score * prev_score
            evaluated_placements.append( (placement, vol, score) )

        if _sampling_type == 'random_pdf':
            chooseElement_func = chooseRandomPdfElementIdx
        elif _sampling_type == 'best':
            chooseElement_func = chooseBestElementIdx
        elif _sampling_type == 'random':
            chooseElement_func = chooseRandomElementIdx
        else:
            raise Exception(f'Unknown sampling type: {_sampling_type}')

        sample_idx = chooseElement_func([score for pl, vol, score in evaluated_placements])
        sel_placement, sel_vol, sel_score = evaluated_placements[sample_idx]
        
        placement_pt = sel_placement.T_W_F.calculate({'vert_rot': 0.0}).p
        print(f'GenPlace: selected placement: {placement_pt}, score: {sel_score}')
        if sel_score < 0.01:
            if self._returned_placements_pts:
                return GeneratorResult.NoMore()
            else:
                return GeneratorResult.Failed()
        # else:
        self._returned_placements_pts.append( placement_pt )

        # This pose has one DOF
        return GeneratorResult.Sample([
            TypedValue('Placement', sel_placement),
            TypedValue('Volume', sel_vol)])


class GenPourTraj:
#   (:stream GenPourTraj
#     :inputs ( ?src - ObjectId
#               ?sd - Side
#               ?gr - GraspId
#               ?dst - ObjectId
#               ?p_dst - Pose)
#     :domain (and (HasHole ?src) (HasHole ?dst))
#     :outputs (?tr - JointTraj ?q1 - ConfA ?qt1 - ConfT ?q2 - ConfA ?qt2 - ConfT)
#     :certified (and (TrajBegin ?tr ?sd ?q1 ?qt1) (TrajEnd ?tr ?sd ?q2 ?qt2))
#   )
    def __init__(self, state: State):
        self._state = state
        self._generated = False
        self._vsolv = getVelmaSolver()

    def _initialize(self, inputs: list):
        src, src_side, gr, dst, p_dst = inputs

        sti = getStatesInterface()
        src_hole = sti.getObjectHole(self._state, src)

        dst_pose = KdlFrame.fromDict(p_dst)
        dst_hole = sti.getObjectHole(self._state, dst)

        assert not src_hole is None
        assert not dst_hole is None

        src_grasp_def = sti.getGraspDef(self._state, src, gr)
        T_G_OS = sti.getGraspPose(src_grasp_def)
        T_E_G = self._vsolv.vsolv.getT_E_G(src_side)
        T_E_OS = T_E_G * T_G_OS
        T_OS_E = T_E_OS.Inverse()

        # pose of object dst
        T_W_OD = dst_pose

        # pose of hole of dst
        T_OD_ODH = dst_hole.T_O_F

        dst_hole_r = dst_hole.size[0]
        dst_hole_center_pos = (T_W_OD * T_OD_ODH).p

        elbow_circle_angles = list( np.linspace(-math.pi, math.pi, 12, endpoint=False) )

        pour_dst_angle = math.atan2( dst_hole_center_pos.y(), dst_hole_center_pos.x() )
        torso_angle_min = max(-1.3, pour_dst_angle - math.radians(90))
        torso_angle_max = min(1.3, pour_dst_angle + math.radians(90))

        print('GenPourTraj: pour_dst_angle: {}'.format(pour_dst_angle))        

        # Generate a lot of trajectories
        all_trajectories = []
        for appr_rot_z in np.linspace(-math.pi, math.pi, 16, endpoint=False):
            for src_rot_z in np.linspace(-math.pi, math.pi, 16, endpoint=False):
                T_appr_rot_z = KdlFrame(KdlRotation.RotZ(appr_rot_z))
                T_src_rot_z = KdlFrame(KdlRotation.RotZ(src_rot_z))
                # T_W_OSH1 = KdlFrame(dst_hole_center_pos) * T_appr_rot_z *\
                #                         KdlFrame(KdlVector(dst_hole_r-0.0, 0, 0.3))

                T_W_OSH2 = KdlFrame(dst_hole_center_pos) * T_appr_rot_z *\
                                        KdlFrame(KdlVector(dst_hole_r+0.02, 0, 0.25)) *\
                                        KdlFrame(KdlRotation.RotY(math.radians(-45.0))) *\
                                        T_src_rot_z

                T_W_OSH3 = KdlFrame(dst_hole_center_pos) * T_appr_rot_z *\
                                        KdlFrame(KdlVector(dst_hole_r+0.02, 0, 0.15)) *\
                                        KdlFrame(KdlRotation.RotY(math.radians(-90.0))) *\
                                        T_src_rot_z

                T_W_OSH4 = KdlFrame(dst_hole_center_pos) * T_appr_rot_z *\
                                        KdlFrame(KdlVector(dst_hole_r+0.02, 0, 0.15)) *\
                                        KdlFrame(KdlRotation.RotY(math.radians(-135.0))) *\
                                        T_src_rot_z

                # T_W_E1 = T_W_OSH1 * T_OS_E
                T_W_E2 = T_W_OSH2 * T_OS_E
                T_W_E3 = T_W_OSH3 * T_OS_E
                T_W_E4 = T_W_OSH4 * T_OS_E

                # Add more intermediate points
                # dT = T_W_E1.diff(T_W_E2, 1.0)
                # T_W_E1a = T_W_E1.addDelta(dT, 0.5)

                dT = T_W_E2.diff(T_W_E3, 1.0)
                T_W_E2a = T_W_E2.addDelta(dT, 0.5)

                dT = T_W_E3.diff(T_W_E4, 1.0)
                T_W_E3a = T_W_E3.addDelta(dT, 0.5)

                # # Select the best torso angle
                # best_value = None
                # best_torso_angle = None
                # for torso_angle in np.linspace( torso_angle_min, torso_angle_max, 9, endpoint=True):
                #     T_B_AB = self._vsolv.vsolv.getArmBaseFk(src_side.getStr(), torso_angle)
                #     cell_idx = lwr_ws.getCellIdx( T_B_AB.Inverse() * T_W_E1.p )
                #     cell_value = lwr_ws.getCellValue(cell_idx)
                #     if best_value is None or cell_value > best_value:
                #         best_value = cell_value
                #         best_torso_angle = torso_angle

                solutions: list[tuple[float, float, list]] = []
                for torso_angle in np.linspace( torso_angle_min, torso_angle_max, 9, endpoint=True):
                    # sol_list = self._vsolv.vsolv.calculateIkSet(src_side, T_W_E1, torso_angle,
                    #                                                                 elbow_circle_angles)
                    sol_list = self._vsolv.vsolv.calculateIkSet(src_side, T_W_E2, torso_angle,
                                                                                    elbow_circle_angles)
                    best_sol = None
                    for arm_q in sol_list:
                        score = self._vsolv.evaluateArmConf(src_side, arm_q)
                        if score < 0.01:
                            continue
                        # else:
                        if best_sol is None or score > best_sol[0]:
                            best_sol = score, arm_q

                    if not best_sol is None:
                        solutions.append( (torso_angle, best_sol[0], best_sol[1]) )

                solutions = sorted(solutions, key=lambda x: x[1], reverse=True)
                if len(solutions) > 5:
                    solutions = solutions[:5]

                #print(f'GenPourTraj: IK solutions: {len(q_list) for torso_angle, q_list in solutions}')
                # T_W_E_list = [T_W_E1, T_W_E1a, T_W_E2, T_W_E2a, T_W_E3, T_W_E3a, T_W_E4]
                T_W_E_list = [T_W_E2, T_W_E2a, T_W_E3, T_W_E3a, T_W_E4]
                for torso_angle, score, arm_q in solutions:
                    all_trajectories.append( (score, T_W_E_list, torso_angle, arm_q) )
                    # print(f'GenPourTraj: IK solutions: {solutions[0][1]}')

        all_trajectories = sorted( all_trajectories, key=lambda x: x[0] , reverse=True)

        out_traj = None
        out_qt = None
        for score, T_W_E_list, torso_angle, arm_q in all_trajectories:
            # Try to generate trajectory
            
            q_st = ConfA(src_side, arm_q)
            qt_st = ConfT(torso_angle)
            js_start = joinJsMap([q_st.toJsMap(), qt_st.toJsMap()])
            fraction, traj = sti.computeCartesianPath(src_side, js_start, T_W_E_list, [])
            print(f'score: {score:.3f}, fraction: {fraction:.2f}')
            if fraction > 0.95:
                out_traj = traj
                out_qt = torso_angle
                break

        return out_traj, out_qt


    def generate(self, inputs: list) -> GeneratorResult:
        assert len(inputs) == 5
        src, src_side, gr, dst, p_dst = inputs

        # Single-shot
        if self._generated:
            return GeneratorResult.NoMore()
        # else:
        self._generated = True
        traj, qt = self._initialize(inputs)

        if traj is None:
            return GeneratorResult.Failed([4])
        # else:
        assert not qt is None
        q_map1 = get_trajectory_first_point(traj)
        q_map2 = get_trajectory_last_point(traj)
        return GeneratorResult.SingleSample([
            TypedValue('JointTraj', traj),
            TypedValue('ConfA', ConfA.fromJsMap(src_side, q_map1)),
            TypedValue('ConfT', ConfT(qt)),
            TypedValue('ConfA', ConfA.fromJsMap(src_side, q_map2)),
            TypedValue('ConfT', ConfT(qt)),
        ])


generators = [GenCurrentGrasp, GenPlace, GenPlaceSimple, GenUngraspPose,
    GenGraspSd, GenGraspSdCl,
    GenApproaches, GenGraspHandConfig, GenPregraspHandConfig, GenIk,
    GenUngraspTraj, GenGraspTrajRev, GenPickTraj, GenTrajRev, GenCurrentConfAT,
    GenCurrentConfT, GenCurrentConfH, GenCurrentConfG, GenLookAt, GenTraj,
    GenObjectCurrentPose, GenFk, GenTuckArmTraj, GenOtherSide,
    GenCurrentGraspAtSide, GenResourceId, GenGraspReservedSpace, GenClear,
    GenManipSide, GenClosedG, GenAnySidePref, GenPourSpace, GenPourPlace,
    GenPourTraj]
def create_generator(name, state):
    for generator_class in generators:
        if name == generator_class.__name__:
            return generator_class(state)
    raise Exception(f'Unknown generator: {name}')
