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


import numpy as np

class LinearIntervalFunction:
    def __init__(self, points: list[tuple[float, float]] | None = None):
        self.__points__ = []
    
        if not points is None:
            for pt in points:
                self.addPoint( pt[0], pt[1] )

    def addPoint(self, x: float, val: float) -> None:
        if len(self.__points__) == 0:
            self.__points__.append( (float(x), float(val)) )
            return
            
        for idx, pt in enumerate(self.__points__):
            if x < pt[0]:
                self.__points__.insert( idx, (x, val) )
                break
            elif idx == len(self.__points__)-1:
                self.__points__.append( (x, val) )
                break
            elif x > pt[0] and x < self.__points__[idx+1][0]:
                self.__points__.insert( idx+1, (x, val) )
                break
            elif x == pt[0] or x == self.__points__[idx+1][0]:
                raise Exception('Added point with the same x twice')
        
    def interpolate(self, x: float) -> float:
        x = float(x)
        if len(self.__points__) < 2:
            raise Exception('Could not interpolate the function with only one point')
            
        if x < self.__points__[0][0]:
            raise Exception('x is below the domain range')

        for idx, pt in enumerate(self.__points__):
            if idx == len(self.__points__)-1:
                break
            elif x >= pt[0] and x <= self.__points__[idx+1][0]:
                f = (x - pt[0]) / (self.__points__[idx+1][0] - pt[0])
                return (1.0-f) * pt[1] + f * self.__points__[idx+1][1]
        raise Exception('x is above the domain range: ' + str(x) + ', points: ' + str(self.__points__))

    def isInRange(self, x: float) -> bool:
        return x >= self.__points__[0][0] and x <= self.__points__[-1][0]

    def getDomainMin(self) -> float:
        return self.__points__[0][0]

    def getDomainMax(self) -> float:
        return self.__points__[-1][0]


# def angleDiff(a1, a2):

#     ad = a1 - a2
#     while ad > np.pi:
#         ad -= 2*np.pi

#     while ad < -np.pi:
#         ad += 2*np.pi

#     return ad

# Returns angle wrapped to [-pi, pi)
def wrapAngle(angle: float) -> float:
    return (angle+np.pi) % (2 * np.pi) - np.pi
