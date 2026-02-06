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

import cv2 as cv
import numpy as np

class VelmaWristConstraint:
    def __init__(self, image_path: str):
        self._img = cv.imread(image_path, cv.IMREAD_GRAYSCALE).astype(float)
        self._img = self._img / np.max(self._img)
        self._x_range = (-3.0, 3.0)
        self._y_range = (-3.0, 3.0)

    def test_showImage(self):
        cv.imshow("VelmaWristConstraint", self._img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def getScore(self, side: str, q5: float, q6: float):
        if side == 'left':
            q5 = -q5
            q6 = -q6
        ix = int( round( self._img.shape[1] * (q5 - self._x_range[0]) / (self._x_range[1] - self._x_range[0]) ) )
        iy = int( round( self._img.shape[0] * (q6 - self._y_range[0]) / (self._y_range[1] - self._y_range[0]) ) )
        return self._img[iy, ix]
