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

import math
from tamp_htn_stream.core import TypedValue
from .data_types import ConfG

def get_constants() -> dict[str, TypedValue]:
    side = 'left'
    prox = math.radians(120.0)
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
    return {
        'CLOSED_HAND': TypedValue('ConfG', ConfG.fromJsMap(side, js))
        }
