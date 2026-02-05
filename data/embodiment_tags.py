# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum

# for inference service to map robot_id to embodiment tags
EMBODIMENT_TAGS = {
    # Simulation / Lab environments
    "panda_wristcam": "panda",
    "xarm6_robotiq_wristcam": "xarm6",
    "widowxai_wristcam": "widowxai",
    "xarm7_robotiq_wristcam": 'xarm7',

    "panda_wristcam_rel": "panda_rel",
    "xarm6_robotiq_wristcam_rel": "xarm6_rel",
    "widowxai_wristcam_rel": "widowxai_rel",
    "xarm7_robotiq_wristcam_rel": 'xarm7_rel',

    # Real world environments
    "piper_real": 'piper_real',
    "arx5_real": 'arx5_real',
    "xarm7_real": 'xarm7_real',

    "arx5_real_rel": 'arx5_real_rel',
    "piper_real_rel": 'piper_real_rel',
    "xarm7_real_rel": 'xarm7_real_rel',
}

class EmbodimentTag(Enum):

    XARM6 = "xarm6"
    XARM7 = "xarm7"
    PANDA = "panda"
    WIDOWXAI = "widowxai" 
    XARM6_REL = "xarm6_rel"
    XARM7_REL = "xarm7_rel"
    PANDA_REL = "panda_rel"
    WIDOWXAI_REL = "widowxai_rel" 

    ARX5_REAL = "arx5_real"
    PIPER_REAL = "piper_real"
    XARM7_REAL = "xarm7_real"
    ARX5_REAL_REL = "arx5_real_rel"
    PIPER_REAL_REL = "piper_real_rel"
    XARM7_REAL_REL = "xarm7_real_rel"

# Embodiment tag string: to projector index in the Action Expert Module
EMBODIMENT_TAG_MAPPING = {
    EmbodimentTag.XARM6.value: 0,
    EmbodimentTag.PANDA.value: 1,
    EmbodimentTag.XARM7.value: 2,
    EmbodimentTag.WIDOWXAI.value: 3,

    EmbodimentTag.XARM6_REL.value: 0,
    EmbodimentTag.PANDA_REL.value: 1,
    EmbodimentTag.XARM7_REL.value: 2,
    EmbodimentTag.WIDOWXAI_REL.value: 3,

    EmbodimentTag.ARX5_REAL.value: 6,
    EmbodimentTag.PIPER_REAL.value: 7,
    EmbodimentTag.XARM7_REAL.value: 8,

    EmbodimentTag.ARX5_REAL_REL.value: 6,
    EmbodimentTag.PIPER_REAL_REL.value: 7,
    EmbodimentTag.XARM7_REAL_REL.value: 8,
}
