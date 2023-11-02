#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 Tumult Labs
# 
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
# 
#        http://www.apache.org/licenses/LICENSE-2.0
# 
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os

from tmlt.core.utils.testing import PySparkTest
from tmlt.safetab_p.paths import RESOURCES_DIR
from tmlt.safetab_p.safetab_p_analytics import run_plan_p_analytics

PySparkTest.setUpClass()
data_path = os.path.join(RESOURCES_DIR, "toy_dataset")
parameters_path = os.path.join(data_path, "input_dir_puredp")
output_dir = "example_output/safetab_p"
x = list(range(3))

manual_thresholds = {
    "(USA, 1)": x,
    "(USA, 2)": x,
    "(STATE, 1)": x,
    "(STATE, 2)": x,
    "(COUNTY, 1)": x,
    "(COUNTY, 2)": x,
    "(TRACT, 1)": x,
    "(TRACT, 2)": x,
    "(PLACE, 1)": x,
    "(PLACE, 2)": x,
    "(AIANNH, 1)": x,
    "(AIANNH, 2)": x,
}
run_plan_p_analytics(
    parameters_path,
    data_path,
    output_dir,
    overwrite_config={"thresholds_p": manual_thresholds},
)
PySparkTest.tearDownClass()
