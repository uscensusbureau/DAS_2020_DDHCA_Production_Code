#!/usr/bin/env python3
"""Create and display example error report."""

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

import pandas as pd

from tmlt.core.utils.testing import PySparkTest
from tmlt.safetab_p.accuracy_report import run_safetab_p_with_error_report
from tmlt.safetab_p.paths import RESOURCES_DIR

PySparkTest.setUpClass()

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
# mypy complains about using *paths
run_safetab_p_with_error_report(
    parameters_path=os.path.join(RESOURCES_DIR, "toy_dataset", "input_dir_puredp"),
    data_path=os.path.join(RESOURCES_DIR, "toy_dataset"),
    noisy_path="example_output/error_report_p/noisy",
    ground_truth_path="example_output/error_report_p/target",
    output_path="example_output/error_report_p/output",
    overwrite_config={"thresholds_p": manual_thresholds},
)

df = pd.read_csv(
    "example_output/error_report_p/output/error_report.csv", dtype={"REGION_ID": str}
)
pd.options.display.max_rows = 999
print(df)
PySparkTest.tearDownClass()
