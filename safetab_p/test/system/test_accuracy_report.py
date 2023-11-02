"""System tests for :mod:`tmlt.safetab_p.accuracy_report`."""

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
import shutil
import tempfile
from typing import Dict, List, Tuple

from nose.plugins.attrib import attr
from parameterized import parameterized

from tmlt.core.utils.testing import PySparkTest
from tmlt.safetab_p.accuracy_report import run_full_error_report_p
from tmlt.safetab_p.paths import RESOURCES_DIR


class TestRunFullErrorReport(PySparkTest):
    """Run the full error report."""

    def setUp(self):
        """Create temporary directories."""
        self.input_dir = tempfile.TemporaryDirectory()
        self.data_path = os.path.join(self.input_dir.name, "dataset")
        self.parameters_path = os.path.join(self.data_path, "input_dir_puredp")
        self.config_dir = tempfile.TemporaryDirectory()
        self.config_path = self.config_dir.name
        self.output_dir = tempfile.TemporaryDirectory()
        shutil.copytree(os.path.join(RESOURCES_DIR, "toy_dataset"), self.data_path)

    @parameterized.expand(
        [
            (
                {
                    "(USA, 1)": list(range(3)),
                    "(USA, 2)": list(range(3)),
                    "(STATE, 1)": list(range(3)),
                    "(STATE, 2)": list(range(3)),
                    "(COUNTY, 1)": list(range(3)),
                    "(COUNTY, 2)": list(range(3)),
                    "(TRACT, 1)": list(range(3)),
                    "(TRACT, 2)": list(range(3)),
                    "(PLACE, 1)": list(range(3)),
                    "(PLACE, 2)": list(range(3)),
                    "(AIANNH, 1)": list(range(3)),
                    "(AIANNH, 2)": list(range(3)),
                },
                "US",
            ),
            (
                {
                    "(PR-STATE, 1)": list(range(3)),
                    "(PR-STATE, 2)": list(range(3)),
                    "(PR-COUNTY, 1)": list(range(3)),
                    "(PR-COUNTY, 2)": list(range(3)),
                    "(PR-TRACT, 1)": list(range(3)),
                    "(PR-TRACT, 2)": list(range(3)),
                    "(PR-PLACE, 1)": list(range(3)),
                    "(PR-PLACE, 2)": list(range(3)),
                },
                "PR",
            ),
        ]
    )
    @attr("slow")  # This test is not run frequently as it takes longer than 10 minutes
    def test_safetab_p_multi_run_error_report(
        self,
        manual_thresholds: Dict[Tuple[str, str], List[int]],
        us_or_puerto_rico: str,
    ):
        """Run full error report on SafeTab-P."""
        temp_output_dir = os.path.join(self.output_dir.name, "safetab-p")
        run_full_error_report_p(
            self.parameters_path,
            self.data_path,
            temp_output_dir,
            self.config_path,
            trials=5,
            overwrite_config={"thresholds_p": manual_thresholds},
            us_or_puerto_rico=us_or_puerto_rico,
        )
        expected_subdir = ["full_error_report", "ground_truth", "single_runs"]
        actual_subdir = next(os.walk(temp_output_dir))[1]
        self.assertListEqual(expected_subdir, sorted(actual_subdir))
