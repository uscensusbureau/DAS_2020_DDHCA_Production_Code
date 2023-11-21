SPDX-License-Identifier: Apache-2.0
Copyright 2023 Tumult Labs

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

## Tumult Contents

This directory contains SafeTab-P's supporting Tumult-developed libraries. There are three folders, each of which contains a Python library used by SafeTab-P. For instructions on running SafeTab-P, see its [README](../safetab_p/README.md).

- **Core 0.6.0**: A Python library for performing differentially private computations. The design of Tumult Core is based on the design proposed in the [OpenDP White Paper](https://projects.iq.harvard.edu/files/opendp/files/opendp_programming_framework_11may2020_1_01.pdf), and can automatically verify the privacy properties of algorithms constructed from Tumult Core components. Tumult Core is scalable, includes a wide variety of components to handle various query types, and supports multiple privacy definitions. This library is available as an independent open-source release. For more, see its software documentation at https://docs.tmlt.dev/core/v0.6/.
- **Analytics 0.5.3**: A Python library for privately answering statistical queries on tabular data, implemented using Tumult Core. It is built on PySpark, allowing it to scale to large datasets. Its privacy guarantees are based on differential privacy, a technique that perturbs statistics to provably protect the data of individuals in the original dataset. This library is available as an independent open-source release. For more, see its software documentation at https://docs.tmlt.dev/analytics/v0.5/.
- **Common 0.8.1**: A Python library with utilities for reading and validating data. Code in Common is designed not to be specific to Census applications.

For details, consult each library's `README` within its respective subfolder. To see which new features have been added since the previous versions, consult their respective `CHANGELOG`s.
