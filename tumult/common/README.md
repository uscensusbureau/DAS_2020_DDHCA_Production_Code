# Common Utility

This module primarily contains common utility functions used by different Tumult projects.

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

## Overview

The utility functions include:
* Methods to serialize/deserialize objects into json format (marshallable).
* Expected error computations.
* A tool for creating error reports.

See [CHANGELOG](CHANGELOG.md) for version number information and changes from past versions.

## Testing

To run the tests, install the required dependencies from the `test_requirements.txt`

```
pip install -r test_requirements.txt
```

*All tests (including Doctest):*

```bash
nosetests tmlt/common --with-doctest
```

See `examples` for examples of features of `common`.
