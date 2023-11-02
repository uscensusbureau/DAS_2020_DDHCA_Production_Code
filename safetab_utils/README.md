# SafeTab-Utils

This module primarily contains common utility functions used by different SafeTab products.

The SafeTab product produces differentially private tables of statistics (counts) of demographic and housing characteristics crossed with detailed races and ethnicities at varying levels of geography (national, state, county, tract, place and AIANNH).

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

See [CHANGELOG](CHANGELOG.md) for version number information and changes from past versions.

## Testing

*Fast Tests:*

```
nosetests test/unit test/system -a '!slow'
```

*Slow Tests:*

Slow tests are tests that we run less frequently because they take a long time to run, or the functionality has been tested by other fast tests.

```
nosetests test/unit test/system -a 'slow'
```

*Detailed test plan is included in [TESTPLAN](TESTPLAN.md)*


### Warnings

1. Nosetests warning:

```
RuntimeWarning: Unable to load plugin windmill = windmill.authoring.nose_plugin:WindmillNosePlugin: No module named 'bin'
  RuntimeWarning)
```


2. In order to prevent the following warning:

```
WARN NativeCodeLoader: Unable to load native-hadoop library for your platform
```

`LD_LIBRARY_PATH` must be set correctly. Use the following:

```bash
export LD_LIBRARY_PATH=/usr/lib/hadoop/lib/native/
```

If `HADOOP_HOME` is set correctly (usually `/usr/lib/hadoop`), this may be replaced with

```bash
export LD_LIBRARY_PATH=$HADOOP_HOME/lib/native/
```

3. Other known warnings:

```
FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise
comparison
```
```
UserWarning: In Python 3.6+ and Spark 3.0+, it is preferred to specify type hints for pandas UDF instead of specifying
pandas UDF type which will be deprecated in the future releases. See SPARK-28264 for more details.
```
```
UserWarning: It is preferred to use 'applyInPandas' over this API. This API will be deprecated in the future releases.
See SPARK-28264 for more details.
```

## Additional resources

We also provide a standalone script for converting short form iteration mapping to long form.

The script is located in `tmlt/safetab_utils`.

To view the arguments for running,

```bash
python convert_short_form.py -h
```

Example cmd using all arguments,

```bash
python convert_short_form.py \
--short-form [INPUT_FILE] \      # Name of short-form race file.
--race-codes [RACE_CODES_FILE] \ # Name of race codes file.
--output [OUTPUT_FILE]           # Name of output file for long-form races.
```
