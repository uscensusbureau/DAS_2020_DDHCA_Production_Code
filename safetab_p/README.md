# SafeTab-P for Detailed Race and Ethnicity

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

SafeTab-P produces differentially private tables of statistics (counts) of demographic and housing characteristics crossed with detailed races and ethnicities at varying levels of geography (national, state, county, tract, place and AIANNH).

The data product derived from the output of SafeTab-P is known as Detailed DHC-A.

More information about the SafeTab algorithm can be found in the [SafeTab-P specifications document](SafeTab_P_Documentation.pdf), which describes the problem, general approach, and (in Appendix A) the input and output file formats.

See [CHANGELOG](CHANGELOG.md) for version number information and changes from past versions.

## Requirements

SafeTab-P is designed to run on a Linux machine with Python 3.7 and PySpark 3. It can be run either locally on a single machine, or on a Spark cluster. We have developed and tested with Amazon EMR 6.10 in mind. When running on a nation-scale dataset, we recommend running SafeTab-P on an EMR cluster having at least 1 master and 2 core nodes of instance type r4.16xlarge or higher.

## Installation Instructions

For both local and cluster mode execution, the following preconditions must be met:
 
### 1. Dependencies

All python dependencies, as specified in [requirements.txt](requirements.txt) must be installed and available on the PYTHONPATH.

When running locally, dependencies can be installed by running:

```bash
sudo python3 -m pip install -r <absolute path of cloned DAS_2020_DDHCA_Production_Code repository>/safetab_p/requirements.txt
```

Note that one of the dependencies is PySpark, which requires Java 8 or later with `JAVA_HOME` properly set. If Java is not yet installed on your system, you can [install OpenJDK 8](https://openjdk.org/install/) (installation will vary based on the system type).

When running on an EMR cluster, make sure your cluster comes with Spark and Hadoop installed. Other dependencies can be installed as part of a [bootstrap action](https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-plan-bootstrap.html). A sample bootstrap scipt has been provided with SafeTab-P that installs the public dependencies as well as Tumult Core (see below): [public_install_bootstrap.sh](tmlt/safetab_p/resources/installation/public_install_bootstrap.sh). If you prefer to install Tumult Core from a wheel, rather than from PyPI (see below for details on both options), comment out the line in the bootstrap script that installs `tmlt.core`. EMR clusters come with Java pre-installed, so no additional steps are necessary.

### 2. Tumult Core installation

SafeTab-P also requires the Tumult Core library to be installed. Tumult Core can either be installed from the wheel file provided with this repository, or from PyPI (like external dependencies in the previous step). Users like the Census who prefer to avoid installing packages from PyPI will likely prefer installing from a wheel. Users who do not have such concerns will likely find it easier to install from PyPI.

#### Wheel installation

When running locally, Core can be installed by calling:

```bash
sudo python3 -m pip install --no-deps <absolute path of cloned DAS_2020_DDHCA_Production_Code repository>/tumult/core/tmlt_core-0.6.0-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```
When running on an EMR cluster, Tumult Core can be installed from the wheel file in a bootstrap action. Tumult Core is dependent on some of the [SafeTab-P requirements](requirements.txt) so ensure that the first EMR bootstrap action installs those dependencies. Then add a second bootstrap action to install Tumult Core using the following steps:
1. Upload [`core/bootstrap_script.sh`](../tumult/core/bootstrap_script.sh), [`../tumult/core/test_script.py`](../tumult/core/test_script.py), and [`../tumult/core/tmlt_core-0.6.0-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`](../core/tmlt_core-0.6.0-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl) to an S3 bucket. 
2. When creating the EMR cluster, add a bootstrap action. Set the bootstrap action's "Script location" to the s3 location of `core/bootstrap_script.sh`, and add the wheel (whl) file's s3 path as an Optional Argument.

To verify Tumult Core's installation on an EMR cluster, add [`../tumult/core/test_script.py`](../tumult/core/test_script.py) as a step to the EMR cluster. For details on how to add a step, see [adding a step](https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-work-with-steps.html) in the AWS documentation or the [Steps](#steps) section below.

    spark-submit --deploy-mode client --master yarn <s3-path-to-test_script.py>

If this step completes without any errors, verification is complete.

#### Pip installation

Tumult Core can also be installed from PyPI. If installing locally, run:

```sudo python3 -m pip install tmlt.core==0.6.0```

If installing on an EMR cluster, Tumult Core can be installed with a boostrap action. Our [sample boostrap script](tmlt/safetab_p/resources/installation/public_install_bootstrap.sh) installs Tumult Core along with other dependencies.

### 3. Environment Setup (local mode only)

Some environment variables need to be set. In particular, `PYSPARK_PYTHON` must be set to the correct python version:

```bash
export PYSPARK_PYTHON=$(which python3)
```

If running on the master node of an EMR cluster, some libraries need to be added to the `PYTHONPATH`:

```bash
export PYTHONPATH=/usr/lib/spark/python:/usr/lib/spark/python/lib:/usr/lib/spark/python/lib/py4j-0.10.9.5-src.zip:$PYTHONPATH
```

`PYTHONPATH` needs to updated to include the Tumult source directories. This assumes that required Python dependencies have been installed.

```bash
DIR=<absolute path of cloned DAS_2020_DDHCA_Production_Code repository>
export PYTHONPATH=$PYTHONPATH:$DIR/safetab_p
export PYTHONPATH=$PYTHONPATH:$DIR/safetab_utils
export PYTHONPATH=$PYTHONPATH:$DIR/tumult/core
export PYTHONPATH=$PYTHONPATH:$DIR/tumult/common
export PYTHONPATH=$PYTHONPATH:$DIR/tumult/analytics
```

`PYTHONPATH` also needs to be updated to include a Census Edited File (CEF) reader module for SafeTab `safetab_cef_reader.py`. This can be either MITRE’s CEF reader (developed separately), or the built-in mock CEF reader: `safetab_p/tmlt/mock_cef_reader`. Note that the mock CEF reader does not actually read CEF files, so it can only be used if input is being read from CSV files.

To use the MITRE CEF reader and add it to the Python Path, run the following command:

```bash
export PYTHONPATH=<absolute path of cloned DAS_2020_DDHCA_Production_Code repository>/mitre:$PYTHONPATH
```

Consult the CEF reader [README](../mitre/cef-readers/README.md) for more details.

## Input Directory Structure
There are two path inputs to SafeTab-P, a `parameters path` and a `data path`. This setup is replicated in the  `safetab_p/tmlt/safetab_p/resources/toy_dataset` directory. 

The parameters path should point to a directory containing:
  - config.json
  - ethnicity-characteristic-iterations.txt
  - race-and-ethnicity-code-to-iteration.txt
  - race-and-ethnicity-codes.txt
  - race-characteristic-iterations.txt
  
The `data path` has different requirements depending on the type of input reader being used. If a CEF reader is specified, the data path should point to the CEF reader's config file. If a CSV reader is being used, the data path should point to a directory containing:
  - person-records.txt
  - GRF-C.txt

Note:
- The parameters directory contains non-CEF input files. `config.json` specifies the privacy parameters. [`safetab_p/tmlt/safetab_p/resources/toy_dataset/input_dir_puredp/config.json`](tmlt/safetab_p/resources/toy_dataset/input_dir_puredp/config.json) contains PureDP privacy parameters and sets the algorithm to run using PureDP, while [`safetab_p/tmlt/safetab_p/resources/toy_dataset/input_dir_zcdp/config.json`](tmlt/safetab_p/resources/toy_dataset/input_dir_zcdp/config.json) contains Rho zCDP privacy parameters and sets the algorithm to run using Rho zCDP.

## Command line tool
The primary command line interface is driven by [`safetab_p/tmlt/safetab_p/safetab-p.py`](tmlt/safetab_p/safetab-p.py). We recommend running SafeTab-P from `safetab_p/tmlt/safetab_p` because the provided Spark properties files need to assume a directory from which the program is running.

The SafeTab-P command line program expects one of the following subcommands: `validate` or `execute`. These modes are explained below.

To view the list of available subcommands on console, enter:

```bash
safetab-p.py -h
```

To view the arguments for running in a given mode on console, enter:

```bash
safetab-p.py <subcommand> -h
```

The following subcommands are supported:

### Validate
`validate` mode validates the input data files against the input specification and reports any discrepancies.  Validation errors are written to the user-specified log file.

An example command to validate in local mode:

```bash
spark-submit \
   --properties-file resources/spark_configs/spark_local_properties.conf \
   safetab-p.py validate \
   <path to parameters folder>  \
   <path to reader configuration> \
   --log <log_file_name>
```

Note: If using csv readers replace `<path to reader configuration>` with `<path to data folder>`.

### Execute
The `execute` subcommand first validates the input files and then executes SafeTab-P. Both input validation and execution of the private algorithm use spark.

The SafeTab-P algorithm (executed with `safetab-p.py`) produces t1 and t2 tabulations. The output files `t1/*.csv` and `t2/*.csv` are generated and saved to the output folder specified on the command-line. The optional `--validate-private-output` flag can be passed to validate the generated output files.

Input and output directories must correspond to locations on the local machine (and not S3 or HDFS).

An example command to execute in local mode:

```bash
spark-submit \
      --properties-file resources/spark_configs/spark_local_properties.conf \
      safetab-p.py execute \
      <path to parameters folder>/  \
      <path to reader configuration> \
      <path to output folder> \
      --log <log_file_name> \
      --validate-private-output
```

Notes:

If using csv readers replace `<path to reader configuration>` with `<path to data folder>`.

If, by default, the machine does not have enough disk space for the SafeTab output,
a larger directory can be mounted to a drive on the local machine prior to
execution with commands similar to the following:

```
> lsblk

NAME    MAJ:MIN RM  SIZE RO TYPE MOUNTPOINT
xvda    202:0    0   10G  0 disk
└─xvda1 202:1    0   10G  0 part /
xvdb    202:16   0  128G  0 disk
├─xvdb1 202:17   0    5G  0 part /emr
└─xvdb2 202:18   0  123G  0 part /mnt
xvdc    202:32   0  128G  0 disk /mnt1
xvdd    202:48   0  128G  0 disk /mnt2
xvde    202:64   0  128G  0 disk /mnt3

> mkdir <path to output folder>
> sudo mount /dev/xvdc <path to output folder>
```

Note: When running in local mode, input and output directories must correspond to locations on the local machine (and not S3).
A sample Spark custom properties file for local mode execution is located in [`safetab_p/tmlt/safetab_p/resources/spark_configs/spark_local_properties.conf`](tmlt/safetab_p/resources/spark_configs/spark_local_properties.conf).
While Spark properties are often specific to the environment (number of cores,
memory allocation, etc.), we strongly recommend that the `spark.sql.execution.arrow.enabled`
be set to `true` as is done in the example config file for local mode.
When pyarrow is enabled, the data exchange between Python and Java is much faster and results
in orders-of-magnitude differences in runtime performance. 

### Example scripts

The shell script [`safetab_p/examples/validate_input_safetab_p.sh`](examples/validate_input_safetab_p.sh) demonstrates running the SafeTab-P command line program in validate mode on the toy dataset using the csv reader.  An excerpt is shown here with comments:

```bash
safetab-p.py validate \  # validate the Safetab-P inputs
resources/toy_dataset/input_dir_puredp \   # the parameters directory (see note below)
resources/toy_dataset \   # the data_path (see note below)
-l example_output_p/safetab_toydataset.log  # desired log location
```

The shell script [`safetab_p/examples/run_safetab_p_local.sh`](examples/run_safetab_p_local.sh) demonstrates running the SafeTab-P command line program in execute mode using the csv reader with input and output validation.  An excerpt is shown here with comments:

```bash
safetab-p.py execute \  # execute the Safetab-P algorithm
resources/toy_dataset/input_dir_puredp \   # the parameters directory (see note below)
resources/toy_dataset \   # the data_path (see note below)
example_output_p \        # desired output location
-l example_output_p/safetab_toydataset.log \  # desired log location
--validate-private-output      # validate output after executing algorithm
```

See `safetab_p/examples` for examples of other features of the SafeTab-P command line program.

When running these examples, the SafeTab-P config file containing the input privacy-parameters is defined in [`safetab_p/tmlt/safetab_p/resources/toy_dataset/input_dir_puredp/config.json`](tmlt/safetab_p/resources/toy_dataset/input_dir_puredp/config.json). The default privacy definition is PureDP. If zCDP is desired, switch the parameters folder argument to `resources/toy_dataset/input_dir_zcdp`, which will utilize [`safetab_p/tmlt/safetab_p/resources/toy_dataset/input_dir_zcdp/config.json`](tmlt/safetab_p/resources/toy_dataset/input_dir_zcdp/config.json) as the SafeTab-P config.

Note that the toy dataset (located at `safetab_p/tmlt/safetab_p/resources/toy_dataset`) used in these examples is small and not representative of a realistic input. The output from SafeTab-P when run on this dataset will not be a representative or comparable to a run on a realistic input. We include the toy dataset to provide an example of input formats, an example of output formats, and a way to quickly experiment with running SafeTab-P, but not as a way to generate representative outputs.

### Running on an EMR cluster

To run on an existing EMR cluster, use the AWS Management Console to configure a step that will invoke the `spark-submit` command. There are two important preconditions:

1. All of the inputs to `spark-submit` must be located on s3.
1. A zip file containing the repository source code must be created and placed in s3.
1. The main driver python program, [`safetab_p/tmlt/safetab_p/safetab-p.py`](tmlt/safetab_p/safetab-p.py), must be placed in s3.

The zip file can be created using the following command, which creates a packaged repository `repo.zip` that contains Tumult’s products and MITRE’s CEF reader (it does not contain any other dependencies).

```bash
bash <path to cloned tumult repo>/safetab_p/examples/repo_zip.sh \
-t <path to cloned tumult repo> \
-r <path to CEF reader>
```

The `-r` argument is optional if you're using the built-in CSV reader rather than a CEF reader.

Note: The `repo_zip.sh` script has a dependency on associative arrays and works with bash version 4.0 or newer.

#### <a id="steps"></a>Steps:

It is possible to run `safetab-p.py` in cluster mode on an existing EMR cluster by [adding a step](https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-work-with-steps.html). To add a spark-submit step through the AWS Management Console:

1. In the [Amazon EMR console](https://console.aws.amazon.com/elasticmapreduce), on the Cluster List page, select the link for your cluster.
1. On the `Cluster Details` page, choose the `Steps` tab.
1. On the `Steps` tab, choose `Add step`.
1. Type appropriate values in the fields in the `Add Step` dialog, and then choose `Add`. Here are the sample values:

                Step type: Custom JAR

                Name: <any name that you want>

                JAR location: command-runner.jar

                Arguments:
                        spark-submit
                        --deploy-mode client --master yarn
                        --conf spark.driver.maxResultSize=20g
                        --conf spark.sql.execution.arrow.enabled=true
                        --py-files s3://<s3 repo.zip path>
                        s3://<s3 safetab-p main file path>/safetab-p.py
                        execute
                        s3://<s3 sample parameters files path>
                        s3://<s3 data_path>
                        s3://<s3 output directory>
                        --log s3://<s3 output directory>/<log file path>
                        --validate-private-output

                Action on Failure: Cancel and wait

Note: Output locations must be S3 paths. The `--log` and `--validate-private-output` arguments are optional. 

This requires a copy of safetab-p.py and the `repo.zip` (created from the instructions above) to be stored in an s3 bucket.

The above instructions can be repeated with the sample input files or the toy dataset. Also, similar instructions can be used for the `validate` subcommand with some changes to the command - replacing `execute` with `validate`, removing the output directory argument and `--validate-private-output` flag.

#### Spark properties

While Spark properties are often specific to the environment (number of cores, memory allocation, etc.), we strongly recommend that the `spark.sql.execution.arrow.enabled` be set to `true`. An example custom Spark properties config file for cluster mode is located in [`safetab_p/tmlt/safetab_p/resources/spark_configs/spark_cluster_properties.conf`](tmlt/safetab_p/resources/spark_configs/spark_cluster_properties.conf).

A properties file must be located on the local machine (we recommend using a bootstrap action to accomplish this), and can be specified by adding the `--properties-file` option to `spark-submit` in the step specification.

## Testing

*See [TESTPLAN](TESTPLAN.md)*

## Warnings and errors

### List of error codes

SafeTab-P is distributed as Python source code plus associated scripts and resource files. The errors that occur during the SafeTab-P command line tool usage are human readable Python exceptions.

### Known Warnings

These warnings can be safely ignored:

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

### Known Errors

SafeTab-P occasionally produces a `ValueError: I/O operation on closed file` about attempting to write to the logs file after all logs are written. This error can be safely ignored, as all log records should still be written.

SafeTab-P has been tested on and can be used successfully with EMR 6.2, 6.8, and 6.10. However, Tumult Core 0.6.0 does not officially support PySpark >= 3.1.0. Some Core tests and utility functions may fail when run on EMR 6.8 or 6.10. SafeTab-P does not use these utility functions, so the program can complete on these EMR cluster types.
