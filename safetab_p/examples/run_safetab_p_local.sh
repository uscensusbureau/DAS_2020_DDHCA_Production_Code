#!/usr/bin/env bash

# Run SafeTab-P on local mode.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $SCRIPT_DIR/../tmlt/safetab_p/

# Note: Ensure java 8 is installed and JAVA_HOME environment variable is set.

# Local mode: the spark master & the worker are all running inside the client application JVM.
# SparkUI: http://localhost:4040/

spark-submit \
        --properties-file resources/spark_configs/spark_local_properties.conf \
        safetab-p.py execute \
        resources/toy_dataset/input_dir_puredp resources/toy_dataset example_output/safetab_p \
        --log example_output/safetab_p/sample.log --validate-private-output
