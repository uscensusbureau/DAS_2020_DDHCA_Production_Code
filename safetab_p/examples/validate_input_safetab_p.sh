#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $SCRIPT_DIR/../tmlt/safetab_p/

spark-submit \
        --properties-file resources/spark_configs/spark_local_properties.conf \
        safetab-p.py validate resources/toy_dataset/input_dir_puredp resources/toy_dataset
