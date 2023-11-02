#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $SCRIPT_DIR/../tmlt/safetab_p/

spark-submit \
        --properties-file resources/spark_configs/spark_local_properties.conf \
        target_counts_p.py \
        --input resources/toy_dataset/input_dir_puredp \
        --reader resources/toy_dataset \
        --output example_output/target_counts_p
