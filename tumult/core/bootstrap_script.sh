#! /bin/bash -xe

if [ $# -eq 0 ]; then
    echo "s3 wheel path must be supplied as an argument to this script"
    exit 2
fi

aws s3 cp $1 .
sudo python3 -m pip install --no-deps tmlt_core-0.6.0-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
