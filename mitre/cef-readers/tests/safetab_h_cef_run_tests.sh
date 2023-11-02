#!/bin/bash

# zip the code
ZIPFILE=das_phsafe_h_cef_reader.zip
export ZIPFILE

zip -r -q $ZIPFILE . -i '*.sh' '*.ini'
echo $ZIPFILE

# setup the venv
VENV=${VENV:-"~/virtualenv"}
VENV_TARBALL=${VENV_TARBALL:-"${VENV}.tgz"}
VENV_NAME=${VENV_NAME:-"virtualenv"}

# run the spark submit
spark_submit="spark-submit
	--conf spark.local.dir=/mnt/tmp/
  --conf spark.eventLog.enabled=true
  --conf spark.eventLog.dir=/mnt/tmp/logs/
  --conf spark.submit.deployMode=client
	--conf spark.network.timeout=3000
	--conf spark.driver.extraJavaOptions=-Dlog4j.configuration=file:log4j.configuration.xml
	--conf spark.executor.extraJavaOptions=-Dlog4j.configuration=file:log4j.configuration.xml
	--conf spark.driver.extraJavaOptions=-Dlog4j.configuration=file:log4j.properties
	--conf spark.executor.extraJavaOptions=-Dlog4j.configuration=file:log4j.properties
	--files src/phsafe_safetab_reader/log4j.configuration.xml
	--files src/phsafe_safetab_reader/log4j.properties
	--py-files $ZIPFILE
	--master yarn
  --conf spark.archives=${VENV_TARBALL}#${VENV_NAME}
  --conf spark.pyspark.driver.python=${VENV}/bin/python3
  --conf spark.pyspark.python=./virtualenv/bin/python3
	--name $JBID:convert_measurements"
python_cmd="tests/safetab_h_cef_tests.py"
echo
echo
echo $spark_submit $python_cmd
echo
extra_java_options="-XX:+PrintGCTimeStamps -XX:+PrintGCDetails -verbose:gc"

# Send the code to spark submit and call convert_pickled.python
$spark_submit --conf "spark.executor.extraJavaOptions=$extra_java_options" $python_cmd
