#! /bin/sh

LOG_FILE="log.tmp"

./spark/bin/spark-submit --class "App" target/scala-2.11/machine-learning-assembly-1.0.jar > ${LOG_FILE}