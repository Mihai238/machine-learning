#! /bin/sh

LOG_FILE="log.tmp"

./spark/bin/spark-submit --class "App" target/scala-2.11/machine-learning_2.11-1.0.jar > ${LOG_FILE}