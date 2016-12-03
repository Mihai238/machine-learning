#! /bin/sh

if [ ! -d "spark" ];
then
    FILE_NAME=spark-2.0.2-bin-hadoop2.7
    URL=http://d3kbcqa49mib13.cloudfront.net/spark-2.0.2-bin-hadoop2.7.tgz
    curl ${URL} | tar xz
    mv ${FILE_NAME} spark
fi

sbt package

./run.sh