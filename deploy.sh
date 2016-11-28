#! /bin/sh

sbt package
cp -f target/scala-2.11/machine-learning_2.11-1.0.jar ~/apps/spark-2.0.2-bin-hadoop2.7
