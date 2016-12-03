name := "machine-learning"

version := "1.0"

scalaVersion := "2.11.8"

libraryDependencies += "org.apache.spark" % "spark-core_2.11" % "2.0.2" % "provided"
libraryDependencies += "org.apache.spark" % "spark-mllib_2.11" % "2.0.2" % "provided"
libraryDependencies += "org.apache.spark" % "spark-sql_2.11" % "2.0.2" % "provided"
libraryDependencies += "com.quantifind" %% "wisp" % "0.0.4"
