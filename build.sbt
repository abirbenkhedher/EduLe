name := "analyse_entiment"

version := "0.1"

scalaVersion := "2.12.7"

val sparkVersion = "2.4.0"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "log4j" % "log4j" % "1.2.17"
)
