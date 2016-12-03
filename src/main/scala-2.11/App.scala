import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.{Vector, Vectors}

object App {

  def main(args: Array[String]): Unit = {
    val filePath = "/home/mihai/Development/machine-learning/src/main/resources/communities.data"

    val conf = new SparkConf().setAppName("Simple Application")
    val sc = new SparkContext(conf)

    val spark = SparkSession
      .builder()
      .appName("Spark SQL basic example")
      .getOrCreate()

    val training = spark
      .read
      .format("csv")
      .load(filePath)

    val x = training
      .collect()
      .drop(1)
      .map(line => {
        val map = line.toSeq.drop(5).dropRight(1).map(_.toString).map(toDouble).toArray
        val label = line.toSeq.last
        (label.toString.toDouble, Vectors.dense(map))
      })

    val y = spark.createDataFrame(x)
      .toDF("label", "features")

    // Building the model
    val lr = new LinearRegression()

    val lrModel = lr.fit(y)

    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")
    val trainingSummary = lrModel.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
    trainingSummary.residuals.show()
    println(s"MSE: ${trainingSummary.meanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")

    sc.stop()
  }

  def toDouble(input: String): Double = input match {
    case "NA" => 0
    case x => x.toDouble
  }

}
