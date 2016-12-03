import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}

object App {

  def main(args: Array[String]): Unit = {
    val filePath = "/home/mihai/Development/machine-learning/src/main/resources/communities.data"

    val conf = new SparkConf().setAppName("Simple Application")
    val sc = new SparkContext(conf)

    val spark = SparkSession
      .builder()
      .appName("Spark SQL basic example")
      .getOrCreate()

    val originalDataFrame = spark
      .read
      .format("csv")
      .load(filePath)

    val dataFrame = spark.createDataFrame(splitFeaturesAndLabel(originalDataFrame))
      .toDF("label", "features")

    // Building the model
    val lr = new LinearRegression()
    val lrModel = lr.fit(dataFrame)

    println(s"Coefficients: ${lrModel.coefficients}")
    println(s"Intercept: ${lrModel.intercept}")
    val trainingSummary = lrModel.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
    trainingSummary.residuals.show()
    println(s"MSE: ${trainingSummary.meanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")

    sc.stop()
  }

  def splitFeaturesAndLabel(dataFrame: DataFrame): Array[(Double, Vector)] = {
    dataFrame
      .collect()
      .drop(1)
      .map(line => {
        val map = line.toSeq.drop(5).dropRight(1).map(_.toString).map(toDouble).toArray
        val label = line.toSeq.last
        (label.toString.toDouble, Vectors.dense(map))
      })
  }

  def toDouble(input: String): Double = input match {
    case "NA" => 0
    case x => x.toDouble
  }

}
