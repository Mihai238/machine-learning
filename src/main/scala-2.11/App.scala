import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}

object App {

  def main(args: Array[String]): Unit = {
    val filePath = "/home/mihai/Development/machine-learning/src/main/resources/communities.data"

    implicit val spark = SparkSession
      .builder()
      .appName("Spark SQL basic example")
      .getOrCreate()
    import spark.implicits._

    val data: DataFrame = parseData(filePath)
    val Array(trainingData, testData) = data.randomSplit(Array(0.9, 0.1))

    val list = trainingData.collectAsList()
    import com.quantifind.charts.Highcharts._
    scatter((0 until list.size()).map(i => list.get(i).getAs[Double]("label")))

    // Building the model
    val algorithm = new LinearRegression()
    val model = algorithm.fit(trainingData)

    val predictions = model.transform(testData)

    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println("Root Mean Squared Error (RMSE) on test data = " + rmse)

    val max = predictions.map { row =>
      row.getAs[Double]("label") - row.getAs[Double]("prediction")
    }.rdd.max()
    println(s"Max prediction deviation: $max")

    val trainingSummary = model.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"MSE (Mean Squared Error) on model: ${trainingSummary.meanSquaredError}")
    println(s"R-squared: ${trainingSummary.r2}")

    println("Correlation: " + computeCorrelation(testData, predictions))

    plotPredictions(predictions)
  }

  private def plotPredictions(predictions: DataFrame) = {
    val rows = predictions.collect()
    val size = predictions.count().toInt
    val x = (0 until size).map(_.toDouble)
    val labels = (0 until size).map(j => rows(j).getAs[Double]("label"))
    val preds = (0 until size).map(j => rows(j).getAs[Double]("prediction"))

    import com.quantifind.charts.Highcharts._
    scatter((0 until size).map(i => labels(i)))
    hold()
    scatter((0 until size).map(i => preds(i)))
    legend(Seq("Actual", "Predicted"))
    xAxis("Index")
    yAxis("Violent crimes per 100K population")

  }

  private def parseData(filePath: String)(implicit spark: SparkSession) = {
    val originalDataFrame = spark
      .read
      .format("csv")
      .load(filePath)

    val dataFrame = spark.createDataFrame(splitFeaturesAndLabel(originalDataFrame))
      .toDF("label", "features")
    dataFrame
  }

  def splitFeaturesAndLabel(dataFrame: DataFrame): Array[(Double, Vector)] = {
    dataFrame
      .collect()
      .drop(1)
      .map(line => {
        val features = line.toSeq.drop(5).dropRight(1).map(_.toString).map(toDouble).toArray
        val label = line.toSeq.last
        (label.toString.toDouble, Vectors.dense(features))
      })
  }

  def computeCorrelation(data1: DataFrame, data2: DataFrame): Double = Statistics.corr(getLabelAsRddDouble(data1), getLabelAsRddDouble(data2), "spearman")

  def getLabelAsRddDouble(dataFrame: DataFrame): RDD[Double] = dataFrame.select("label").rdd.map(_.getAs[Double](0))

  def toDouble(input: String): Double = input match {
    case "NA" => 0
    case x => x.toDouble
  }

}
