import java.util.Locale

// Importing required Spark libraries
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.udf
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.types.IntegerType

object SentimentTrainer {
  def main(args: Array[String]) {

    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)

    val spark = SparkSession
      .builder()
      .appName("Spark Sentiment")
      .config("spark.master", "local")
      .getOrCreate()

    val twitterTrainPath = "twitter_data/train.csv"
    // Defining paths for training data

    // Printing the path
    println("Reading Twitter data from: " + twitterTrainPath)

    // Reading and preprocessing Twitter data
    val twitterData = readTwitterData(twitterTrainPath, spark)

    val tokenizer = new RegexTokenizer()
      .setInputCol("Preprocessed")
      .setOutputCol("Tokenized All")
      .setPattern("\\s+")

    val wordTokenizer = new RegexTokenizer()
      .setInputCol("Preprocessed")
      .setOutputCol("Tokenized Words")
      .setPattern("\\W")

    Locale.setDefault(Locale.ENGLISH)

    val stopW = new StopWordsRemover()
      .setInputCol("Tokenized Words")
      .setOutputCol("Stopped")

    val ngram = new NGram()
      .setN(2)
      .setInputCol("Stopped")
      .setOutputCol("Grams")

    val tokenVectorizer = new CountVectorizer()
      .setInputCol("Tokenized All")
      .setOutputCol("Token Vector")

    val gramVectorizer = new CountVectorizer()
      .setInputCol("Grams")
      .setOutputCol("Gram Vector")

    val assembler = new VectorAssembler()
      .setInputCols(Array("Token Vector"))
      .setOutputCol("features")

    val model = new LogisticRegression()
      .setFamily("multinomial")
      .setLabelCol("Sentiment")

    val pipe = new Pipeline()
      .setStages(Array(tokenizer,
        wordTokenizer, stopW,
        ngram, tokenVectorizer,
        gramVectorizer,
        assembler, model))

    val paramMap = new ParamMap()
      .put(tokenVectorizer.vocabSize, 10000)
      .put(gramVectorizer.vocabSize, 10000)
      .put(model.elasticNetParam, .8)
      .put(model.tol, 1e-20)
      .put(model.maxIter, 100)

    // Fit pipeline with parameters
    val lr = pipe.fit(twitterData, paramMap)

    // Transform data with trained pipeline
    val tr = lr.transform(twitterData).select("Sentiment", "probability", "prediction")

    // Print the first 10 rows with sentiment label
    tr.take(10).foreach(row => {
      val sentimentLabel = if (row.getAs[Double]("prediction") == 1.0) "positive" else "negative"
      println(s"Sentiment: ${row.getAs[Int]("Sentiment")}, Prediction: ${sentimentLabel}, Probability: ${row.getAs[org.apache.spark.ml.linalg.DenseVector]("probability")}")
    })

    // Evaluate model performance using BinaryClassificationEvaluator
    val eval = new BinaryClassificationEvaluator()
      .setLabelCol("Sentiment")
      .setRawPredictionCol("prediction")

    val roc = eval.evaluate(tr)
    println(s"ROC: ${roc}")

    // Print the schema of the transformed dataset
    tr.printSchema()

    // Define a parameter grid for tuning hyperparameters
    val paramGrid = new ParamGridBuilder()
      .addGrid(tokenVectorizer.vocabSize, Array(10000))
      .addGrid(gramVectorizer.vocabSize, Array(10000))
      .addGrid(model.elasticNetParam, Array(.8))
      .addGrid(model.tol, Array(1e-20))
      .addGrid(model.maxIter, Array(100))
      .build()

    // Define cross-validation
    val cv = new CrossValidator()
      .setEstimator(pipe)
      .setEvaluator(new BinaryClassificationEvaluator()
        .setRawPredictionCol("prediction")
        .setLabelCol("Sentiment"))
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)
      .setParallelism(1)

    // Fit cross-validation model
    val cvmodel = cv.fit(twitterData)

    // Transform the dataset using cross-validation model and print results
    cvmodel.transform(twitterData)
      .select("ItemID","Preprocessed", "probability", "prediction")
      .collect().take(10)
      .foreach(row => {
        val sentimentLabel = if (row.getAs[Double]("prediction") == 1.0) "positive" else "negative"
        println(s"${row.getAs[Int]("ItemID")}, ${row.getAs[String]("Preprocessed")}, Probability: ${row.getAs[org.apache.spark.ml.linalg.DenseVector]("probability")}, ${sentimentLabel}")
      })

    // Print average metrics of the cross-validation
    println("\n")
    println("Metrics:\n")
    cvmodel.avgMetrics.foreach(println)
    println("\n\n")

    // Save the cross-validation model
    cvmodel.write.overwrite().save("sentiment-classifier")
  }

  // Function to read and preprocess Twitter data
  def readTwitterData(path: String, spark: SparkSession) = {

    val data = spark.read.format("csv")
      .option("header", "true")
      .load(path)

    // Define UDF for preprocessing text data
    val preprocess: String => String = {
      _.replaceAll("((.))\\1+","$1")
    }
    val preprocessUDF = udf(preprocess)

    // Apply UDF to preprocess "SentimentText" column
    val newCol = preprocessUDF.apply(data("SentimentText"))

    // Cast "Sentiment" column to IntegerType
    val label = data("Sentiment").cast(IntegerType)

    // Create a new DataFrame with "Preprocessed" and "Sentiment" columns
    data.withColumn("Preprocessed", newCol)
      .withColumn("Sentiment",label)
      .select("ItemID","Sentiment","Preprocessed")
  }

}
