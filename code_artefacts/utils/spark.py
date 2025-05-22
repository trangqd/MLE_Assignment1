from pyspark.sql import SparkSession


def get_spark(app_name: str = "loan-features-pipeline") -> SparkSession:
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")
    return spark
