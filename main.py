from datetime import datetime
from utils.bronze_processor import generate_first_of_month_dates, process_bronze
from utils.spark import get_spark
from utils.silver_processor import process_silver_all

if __name__ == "__main__":
    spark = get_spark("bronze-layer")

    df_dates = (
        spark.read.option("header", "true")
             .csv("data/raw/lms_loan_daily.csv")
             .selectExpr("min(snapshot_date) as min_dt", "max(snapshot_date) as max_dt")
             .collect()[0]
    )

    start_date_str = datetime.strptime(df_dates["min_dt"], "%Y-%m-%d").strftime("%Y-%m-01")
    end_date_str   = datetime.strptime(df_dates["max_dt"], "%Y-%m-%d").replace(day=1).strftime("%Y-%m-%d")

    dates = generate_first_of_month_dates(start_date_str, end_date_str)
    print(f"Back-filling Bronze for {len(dates)} months: {dates[0]} → {dates[-1]}")

    process_bronze(dates, spark)
    spark.stop()

if __name__ == "__main__":
    spark = get_spark("silver-layer")

    df_dates = (spark.read.option("header", "true")
                       .csv("data/raw/lms_loan_daily.csv")
                       .selectExpr("min(snapshot_date) as min_dt",
                                   "max(snapshot_date) as max_dt")
                       .collect()[0])

    start = datetime.strptime(df_dates["min_dt"], "%Y-%m-%d").strftime("%Y-%m-01")
    end   = datetime.strptime(df_dates["max_dt"], "%Y-%m-%d").replace(day=1).strftime("%Y-%m-%d")

    for ds in generate_first_of_month_dates(start, end):
        process_silver_all(ds, spark)

    spark.stop()