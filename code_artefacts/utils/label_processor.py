from datetime import datetime
from pathlib import Path

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import IntegerType, StringType

SILVER_ROOT = Path("data_mart/silver")
GOLD_ROOT = Path("data_mart/gold")
LABEL_STORE_DIR = GOLD_ROOT / "label_store"

def process_labels_gold_table(snapshot_date_str: str, spark: SparkSession, dpd: int = 30, mob: int = 6):
    partition_name = f"silver_lms_loan_daily_{snapshot_date_str.replace('-', '_')}.csv"
    silver_file = SILVER_ROOT / "lms_loan_daily" / partition_name

    if not silver_file.exists():
        print(f"[SKIP] {silver_file} does not exist.")
        return None

    df = spark.read.option("header", "true").csv(str(silver_file))
    print(f"[LOAD] snapshot {snapshot_date_str} — {df.count()} rows from {silver_file}")

    df = df.withColumn("mob", F.col("mob").cast(IntegerType()))
    df = df.withColumn("dpd", F.col("dpd").cast(IntegerType()))

    df = df.filter(F.col("mob") == mob)
    df = df.withColumn("label", F.when(F.col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(f"{dpd}dpd_{mob}mob").cast(StringType()))
    df = df.withColumn("snapshot_date", F.lit(snapshot_date_str))

    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # Write label to gold output
    LABEL_STORE_DIR.mkdir(parents=True, exist_ok=True)
    out_file = LABEL_STORE_DIR / f"gold_label_store_{snapshot_date_str.replace('-', '_')}.csv"
    df.toPandas().to_csv(out_file, index=False)

    print(f"[SAVE] {out_file} ({df.count()} rows)")
    return df
