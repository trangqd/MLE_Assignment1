from datetime import datetime
from pathlib import Path
import shutil
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import IntegerType, StringType

SILVER_ROOT = Path("data_mart/silver")
GOLD_ROOT   = Path("data_mart/gold")
LABEL_STORE_DIR = GOLD_ROOT / "label_store"

def process_labels_gold_table(snapshot_date_str: str, spark: SparkSession, dpd: int = 30, mob: int = 6):
    silver_dir = SILVER_ROOT / "lms_loan_daily" / f"snapshot_date={snapshot_date_str}"

    if not silver_dir.exists():
        print(f"[SKIP] {silver_dir} does not exist.")
        return None

    df = spark.read.parquet(str(silver_dir))
    print(f"[LOAD] snapshot {snapshot_date_str} — {df.count()} rows from {silver_dir}")

    df = df.withColumn("mob", F.col("mob").cast(IntegerType()))
    df = df.withColumn("dpd", F.col("dpd").cast(IntegerType()))

    df = df.filter(F.col("mob") == mob)
    df = df.withColumn("label", F.when(F.col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(f"{dpd}dpd_{mob}mob").cast(StringType()))
    df = df.withColumn("snapshot_date", F.lit(snapshot_date_str))

    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # Write as Parquet partitioned by snapshot_date
    out_path = LABEL_STORE_DIR / f"snapshot_date={snapshot_date_str}"
    out_path_str = str(out_path)
    
    if out_path.exists():
        shutil.rmtree(out_path)
    
    df.write.mode("overwrite").parquet(out_path_str)

    print(f"[SAVE] {out_path}/snapshot_date={snapshot_date_str} ({df.count()} rows)")
    return df
