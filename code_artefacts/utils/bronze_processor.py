from __future__ import annotations
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable
from pyspark.sql import SparkSession, functions as F

RAW_DIR = Path("data/raw")
BRONZE_ROOT = Path("data_mart/bronze")

CSV_SOURCES = {
    "feature_clickstream":   "feature_clickstream.csv",
    "features_attributes":   "features_attributes.csv",
    "features_financials":   "features_financials.csv",
    "lms_loan_daily":        "lms_loan_daily.csv",
}

PARTITION_COL = "snapshot_date"

def generate_first_of_month_dates(start_str: str, end_str: str) -> list[str]:
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end   = datetime.strptime(end_str,   "%Y-%m-%d")
    dates = []
    cur = datetime(start.year, start.month, 1)
    while cur <= end:
        dates.append(cur.strftime("%Y-%m-%d"))
        cur = datetime(cur.year + (cur.month // 12), (cur.month % 12) + 1, 1)
    return dates

def process_bronze(dates_str_lst: Iterable[str], spark: SparkSession) -> None:
    for table, raw_csv in CSV_SOURCES.items():
        src_path = RAW_DIR / raw_csv
        bronze_dir = BRONZE_ROOT / table
        bronze_dir.mkdir(parents=True, exist_ok=True)

        sdf_full = (
            spark.read.option("header", "true").option("inferSchema", "true")
                 .csv(str(src_path))
        )

        for ds in dates_str_lst:
            snap_dt = datetime.strptime(ds, "%Y-%m-%d")
            slice_df = sdf_full.filter(F.col(PARTITION_COL) == snap_dt)

            if not slice_df.head(1):
                print(f"[{table}] {ds} – 0 rows (skip)")
                continue

            fname = f"bronze_{table}_{ds.replace('-', '_')}.csv"
            out_path = bronze_dir / fname

            slice_df.toPandas().to_csv(out_path, index=False)
            print(f"[{table}] {ds}  → wrote {out_path}")
