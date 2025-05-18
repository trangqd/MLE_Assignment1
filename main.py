from datetime import datetime
from pathlib import Path

from utils.bronze_processor import generate_first_of_month_dates, process_bronze
from utils.spark import get_spark
from utils.silver_processor import process_silver_all
from utils.label_processor import process_labels_gold_table

from utils.gold_processor import (
    load_merged_silver_table,
    engineer_features as engineer_features,
    apply_capping_and_flags,
    engineer_lag_rolling_features,
    split_train_test,
    GOLD_OUTPUT_DIR
)


def process_bronze_tables(spark):
    df_dates = (
        spark.read.option("header", "true")
             .csv("data/raw/lms_loan_daily.csv")
             .selectExpr("min(snapshot_date) as min_dt", "max(snapshot_date) as max_dt")
             .collect()[0]
    )
    start_date_str = datetime.strptime(df_dates["min_dt"], "%Y-%m-%d").strftime("%Y-%m-01")
    end_date_str   = datetime.strptime(df_dates["max_dt"], "%Y-%m-%d").replace(day=1).strftime("%Y-%m-%d")
    dates = generate_first_of_month_dates(start_date_str, end_date_str)

    print(f"[BRONZE] Back-filling Bronze for {len(dates)} months: {dates[0]} → {dates[-1]}")
    process_bronze(dates, spark)

def process_silver_tables(spark):
    df_dates = (
        spark.read.option("header", "true")
             .csv("data/raw/lms_loan_daily.csv")
             .selectExpr("min(snapshot_date) as min_dt", "max(snapshot_date) as max_dt")
             .collect()[0]
    )
    start = datetime.strptime(df_dates["min_dt"], "%Y-%m-%d").strftime("%Y-%m-01")
    end   = datetime.strptime(df_dates["max_dt"], "%Y-%m-%d").replace(day=1).strftime("%Y-%m-%d")

    for ds in generate_first_of_month_dates(start, end):
        print(f"[SILVER] Processing {ds}")
        process_silver_all(ds, spark)

def process_gold_labels(spark):
    SILVER_LMS_DIR = Path("data_mart/silver/lms_loan_daily")
    dpd_thresh = 30
    mob_filter = 6

    silver_files = list(SILVER_LMS_DIR.glob("silver_lms_loan_daily_*.csv"))
    if not silver_files:
        print("[ERROR] No silver LMS files found.")
        return

    snapshot_dates = sorted([
        f.name.replace("silver_lms_loan_daily_", "").replace(".csv", "").replace("_", "-")
        for f in silver_files
    ])
    print(f"[LABEL] Found {len(snapshot_dates)} snapshot dates")

    for ds in snapshot_dates:
        print(f"[LABEL] Processing {ds}")
        process_labels_gold_table(ds, spark, dpd=dpd_thresh, mob=mob_filter)

def process_gold_feature_files():
    lms = load_merged_silver_table("lms_loan_daily")
    clk = load_merged_silver_table("feature_clickstream")
    fin = load_merged_silver_table("features_financials")
    attr = load_merged_silver_table("features_attributes")

    fin = fin.drop(columns=["snapshot_date"], errors="ignore")
    attr = attr.drop(columns=["snapshot_date"], errors="ignore")

    df = lms.merge(clk, on=["Customer_ID", "snapshot_date"], how="left")
    df = df.merge(fin, on="Customer_ID", how="left")
    df = df.merge(attr, on="Customer_ID", how="left")

    df = engineer_features(df)
    df = apply_capping_and_flags(df)
    df = engineer_lag_rolling_features(df)

    drop_cols = ["Credit_Mix", "Payment_Behaviour", "age_band", "Occupation", "SSN", "Name", "Type_of_Loan", "Payment_of_Min_Amount", "Credit_History_Age", "credit_history_months"]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    train, test = split_train_test(df)
    train.to_csv(GOLD_OUTPUT_DIR / "gold_train.csv", index=False)
    test.to_csv(GOLD_OUTPUT_DIR / "gold_test.csv", index=False)

    print("[✓] Feature engineering completed. Train/test CSVs saved.")

if __name__ == "__main__":
    # Bronze
    spark = get_spark("bronze-layer")
    process_bronze_tables(spark)
    spark.stop()

    # Silver
    spark = get_spark("silver-layer")
    process_silver_tables(spark)
    spark.stop()

    # Label store
    spark = get_spark("label-store")
    process_gold_labels(spark)
    spark.stop()

    # Feature store
    spark = get_spark("feature-store")
    process_gold_feature_files()
    spark.stop()