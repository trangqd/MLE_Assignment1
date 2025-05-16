from __future__ import annotations
from pathlib import Path
from typing import Dict
from datetime import datetime

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import IntegerType, FloatType, DateType

BRONZE_ROOT = Path("data_mart/bronze")
SILVER_ROOT = Path("data_mart/silver")
PARTITION_COL = "snapshot_date"

Y_RE = r"(\d+)\s*Years?"
M_RE = r"(\d+)\s*Months?"

def credit_age_to_months(col_name: str = "Credit_History_Age") -> F.Column:
    yrs = F.regexp_extract(F.col(col_name), Y_RE, 1).cast(IntegerType())
    mth = F.regexp_extract(F.col(col_name), M_RE, 1).cast(IntegerType())
    return (yrs * 12 + mth).cast(IntegerType())


# 1. LMS Loan Daily
def augment_lms(df):
    return (
        df.withColumn("mob", F.col("installment_num").cast(IntegerType()))
          .withColumn("installments_missed",
              F.ceil(F.col("overdue_amt") / F.col("due_amt")).cast(IntegerType()))
          .fillna(0, subset=["installments_missed"])
          .withColumn("first_missed_date",
              F.when(F.col("installments_missed") > 0,
                     F.add_months(F.col("snapshot_date"), -1 * F.col("installments_missed")))
               .cast(DateType()))
          .withColumn("dpd",
              F.when(F.col("overdue_amt") > 0,
                     F.datediff(F.col("snapshot_date"), F.col("first_missed_date")))
               .otherwise(0).cast(IntegerType()))
          .withColumn("remaining_term",  F.col("tenure") - F.col("installment_num"))
          .withColumn("days_since_origination",
              F.datediff(F.col("snapshot_date"), F.col("loan_start_date")))
    )


# 2. Click-stream
def augment_clickstream(df):
    feat_cols = [f"fe_{i}" for i in range(1, 21)]
    for c in feat_cols:
        df = df.withColumn(c, F.regexp_replace(c, r"_$", "").cast(FloatType()))
    return (
        df.withColumn("clickstream_mean",
            sum(F.col(c) for c in feat_cols) / len(feat_cols))
    )


# 3. Customer Attributes
def augment_attributes(df):
    df = (df
          .withColumn("Age", F.regexp_replace("Age", r"_$", "").cast(IntegerType()))
          .withColumn("Occupation",
              F.when(F.col("Occupation").isin("_______", "", None), "NA")
               .otherwise(F.col("Occupation"))))
    df = df.withColumn("Age",
            F.when((F.col("Age") < 0) | (F.col("Age") > 100), None)
             .otherwise(F.col("Age")))
    return (
        df.withColumn("age_band",
              F.when(F.col("Age") < 25, "18-24")
               .when(F.col("Age") < 35, "25-34")
               .when(F.col("Age") < 45, "35-44")
               .when(F.col("Age") < 55, "45-54")
               .otherwise("55+"))
          .withColumn("has_valid_ssn",
              F.when(F.col("SSN").rlike(r"^\d{3}-\d{2}-\d{4}$"), 1).otherwise(0))
    )

# 4. Financials
def augment_financials(df):
    # 1) Format fixes
    df = (df
        .withColumn("Annual_Income",
            F.regexp_replace("Annual_Income", r"_$", "").cast(FloatType()))
        .withColumn("Outstanding_Debt",
            F.regexp_replace("Outstanding_Debt", r"_$", "").cast(FloatType()))
        .withColumn("Amount_invested_monthly",
            F.regexp_replace("Amount_invested_monthly", r"^_+|_+$", "").cast(FloatType()))
        .withColumn("Changed_Credit_Limit",
            F.regexp_replace("Changed_Credit_Limit", r"_$", "").cast(FloatType()))
        .withColumn("Credit_Mix",
            F.when(F.col("Credit_Mix").contains("_"), None)
             .otherwise(F.col("Credit_Mix")))
        .withColumn("Num_of_Loan",
            F.regexp_replace("Num_of_Loan", r"_$", "").cast(IntegerType()))
        .withColumn("Num_of_Delayed_Payment",
            F.regexp_replace("Num_of_Delayed_Payment", r"_$", "").cast(IntegerType()))
        .withColumn("Payment_Behaviour",
            F.when(F.col("Payment_Behaviour").rlike("^[A-Za-z0-9_ ,]+$"), F.col("Payment_Behaviour"))
             .otherwise(None))
    )
    # 2) Non-sense values handling 
    df = (df
        .withColumn("Num_of_Loan",
            F.when(F.col("Num_of_Loan") < 0, None).otherwise(F.col("Num_of_Loan")))
        .withColumn("Num_Credit_Card",
            F.when(F.col("Num_Credit_Card") < 0, None)
             .otherwise(F.col("Num_Credit_Card").cast(IntegerType())))
        .withColumn("Num_Bank_Accounts",
            F.when(F.col("Num_Bank_Accounts") < 0, None)
             .otherwise(F.col("Num_Bank_Accounts").cast(IntegerType()))))
    # 3) Feature aug 
    return (df
        .withColumn("is_min_pay_only",
            F.when(F.col("Payment_of_Min_Amount") == "Yes", 1)
             .when(F.col("Payment_of_Min_Amount") == "No", 0)
             .otherwise(None))
        .withColumn("min_pay_info_missing",
            F.when(F.col("Payment_of_Min_Amount").isin("NM", "", None), 1).otherwise(0))
        .withColumn("credit_history_months", credit_age_to_months("Credit_History_Age"))
        .withColumn("credit_history_years",
            (credit_age_to_months("Credit_History_Age") / 12.0).cast(FloatType()))
        .withColumn("dti",
            F.when(F.col("Annual_Income") > 0,
                   F.col("Outstanding_Debt") / F.col("Annual_Income"))
             .otherwise(None).cast(FloatType()))
        .withColumn("card_num_suspicious",
            F.when(F.col("Num_Credit_Card") > 20, 1).otherwise(0))
        .withColumn("bank_num_suspicious",
            F.when(F.col("Num_Bank_Accounts") > 20, 1).otherwise(0))
    )


TABLES: Dict[str, callable] = {
    "lms_loan_daily":       augment_lms,
    "feature_clickstream":  augment_clickstream,
    "features_attributes":  augment_attributes,
    "features_financials":  augment_financials,
}

def process_one(table: str, snapshot: str, spark: SparkSession):
    src = BRONZE_ROOT / table / f"bronze_{table}_{snapshot.replace('-', '_')}.csv"
    if not src.exists():
        print(f"[SKIP] {src} not found.")
        return
    sdf = spark.read.option("header", "true").csv(str(src))
    sdf = TABLES[table](sdf)
    out = SILVER_ROOT / table / f"silver_{table}_{snapshot.replace('-', '_')}.parquet"
    sdf.write.mode("overwrite").parquet(str(out))
    print(f"[WRITE] {table} → {out}")

def process_silver_all(snapshot: str, spark: SparkSession):
    for tbl in TABLES:
        process_one(tbl, snapshot, spark)
