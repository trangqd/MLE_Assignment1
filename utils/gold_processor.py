from pathlib import Path
import pandas as pd
import numpy as np

SILVER_ROOT = Path("data_mart/silver")
GOLD_OUTPUT_DIR = Path("data_mart/gold/feature_store")
GOLD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLS_WITH_OUTLIERS = [
    "Num_Bank_Accounts", "Num_Credit_Card", "Num_of_Loan", "Num_of_Delayed_Payment"
]

def engineer_features(df):
    credit_mix_map = {"Bad": 0, "Standard": 1, "Good": 2}
    payment_map = {
        "Low_spent_Small_value_payments": 0,
        "Low_spent_Medium_value_payments": 1,
        "Low_spent_Large_value_payments": 2,
        "High_spent_Small_value_payments": 3,
        "High_spent_Medium_value_payments": 4,
        "High_spent_Large_value_payments": 5,
    }
    age_band_map = {
        "18-24": 0, "25-34": 1, "35-44": 2, "45-54": 3, "55+": 4
    }

    df["Balance_to_EMI_Ratio"] = df["Monthly_Balance"] / df["Total_EMI_per_month"].replace(0, np.nan)
    df["Investment_to_Income_Ratio"] = df["Amount_invested_monthly"] / df["Monthly_Inhand_Salary"].replace(0, np.nan)
    df["Balance_to_Outstanding_Debt_Ratio"] = df["Monthly_Balance"] / df["Outstanding_Debt"].replace(0, np.nan)
    df["Credit_Mix_Encoded"] = df["Credit_Mix"].map(credit_mix_map)
    df["Payment_Behaviour_Encoded"] = df["Payment_Behaviour"].map(payment_map)
    df["age_band_encoded"] = df["age_band"].map(age_band_map)

    return df

def apply_capping_and_flags(df):
    for col in COLS_WITH_OUTLIERS:
        if col in df.columns:
            cap = df[col].quantile(0.995)
            df[f"{col}_capped"] = df[col].clip(upper=cap)
            df[f"{col}_outlier_flag"] = (df[col] > cap).astype(int)
    return df

def engineer_lag_rolling_features(df):
    df = df.sort_values(["Customer_ID", "snapshot_date"])
    df["payment_ratio"] = df["paid_amt"] / df["due_amt"].replace(0, np.nan)
    df["shortfall"] = (df["due_amt"] - df["paid_amt"]).clip(lower=0)
    df["missed_payment"] = (df["paid_amt"] < df["due_amt"]).astype(int)
    df["full_payment"] = (df["paid_amt"] >= df["due_amt"]).astype(int)
    df["overpayment"] = (df["paid_amt"] - df["due_amt"]).clip(lower=0)

    df_group = df.groupby("Customer_ID")

    df["rolling_avg_payment_ratio_3m"] = (
        df_group["payment_ratio"].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    df["rolling_sum_shortfall_3m"] = (
        df_group["shortfall"].rolling(3, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    df["rolling_max_dpd_3m"] = (
        df_group["dpd"].rolling(3, min_periods=1).max().reset_index(level=0, drop=True)
    )

    # Consecutive missed payments
    def compute_consecutive_missed(x):
        streak = 0
        result = []
        for m in x:
            if m:
                streak += 1
            else:
                streak = 0
            result.append(streak)
        return result

    df["consecutive_missed_payments"] = (
        df_group["missed_payment"].transform(compute_consecutive_missed)
    )

    return df

def split_train_test(df, test_start="2025-01-01"):
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    train = df[df["snapshot_date"] < test_start].copy()
    test = df[df["snapshot_date"] >= test_start].copy()
    return train, test

def load_merged_silver_table(table_name):
    table_path = SILVER_ROOT / table_name
    csv_files = list(table_path.glob("silver_*.csv"))
    df_list = []

    for file in csv_files:
        df = pd.read_csv(file)
        df["Customer_ID"] = df["Customer_ID"].astype(str).str.strip().str.lower()
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
        df_list.append(df)

    df_all = pd.concat(df_list, ignore_index=True)
    df_all = df_all.sort_values(["Customer_ID", "snapshot_date"])
    return df_all
