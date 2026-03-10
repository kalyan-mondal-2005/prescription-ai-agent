import pandas as pd
import os

# Paths
RAW_DATA_PATH = "data/raw/medicine_dataset_1mb.csv"
PROCESSED_DATA_PATH = "data/processed/clean_data.csv"


def load_data(path):
    try:
        df = pd.read_csv(path)
        print(" Raw data loaded successfully")
        return df
    except Exception as e:
        print(" Error loading data:", e)
        return None


def clean_data(df):
    
    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove rows with missing medicine names
    df = df.dropna(subset=["Medicine_Name"])

    # Fill missing values
    df["Strength"] = df["Strength"].fillna("Unknown")
    df["Use_Case"] = df["Use_Case"].fillna("General")
    df["Alternative"] = df["Alternative"].fillna("None")
    df["Stock"] = df["Stock"].fillna("Unknown")
    df["Dosage_Instruction"] = df["Dosage_Instruction"].fillna("Consult doctor")

    # Standardize text
    df["Medicine_Name"] = df["Medicine_Name"].str.strip().str.title()
    df["Use_Case"] = df["Use_Case"].str.strip().str.title()

    # Convert stock to uppercase
    df["Stock"] = df["Stock"].str.upper()

    print(" Data cleaning completed")

    return df


def save_data(df, path):
    """Save cleaned data"""

    os.makedirs(os.path.dirname(path), exist_ok=True)

    df.to_csv(path, index=False)

    print(f" Clean data saved to: {path}")


def main():
    df = load_data(RAW_DATA_PATH)

    if df is not None:
        clean_df = clean_data(df)
        save_data(clean_df, PROCESSED_DATA_PATH)


if __name__ == "__main__":
    main()