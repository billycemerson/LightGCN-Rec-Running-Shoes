import pandas as pd
import os

# Paths to your CSV files (relative to this script)
daily_path = os.path.join("..", "data", "transform", "daily_run_clean.csv")
recovery_path = os.path.join("..", "data", "transform", "recovery_run_clean.csv")
speed_path = os.path.join("..", "data", "transform", "speed_run_clean.csv")

# Output directories
transform_dir = os.path.join("..", "data", "transform")
load_dir = os.path.join("..", "data", "load")
os.makedirs(load_dir, exist_ok=True)

def preprocess(path):
    df = pd.read_csv(path)

    # Keep only required columns
    df = df[['user_id', 'product_id', 'rating', 'date']].copy()

    # Convert 'date' to datetime and extract the year as timestamp
    df['timestamp'] = pd.to_datetime(df['date'], errors='coerce').dt.year

    # Drop the original date column
    # df.drop(columns='date', inplace=True)

    # Keep only ratings >= 3
    df = df[df['rating'] >= 3]

    return df

def load_data():
    df_daily = preprocess(daily_path)
    df_recovery = preprocess(recovery_path)
    df_speed = preprocess(speed_path)

    # Combine all into one DataFrame
    full_review = pd.concat([df_daily, df_recovery, df_speed], ignore_index=True)

    return full_review

if __name__ == "__main__":
    full_review = load_data()
    print(f"Total records: {len(full_review)}")

    # Save to load directory
    output_path = os.path.join(load_dir, "full_review.csv")
    full_review.to_csv(output_path, index=False)