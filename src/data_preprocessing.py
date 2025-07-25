from pathlib import Path
import pandas as pd
import os

class TrainTestGenerator:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)

    def prepare_data(self):
        # Load your CSV review dataset
        df = pd.read_csv(self.data_path)

        # Ensure timestamp is in integer year format (e.g. 2024, etc.)
        df['timestamp'] = df['timestamp'].astype(int)
        
        # Drop date column
        if 'date' in df.columns:
            df = df.drop(columns=['date'])

        # Optional: filter data if needed
        df = df[df['timestamp'] >= 2020]  # Example filter
        df = df.reset_index(drop=True)

        return df

    def forward_chaining(self, start_year=2024, end_year=2025):
        data = self.prepare_data()

        for test_year in range(start_year, end_year + 1):
            train = data[data["timestamp"] < test_year]
            test = data[data["timestamp"] == test_year]

            yield test_year, train, test