import pandas as pd
import os


def import_data(folder_path='../../data/'):
    # Generate file names for the date range
    date_range = pd.date_range(start="2024-10-15", end="2024-10-24")
    file_names = [f'bid_ask_data_BTCUSDT_{date.strftime("%Y%m%d")}.csv' for date in date_range]

    # Initialize an empty list to hold DataFrames
    data_frames = []

    # Loop through the file names
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        try:
            # Read each CSV file
            df = pd.read_csv(file_path)
            data_frames.append(df)
            print(f"Loaded {file_name}")
        except FileNotFoundError:
            print(f"File {file_name} not found. Skipping.")

    # Concatenate all DataFrames into a single DataFrame
    combined_data = pd.concat(data_frames, ignore_index=True)

    return combined_data