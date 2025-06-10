import pandas as pd
import os

DATA_PATH = './data/'

def load_steam_data(data_path=DATA_PATH):
    """Loads steam.csv into a pandas DataFrame."""
    file_path = os.path.join(data_path, 'steam.csv')
    df = pd.read_csv(file_path)
    df = df.rename(columns={'appid': 'steam_appid'})
    print("--- steam.csv (renamed 'appid' to 'steam_appid') ---")
    print(f"Shape: {df.shape}")
    print(f"Head:\n{df.head()}")
    print(f"Data Types:\n{df.dtypes}")
    print(f"Missing Values:\n{df.isnull().sum()}")
    print("--- End steam.csv ---\n")
    return df

def load_steam_description_data(data_path=DATA_PATH):
    """Loads steam_description_data.csv into a pandas DataFrame."""
    file_path = os.path.join(data_path, 'steam_description_data.csv')
    df = pd.read_csv(file_path)
    print("--- steam_description_data.csv ---")
    print(f"Shape: {df.shape}")
    print(f"Head:\n{df.head()}")
    print(f"Data Types:\n{df.dtypes}")
    print(f"Missing Values:\n{df.isnull().sum()}")
    print("--- End steam_description_data.csv ---\n")
    return df

def load_steamspy_tag_data(data_path=DATA_PATH):
    """Loads steamspy_tag_data.csv into a pandas DataFrame."""
    file_path = os.path.join(data_path, 'steamspy_tag_data.csv')
    df = pd.read_csv(file_path)
    print("--- steamspy_tag_data.csv ---")
    print(f"Shape: {df.shape}")
    print(f"Head:\n{df.head()}")
    print(f"Data Types:\n{df.dtypes}")
    print(f"Missing Values:\n{df.isnull().sum()}")
    print("--- End steamspy_tag_data.csv ---\n")
    return df

def load_steam_requirements_data(data_path=DATA_PATH):
    """Loads steam_requirements_data.csv into a pandas DataFrame."""
    file_path = os.path.join(data_path, 'steam_requirements_data.csv')
    df = pd.read_csv(file_path)
    print("--- steam_requirements_data.csv ---")
    print(f"Shape: {df.shape}")
    print(f"Head:\n{df.head()}")
    print(f"Data Types:\n{df.dtypes}")
    print(f"Missing Values:\n{df.isnull().sum()}")
    print("--- End steam_requirements_data.csv ---\n")
    return df

def load_steam_media_data(data_path=DATA_PATH):
    """Loads steam_media_data.csv into a pandas DataFrame."""
    file_path = os.path.join(data_path, 'steam_media_data.csv')
    df = pd.read_csv(file_path)
    print("--- steam_media_data.csv ---")
    print(f"Shape: {df.shape}")
    print(f"Head:\n{df.head()}")
    print(f"Data Types:\n{df.dtypes}")
    print(f"Missing Values:\n{df.isnull().sum()}")
    print("--- End steam_media_data.csv ---\n")
    return df

def load_steam_support_info_data(data_path=DATA_PATH):
    """Loads steam_support_info.csv into a pandas DataFrame."""
    file_path = os.path.join(data_path, 'steam_support_info.csv')
    df = pd.read_csv(file_path)
    print("--- steam_support_info.csv ---")
    print(f"Shape: {df.shape}")
    print(f"Head:\n{df.head()}")
    print(f"Data Types:\n{df.dtypes}")
    print(f"Missing Values:\n{df.isnull().sum()}")
    print("--- End steam_support_info.csv ---\n")
    return df

def preprocess_and_merge_data(steam_df, description_df, tags_df):
    """
    Merges and preprocesses the core game data DataFrames.
    - Merges steam_df with description_df on 'steam_appid'.
    - Merges the result with tags_df on 'steam_appid' (after renaming 'appid' in tags_df).
    - Fills missing values for specific text columns.
    - Performs feature engineering on 'genres' and 'steamspy_tags'.
    """
    print("--- Starting Preprocessing and Merging ---")

    # Merge steam_df with description_df
    merged_df = pd.merge(steam_df, description_df, on='steam_appid', how='left')
    print("\n--- After merging steam_df and description_df ---")
    print(f"Shape: {merged_df.shape}")
    print(f"Head:\n{merged_df.head()}")
    print(f"Missing Values:\n{merged_df.isnull().sum()}")

    # Prepare tags_df for merging
    tags_df_copy = tags_df.copy()
    tags_df_copy = tags_df_copy.rename(columns={'appid': 'steam_appid'})

    # Merge with tags_df_copy
    merged_df = pd.merge(merged_df, tags_df_copy, on='steam_appid', how='left')
    print("\n--- After merging with tags_df ---")
    print(f"Shape: {merged_df.shape}")
    print(f"Head:\n{merged_df.head()}")

    # Handle Missing Values for text descriptions
    text_cols_to_fill = ['detailed_description', 'about_the_game', 'short_description']
    for col in text_cols_to_fill:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].fillna('')
    print("\n--- After filling missing text descriptions ---")
    print(f"Missing values in '{text_cols_to_fill}':\n{merged_df[text_cols_to_fill].isnull().sum()}")

    # Feature Engineering: Split 'genres' and 'steamspy_tags'
    if 'genres' in merged_df.columns:
        merged_df['genre_list'] = merged_df['genres'].apply(lambda x: x.split(';') if pd.notna(x) else [])
        print("\nCreated 'genre_list' column.")
    else:
        print("\n'genres' column not found for feature engineering.")

    if 'steamspy_tags' in merged_df.columns:
        merged_df['steamspy_tag_list'] = merged_df['steamspy_tags'].apply(lambda x: x.split(';') if pd.notna(x) else [])
        print("Created 'steamspy_tag_list' column.")
    else:
        print("\n'steamspy_tags' column not found for feature engineering.")

    print("\n--- Final Merged DataFrame Missing Values Summary ---")
    print(merged_df.isnull().sum())

    print("--- End Preprocessing and Merging ---\n")
    return merged_df

if __name__ == '__main__':
    steam_df = load_steam_data()
    print("Successfully loaded steam_data.csv")
    description_df = load_steam_description_data()
    print("Successfully loaded steam_description_data.csv")
    tags_df = load_steamspy_tag_data()
    print("Successfully loaded steamspy_tag_data.csv")

    # Load other dataframes as they are not part of the core merge for now
    load_steam_requirements_data()
    print("Successfully loaded steam_requirements_data.csv")
    load_steam_media_data()
    print("Successfully loaded steam_media_data.csv")
    load_steam_support_info_data()
    print("Successfully loaded steam_support_info.csv")

    # Preprocess and merge the core data
    core_game_data_df = preprocess_and_merge_data(steam_df, description_df, tags_df)
    print("Successfully preprocessed and merged core game data.")
    print("\n--- Core Game Data DataFrame Info ---")
    core_game_data_df.info()
    print("\n--- Core Game Data DataFrame Head ---")
    print(core_game_data_df.head())

    print("\nAll data loading and initial preprocessing functions executed.")
