import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging
import yaml

# ensure the logs directory exists
log_dir = 'logs'

os.makedirs(log_dir, exist_ok=True)

# configure logging
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters loaded from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('Failed to parse the YAML file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading parameters: %s', e)
        raise
def load_data(data_url: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        data_url (str): The URL or local path to the CSV file.

    Returns:
        pd.DataFrame: The loaded data as a DataFrame.
    """
    try:
        df = pd.read_csv(data_url)
        logger.debug(f"Data loaded successfully from {data_url}")
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s',e )
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading data: %s', e)
        raise
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by handling missing values and encoding categorical variables.
    """
    try:
        df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True, errors='ignore')
        df.rename(columns={
            'V1': 'target',
            'V2': 'text',
            'v1': 'target',
            'v2': 'text'
        }, inplace=True)
        if 'target' not in df.columns or 'text' not in df.columns:
            missing = [c for c in ['target', 'text'] if c not in df.columns]
            raise KeyError(f"Missing columns after rename: {missing}")
        logger.debug('Data preprocessing completed successfully')
        return df
    except KeyError as e:
        logger.error('Missing expected columns in the DataFrame: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred during data preprocessing: %s', e)
        raise
def save_data(train_data:pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """
    Save the preprocessed data to CSV files.

    Args:
        train_data (pd.DataFrame): The training data.
        test_data (pd.DataFrame): The testing data.
        data_path (str): The directory path where the CSV files will be saved.
    """
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_file_path = os.path.join(raw_data_path, 'train.csv')
        test_file_path = os.path.join(raw_data_path, 'test.csv')
        train_data.to_csv(train_file_path, index=False)
        test_data.to_csv(test_file_path, index=False)
        logger.debug(f"Train and test data saved successfully to {raw_data_path}")
    except Exception as e:
        logger.error('Unexpected error occurred while saving data: %s', e)
        raise

def main():
    """save the train test datasets."""
    try:
        # test_size = 0.2
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']
        data_path = 'https://raw.githubusercontent.com/tsetan13821/MLOps-ML-Pipeline-with-DVC-S3-etc/refs/heads/main/experiments/spam.csv'
        df = load_data(data_url = data_path)
        final_df = preprocess_data(df)
        train_data , test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        save_data(train_data, test_data, data_path = './data')
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")
if __name__ == "__main__":
    main()  