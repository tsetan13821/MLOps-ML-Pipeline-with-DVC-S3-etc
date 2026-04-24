import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# configure logging
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    """
    Transform the input text by removing punctuation, converting to lowercase,
    removing stopwords, and applying stemming.
    """
    ps = PorterStemmer()
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    # remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    # apply stemming
    text = [ps.stem(word) for word in text]
    return ' '.join(text)

def preprocess_df(df, text_column = 'text',target_column = 'target'):
    """
    Preprocess the DataFrame by transforming the text column and encoding the target column.
    """
    try:
        logger.debug('Starting data preprocessing')
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('Target column encoded successfully')
        # Remove duplicate rows
        df = df.drop_duplicates(keep='first')
        logger.debug('Duplicate rows removed successfully')
        # Apply text transformation to the specified text column
        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug('Text column transformed successfully')
        return df
    except KeyError as e:
        logger.error('Missing expected columns in the DataFrame: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred during data preprocessing: %s', e)
        raise

def main(text_column = 'text',target_column = 'target'):
    """
    Main function to load, preprocess, and save the data.
    """
    try:
        # fetch the data from data/raw
        train_data = pd.read_csv('data/raw/train.csv')
        test_data = pd.read_csv('data/raw/test.csv')

        # normalize columns from raw input so preprocessing works on both v1/v2 and target/text
        rename_map = {
            'V1': 'target',
            'V2': 'text',
            'v1': 'target',
            'v2': 'text'
        }
        train_data.rename(columns=rename_map, inplace=True)
        test_data.rename(columns=rename_map, inplace=True)

        logger.debug('Data loaded successfully')
        
        # Transform the data
        train_processed_data = preprocess_df(train_data, text_column, target_column)
        test_processed_data = preprocess_df(test_data, text_column, target_column)

        # Store the data in data/processed
        data_path = os.path.join('./data', 'interim')
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, 'train_processed.csv'), index=False)
        test_processed_data.to_csv(os.path.join(data_path, 'test_processed.csv'), index=False)

        logger.debug('Processed data saved successfully to %s', data_path)
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        raise
    except pd.errors.EmptyDataError as e:
        logger.error('Empty data error: %s', e)
        raise
    except KeyError as e:
        logger.error('Missing expected columns in the input DataFrame: %s', e)
        raise
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        raise
if __name__ == "__main__":    
    main()
