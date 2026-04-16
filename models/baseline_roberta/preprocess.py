# Standard libraries

# Custom libraries
import pandas as pd
from sklearn.model_selection import train_test_split

# Local imports
from config import TEST_SIZE, TEST_VAL_SIZE, RANDOM_STATE

def generate_splits(df: pd.DataFrame):
    train_df, temp_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df['is_sarcastic']
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=TEST_VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=temp_df['is_sarcastic']
    )

    return train_df, val_df, test_df