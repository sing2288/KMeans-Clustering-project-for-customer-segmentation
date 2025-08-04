import sys
import os

# Add the 'modules' folder to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from modules.logger import get_logger

logger = get_logger(__name__)

def load_data(path):
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df
