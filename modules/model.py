import sys
import os

# Add the 'modules' folder to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.cluster import KMeans
from modules.logger import get_logger

logger = get_logger(__name__)

def train_kmeans(data, k):
    logger.info(f"Training KMeans with k={k}")
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(data)
    logger.info("KMeans training complete")
    return model, model.labels_, model.inertia_
