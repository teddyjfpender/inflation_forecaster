# src/store_manager.py

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_prediction_store(store_path: str) -> Dict[str, Any]:
    """
    Load the prediction store from a JSON file. If the file doesn't exist,
    return an empty dictionary.
    
    Args:
        store_path: Path to the JSON store file.
        
    Returns:
        Dictionary containing the prediction store.
    """
    store_path = Path(store_path)
    
    # Create parent directories if they don't exist
    store_path.parent.mkdir(parents=True, exist_ok=True)
    
    if store_path.exists():
        try:
            with open(store_path, 'r') as f:
                store = json.load(f)
            logger.info(f"Loaded prediction store from {store_path}")
            return store
        except Exception as e:
            logger.error(f"Error loading prediction store: {e}")
            return {}
    else:
        logger.info(f"Prediction store not found at {store_path}. Creating new store.")
        return {}

def save_prediction_store(store: Dict[str, Any], store_path: str) -> None:
    """
    Save the prediction store to a JSON file.
    
    Args:
        store: Dictionary containing the prediction store.
        store_path: Path to save the JSON store file.
    """
    store_path = Path(store_path)
    
    # Create parent directories if they don't exist
    store_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(store_path, 'w') as f:
            json.dump(store, f, indent=2)
        logger.info(f"Saved prediction store to {store_path}")
    except Exception as e:
        logger.error(f"Error saving prediction store: {e}")
        raise

def update_store(new_predictions: List[Dict[str, Any]], store: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update the prediction store with new predictions.
    Only add predictions with dates later than the latest date in the store.
    
    Args:
        new_predictions: List of new prediction records.
        store: Dictionary containing the model's prediction store.
        
    Returns:
        Updated store dictionary.
    """
    # Initialize the store structure if empty
    if not store:
        store = {
            "last_updated": datetime.now().isoformat(),
            "historical_predictions": []
        }
    
    # Get existing predictions
    existing = store.get("historical_predictions", [])
    
    # Find the latest date in existing predictions
    last_date = None
    if existing:
        last_date = max(entry['date'] for entry in existing)
    
    # Filter new predictions to only include those after the last date
    filtered_predictions = []
    for pred in new_predictions:
        if last_date is None or pred['date'] > last_date:
            filtered_predictions.append(pred)
    
    # Merge with existing predictions
    merged = existing + filtered_predictions
    
    # Update the store
    store["historical_predictions"] = merged
    store["last_updated"] = datetime.now().isoformat()
    
    logger.info(f"Added {len(filtered_predictions)} new predictions to the store")
    return store

def get_last_prediction(store: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Get the most recent prediction from the store.
    
    Args:
        store: Dictionary containing the model's prediction store.
        
    Returns:
        The most recent prediction, or None if the store is empty.
    """
    if not store or "historical_predictions" not in store:
        return None
    
    predictions = store["historical_predictions"]
    if not predictions:
        return None
    
    # Find the prediction with the latest date
    return max(predictions, key=lambda x: x['date'])

def get_predictions_in_range(store: Dict[str, Any], start_date: str, end_date: str) -> List[Dict[str, Any]]:
    """
    Get predictions within a specific date range.
    
    Args:
        store: Dictionary containing the model's prediction store.
        start_date: Start date in ISO format (YYYY-MM-DD).
        end_date: End date in ISO format (YYYY-MM-DD).
        
    Returns:
        List of predictions within the specified date range.
    """
    if not store or "historical_predictions" not in store:
        return []
    
    predictions = store["historical_predictions"]
    if not predictions:
        return []
    
    # Filter predictions in the date range
    filtered = [
        pred for pred in predictions 
        if start_date <= pred['date'] <= end_date
    ]
    
    return filtered 