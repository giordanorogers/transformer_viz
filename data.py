"""
The data loading logic.
"""

import os
import json
import pandas as pd

def load_dataset(file_path):
    """
    Loads a JSON dataset from the specified file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file {file_path} not found.")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_prompt_data(file_path="known_1000.json"):
    """
    Loads prompt data and returns it as a pandas DataFrame.
    """
    data = load_dataset(file_path)
    df = pd.DataFrame(data)
    return df

def get_prompt_dict(file_path="known_1000.json"):
    """
    Returns a dictionary mapping "known_id: subject" to the prompt data.
    """
    df = get_prompt_data(file_path)
    prompt_dict = {f"{row['known_id']}: {row['subject']}": row for _, row in df.iterrows()}
    return prompt_dict
