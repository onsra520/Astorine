import os
import json5
import pandas as pd
from difflib import SequenceMatcher
from typing import Optional, Union, List, Dict, Any

def relocate_column(
    Information=None, Column_Name=None, Before=None, Behind=None, Position=None
) -> pd.DataFrame:
    """
    Move a column to a specific position in a DataFrame.

    Args:
        Information (pd.DataFrame): DataFrame to process.
        Column_Name (str): Column name to move.
        Before (str): Name of the BEFORE column.
        Behind (str): Name of the AFTER column.
        Position (int): Specific position to move

    Returns:
        pd.DataFrame: DataFrame with the column moved to move

    Raises:
        ValueError: If `Information`, `Column_Name` is None.
        KeyError: If `Column_Name` does not exist in the DataFrame.
        IndexError: If `Position` is out of range for the DataFrame.
    """
    if Information is None or Column_Name is None:
        raise ValueError(
            "The parameters Information, Column_Name cannot be left blank."
        )
    if Column_Name not in Information.columns:
        raise KeyError(f"'{Column_Name}' does not exist in the DataFrame.")
    if Position is None:
        Position = 0

        if not (0 <= Position < len(Information.columns)):
            raise IndexError(
                f"Position {Position} is out of range for the DataFrame with {len(Information.columns)} columns."
            )
        Columns = Information.columns.tolist()
        Columns.pop(Columns.index(Column_Name))
        if Before is not None:
            Position = Columns.index(Before) + 1
            Columns.insert(Position, Column_Name)
            if Behind is not None:
                Columns.pop(Columns.index(Behind))
                Columns.insert(Columns.index(Column_Name) + 1, Behind)
        elif Behind is not None:
            Position = Columns.index(Behind)
            Columns.insert(Position, Column_Name)
    else:
        Columns.insert(Position, Columns.pop(Columns.index(Column_Name)))

    return Information[Columns]

def manage_column_list(JsonFilePath=None, Remove=None, Recovers=None) -> list:
    """
    Update the list of columns that are not needed in the data processing.

    Args:
        JsonFilePath (str): File path to the JSON file.
        Remove (list or str, optional): Columns to remove from the unnecessary list. These columns will be added to the list if not already present.
        Recovers (list or str, optional): Columns to recover from the unnecessary list. These columns will be removed from the list if present.

    Returns:
        list: List of columns that are not needed in the data processing.

    Raises:
        ValueError: If `JsonFilePath` is None.
    """
    if JsonFilePath is None:
        raise ValueError("The parameter 'Path' cannot be left blank.")

    if os.path.exists(JsonFilePath):
        with open(JsonFilePath, "r", encoding="utf-8") as file:
            try:
                data = json5.load(file)
            except json5.JSONDecodeError:
                data = {}
    else:
        data = {}

    if "second_process" not in data:
        data["second_process"] = {"drop": [], "Recovers": []}

    if isinstance(Remove, (list, str)):
        Remove = [Remove] if isinstance(Remove, str) else Remove
        data["second_process"]["drop"].extend(
            [col for col in Remove if col not in data["second_process"]["drop"]]
        )

    if isinstance(Recovers, (list, str)):
        Recovers = [Recovers] if isinstance(Recovers, str) else Recovers
        data["second_process"]["drop"] = [
            col for col in data["second_process"]["drop"] if col not in Recovers
        ]

    with open(JsonFilePath, "w", encoding="utf-8") as file:
        json5.dump(data, file, indent=4, ensure_ascii=False)

    return data["second_process"]["drop"]

def find_similar_rows(
    dataframe: pd.DataFrame,
    target_row: Dict[str, Any],
    reference_column: str,
    matching_columns: Union[List[str], str],
    comparison_column: str,
    mode: Optional[str] = None,
    similarity_threshold: int = 80,
) -> Union[str, List[str]]:
    """Find rows in a DataFrame similar to a target row based on specified columns.

    Args:
        dataframe: DataFrame containing the data to search
        target_row: Dictionary containing the row to compare against
        reference_column: Column name used for initial filtering
        matching_columns: Column(s) to use for finding potential matches
        comparison_column: Column used for final similarity scoring
        mode: Matching strategy ('exact', 'partial', 'similarity')
        similarity_threshold: Minimum similarity score (0-100) for matches

    Returns:
        Best matching value(s) or "Undefined" if no matches found

    Raises:
        ValueError: For missing required parameters
        KeyError: If specified columns are missing from DataFrame
    """
    # Validate input parameters
    if dataframe is None or target_row is None:
        raise ValueError("DataFrame and target row must be provided")

    if isinstance(matching_columns, str):
        matching_columns = [matching_columns]

    # Check column existence
    def validate_column(column: str) -> None:
        if column not in dataframe.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame")

    validate_column(reference_column)
    for col in matching_columns:
        validate_column(col)
    validate_column(comparison_column)

    # Filter initial candidates
    target_value = str(target_row[reference_column]).strip()
    candidates = dataframe[dataframe[reference_column] != target_value].copy()

    # Define matching strategies
    def exact_match(col: str, value: str) -> pd.DataFrame:
        return candidates[candidates[col].apply(lambda x: str(x).strip() == value)]

    def partial_match(col: str, value: str) -> pd.DataFrame:
        return candidates[
            candidates[col].apply(lambda x: value in str(x) or str(x) in value)
        ]

    def similarity_match(col: str, value: str) -> pd.DataFrame:
        return candidates[
            candidates[col].apply(
                lambda x: SequenceMatcher(None, value, str(x)).ratio() * 100
                > similarity_threshold
            )
        ]

    strategies = {
        "exact": exact_match,
        "partial": partial_match,
        "similarity": similarity_match,
    }

    # Apply matching strategy
    matches = pd.DataFrame()
    for column in matching_columns:
        current_value = str(target_row[column]).strip()

        if mode:
            strategy = strategies.get(mode.lower(), exact_match)
            matches = strategy(column, current_value)
        else:
            # Try all strategies in order
            for strategy in [exact_match, partial_match, similarity_match]:
                matches = strategy(column, current_value)
                if not matches.empty:
                    break

        if not matches.empty:
            break

    if matches.empty:
        return "Undefined"

    # Calculate similarity scores
    best_match = "Undefined"
    highest_score = 0

    for _, row in matches.iterrows():
        score = SequenceMatcher(
            None, str(row[comparison_column]), str(target_row[comparison_column])
        ).ratio()

        if score > highest_score and row[reference_column] != "Undefined":
            highest_score = score
            best_match = row[reference_column]

    return best_match if best_match != "Undefined" else "Undefined"

def load(JsonFilePath=None, Categorical=None, Numerical=None, Remove=None):
    """
    Load data from a JSON file and update it with the specified categorical and numerical data.

    Args:
        JsonFilePath (str): Path to the JSON file.
        Categorical (list or str): List of categorical data or a single string.
        Numerical (list or str): List of numerical data or a single string.
        Remove (list or str): List of items to remove from the data or a single string.

    Returns:
        tuple: Two lists, the first containing the categorical data and the second containing the numerical data.

    Raises:
        ValueError: If `JsonFilePath` is not provided.
    """
    if JsonFilePath is None:
        raise ValueError("The parameter 'JsonFilePath' cannot be left blank.")

    # If the file exists, read it; otherwise, create a new dictionary
    data = {}
    if os.path.exists(JsonFilePath):
        with open(JsonFilePath, "r", encoding="utf-8") as file:
            try:
                data = json5.load(file)
            except json5.JSONDecodeError:
                pass  # If the JSON is invalid, we'll just start fresh
    else:
        data["Hoodoo"] = {
            "Categorical Data": [],
            "Numerical Data": [],
        }

    # Ensure "Hoodoo" key exists in the dictionary
    if "Hoodoo" not in data:
        data["Hoodoo"] = {
            "Categorical Data": [],
            "Numerical Data": [],
        }

    # Normalize input lists and update the JSON data
    def normalize_and_update(target_list, source_list):
        if isinstance(source_list, (list, str)):
            source_list = [source_list] if isinstance(source_list, str) else source_list
            for item in source_list:
                if item not in target_list:
                    target_list.append(item)

    normalize_and_update(data["Hoodoo"]["Categorical Data"], Categorical)
    normalize_and_update(data["Hoodoo"]["Numerical Data"], Numerical)

    # Remove specified items from the lists if provided
    if Remove is not None:
        for key in data["Hoodoo"]:
            for item in data["Hoodoo"][key]:
                if item in Remove:
                    data["Hoodoo"][key].remove(item)

    # Write the updated data back to the JSON file
    with open(JsonFilePath, "w", encoding="utf-8") as file:
        json5.dump(data, file, indent=4, ensure_ascii=False)

    return data["Hoodoo"]["Categorical Data"], data["Hoodoo"]["Numerical Data"]
