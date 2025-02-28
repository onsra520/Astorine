import os
import json5

def load_features(JsonFilePath=None, Categorical=None, Numerical=None, Remove=None):
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
