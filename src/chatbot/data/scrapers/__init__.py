from pathlib import Path

def directory(default="chatbot") -> Path:
    """
    This function returns the path to the 'chatbot' directory, regardless of how many levels up or down it is.

    Args:
        default (str): The name of the default directory to return. Defaults to 'chatbot'

    Returns:
        Path: The path to the default directory
    """
    current_path = Path(__file__).resolve()

    for parent in current_path.parents:
        if parent.name.lower() == default.lower():
            return parent
    default_path = Path.cwd() / default
    if not default_path.exists():
        default_path.mkdir(parents=True, exist_ok=True)
    return default_path

if __name__ == "__main__":
    directory()
