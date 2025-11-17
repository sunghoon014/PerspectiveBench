import json
from pathlib import Path
from typing import Any


def save_text(filepath: str | Path, data: str, encoding: str = "utf-8"):
    """Saves text to a file.

    Args:
        filepath (str | Path): The path to the file to save.
        data (str): The text data to save.
        encoding (str): The file encoding (default: "utf-8").
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding=encoding) as f:
        f.write(data)


def load_text(filepath: str | Path, encoding: str = "utf-8") -> str:
    """Loads text from a file.

    Args:
        filepath (str | Path): The path to the file to load.
        encoding (str): The file encoding (default: "utf-8").

    Returns:
        str: The text data read from the file.
    """
    path = Path(filepath)
    with open(path, encoding=encoding) as f:
        return f.read()


def save_json(filepath: str | Path, data: dict[str, Any], encoding: str = "utf-8"):
    """Saves JSON data to a file.

    Args:
        filepath (str | Path): The path to the file to save.
        data (dict[str, Any]): The JSON data to save (dict).
        encoding (str): The file encoding (default: "utf-8").
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding=encoding) as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_json(filepath: str | Path, encoding: str = "utf-8") -> dict[str, Any]:
    """Loads JSON data from a file.

    Args:
        filepath (str | Path): The path to the file to load.
        encoding (str): The file encoding (default: "utf-8").

    Returns:
        dict[str, Any]: The JSON data read from the file (dict).
    """
    path = Path(filepath)
    with open(path, encoding=encoding) as f:
        return json.load(f)
