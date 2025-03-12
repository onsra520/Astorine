from pathlib import Path
import os

skip = [
    "__pycache__",
    "downloaded_files",
    "_metadata",
    "__MACOSX",
    "generated_indexed_rulesets",
]

def pathtree(nrf: str = "chatbot") -> dict:
    cur_dir = os.path.dirname(__file__)
    rdir = Path(cur_dir.split(nrf)[0]) / nrf
    ftree = {}
    for path in rdir.rglob("*"):
        if path.is_dir() and path.name in skip:
            continue
        if path.is_dir():
            key = path.name
            if key in ftree:
                parent_name = path.parent.name
                key = f"{parent_name}.{path.name}"
            ftree[key] = str(path.resolve())
    return ftree
