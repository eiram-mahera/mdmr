import os, csv, json, time
from pathlib import Path
from typing import Union, List

def mkdir_p(path: Union[str, os.PathLike]) -> str:
    """Create directory path if it doesn’t exist (like `mkdir -p`)."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)

def timestamp() -> str: return time.strftime("%Y%m%d-%H%M%S")

def save_json(obj, path: str):
    mkdir_p(os.path.dirname(path))
    with open(path, "w") as f: json.dump(obj, f, indent=2)

def append_row(csv_path: Union[str, os.PathLike], header: List[str], row: list) -> None:
    """Append a row to a CSV file, writing the header first if the file is new."""
    csv_path = Path(csv_path)           # normalize to Path
    mkdir_p(csv_path.parent)            # ensure parent dir exists

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)
