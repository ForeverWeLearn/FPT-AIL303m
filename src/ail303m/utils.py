from pathlib import Path


def get_incremental_path(path_str: str) -> str:
    path = Path(path_str)
    
    if not path.exists():
        return str(path)
    
    parent = path.parent
    stem = path.stem
    suffix = path.suffix
    
    counter = 1
    while True:
        new_path = parent / f"{stem}_{counter}{suffix}"
        if not new_path.exists():
            return str(new_path)
        counter += 1
