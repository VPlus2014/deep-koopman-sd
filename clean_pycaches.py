import os
from pathlib import Path
import shutil


def remove_pycache_dirs(root_dir="."):
    root_dir = Path(root_dir).resolve()
    print(root_dir)
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            if dirname == "__pycache__":
                pycache_path = os.path.join(dirpath, dirname)
                print(f"Removing {pycache_path}")
                shutil.rmtree(pycache_path)


if __name__ == "__main__":
    remove_pycache_dirs()
