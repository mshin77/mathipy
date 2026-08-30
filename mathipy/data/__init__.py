"""Locate sample items supplied by the user.

No assessment items ship with mathipy. Set the ``MATHIPY_DATA_DIR`` environment
variable to a directory holding item images and a metadata CSV. The helpers
below resolve paths inside that directory and do not require the files to
exist.
"""

import os
import re
from pathlib import Path


def data_directory() -> Path:
    """Return the directory searched for sample items."""
    return Path(os.environ.get("MATHIPY_DATA_DIR", Path(__file__).parent))


def get_sample_csv() -> Path:
    """Return the expected path of the sample metadata CSV."""
    return data_directory() / "items.csv"


def get_sample_image(item_id: str) -> Path:
    """Return the expected path of an item image by item ID."""
    if not re.match(r"^[a-zA-Z0-9_\-\s#]+$", item_id):
        raise ValueError(f"Invalid item_id: {item_id}")
    base = data_directory().resolve()
    path = base / f"{item_id}.png"
    if not path.resolve().is_relative_to(base):
        raise ValueError(f"Invalid item_id: {item_id}")
    return path


def list_sample_images() -> list[str]:
    """Return a sorted list of image filenames present in the data directory."""
    return sorted(p.name for p in data_directory().glob("*.png"))
