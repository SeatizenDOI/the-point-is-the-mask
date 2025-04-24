
from pathlib import Path

from datasets import Dataset, Image

def create_dataset(image_paths: list[Path], label_paths: list[Path]):
    image_names = [Path(img).name for img in image_paths]  # Extract image names
    dataset = Dataset.from_dict({
        "image_name": sorted(image_names),
        "image": sorted(image_paths),
        "label": sorted(label_paths)
    })
    dataset = dataset.cast_column("image", Image())
    dataset = dataset.cast_column("label", Image())

    return dataset
