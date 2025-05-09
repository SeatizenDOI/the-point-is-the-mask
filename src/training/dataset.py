import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import Dataset, Image
from transformers import AutoImageProcessor
from torchvision.transforms import ColorJitter
from sklearn.model_selection import train_test_split


from ..ConfigParser import ConfigParser

def create_dataset(image_paths: list[Path], label_paths: list[Path]) -> Dataset:
    image_names = [Path(img).name for img in image_paths]  # Extract image names
    dataset = Dataset.from_dict({
        "image_name": sorted(image_names),
        "image": sorted(image_paths),
        "label": sorted(label_paths)
    })
    dataset = dataset.cast_column("image", Image())
    dataset = dataset.cast_column("label", Image())

    return dataset


class DatasetManager:

    def __init__(self, cp: ConfigParser, path_to_image_and_annotation: Path):
        
        self.base_path = path_to_image_and_annotation
        self.cp = cp

        self.train_ds, self.validation_ds, self.test_ds = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


    def load_datasets(self):
        # Sort images and annotations
        image_folder = Path(self.base_path, "images")
        annotation_folder = Path(self.base_path, "annotations")

        if not image_folder.exists() or not image_folder.is_dir():
            raise FileNotFoundError(f"Cannot found image folder for the training: {image_folder}")
        
        if not annotation_folder.exists() or not annotation_folder.is_dir():
            raise FileNotFoundError(f"Cannot found annotation folder for the training: {annotation_folder}")

        image_to_annotation = {
            str(img): str(ann) for img in image_folder.iterdir() for ann in annotation_folder.iterdir() if img.name == ann.name
        }

        images = list(image_to_annotation.keys())
        annotations = list(image_to_annotation.values())

        # Split into train, validation, and test sets
        image_paths_train, image_paths_temp, label_paths_train, label_paths_temp = train_test_split(
            images, annotations, test_size=0.3, random_state=42
        )

        image_paths_validation, image_paths_test, label_paths_validation, label_paths_test = train_test_split(
            image_paths_temp, label_paths_temp, test_size=0.5, random_state=42
        )


        self.train_ds = create_dataset(image_paths_train, label_paths_train)
        self.validation_ds = create_dataset(image_paths_validation, label_paths_validation)
        self.test_ds = create_dataset(image_paths_test, label_paths_test) # TODO Remove, unused


    def attach_transforms(self) -> None:
        
        processor = AutoImageProcessor.from_pretrained(self.cp.base_model_name, do_reduce_labels=True, use_fast=False)

        jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)


        def train_transforms(example_batch):
            images = [jitter(x) for x in example_batch['image']]  # ColorJitter
            inputs = processor(images, example_batch['label'])
            inputs["labels"] = [torch.tensor(np.array(x), dtype=torch.long) for x in inputs["labels"]]

            return {
                "pixel_values": inputs["pixel_values"],
                "labels": inputs["labels"],
                "image_name": example_batch["image_name"]  # Preserve image names
            }

        def val_transforms(example_batch):
            images = example_batch['image']  # Do NOT apply jitter
            inputs = processor(images, example_batch['label'])
            inputs["labels"] = [torch.tensor(np.array(x), dtype=torch.long) for x in inputs["labels"]]

            return {
                "pixel_values": inputs["pixel_values"],
                "labels": inputs["labels"],
                "image_name": example_batch["image_name"]  # Preserve image names
            }
        
        # Set transforms
        self.train_ds.set_transform(train_transforms)
        self.validation_ds.set_transform(val_transforms)
        self.test_ds.set_transform(val_transforms)
