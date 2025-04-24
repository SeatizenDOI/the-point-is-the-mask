import torch
from torchvision.transforms import ColorJitter
import numpy as np
from transformers import AutoImageProcessor
from pathlib import Path
from sklearn.model_selection import train_test_split

from ..ConfigParser import ConfigParser
from .dataset import create_dataset
from .trainer import setup_trainer
from ..utils.lib_tools import print_gpu_is_used


def main_launch_training(cp: ConfigParser, path_to_image: Path, class_mapping: dict) -> None:
    """
        cp: From 
        path_to_image: Path to a folder with a subfolder images and a subfolder annotations
    """

    print("\n\n------ [TRAIN - Setup image dataset] ------\n")

    # Sort images and annotations
    image_folder = Path(path_to_image, "images")
    annotation_folder = Path(path_to_image, "annotations")

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


    # Step 1: create Dataset objects
    train_ds = create_dataset(image_paths_train, label_paths_train)
    validation_ds = create_dataset(image_paths_validation, label_paths_validation)
    test_ds = create_dataset(image_paths_test, label_paths_test)

    processor = AutoImageProcessor.from_pretrained(cp.model_name, do_reduce_labels=True, use_fast=True)

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

    def val_test_transforms(example_batch):
        images = example_batch['image']  # Do NOT apply jitter
        inputs = processor(images, example_batch['label'])
        inputs["labels"] = [torch.tensor(np.array(x), dtype=torch.long) for x in inputs["labels"]]

        return {
            "pixel_values": inputs["pixel_values"],
            "labels": inputs["labels"],
            "image_name": example_batch["image_name"]  # Preserve image names
        }
    
    # Set transforms
    train_ds.set_transform(train_transforms)
    validation_ds.set_transform(val_test_transforms)
    test_ds.set_transform(val_test_transforms)

    print("\n\n------ [TRAIN - Setup model] ------\n")

    trainer = setup_trainer(cp, len(class_mapping), train_ds, validation_ds)

    print("\n\n------ [TRAIN - Start training] ------\n")

    print_gpu_is_used()
    trainer.train()


