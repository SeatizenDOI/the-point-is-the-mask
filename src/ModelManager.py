import torch
import torch.nn as nn

import rasterio
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from argparse import Namespace
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

from .PathManager import PathManager

class ModelManager:

    def __init__(self, opt: Namespace):
        self.opt = opt

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = SegformerForSemanticSegmentation.from_pretrained(self.opt.path_segmentation_model).to(self.device)
        self.processor = AutoImageProcessor.from_pretrained("nvidia/mit-b0", do_reduce_labels=False)


    def preprocess_image(self, image_path: Path):
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            size = image.size
        return size, inputs

    def predict_mask(self, image_path: Path):
        size, inputs = self.preprocess_image(image_path)
        
        with torch.no_grad():
            outputs =self.model(**inputs)

        logits = outputs.logits  # Shape: (1, num_labels, height, width)
        mask_resized_bilinear = nn.functional.interpolate( # Segformer size is 1/4 need to resize to get mask on image
                logits,  
                size=size, 
                mode='bilinear',
                align_corners=False
            )
        mask_resized_bilinear = mask_resized_bilinear.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        return mask_resized_bilinear + 1
    

    def inference(self, path_manager: PathManager):
        print("*\t Perform inference.")
        session_images = sorted(list(path_manager.cropped_ortho_img_folder.iterdir()))
        predicted_rasters = []
        
        for img_path in tqdm(session_images, desc="Performing inference on images"):
            mask = self.predict_mask(img_path)
            predicted_rasters.append((mask, img_path))

        # Convert predictions to GeoTIFF
        for mask, img_path in tqdm(predicted_rasters, desc="Convert mask to tiff files"):
            corresponding_tiff = Path(path_manager.cropped_ortho_folder, f"{img_path.stem}.tif")
            if not corresponding_tiff.exists():
                print(f"Warning: No matching TIFF file found for {img_path.name}, skipping...")
                continue

            with rasterio.open(corresponding_tiff) as src:
                meta = src.meta.copy()
                meta.update({"dtype": 'uint8', "count": 1, "nodata": 255}) 
                
            output_tiff_path = Path(path_manager.predictions_tiff_folder, f"{img_path.stem}_prediction.tif")
            with rasterio.open(output_tiff_path, 'w', **meta) as dst:
                mask = np.where(mask == 0, 255, mask)
                dst.write(mask, 1)        
