import torch
import torch.nn as nn

import rasterio
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from argparse import Namespace
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from segment_anything import sam_model_registry
from scipy.ndimage import distance_transform_edt
from osgeo import gdal

from ..utils.sam_refiner import sam_refiner
from .PathRasterManager import PathRasterManager



class ModelManager:

    def __init__(self, opt: Namespace):
        self.opt = opt

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = SegformerForSemanticSegmentation.from_pretrained(self.opt.path_segmentation_model).to(self.device)
        self.processor = AutoImageProcessor.from_pretrained("nvidia/mit-b0", do_reduce_labels=False, use_fast=False)
        
        
        self.unwanted_value = max(self.model.config.id2label) + 1
        self.label_sand_id = self.model.config.label2id.get("Sand", 5) # We want to refine only on the sand.
        
        if self.opt.use_sam_refiner:
            self.setup_sam_refiner()


    def setup_sam_refiner(self):

        print("We are currently charging the SAM refiner model.")
        path_sam_model = Path(self.opt.path_sam_model)
        if not path_sam_model.exists() or not path_sam_model.is_file():
            raise FileNotFoundError("[ERROR] We cannot use sam, please follow the readme to download the weight.")

        self.sam = sam_model_registry["vit_h"](checkpoint=path_sam_model)
        self.sam.to(device=self.device)


    def preprocess_image(self, image_path: Path):
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            size = image.size
        return size, inputs


    def fill_unlabelled_with_neighbors(self, mask):
        """Fill unlabelled pixels with nearest neighbor values using distance transform."""
        
        # Create a mask of unlabelled values
        mask_unlabelled = mask == self.unwanted_value
        if not np.any(mask_unlabelled):
            return mask
        
        # Generate labelled regions for valid values
        distance, nearest_index = distance_transform_edt(mask_unlabelled, return_indices=True)

        nearest_values = mask[nearest_index[0], nearest_index[1]]

        # Fill unlabelled areas with nearest neighbor values
        filled_mask = mask.copy()
        filled_mask[mask_unlabelled] = nearest_values[mask_unlabelled]

        return filled_mask
    
    def predict_sam(self, mask, image_path):
        
        # Get all sand value and replace its value by unwanted.
        binary_mask = (mask == self.label_sand_id).astype(np.uint8)
        mask[binary_mask == 1] = self.unwanted_value

        # We refine the mask of the sand.
        refined_mask = sam_refiner(image_path, [binary_mask], self.sam)[0]
        mask[refined_mask[0] == 1] = self.unwanted_value  # Apply thresholding for segmentation

        # For all zones without value, we try to find the nearest neighbours.
        output_mask = self.fill_unlabelled_with_neighbors(mask)

        # Finally we reapply our sand mask.
        output_mask[refined_mask[0] == 1] = self.label_sand_id
        output_mask[output_mask == self.unwanted_value] = 0 # background value

        return output_mask


    def predict_mask(self, image_path: Path):
        size, inputs = self.preprocess_image(image_path)
        
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits  # Shape: (1, num_labels, height, width)
        mask_resized_bilinear = nn.functional.interpolate( # Segformer size is 1/4 need to resize to get mask on image
                logits,  
                size=size, 
                mode='bilinear',
                align_corners=False
            )
        mask_resized_bilinear = mask_resized_bilinear.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        return mask_resized_bilinear + 1
    

    def inference(self, path_manager: PathRasterManager):
        print("*\t Perform inference.")
        session_images = sorted(list(path_manager.cropped_ortho_img_folder.iterdir()))
        predicted_rasters = []
        
        for img_path in tqdm(session_images, desc="Performing inference on images"):
            mask = self.predict_mask(img_path)
            
            if self.opt.use_sam_refiner:
                mask = self.predict_sam(np.copy(mask), img_path)
            
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


    def get_id2label(self) -> dict:
        return self.model.config.id2label