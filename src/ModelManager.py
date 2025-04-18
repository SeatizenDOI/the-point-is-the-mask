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

from .lib.sam_refiner import sam_refiner
from .PathManager import PathManager

UNWANTED_VALUE = 6

class ModelManager:

    def __init__(self, opt: Namespace):
        self.opt = opt

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = SegformerForSemanticSegmentation.from_pretrained(self.opt.path_segmentation_model).to(self.device)
        self.processor = AutoImageProcessor.from_pretrained("nvidia/mit-b0", do_reduce_labels=False)

        sam_checkpoint = "./models/sam_vit_h_4b8939.pth"
        model_type = "vit_h"

        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=self.device)


    def preprocess_image(self, image_path: Path):
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            size = image.size
        return size, inputs

    def fill_unlabelled_with_neighbors(self, mask, unlabelled_value=UNWANTED_VALUE):
        """Fill unlabelled pixels with nearest neighbor values using distance transform."""
        
        # Create a mask of unlabelled values
        mask_unlabelled = mask == unlabelled_value
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

        binary_mask = (mask == 5).astype(np.uint8)
        mask[binary_mask == 1] = UNWANTED_VALUE
        refined_mask = sam_refiner(image_path, [binary_mask], self.sam)[0]
        mask[refined_mask[0] == 1] = UNWANTED_VALUE  # Apply thresholding for segmentation
        output_mask = self.fill_unlabelled_with_neighbors(mask, UNWANTED_VALUE)
        output_mask[refined_mask[0] == 1] = 5
        output_mask[output_mask == 6] = 0

        return output_mask


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
            # mask_sam = self.predict_sam(np.copy(mask), img_path)
            predicted_rasters.append((mask, img_path))
            # predicted_rasters.append((mask, mask_sam, img_path))

        # Convert predictions to GeoTIFF
        for mask, img_path in tqdm(predicted_rasters, desc="Convert mask to tiff files"):
        # for mask, mask_sam, img_path in tqdm(predicted_rasters, desc="Convert mask to tiff files"):

            corresponding_tiff = Path(path_manager.cropped_ortho_folder, f"{img_path.stem}.tif")
            if not corresponding_tiff.exists():
                print(f"Warning: No matching TIFF file found for {img_path.name}, skipping...")
                continue

            with rasterio.open(corresponding_tiff) as src:
                meta = src.meta.copy()
                meta.update({"dtype": 'uint8', "count": 1, "nodata": 255}) 
                
            # output_tiff_path = Path(path_manager.predictions_tiff_refine_folder, f"{img_path.stem}_prediction.tif")
            # with rasterio.open(output_tiff_path, 'w', **meta) as dst:
            #     mask_sam = np.where(mask_sam == 0, 255, mask_sam)
            #     dst.write(mask_sam, 1)

            output_tiff_path = Path(path_manager.predictions_tiff_base_folder, f"{img_path.stem}_prediction.tif")
            with rasterio.open(output_tiff_path, 'w', **meta) as dst:
                mask = np.where(mask == 0, 255, mask)
                dst.write(mask, 1)    

            # # Open the input TIFF file
            # output_path = Path(path_manager.predictions_png_base_folder, f"{img_path.stem}_prediction.png")

            # src_ds = gdal.Open(output_tiff_path)
            # if src_ds:
            #     # Read raster data as an array
            #     raster_data = src_ds.ReadAsArray().astype(np.uint8)  # Ensure float values
    
            #     # Save as grayscale PNG (mode "L")
            #     image = Image.fromarray(raster_data, mode="L")
            #     image.save(output_path)    
