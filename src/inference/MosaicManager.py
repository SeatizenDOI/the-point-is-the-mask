import math
import numpy as np
from tqdm import tqdm
from pathlib import Path

import rasterio
from rasterio.merge import merge
from rasterio.transform import Affine
from rasterio.windows import from_bounds

from .PathRasterManager import PathRasterManager
from ..utils.raster_constants import RASTER_CLASS_ID2COLOR, NO_DATA_VALUE

class MosaicManager:
    def __init__(self, path_manager: PathRasterManager, id2label: dict, max_pixels_by_slice: int = 800000000):
        self.tmp_rasters_slice: list[Path] = []
        self.path_manager = path_manager
        self.max_pixels_by_slice = max_pixels_by_slice
        self.id2label = id2label

        self.predictions_tiff_files = sorted([f for f in list(self.path_manager.predictions_tiff_folder.iterdir()) if f.suffix.lower() in [".tif"]])
        
        with rasterio.open(self.predictions_tiff_files[0]) as src:
            self.crs = src.crs

        self.global_min, self.global_max, self.num_classes = min(id2label), max(id2label), len(id2label)
        print(f"✅ Detected class range: {self.global_min} to {self.global_max} ({self.num_classes} classes)")


    def build_raster(self):
        tiles_with_transforms = self.create_intermediate_subraster()
        self.populate_and_save_subraster(tiles_with_transforms)
        self.create_final_rasters()


    def create_intermediate_subraster(self) -> list:
          
        # Merge only the currently opened files
        origin_mosaic, origin_transform = merge(self.predictions_tiff_files, method="first")
        
        # Get the total size of the mosaic
        nb_class, height, width = origin_mosaic.shape[0], origin_mosaic.shape[1], origin_mosaic.shape[2]
        intermediate_tile_height = self.max_pixels_by_slice // (width * nb_class)
        nb_slice = math.ceil(height / intermediate_tile_height)

        print(f"The final raster size is {origin_mosaic.shape}. It will be cut by {nb_slice} slice of {intermediate_tile_height} pixels.")

        # Loop through and extract tiles
        mosaic_tiles = []
        for i in range(0, height, intermediate_tile_height):
            # Define the window, making sure it doesn't exceed bounds
            win_height = min(intermediate_tile_height, height - i)

            # Extract the tile
            tile = origin_mosaic[:, i:i+win_height, :]
            new_transform = origin_transform * Affine.translation(1, i)

            mosaic_tiles.append((tile, new_transform))  # Store tile with position
        
        return mosaic_tiles


    def populate_and_save_subraster(self, tiles_with_transforms: list):

        for i, (mosaic, out_trans) in enumerate(tiles_with_transforms):
            tmp_path = Path(self.path_manager.merged_predictions_folder, f"{i}_{self.path_manager.final_merged_tiff_file.name}") 

            # Optimized Argmax Calculation
            most_common_values = np.full(mosaic.shape[1:], NO_DATA_VALUE, dtype=np.uint8)  # Default to NO_DATA_VALUE

            count_buffer = np.zeros((self.num_classes, *mosaic.shape[1:]), dtype=np.uint16)  # Avoid large int types

            for src_path in tqdm(self.predictions_tiff_files, desc=f"Processing tiles for subraster {i}", unit="file"):
                with rasterio.open(src_path) as src:
                    tile_data = src.read(1)
                    window = from_bounds(*src.bounds, transform=out_trans).round_offsets().round_lengths()
                
                # Get offsets
                row_off, col_off = int(window.row_off), int(window.col_off)

                # Handle Negative Offsets (for overlapping images)
                row_start_tile = max(0, -row_off)  
                col_start_tile = max(0, -col_off)

                row_off = max(0, row_off)  # Adjust offset to fit inside mosaic
                col_off = max(0, col_off)

                row_end = min(row_off + tile_data.shape[0] - row_start_tile, most_common_values.shape[0])
                col_end = min(col_off + tile_data.shape[1] - col_start_tile, most_common_values.shape[1])

                tile_height = row_end - row_off
                tile_width = col_end - col_off

                if tile_height <= 0 or tile_width <= 0:
                    continue  # Skip tiles that are completely outside

                # **Crop tile_data properly for out-of-bounds cases**
                tile_data = tile_data[row_start_tile:row_start_tile + tile_height, col_start_tile:col_start_tile + tile_width]

                # Update class frequencies, ensuring bounds are correct
                for v in range(self.global_min, self.global_max + 1):
                    mask = (tile_data == v)  
                    count_buffer[v - self.global_min, row_off:row_end, col_off:col_end] += mask

            # Faster argmax using efficient NumPy operations
            valid_pixel_mask = count_buffer.sum(axis=0) > 0  # Avoid unnecessary computation
            most_common_values[valid_pixel_mask] = count_buffer[:, valid_pixel_mask].argmax(axis=0) + self.global_min

            # Save merged raster
            with rasterio.open(
                tmp_path,
                "w",
                driver="GTiff",
                height=most_common_values.shape[0],
                width=most_common_values.shape[1],
                count=1,
                dtype=np.uint8,
                crs=self.crs,
                transform=out_trans,
                compress="LZW",
                nodata=NO_DATA_VALUE,
            ) as dst:
                dst.write(most_common_values, 1)

            self.tmp_rasters_slice.append(tmp_path)


    def create_final_rasters(self):
        print("*\t Create the final raster.")

        if len(self.tmp_rasters_slice) == 1:
            Path.rename(self.tmp_rasters_slice[0], self.path_manager.final_merged_tiff_file)

            with rasterio.open(self.path_manager.final_merged_tiff_file, 'r+') as src:
                
                # Apply the colormap to band 1
                src.write_colormap(1, RASTER_CLASS_ID2COLOR)
            return
        
        
        mosaic, out_trans = merge(self.tmp_rasters_slice, method="first")

        # Save the final merged raster
        with rasterio.open(
            self.path_manager.final_merged_tiff_file,
            "w",
            driver="GTiff",
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            count=1,
            dtype=np.uint8,
            crs=self.crs,
            transform=out_trans,
            compress="LZW",
            nodata=NO_DATA_VALUE
        ) as dst:
            dst.write(mosaic[0, :], 1)
         
            dst.write_colormap(1, RASTER_CLASS_ID2COLOR)

        # Clean up temporary files
        for temp_tiff in self.tmp_rasters_slice:
            temp_tiff.unlink()