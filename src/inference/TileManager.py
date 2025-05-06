import json
import pyproj
import numpy as np
from tqdm import tqdm
import geopandas as gpd
from pathlib import Path
from argparse import Namespace
from multiprocessing import Pool, cpu_count

import rasterio
from rasterio.windows import Window
from rasterio.transform import array_bounds

from shapely.ops import transform
from shapely.geometry import shape, box


from ..utils.tiles_tools import convert_one_tiff_to_png
from .PathRasterManager import PathRasterManager

NUM_WORKERS = max(1, cpu_count() - 2)  # Use available CPU cores, leaving some free

class TileManager:

    def __init__(self, opt: Namespace):
        self.opt = opt
        self.tile_size, self.hs, self.vs = 0, 0, 0
        self.geojson_datas = []

        self.setup()

    def setup(self) -> None:
        self.tile_size = self.opt.tile_size
        self.hs = int(self.tile_size * (1 - self.opt.horizontal_overlap)) # Horizontal step.
        self.vs = int(self.tile_size * (1 - self.opt.vertical_overlap)) # Vertical step.

        for geojson_str in self.opt.path_geojson:
            geojson_path = Path(geojson_str)
            if not geojson_path.exists() or not geojson_path.is_file(): continue
            self.geojson_datas.append(gpd.read_file(geojson_path)) 
            
        if len(self.geojson_datas) == 0:
            print("[WARNING] GeoJSON data - We don't crop the ortho with the geojson data due to no data.")


    def split_ortho_into_tiles(self, path_manager: PathRasterManager) -> None:
        print("*\t Splitting ortho into tiles.")
        with rasterio.open(path_manager.raster_path) as ortho:

            tile_coords = [
                (path_manager, x, y)
                for x in range(0, ortho.width - self.tile_size + 1, self.hs) 
                for y in range(0, ortho.height - self.tile_size + 1, self.vs)
            ]
            
        with Pool(NUM_WORKERS) as pool:
            list(tqdm(pool.imap_unordered(self.extract_one_tile, tile_coords), total=len(tile_coords), desc=f"Processing {path_manager.raster_path.name}"))


    def extract_one_tile(self, args: tuple[PathRasterManager, int, int]) -> None:
        path_manager, tile_x, tile_y = args

        orthoname = path_manager.raster_path.stem.replace("_ortho", "")

        with rasterio.open(path_manager.raster_path) as ortho:
            raster_crs = ortho.crs.to_string()  # Get CRS of the raster
            
            window = Window(tile_x, tile_y, self.tile_size, self.tile_size)
            tile_transform = rasterio.windows.transform(window, ortho.transform)
            tile_bounds = box(*array_bounds(self.tile_size, self.tile_size, tile_transform))
            
            for poly_gdf in self.geojson_datas:
                if poly_gdf.crs != raster_crs:
                    poly_gdf = poly_gdf.to_crs(raster_crs)
                
                if poly_gdf.intersects(tile_bounds).any():
                    break  # There is an intersection; keep the tile
            else:
                return  # No intersection.
    
            # Read raster data
            tile_ortho = ortho.read(window=window)

            # Apply threshold to filter out mostly black or white tiles
            greyscale_tile = np.sum(tile_ortho, axis=0) / 3
            
            # Black threshold.
            percentage_black_pixel = np.sum(greyscale_tile == 0) * 100 / self.tile_size**2
            if percentage_black_pixel > 5:
                return

            # White threshold.
            percentage_white_pixel = np.sum(greyscale_tile == 255) * 100 / self.tile_size**2
            if percentage_white_pixel > 10:
                return        

            tile_filename = f"{orthoname}_{tile_x}_{tile_y}.tif"
            tile_output_path = Path(path_manager.cropped_ortho_folder, tile_filename)

            tile_meta = ortho.meta.copy()
            tile_meta.update({
                "height": self.tile_size,
                "width": self.tile_size,
                "transform": tile_transform
            })

            with rasterio.open(tile_output_path, "w", **tile_meta) as dest:
                dest.write(tile_ortho)

    
    def convert_tiff_tiles_into_png(self, path_manager: PathRasterManager) -> None:
        print("*\t Convert ortho tiff tiles into png files.")
        ucc = bool(self.opt.underwater_color_correction)
        filepaths = [(filepath, path_manager.cropped_ortho_img_folder, ucc) for filepath in path_manager.cropped_ortho_folder.iterdir()]

        with Pool(processes=cpu_count()) as pool:
            list(tqdm(pool.imap(convert_one_tiff_to_png, filepaths, ), total=len(filepaths), desc=f"Processing {path_manager.cropped_ortho_folder.name}"))
