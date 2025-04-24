from pathlib import Path
import numpy as np
from shapely import Polygon, GeometryCollection, box
from multiprocessing import Pool, cpu_count
import rasterio
import rasterio.warp
from rasterio.windows import Window
from tqdm import tqdm
from osgeo import gdal
from PIL import Image
import geopandas as gpd
from shapely.ops import transform, unary_union
from pyproj import Transformer

from .ConfigParser import ConfigParser
from .PathManager import PathManager
from .UAVManager import UAVManager

from .utils.underwater_correction import apply_filter, get_color_filter_matrix

NUM_WORKERS = max(1, cpu_count() - 2)  # Use available CPU cores, leaving some free
gdal.DontUseExceptions()

class TileManager:

    def __init__(self, cp: ConfigParser, pm: PathManager):
        self.cp = cp
        self.pm = pm


    def create_tiles_and_annotations(self, uav_manager: UAVManager) -> bool:

        print("\n\n------ [TILES - Create ortho and annotation tiles] ------\n")

        len_cropped_folder = len(list(self.pm.cropped_ortho_tif_folder.iterdir()))
        len_upsampled_anno_folder = len(list(self.pm.upsampled_annotation_tif_folder.iterdir())) 

        if len_cropped_folder == len_upsampled_anno_folder and len_cropped_folder != 0:
            print("Tiff tiles and annotations already exist. we don't split.")
            return False

        for raster_anno in self.pm.asv_coarse_folder.iterdir():

            print(f"\n-- Working with {raster_anno.name} --")

            # Extract raster annotation place to get the correspondant uav orthophoto
            place_with_country_code = raster_anno.name.split("_")[1]
            session_name = raster_anno.name.replace("_classification.tif", "")
            orthos_information = uav_manager.get_orthos_from_place_with_country_code(place_with_country_code)
            
            for ortho_path, ortho_crs, height, width in orthos_information:

                self.reproject_raster_if_needed(raster_anno, ortho_crs, raster_anno)

                valid_polygon = self.get_valid_polygon_from_raster(raster_anno)

                tile_coords = [(session_name, x, y, ortho_path, valid_polygon, raster_anno) 
                    for x in range(0, width - self.cp.get_tile_size() + 1, self.cp.get_horizontal_step()) 
                    for y in range(0, height - self.cp.get_tile_size() + 1, self.cp.get_vertical_step())]
                
                # Process tiles in parallel
                with Pool(NUM_WORKERS) as pool:
                    list(tqdm(pool.imap_unordered(self.process_tile, tile_coords), total=len(tile_coords), desc="Processing Tiles"))

        return True
    
    
    def get_valid_polygon_from_raster(self, raster_path: Path) -> GeometryCollection:
        """ """
        if not raster_path.exists() or not raster_path.is_file():
            raise FileNotFoundError(f"Cannot extract valid polygon from {raster_path}. File not found")
        
        with rasterio.open(raster_path) as raster:
            annotation_data = raster.read(1, masked=True)
            annotation_transform = raster.transform

        valid_indices = np.where(~annotation_data.mask)
        valid_coords = [rasterio.transform.xy(annotation_transform, row, col, offset="center") for row, col in zip(*valid_indices)]
        valid_polygon = Polygon(valid_coords).convex_hull
        
        return valid_polygon
    

    def reproject_raster_if_needed(self, input_raster_path: Path, target_crs, output_raster_path: Path) -> None:

        with rasterio.open(input_raster_path) as src:
            print(f"Annotation raster crs is {src.crs} and target crs is {target_crs}")
            if src.crs == target_crs: return # No reprojection needed

            print(f"Reprojecting {input_raster_path} to match CRS {target_crs}")

            transform, width, height = rasterio.warp.calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )
            # extract class names from old annotation raster
            class_mapping = src.tags()
            class_names = {str(k): v for k, v in class_mapping.items() if k.isdigit()}
            print(f"Class names: {class_names}")

            # Update metadata for reprojected raster
            kwargs = src.meta.copy()
            kwargs.update({
                "crs": target_crs,
                "transform": transform,
                "width": width,
                "height": height
            })

            # Reproject and save
            with rasterio.open(output_raster_path, "w", **kwargs) as dst:
                for i in range(1, src.count + 1):
                    rasterio.warp.reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=rasterio.enums.Resampling.nearest
                    )
                
                dst.update_tags(**class_names)


    def process_tile(self, args: tuple) -> None:
        
        session_name, tile_x, tile_y, orthophoto_path, valid_polygon, annotation_path = args
        tile_size = self.cp.get_tile_size()

        # Open orthophoto inside the worker
        with rasterio.open(orthophoto_path) as ortho:
            # Define tile window with overlap applied
            window = Window(tile_x, tile_y, tile_size, tile_size)
            
            # Define tile transform
            tile_transform = rasterio.windows.transform(window, ortho.transform)

            # Check if tile bounds are fully contained within the valid polygon
            tile_bounds = box(*rasterio.windows.bounds(window, ortho.transform))

            # Skip tiles that are not fully contained within the annotation polygon
            if not valid_polygon.contains(tile_bounds): return  
            
            tile_ortho = ortho.read(window=window)

            # Apply threshold to filter out mostly black or white tiles
            greyscale_tile = np.sum(tile_ortho, axis=0) / 3
            
            # Black threshold.
            percentage_black_pixel = np.sum(greyscale_tile == 0) * 100 / tile_size**2
            if percentage_black_pixel > 5: return
            
            # White threshold.
            percentage_white_pixel = np.sum(greyscale_tile == 255) * 100 / tile_size**2
            if percentage_white_pixel > 10: return

            # Get tile meta.
            tile_meta = ortho.meta.copy()
            tile_meta.update({
                "height": tile_size,
                "width": tile_size,
                "transform": tile_transform
            })


        # Open annotation raster inside the worker
        with rasterio.open(annotation_path) as annotation:
            annotation_data = annotation.read(1, masked=True)
            annotation_transform = annotation.transform

            # Generate the corresponding annotation for this tile
            upsampled_annotation = np.full((tile_size, tile_size), np.nan, dtype=np.float32)

            for row in range(tile_size):
                for col in range(tile_size):
                    # Convert (row, col) from tile ortho space to annotation coordinates
                    x, y = rasterio.transform.xy(tile_transform, row, col, offset="center")

                    # Map geographic coordinates (x, y) to annotation pixel indices
                    annotation_row, annotation_col = rasterio.transform.rowcol(annotation_transform, x, y)

                    # Ensure the annotation pixel is within bounds
                    if annotation_row < 0 or annotation_row >= annotation_data.shape[0] or \
                        annotation_col < 0 or annotation_col >= annotation_data.shape[1]:
                        return
                    
                    annotation_value = annotation_data[annotation_row, annotation_col]

                    # If the annotation value is NaN, skip the entire tile
                    if np.isnan(annotation_value) or annotation_value == 0: return

                    # Assign annotation value if it's valid
                    upsampled_annotation[row, col] = annotation_value
            
            upsampled_annotation_meta = annotation.meta.copy()
            upsampled_annotation_meta.update({
                "driver": "GTiff",
                "height": tile_size,
                "width": tile_size,
                "transform": tile_transform
            })
        

        # Save the tile
        tile_output_path = Path(self.pm.cropped_ortho_tif_folder, f"{session_name}_{tile_x}_{tile_y}.tif")
        with rasterio.open(tile_output_path, "w", **tile_meta) as dest:
            dest.write(tile_ortho)


        # Save the upsampled annotation file for this tile
        annotation_output_path = Path(self.pm.upsampled_annotation_tif_folder, f"{session_name}_{tile_x}_{tile_y}.tif")
        with rasterio.open(annotation_output_path, "w", **upsampled_annotation_meta) as dest:
            dest.write(upsampled_annotation, 1)
    

    def convert_tiff_to_png(self, default_uav_crs) -> list:
        print("\n\n------ [TILES - Convert ortho tiff tiles to png] ------\n")
        
        # From drone zone, we try to extract images
        all_drone_test_polygon = []
        for drone_test_zone in self.cp.get_drone_zone_polygon_path():

            # Load the boundary shapefile as a GeoDataFrame
            drone_test_polygon_gpd = gpd.read_file(drone_test_zone)

            # Convert it to a single Shapely Polygon
            drone_test_polygon = drone_test_polygon_gpd.unary_union.convex_hull


            # Check if reprojection is needed
            if default_uav_crs != drone_test_polygon_gpd.crs:

                # Create a transformer for reprojecting
                transformer = Transformer.from_crs(drone_test_polygon_gpd.crs, default_uav_crs, always_xy=True)
                
                # Correctly reproject the Polygon
                drone_test_polygon = transform(transformer.transform, drone_test_polygon)
            
            all_drone_test_polygon.append(drone_test_polygon)
        drone_test_footprint = unary_union(all_drone_test_polygon)
        
        filepaths, test_images_list = [], []
        for file in self.pm.cropped_ortho_tif_folder.iterdir():
            if file.suffix.lower() != ".tif": continue

            tile_bounds = box(*rasterio.open(file).bounds)

            if drone_test_footprint.intersects(tile_bounds):
                test_images_list.append(file.stem)
                
            output_dir = self.pm.test_images_folder if file.stem in test_images_list else self.pm.train_images_folder
            filepaths.append((file, output_dir))
        

        with Pool(processes=cpu_count()) as pool:
            list(tqdm(pool.imap(self.convert_one_tiff_to_png, filepaths), total=len(filepaths), desc=f"Processing {self.pm.cropped_ortho_tif_folder.name}"))

        
        return test_images_list


    def convert_tiff_to_png_annotations(self, test_images_list: list) -> None:
        print("\n\n------ [TILES - Convert anno tiff tiles to png] ------\n")

        filepaths = []
        for file in self.pm.upsampled_annotation_tif_folder.iterdir():
            if file.suffix.lower() != ".tif": continue

            # Determine output folder based on test session
            output_dir = self.pm.test_annotation_folder if file.stem in test_images_list else self.pm.train_annotation_folder

            filepaths.append((file, output_dir))

        with Pool(processes=cpu_count()) as pool:
            list(tqdm(pool.imap(self.convert_one_tiff_to_png, filepaths), total=len(filepaths), desc=f"Processing {self.pm.upsampled_annotation_tif_folder.name}"))


    def convert_one_tiff_to_png(self, filepath_and_output_dir: tuple[Path, Path]) -> None:
        filepath, output_dir = filepath_and_output_dir
        png_output_path = Path(output_dir, f'{filepath.stem}.png')
        
        with gdal.Open(str(filepath)) as src_ds:

            raster_data = src_ds.ReadAsArray()
            if raster_data.ndim == 3:
                image = Image.fromarray(np.transpose(raster_data[:3], (1, 2, 0)).astype(np.uint8), mode="RGB")
                if self.cp.use_color_correction():
                    pixels = np.array(image, dtype=np.uint8)
                    height, width = pixels.shape[:2]  # Get image dimensions
                    filter = get_color_filter_matrix(pixels, width, height)
                    img_out = apply_filter(pixels, filter)
                    image = Image.fromarray(img_out, "RGB")
            elif raster_data.ndim == 2:
                image = Image.fromarray(raster_data.astype(np.uint8), mode="L")
            else:
                raise ValueError(f"Unexpected image format: {raster_data.shape}")
            image.save(png_output_path)
    

    def verify_if_annotation_tiles_contains_valid_values(self, classes_mapping: dict) -> None:

        print("\n\n------ [TILES - Annotations check values] ------\n")


        annotation_files = list(self.pm.train_annotation_folder.glob("*.png"))
        classes = [int(k) for k in classes_mapping.keys()]
        min_class, max_class = min(classes), max(classes)
        cpt_error = 0
        for file in tqdm(annotation_files, desc="Verify annotations file", unit="f"):
            
            image = Image.open(file).convert("L")
            data = np.array(image)
            invalid_mask = (data < min_class) | (data > max_class)
            invalid_values = data[invalid_mask]

            if invalid_values.size == 0: continue

            cpt_error += 1
            files_to_delete = [
                Path(self.pm.cropped_ortho_tif_folder, f"{file.stem}.tif"),
                Path(self.pm.upsampled_annotation_tif_folder, f"{file.stem}.tif"),
                file,
                Path(self.pm.train_images_folder, file.name)
            ]

            for file_path in files_to_delete:
                if file_path.exists():
                    file_path.unlink()
        
        print(f"On {len(annotation_files)} files, we have deleted {cpt_error} files")