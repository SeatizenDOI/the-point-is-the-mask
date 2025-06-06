import tifffile
import numpy as np
from PIL import Image
from pathlib import Path
from shapely import Polygon, box

import rasterio
import rasterio.warp
from rasterio.windows import Window

from ..utils.underwater_correction import apply_filter, get_color_filter_matrix

def convert_one_tiff_to_png(args: tuple[Path, Path, bool]) -> None:
    """Convert an image or annotation from TIFF to PNG without using GDAL."""
    filepath, output_dir, use_underwater_correction = args
    png_output_path = Path(output_dir, f'{filepath.stem}.png')

    raster_data = tifffile.imread(filepath)

    if raster_data.ndim == 3:
        # TIFFs are typically in (height, width, channels) for tifffile
        if raster_data.shape[0] in (3, 4) and raster_data.shape[0] < raster_data.shape[2]:
            # Fix for (bands, H, W) layout
            raster_data = np.transpose(raster_data[:3], (1, 2, 0))

        image = Image.fromarray(raster_data[..., :3].astype(np.uint8), mode="RGB")

        if use_underwater_correction:
            pixels = np.array(image, dtype=np.uint8)
            height, width = pixels.shape[:2]
            filter = get_color_filter_matrix(pixels, width, height)
            img_out = apply_filter(pixels, filter)
            image = Image.fromarray(img_out, "RGB")

    elif raster_data.ndim == 2:
        image = Image.fromarray(raster_data.astype(np.uint8), mode="L")

    else:
        raise ValueError(f"Unexpected image format: {raster_data.shape}")

    image.save(png_output_path)


def align_annotation_to_ortho(input_annotation_path: Path, output_annotation_path: Path, ortho_path: Path) -> None:
    """ Align annotation raster with orthophoto. """
    with rasterio.open(ortho_path) as ortho, rasterio.open(input_annotation_path) as anno:
        if ortho.transform == anno.transform and ortho.crs == anno.crs and ortho.res == anno.res:
            return input_annotation_path
        
        print("Resampling and aligning annotation to match orthophoto...")

        dst_transform = ortho.transform
        dst_crs = ortho.crs
        dst_width = ortho.width
        dst_height = ortho.height

        aligned_data = np.empty((1, dst_height, dst_width), dtype=anno.dtypes[0])

        rasterio.warp.reproject(
            source=rasterio.band(anno, 1),
            destination=aligned_data[0],
            src_transform=anno.transform,
            src_crs=anno.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=rasterio.warp.Resampling.nearest,
        )

        # Copy metadata
        aligned_meta = ortho.meta.copy()
        aligned_meta.update({
            'count': 1,
            'dtype': anno.dtypes[0],
            "driver": "GTiff"
        })

    # Save to temporary aligned file
    with rasterio.open(output_annotation_path, 'w', compress="LZW", **aligned_meta) as dst:
        dst.write(aligned_data)


def extract_one_tile(args: tuple[int, int, int, Polygon, Path, Path, Path, Path]) -> None:
    """ Extract one tile from orthophoto with corresponding annotation"""
    
    tile_x, tile_y, tile_size, ortho_footprint, ortho_path, \
    anno_or_pred_path, tile_ortho_path, tile_pred_filepath   = args


    with rasterio.open(ortho_path) as ortho:
        
        window = Window(tile_x, tile_y, tile_size, tile_size)

        # Check if tile bounds are fully contained within the valid polygon
        tile_bounds = box(*rasterio.windows.bounds(window, ortho.transform))
        if not ortho_footprint.contains(tile_bounds):
            return
        
        # Read raster data
        tile_ortho = ortho.read(window=window)

        # Apply threshold to filter out mostly black or white tiles
        greyscale_tile = np.sum(tile_ortho, axis=0) / 3
        
        # Black threshold.
        percentage_black_pixel = np.sum(greyscale_tile == 0) * 100 / tile_size**2
        if percentage_black_pixel > 5:
            return

        # White threshold.
        percentage_white_pixel = np.sum(greyscale_tile == 255) * 100 / tile_size**2
        if percentage_white_pixel > 10:
            return        

        tile_transform = rasterio.windows.transform(window, ortho.transform)
        tile_meta = ortho.meta.copy()
        tile_meta.update({
            "height": tile_size,
            "width": tile_size,
            "transform": tile_transform
        })

        # Open annotation raster inside the worker
        with rasterio.open(anno_or_pred_path) as annotation:
  
            # Define tile transform
            anno_transform = rasterio.windows.transform(window, annotation.transform)

            tile_anno = annotation.read(window=window)

            # Step 4: Generate the corresponding annotation for this tile
            upsampled_annotation_meta = annotation.meta.copy()
            upsampled_annotation_meta.update({
                "driver": "GTiff",
                "height": tile_size,
                "width": tile_size,
                "transform": anno_transform
            })

        with rasterio.open(tile_ortho_path, "w", **tile_meta) as dest:
            dest.write(tile_ortho)
        
        with rasterio.open(tile_pred_filepath, "w", **upsampled_annotation_meta) as dest:
            dest.write(tile_anno[0, :], 1)