import numpy as np
from PIL import Image
from osgeo import gdal
from pathlib import Path

from ..utils.underwater_correction import apply_filter, get_color_filter_matrix

gdal.DontUseExceptions()

def convert_one_tiff_to_png(args: tuple[Path, Path, bool]) -> None:
    """ Convert a image or an annotation from tif to png"""
    filepath, output_dir, use_underwater_correction = args
    png_output_path = Path(output_dir, f'{filepath.stem}.png')
    
    with gdal.Open(str(filepath)) as src_ds:

        raster_data = src_ds.ReadAsArray()
        if raster_data.ndim == 3:
            image = Image.fromarray(np.transpose(raster_data[:3], (1, 2, 0)).astype(np.uint8), mode="RGB")
            if use_underwater_correction:
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