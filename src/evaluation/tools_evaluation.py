import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import geopandas as gpd
from pathlib import Path
from shapely.geometry import box
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.windows import Window

def clip_ortho_on_test_zone(ortho_path: Path, drone_zone_path: Path, output_clip_path: Path) -> bool:
    """Clip the orthophoto on test zone. Return polygon because he is in the same crs as the ortho."""
    
    if not ortho_path.exists(): raise FileNotFoundError(f"{ortho_path} not found")
    if not drone_zone_path.exists(): raise FileNotFoundError(f"{drone_zone_path} not found")
    
    polygon_gdf = gpd.read_file(drone_zone_path)

    with rasterio.open(ortho_path) as src:    
        # Reproject polygon to avoid errors.
        polygon_gdf = polygon_gdf.to_crs(src.crs)
        polygon_gdf = polygon_gdf.buffer(2) # Buffer add N meters around the shape.

        # Check if polygon overlaps with raster bounds
        raster_bounds_geom = box(*src.bounds)
        if not polygon_gdf.geometry.unary_union.intersects(raster_bounds_geom):
            print("The polygon does not overlap with the orthophoto raster extent.")
            return False
        
        out_image, out_transform = rio_mask(src, polygon_gdf.geometry, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

    with rasterio.open(output_clip_path, "w", **out_meta) as dst:
        dst.write(out_image)
    
    return True

def split_ortho_into_tiles(clipped_ortho_path: Path, tile_size: int, ho: int, vo: int, tiles_folder: Path) -> None:
    """Split ortho into tiles."""

    if not clipped_ortho_path.exists(): raise FileNotFoundError(f"{clipped_ortho_path} not found")
    tiles_folder.mkdir(exist_ok=True, parents=True)

    with rasterio.open(clipped_ortho_path) as src:
        meta = src.meta.copy()
        step_x = int(tile_size * (1 - ho))
        step_y = int(tile_size * (1 - vo))

        for x in range(0, src.width - tile_size + 1, step_x):
            for y in range(0, src.height - tile_size + 1, step_y):
                window = Window(x, y, tile_size, tile_size)
                data = src.read(window=window)
                transform = rasterio.windows.transform(window, src.transform)
                tile_meta = meta.copy()
                tile_meta.update({
                    "height": tile_size,
                    "width": tile_size,
                    "transform": transform
                })
                tile_path = Path(tiles_folder, f"tile_{x}_{y}.tif")
                with rasterio.open(tile_path, 'w', **tile_meta) as dst:
                    dst.write(data)


def convert_tiles_into_png(tiles_folder: Path, images_folder: Path) -> None:
    """Convert tiles into png."""

    if not tiles_folder.exists(): raise FileNotFoundError(f"{tiles_folder} not found")
    images_folder.mkdir(exist_ok=True, parents=True)

    for tif_file in tiles_folder.glob("*.tif"):
        with rasterio.open(tif_file) as src:
            rgb = np.transpose(src.read([1, 2, 3]), (1, 2, 0))
            img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
            img.save(images_folder / (tif_file.stem + ".png"))


def perform_inference(model_path: Path, png_folder: Path, output_folder: Path, base_model_name: str) -> dict:
    
    if not model_path.exists(): raise FileNotFoundError(f"{model_path} not found")
    if not png_folder.exists(): raise FileNotFoundError(f"{png_folder} not found")
    output_folder.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SegformerForSemanticSegmentation.from_pretrained(model_path).to(device)
    processor = AutoImageProcessor.from_pretrained(base_model_name, do_reduce_labels=False, use_fast=False)

    for png_file in tqdm(sorted(png_folder.glob("*.png")), desc="Predicting"):
        name = png_file.stem
        _, x, y = name.split("_")
        x, y = int(x), int(y)

        img = Image.open(png_file).convert("RGB")
        inputs = processor(img, return_tensors="pt").to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            logits = torch.nn.functional.interpolate(logits, size=img.size, mode="bilinear", align_corners=False)
            pred_mask = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8) + 1 # Add one to get value between 1 and 5

        for cls_id in np.unique(pred_mask):
            binary_mask = (pred_mask == cls_id).astype(np.uint8)
            np.save(output_folder / f"class_{cls_id}_tile_{x}_{y}.npy", binary_mask)

    return model.config.id2label


def merge_predictions(clip_ortho_path: Path, prediction_folder: Path, outputfilename: Path, tile_size: int) -> None:

    if not clip_ortho_path.exists(): raise FileNotFoundError(f"{clip_ortho_path} not found")
    if not prediction_folder.exists(): raise FileNotFoundError(f"{prediction_folder} not found")

    with rasterio.open(clip_ortho_path) as src:
        meta = src.meta.copy()
        height, width = src.height, src.width

    all_classes = sorted(set(int(f.name.split("_")[1]) for f in prediction_folder.glob("class_*.npy")))
    class_index_map = {cls: i for i, cls in enumerate(all_classes)}
    vote_stack = np.zeros((height, width, len(all_classes)), dtype=np.uint16)

    for npy_file in tqdm(sorted(prediction_folder.glob("*.npy")), desc="Voting"):
        parts = npy_file.stem.split("_")
        cls_id, x, y = int(parts[1]), int(parts[3]), int(parts[4])
        idx = class_index_map[cls_id]
        data = np.load(npy_file)
        vote_stack[y:y+tile_size, x:x+tile_size, idx] += data

    final_mask = np.argmax(vote_stack, axis=-1) + 1
    final_mask[np.sum(vote_stack, axis=-1) == 0] = 0

    meta.update({"count": 1, "dtype": "uint8", "nodata": 0})
    with rasterio.open(outputfilename, "w", **meta) as dst:
        dst.write(final_mask.astype(np.uint8), 1)


def resize_merged_raster(merged_pred_path: Path, annotation_mask_png_path: Path, output_png_path: Path, drone_polygon_path: Path) -> None:

    if not merged_pred_path.exists(): raise FileNotFoundError(f"{merged_pred_path} not found")
    if not annotation_mask_png_path.exists(): raise FileNotFoundError(f"{annotation_mask_png_path} not found")

    drone_polygon_gdf = gpd.read_file(drone_polygon_path)
    with rasterio.open(merged_pred_path) as src:
        drone_polygon_gdf = drone_polygon_gdf.to_crs(src.crs)
        clipped_data, clipped_transform = rio_mask(src, [geom for geom in drone_polygon_gdf.geometry], crop=True)
        clipped_meta = src.meta.copy()
        clipped_meta.update({
            "height": clipped_data.shape[1],
            "width": clipped_data.shape[2],
            "transform": clipped_transform
        })

    gt_array = np.array(Image.open(annotation_mask_png_path))
    pred_array = clipped_data[0]
    if pred_array.shape != gt_array.shape:
        resized = Image.fromarray(pred_array.astype(np.uint8), mode="L").resize(
            (gt_array.shape[1], gt_array.shape[0]), resample=Image.NEAREST
        )
        final_resized_array = np.array(resized).astype(np.uint8)
    else:
        final_resized_array = pred_array

    clipped_meta.update({
        "height": gt_array.shape[0],
        "width": gt_array.shape[1]
    })

    Image.fromarray(final_resized_array, mode='L').save(output_png_path)