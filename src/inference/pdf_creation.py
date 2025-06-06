"""
    Tools to generate the pdf after inference.
"""

import shutil
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from shapely import Polygon
from multiprocessing import Pool, cpu_count

import numpy as np
from numpy.typing import NDArray

import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, Resampling as WarpResampling

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages

from ..utils.raster_constants import RASTER_CLASS_ID2COLOR, RASTER_CLASS_ID2LABEL
from ..utils.tiles_tools import align_annotation_to_ortho, extract_one_tile, convert_one_tiff_to_png


TILE_SIZE = 2048
NUM_WORKERS = max(1, cpu_count() - 2)  # Use available CPU cores, leaving some free
FIGSIZE = (8.27, 11.69)  # A4 portrait


def load_downsampled_rgb_and_transform(ortho_path: Path, scale=0.05) -> tuple[NDArray, any, int, int]:
    """ Load the orthophoto and downsampled the data."""
    with rasterio.open(ortho_path) as src:
        new_height = int(src.height * scale)
        new_width = int(src.width * scale)
        transform = src.transform * src.transform.scale(
            (src.width / new_width),
            (src.height / new_height)
        )
        rgb = src.read(
            [1, 2, 3],
            out_shape=(3, new_height, new_width),
            resampling=Resampling.average
        )
        rgb = np.transpose(rgb, (1, 2, 0))  # HWC
        rgb = np.clip(rgb / 255.0, 0, 1)
    return rgb, transform, new_width, new_height


def resample_seg_to_rgb(seg_path: Path, dst_shape: tuple[int, int], dst_transform, dst_crs) -> NDArray:
    """ Load the segmentation mask as rgb to be print in pdf."""
    with rasterio.open(seg_path) as src:
        src_seg = src.read(1)
        dst_seg = np.zeros(dst_shape, dtype=np.uint8)

        reproject(
            source=src_seg,
            destination=dst_seg,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=WarpResampling.nearest
        )
    return dst_seg


def create_fullmap_thumbnail_with_predictions(ortho_path: Path, pred_path: Path) -> tuple[NDArray, NDArray]:
    thumb_rgb, rgb_transform, width, height = load_downsampled_rgb_and_transform(ortho_path)
    with rasterio.open(ortho_path) as ortho_src:
        seg_resampled = resample_seg_to_rgb(pred_path, (height, width), rgb_transform, ortho_src.crs)
    
    return thumb_rgb, seg_resampled


def create_predictions_patch(ortho_path: Path, pred_path: Path, tmp_folder: Path, ortho_footprint: Polygon) -> Path:
    """ Blend ortho and predictions into small patches to easily vizualize the results. """
    
    if tmp_folder.exists():
        shutil.rmtree(tmp_folder)
    tmp_folder.mkdir(exist_ok=True, parents=True)
    
    aligned_pred_path = Path(tmp_folder, f"{pred_path.stem}_aligned.tif")
    align_annotation_to_ortho(pred_path, aligned_pred_path, ortho_path)

    OUTPUT_FOLDER_TIF = Path(tmp_folder, "tif")
    if OUTPUT_FOLDER_TIF.exists():
        shutil.rmtree(OUTPUT_FOLDER_TIF)
    OUTPUT_FOLDER_TIF.mkdir(parents=True)

    OUTPUT_FOLDER_PNG = Path(tmp_folder, "png")
    if OUTPUT_FOLDER_PNG.exists():
        shutil.rmtree(OUTPUT_FOLDER_PNG)
    OUTPUT_FOLDER_PNG.mkdir(parents=True)

    OUTPUT_FOLDER_FINAL = Path(tmp_folder, "final")
    if OUTPUT_FOLDER_FINAL.exists():
        shutil.rmtree(OUTPUT_FOLDER_FINAL)
    OUTPUT_FOLDER_FINAL.mkdir(parents=True)

    print("*\t Splitting ortho into tiles.")
    with rasterio.open(ortho_path) as ortho:

        tile_coords = [
            (
                x, 
                y, 
                TILE_SIZE, 
                ortho_footprint,
                ortho_path, 
                aligned_pred_path,
                Path(OUTPUT_FOLDER_TIF, f"tile_{x}_{y}.tif"),
                Path(OUTPUT_FOLDER_TIF, f"tile_{x}_{y}_pred.tif")
            )
            for x in range(0, ortho.width - TILE_SIZE + 1, TILE_SIZE) 
            for y in range(0, ortho.height - TILE_SIZE + 1, TILE_SIZE)
        ]

    with Pool(NUM_WORKERS) as pool:
        list(tqdm(pool.imap_unordered(extract_one_tile, tile_coords), total=len(tile_coords), desc=f"Processing {ortho_path.name}"))

    print("*\t Converting tiles into png.")

    filepaths = [(filepath, OUTPUT_FOLDER_PNG, False) for filepath in OUTPUT_FOLDER_TIF.iterdir()]
    with Pool(processes=cpu_count()) as pool:
        list(tqdm(pool.imap(convert_one_tiff_to_png, filepaths, ), total=len(filepaths), desc=f"Processing {OUTPUT_FOLDER_TIF.name}"))
    
    
    print("*\t Blending ortho tiles and predictions.")
    cpt = 0
    for img in tqdm(list(OUTPUT_FOLDER_PNG.iterdir()), total=20):
        if cpt >= 10: break
        if "_pred" in img.name: continue
        cpt += 1
        pred = Path(img.parent, f"{img.stem}_pred.png")

        # === Step 1: Load base image (background) ===
        background = Image.open(img)

        # === Step 2: Load the overlay label mask (values 1 to 5) ===
        overlay = np.array(Image.open(pred)) 
        
        overlay_rgb = np.zeros((*overlay.shape, 3), dtype=np.uint8)

        for label, color in RASTER_CLASS_ID2COLOR.items():
            overlay_rgb[overlay == label] = color[0:3]

        # Save the result (optional)
        overlay_colored = Image.fromarray(overlay_rgb)
        blended = Image.blend(background, overlay_colored, alpha=0.2)

        # Save the result
        blended.save(Path(OUTPUT_FOLDER_FINAL, f"{img.stem}_final.png"),  optimize=True, compress_level=9)
    
    return OUTPUT_FOLDER_FINAL


def plot_patches(pdf: PdfPages, ortho_path: Path, pred_path: Path, output_folder: Path, ortho_footprint: Polygon) -> None:
    """ Plot patches. """
    blend_prediction_folder = create_predictions_patch(ortho_path, pred_path, output_folder, ortho_footprint)

    for i, file in enumerate(list(blend_prediction_folder.iterdir())):
        if i > 10: break
        img = mpimg.imread(file)
        
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.imshow(img)
        ax.set_title(f"{file.name}", fontsize=12, fontweight="bold")
        ax.axis('off')  # Hide axes
        
        pdf.savefig(fig)
        plt.close(fig)
    
    shutil.rmtree(output_folder)


def plot_legend(ax_legend: plt.Axes) -> None:
    """Plot the segmentation legend with the associated color."""
    
    ax_legend.set_title("Segmentation Classes by Color", fontsize=20, fontweight='bold')

    for i, (index, label) in enumerate(list(RASTER_CLASS_ID2LABEL.items())[::-1]):
        rgba = np.array(RASTER_CLASS_ID2COLOR.get(index, (0, 0, 0, 255))) / 255.0
        ax_legend.add_patch(plt.Rectangle((0, i*0.6), 0.6, 0.6, color=rgba, ec="black"))
        ax_legend.text(0.7, i*0.6 + 0.3, label, va='center', fontsize=12)

    ax_legend.set_xlim(0, 2)
    ax_legend.set_ylim(0, len(RASTER_CLASS_ID2LABEL))
    ax_legend.set_frame_on(False)
    ax_legend.axis('off')


def plot_distribution(ax_stats: plt.Axes, cmap: ListedColormap, seg_resampled: NDArray) -> None:
    """Plot the distribution of each class in a pie chart."""
    colors = [cmap(int(label)) for label in list(RASTER_CLASS_ID2COLOR)]
    unique, counts = np.unique(seg_resampled, return_counts=True)
    labels = [RASTER_CLASS_ID2LABEL[i] for i in unique[1:]]
    total = counts[1:].sum()
    percentages = [100 * count / total for count in counts[1:]]
    
    ax_stats.set_title("Predicted Class Distribution", fontsize=20, fontweight="bold")
    ax_stats.axis('equal')
    ax_stats.axis('off')


    radius_pie = 0.8
    wedges, texts = ax_stats.pie(
        counts[1:], wedgeprops=dict(width=0.3), radius=radius_pie, 
        startangle=60, colors=colors
    )

    kw = dict(arrowprops=dict(arrowstyle="-"), zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = f"angle,angleA=0,angleB={ang}"
        kw["arrowprops"].update({"connectionstyle": connectionstyle})

        label_text = f"{labels[i]} ({percentages[i]:.1f}%)"

        arrow_start_radius = radius_pie * 0.9 
        text_radius = radius_pie * 1.2 

        ax_stats.annotate(
            label_text,
            xy=(arrow_start_radius * x, arrow_start_radius * y),
            xytext=(text_radius * x, text_radius * y), 
            horizontalalignment=horizontalalignment,
            **kw
        )


def plot_distribution_and_legend(pdf: PdfPages, cmap: ListedColormap, seg_resampled: NDArray) -> None:
    """Combine the distribution and legend plot on one pdf page."""
    fig = plt.figure(figsize=FIGSIZE)
    gs = GridSpec(2, 1, figure=fig)  # 10 rows, 1 column grid
    
    ax_stats = fig.add_subplot(gs[0, 0])
    plot_distribution(ax_stats, cmap, seg_resampled)

    ax_legend = fig.add_subplot(gs[1, 0])
    plot_legend(ax_legend)
    
    pdf.savefig(fig)
    plt.close(fig)


def plot_first_page(pdf: PdfPages, session_name: str, model_name: str, ortho_rgb: NDArray, seg_rgb: NDArray, cmap: ListedColormap) -> None:
    fig = plt.figure(figsize=FIGSIZE)
    fig.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95)
    gs = GridSpec(15, 1, figure=fig)

    # Title section
    ax_title = fig.add_subplot(gs[0, 0])
    ax_title.axis('off')
    ax_title.text(0.5, 0.9, "Segmentation Report", ha='center', va='center', fontsize=20)
    ax_title.text(0.5, 0.4, f"Session: {session_name}", ha='center', va='center', fontsize=10)
    ax_title.text(0.5, 0.1, f"Segmentation model: {model_name}", ha='center', va='center', fontsize=10)

    # Overlay section
    ax_overlay = fig.add_subplot(gs[1:, 0])
    ax_overlay.imshow(ortho_rgb)
    ax_overlay.imshow(seg_rgb, cmap=cmap, alpha=0.4, interpolation='none')
    ax_overlay.set_title("Segmentation Overlay")
    ax_overlay.axis('off')

    pdf.savefig(fig)
    plt.close(fig)