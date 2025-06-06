import shutil
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely import Point, Polygon

from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages

from ..utils.raster_constants import RASTER_CLASS_ID2COLOR
from .pdf_creation import create_fullmap_thumbnail_with_predictions, plot_patches, plot_distribution_and_legend, plot_first_page, FIGSIZE

class SeatizenSessionManager:

    def __init__(self, session_path: Path) -> None:
        
        self.session_path = session_path
        self.__ortho_path = None
        self.__new_prediction_raster_path = None
        self.__footprint_buffered = None


    @property
    def orthophoto_path(self) -> Path:
        
        if self.__ortho_path == None:
            self.__ortho_path = Path(self.session_path, "PROCESSED_DATA", "PHOTOGRAMMETRY", "odm_orthophoto", "odm_orthophoto.tif")

            if not self.__ortho_path.exists():
                raise FileNotFoundError(f"{self.__ortho_path} not found.")
        
        return self.__ortho_path


    @property
    def footprint_buffered(self) -> Polygon:

        if self.__footprint_buffered == None:
            metadata_path = Path(self.session_path, "METADATA", "metadata.csv")

            if not metadata_path.exists():
                raise FileNotFoundError(f"{metadata_path} not found.")

            df = pd.read_csv(metadata_path)

            # 2. Create Point geometry using longitude (x) and latitude (y)
            geometry = [Point(xy) for xy in zip(df["GPSLongitude"], df["GPSLatitude"])]
            
            # 3. Create GeoDataFrame with WGS84 (lat/lon) CRS
            gdf = gpd.GeoDataFrame(geometry=geometry, crs="EPSG:4326")

            # 4. Project to a metric CRS (e.g., UTM zone based on location)
            # Automatically determine the appropriate UTM zone
            centroid = gdf.unary_union.centroid
            utm_zone = int((centroid.x + 180) / 6) + 1
            utm_crs = f"EPSG:{32600 + utm_zone if centroid.y >= 0 else 32700 + utm_zone}"
            gdf_utm = gdf.to_crs(utm_crs)

            # 5. Compute convex hull (footprint) and area
            footprint = gdf_utm.unary_union.convex_hull
            area_m2 = footprint.area
            buffer_value = -20 if area_m2 > 50000 else -5 # For large area 50000 m^2, we crop 20 meters else 5
            self.__footprint_buffered = footprint.buffer(buffer_value)
        return self.__footprint_buffered


    def move_prediction_raster(self, prediction_raster: Path, model_name: str) -> None:

        IA_path = Path(self.session_path, "PROCESSED_DATA", "IA")
        IA_path.mkdir(exist_ok=True, parents=True)

        self.__new_prediction_raster_path = Path(IA_path, f"{self.session_path.name}_{model_name}_ortho_predictions.tif")
        shutil.move(prediction_raster, self.__new_prediction_raster_path)


    def create_resume_pdf(self, model_name: str, output_folder: Path) -> None:
        """Create a resume pdf from the prediction raster."""
        print("*\t Resume pdf generation.")

        if self.__new_prediction_raster_path == None:
            print("We don't found the prediction raster path")
            return 
        
        pdf_resume_path = Path(self.session_path, f"000_{self.session_path.name}_preview.pdf")

        # Construct color map.
        cmap = ListedColormap([
            np.array(RASTER_CLASS_ID2COLOR.get(i, (0, 0, 0, 255)))[:3] / 255.0
            for i in range(6)
        ])

        # Resampled ortho and pred mask to be lisible on pdf.
        thumb_rgb, seg_resampled = create_fullmap_thumbnail_with_predictions(self.orthophoto_path, self.__new_prediction_raster_path)
        
        # Prepare the figure
        with PdfPages(pdf_resume_path) as pdf:
            
            ## -- Page 1
            plot_first_page(pdf, self.session_path.name, model_name, thumb_rgb, seg_resampled, cmap)

            ## -- Page 2
            plot_distribution_and_legend(pdf, cmap, seg_resampled)
            
            ## -- Page 3 - ...
            plot_patches(pdf, self.orthophoto_path, self.__new_prediction_raster_path, output_folder, self.footprint_buffered)
